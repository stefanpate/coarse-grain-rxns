import lightning as L
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from chemprop import nn, data, featurizers
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import logging
import yaml
from ergochemics.mapping import rc_to_nest
from cgr.rule_writing import filter_by_pub_date
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
    sep_aidx_to_bin_label,
    calc_bce_pos_weight,
    scrub_anonymous_template_atoms
)

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

def write_production_config(cfg, ckpt_path: str, training_set: str, output_path: Path):
    config = {
        "data": {
            "outer_split_idx": int(cfg.data.outer_split_idx),
            "training_set": training_set,
        },
        "model": {
            "ckpt": ckpt_path,
            "featurizer_mode": cfg.model.featurizer_mode,
            "mp_d_h": int(cfg.model.mp_d_h),
            "mp_depth": int(cfg.model.mp_depth),
            "pred_head_d_hs": list(cfg.model.pred_head_d_hs),
            "pred_head_name": cfg.model.pred_head_name,
        },
        "training": {
            "batch_size": int(cfg.training.batch_size),
            "final_lr": float(cfg.training.final_lr),
            "init_lr": float(cfg.training.init_lr),
            "max_epochs": int(cfg.training.max_epochs),
            "max_lr": float(cfg.training.max_lr),
            "pw_scl": float(cfg.training.pw_scl),
            "warmup_epochs": int(cfg.training.warmup_epochs),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def prep_and_featurize(df: pd.DataFrame, featurizer) -> tuple[list, list, list]:
    df = df.copy()
    df["template_aidxs"] = df["template_aidxs"].apply(rc_to_nest)
    df["template_aidxs"] = df.apply(lambda x: scrub_anonymous_template_atoms(x.template_aidxs, x.rule), axis=1)
    df["binary_label"] = df.apply(lambda x: sep_aidx_to_bin_label(x.am_smarts, x.template_aidxs), axis=1)
    smis = df["am_smarts"].tolist()
    ys = [elt[0] for elt in df["binary_label"]]
    rxn_ids = df["rxn_id"].tolist()
    X, y = zip(*[(data.ReactionDatapoint.from_smi(smi), y) for smi, y in zip(smis, ys)])
    dataset = list(zip(data.ReactionDataset(list(X), featurizer=featurizer), y))
    return dataset, list(y), rxn_ids

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name='train_production')
def main(cfg: DictConfig):
    # Load data
    log.info("Loading & preparing data")
    df = pd.read_parquet(
        Path(cfg.filepaths.mechinformed_mapped_rxns)
    )

    # Filter to direct MCSA only if specified
    if cfg.direct_mcsa_only:
        mm = pd.read_parquet(
            Path(cfg.filepaths.raw_data) / "distilled_mech_reactions.parquet"
        )
        df = df[df['rxn_id'].isin(mm['rxn_id'])]

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=cfg.model.featurizer_mode, atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())

    if cfg.cutoff_date is not None:
        pub_dates = pd.read_parquet(Path(cfg.filepaths.raw_data) / cfg.pub_dates_file)
        df_before = filter_by_pub_date(pub_dates, df, cfg.cutoff_date, mode="before")
        df_after = filter_by_pub_date(pub_dates, df, cfg.cutoff_date, mode="after")

        train_dataset, train_y, _ = prep_and_featurize(df_before, featurizer)
        test_dataset, test_y, test_rxn_ids = prep_and_featurize(df_after, featurizer)
    else:
        train_dataset, train_y, _ = prep_and_featurize(df, featurizer)
        test_dataset, test_y, test_rxn_ids = None, None, None

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_batch)

    # Construct model
    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=cfg.model.mp_d_h, depth=cfg.model.mp_depth)
    if cfg.model.pred_head_name == 'linear':
        pred_head = LinearPredictor(input_dim=cfg.model.mp_d_h, output_dim=1)
    elif cfg.model.pred_head_name == 'ffn':
        pred_head = FFNPredictor(input_dim=cfg.model.mp_d_h, output_dim=1, d_hs=cfg.model.pred_head_d_hs)

    model = GNN(
        message_passing=mp,
        predictor=pred_head,
        pos_weight=calc_bce_pos_weight(train_y, cfg.training.pw_scl),
        warmup_epochs=cfg.training.warmup_epochs,
        init_lr=cfg.training.init_lr,
        max_lr=cfg.training.max_lr,
        final_lr=cfg.training.final_lr,
    )

    # Logging
    exp_name = "production" if cfg.cutoff_date is None else f"before_{cfg.cutoff_date}"
    if cfg.direct_mcsa_only:
        exp_name += "_direct_mcsa_only"
    logger = MLFlowLogger(
        experiment_name=exp_name,
        tracking_uri="file:" + cfg.filepaths.mlruns,
        log_model=True,
    )

    mlflow.set_experiment(experiment_id=logger.experiment_id)

    # Train
    log.info("Training model")
    with mlflow.start_run(run_id=logger.run_id):
        trainer = L.Trainer(max_epochs=cfg.training.max_epochs, logger=logger, accelerator="auto", devices=1)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

        # Write production config for downstream inference
        best_ckpt = trainer.checkpoint_callback.best_model_path
        rel_ckpt = str(Path(best_ckpt).relative_to(cfg.filepaths.mlruns))
        if cfg.cutoff_date is None and cfg.direct_mcsa_only:
            training_set = "direct_mcsa_only"
            config_filename = f"direct_mcsa_only_{cfg.data.outer_split_idx}.yaml"
        elif cfg.cutoff_date is None and not cfg.direct_mcsa_only:
            training_set = "all_data"
            config_filename = f"all_data_{cfg.data.outer_split_idx}.yaml"
        elif cfg.cutoff_date is not None and cfg.direct_mcsa_only:
            training_set = f"before_{cfg.cutoff_date}_direct_mcsa_only"
            config_filename = f"before_{cfg.cutoff_date}_direct_mcsa_only_{cfg.data.outer_split_idx}.yaml"
        else:
            training_set = f"before_{cfg.cutoff_date}"
            config_filename = f"before_{cfg.cutoff_date}_{cfg.data.outer_split_idx}.yaml"
        write_production_config(
            cfg,
            ckpt_path=rel_ckpt,
            training_set=training_set,
            output_path=current_dir / "configs" / "production" / config_filename,
        )
        log.info(f"Wrote production config to configs/production/{config_filename}")

        if cfg.cutoff_date is not None:
            test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)
            test_output = trainer.predict(model=model, dataloaders=test_dataloader)
            y_pred = np.vstack([batch.cpu().numpy() for batch in test_output])

            aidxs = np.vstack([np.arange(elt.shape[0]).reshape(-1, 1) for elt in test_y], dtype=np.int32)
            y = np.vstack(test_y)
            df_rxn_ids = []
            for i in range(len(test_y)):
                df_rxn_ids.extend([test_rxn_ids[i]] * test_y[i].shape[0])

            pred_df = pd.DataFrame(
                data={
                    "rxn_id": df_rxn_ids,
                    "aidx": aidxs.flatten(),
                    "y": y.flatten(),
                    "y_pred": y_pred.flatten()
                }
            )

        # Save and log artifacts & params
        flat_resolved_cfg = pd.json_normalize(
            {k: v for k,v in OmegaConf.to_container(cfg, resolve=True).items() if k != 'filepaths'}, # Resolved interpolated values
            sep='/'
        ).to_dict(orient='records')[0]
        mlflow.log_params(flat_resolved_cfg)

        if cfg.cutoff_date is not None:
            artifact_path = Path(mlflow.get_artifact_uri().removeprefix("file:"))
            pred_df.to_parquet(artifact_path / "predictions.parquet", index=False)
            mlflow.log_artifact(artifact_path / "predictions.parquet")

if __name__ == "__main__":
    main()
