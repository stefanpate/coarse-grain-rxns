import lightning as L
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from chemprop import nn, data, featurizers
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import optuna
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest
from torch.utils.data import DataLoader
import logging
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
    sep_aidx_to_bin_label,
)

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="train")
def main(cfg: DictConfig):

    # Override config with best trial from hpo
    if cfg.use_study:
        study = optuna.load_study(
            study_name=cfg.study_name,
            storage=f"sqlite:///{cfg.filepaths.hpo_studies}/{cfg.study_name}.db"
        )

        pred_head_d_hs = []
        for k, v in study.best_trial.params.items():
            group, hp_name = k.split('/')
            if group == "model" and hp_name.startswith("pred_head_d_h_"):
                pred_head_d_hs.append((int(hp_name.split('_')[-1]), v))

            if group in cfg.keys() and hp_name in cfg[group].keys():
                cfg[group][hp_name] = v

        pred_head_d_hs = sorted(pred_head_d_hs, key=lambda x: x[0])
        pred_head_d_hs = [v for _, v in pred_head_d_hs]
        cfg.model.pred_head_d_hs = pred_head_d_hs

    # Load data
    log.info("Loading & preparing data")
    df = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / "mapped_sprhea_240310_v3_mapped_no_subunits_x_mechanistic_rules.parquet"
    )
    
    # Prep data
    df["reaction_center"] = df["reaction_center"].apply(rc_to_nest)
    smis = df["am_smarts"].tolist()
    df["binary_label"] = df.apply(lambda x: sep_aidx_to_bin_label(x.am_smarts, x.reaction_center), axis=1) # Convert aidxs to binary labels for block mol
    ys = [elt[0] for elt in df["binary_label"]]
    groups = df["rule_id"].tolist() if cfg.data.split_strategy != "random_split" else None
    X, y = zip(*[(data.ReactionDatapoint.from_smi(smi), y) for smi, y in zip(smis, ys)])

    # Split
    outer_splitter = instantiate(cfg.data.outer_splitter)
    train_val_idx, test_idx = list(outer_splitter.split(X, y, groups=groups))[cfg.data.outer_split_idx]
    train_X, train_y = [X[i] for i in train_val_idx], [y[i] for i in train_val_idx]
    test_X, test_y = [X[i] for i in test_idx], [y[i] for i in test_idx]

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=cfg.model.featurizer_mode, atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    train_dataset = list(zip(data.ReactionDataset(train_X, featurizer=featurizer), train_y))
    test_dataset = list(zip(data.ReactionDataset(test_X, featurizer=featurizer), test_y))
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

    # Construct model
    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=cfg.model.mp_d_h, depth=cfg.model.mp_depth)
    if cfg.model.pred_head_name == 'linear':
        pred_head = LinearPredictor(input_dim=cfg.model.mp_d_h, output_dim=1)
    elif cfg.model.pred_head_name == 'ffn':
        pred_head = FFNPredictor(input_dim=cfg.model.mp_d_h, output_dim=1, d_hs=cfg.model.pred_head_d_hs)

    model = GNN(
        message_passing=mp,
        predictor=pred_head,
        warmup_epochs=cfg.training.warmup_epochs,
        init_lr=cfg.training.init_lr,
        max_lr=cfg.training.max_lr,
        final_lr=cfg.training.final_lr,
    )

    # Logging
    logger = MLFlowLogger(
        experiment_name="outer_splits",
        tracking_uri="file:" + cfg.filepaths.mlruns,
        log_model=False,
        tags={"source": "train.py"},
    )

    mlflow.set_experiment(experiment_id=logger.experiment_id)

    # Train
    log.info("Training model")
    with mlflow.start_run(run_id=logger.run_id):
        flat_resolved_cfg = pd.json_normalize(
            {k: v for k,v in OmegaConf.to_container(cfg, resolve=True).items() if k != 'filepaths'}, # Resolved interpolated values
            sep='/'
        ).to_dict(orient='records')[0]
        mlflow.log_params(flat_resolved_cfg)
        
    trainer = L.Trainer(max_epochs=cfg.training.max_epochs, logger=logger, accelerator="auto", devices=1)
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)
    
if __name__ == "__main__":
    main()
