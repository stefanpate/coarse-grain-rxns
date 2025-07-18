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
import numpy as np
from ergochemics.mapping import rc_to_nest
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
    sep_aidx_to_bin_label,
    calc_bce_pos_weight,
    SklearnGNN
)

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

def reformat_data_for_calibration(dataset) -> tuple[list, np.ndarray]:
    """Reformat data for sklearn calibration compatibility."""
    refmt_dataset = []
    ys = []
    for i, (x, y) in enumerate(dataset):
        ys.append(y)
        for _ in range(y.shape[0]):
            refmt_dataset.append((x, i))
    refmt_y = np.vstack(ys).flatten()
    return refmt_dataset, refmt_y

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
        Path(cfg.mechinformed_mapped_rxns)
    )[:1000] # TODO: remove limit

    # Prep data
    df["template_aidxs"] = df["template_aidxs"].apply(rc_to_nest)
    smis = df["am_smarts"].tolist()
    df["binary_label"] = df.apply(lambda x: sep_aidx_to_bin_label(x.am_smarts, x.template_aidxs), axis=1) # Convert aidxs to binary labels for block mol
    ys = [elt[0] for elt in df["binary_label"]]
    groups = df["rule_id"].tolist() if cfg.data.split_strategy != "random_split" else None
    X, y = zip(*[(data.ReactionDatapoint.from_smi(smi), y) for smi, y in zip(smis, ys)])

    # Split
    outer_splitter = instantiate(cfg.data.outer_splitter)
    train_val_idx, test_idx = list(outer_splitter.split(X, y, groups=groups))[cfg.data.outer_split_idx]
    train_X, train_y = [X[i] for i in train_val_idx], [y[i] for i in train_val_idx]
    test_X, test_y = [X[i] for i in test_idx], [y[i] for i in test_idx]
    test_rxn_ids = [df.iloc[i]["rxn_id"] for i in test_idx]

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=cfg.model.featurizer_mode, atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    
    # Extra split & featurization for calibration
    if cfg.do_calibrate:
        train_X, calib_X, train_y, calib_y = train_test_split(
            train_X, train_y, test_size=cfg.calibration_fraction, random_state=cfg.data.split_seed
        )
        calib_dataset = list(zip(data.ReactionDataset(calib_X, featurizer=featurizer), calib_y))
        calib_dataset, calib_y = reformat_data_for_calibration(calib_dataset)

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
        pos_weight=calc_bce_pos_weight(train_y, cfg.training.pw_scl),
        warmup_epochs=cfg.training.warmup_epochs,
        init_lr=cfg.training.init_lr,
        max_lr=cfg.training.max_lr,
        final_lr=cfg.training.final_lr,
    )

    # Logging
    logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri="file:" + cfg.filepaths.mlruns,
        log_model=True,
        tags={"source": "train.py"},
    )

    mlflow.set_experiment(experiment_id=logger.experiment_id)

    # Train
    log.info("Training model")
    with mlflow.start_run(run_id=logger.run_id):
        trainer = L.Trainer(max_epochs=cfg.training.max_epochs, logger=logger, accelerator="auto", devices=1)
        trainer.fit(model=model, train_dataloaders=train_dataloader)
        trainer.test(model=model, dataloaders=test_dataloader)

        if cfg.do_calibrate:
            calib_trainer = L.Trainer(logger=None, accelerator="auto", devices=1)
            calib_skgnn = CalibratedClassifierCV(
                FrozenEstimator(
                    SklearnGNN(model=model, message_passing=mp, predictor=pred_head, trainer=calib_trainer)
                )
            )
            calib_skgnn.fit(calib_dataset, calib_y)

            test_dataset, _ = reformat_data_for_calibration(test_dataset) # Reformat here too not interfere w/ trainer above
            y_pred = calib_skgnn.predict_proba(test_dataset)[:, 1].reshape(-1, 1) # 2nd row has P(y=1)

        else:
            test_output = trainer.predict(model=model, dataloaders=test_dataloader)
            y_pred = np.vstack([batch.cpu().numpy() for batch in test_output])

        # Format preds for dataframe
        aidxs = np.vstack([np.arange(elt.shape[0]).reshape(-1, 1) for elt in test_y], dtype=np.int32)
        y = np.vstack(test_y)
        df_rxn_ids = []
        for i in range(len(test_y)):
            df_rxn_ids.extend([test_rxn_ids[i]] * test_y[i].shape[0])
        df_rxn_ids = np.array(df_rxn_ids, dtype=np.int32).reshape(-1, 1)

        pred_df = pd.DataFrame(
            data={
                "rxn_id": df_rxn_ids.flatten(),
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
        artifact_path = Path(mlflow.get_artifact_uri().removeprefix("file:"))

        # Save preds
        pred_df.to_parquet(artifact_path / "predictions.parquet", index=False)
        mlflow.log_artifact(artifact_path / "predictions.parquet")

        # Save calibrated model
        if cfg.do_calibrate:
            log.info("Saving calibrated model")
            with open(artifact_path / "calibrated_model.pkl", "wb") as f:
                pickle.dump(calib_skgnn, f)

            mlflow.log_artifact(artifact_path / "calibrated_model.pkl")

        mlflow.log_artifact(Path(".hydra/config.yaml"))

if __name__ == "__main__":
    main()
