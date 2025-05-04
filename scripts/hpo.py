import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from chemprop import nn, data, featurizers
from chemprop.data import ReactionDatapoint
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest
from torch.utils.data import DataLoader
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import numpy as np
from functools import partial
import mlflow
from omegaconf import OmegaConf
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
    sep_aidx_to_bin_label,
)

current_dir = Path(__file__).parent.parent.resolve()

def objective(trial: optuna.trial.Trial, train_val_X: list[ReactionDatapoint], train_val_y: list[np.ndarray], cfg: DictConfig, train_val_groups: list[int] = []) -> float:
    hidden_dim = trial.suggest_int("hidden_dim", 200, 250, log=True)
    
    metrics = []
    inner_splitter = instantiate(cfg.data.inner_splitter)
    for i, (train_idx, val_idx) in enumerate(inner_splitter.split(train_val_X, train_val_y, groups=train_val_groups)):
        print(f"Inner split {i + 1}/{cfg.data.inner_splitter.n_splits}")
        
        # Featurize
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="PROD_DIFF", atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
        train_X = [train_val_X[j] for j in train_idx]
        train_y = [train_val_y[j] for j in train_idx]
        val_X = [train_val_X[j] for j in val_idx]
        val_y = [train_val_y[j] for j in val_idx]
        
        train_dataset = list(zip(data.ReactionDataset(train_X, featurizer=featurizer), train_y))
        val_dataset = list(zip(data.ReactionDataset(val_X, featurizer=featurizer), val_y))
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

        # Construct model
        mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=hidden_dim)
        pred_head = LinearPredictor(input_dim=mp.output_dim, output_dim=1)
        
        model = GNN(
            message_passing=mp,
            predictor=pred_head,
            warmup_epochs=cfg.training.warmup_epochs,
            init_lr=cfg.training.init_lr,
            max_lr=cfg.training.max_lr,
            final_lr=cfg.training.final_lr
        )

        # Train
        trainer = L.Trainer(
            max_epochs=cfg.training.hpo_max_epochs, 
            logger=False, 
            enable_checkpointing=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        )
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Evaluate
        metrics.append(trainer.callback_metrics["val_loss"].item())

    return np.mean(metrics)

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="hpo")
def main(cfg: DictConfig):
    # Load data
    print("Loading & preparing data")
    df = pd.read_parquet(
    Path(cfg.filepaths.raw_data) / "mapped_sprhea_240310_v3_mapped_no_subunits_x_mechanistic_rules.parquet"
    ).iloc[::50] # TODO remove post dev
    
    # Prep data
    df["reaction_center"] = df["reaction_center"].apply(rc_to_nest)
    smis = df["am_smarts"].tolist()
    df["binary_label"] = df.apply(lambda x: sep_aidx_to_bin_label(x.am_smarts, x.reaction_center), axis=1) # Convert aidxs to binary labels for block mol
    ys = [elt[0] for elt in df["binary_label"]]
    groups = df["rule_id"].tolist() if cfg.data.split_strategy != "split_random" else None
    X, y = zip(*[(data.ReactionDatapoint.from_smi(smi), y) for smi, y in zip(smis, ys)])

    # Split
    outer_splitter = instantiate(cfg.data.outer_splitter)
    train_val_idx, test_idx = list(outer_splitter.split(X, y, groups=groups))[cfg.data.outer_split_idx]
    train_val_X, train_val_y = [X[i] for i in train_val_idx], [y[i] for i in train_val_idx]
    train_val_groups = [groups[i] for i in train_val_idx] if groups else None
    test_X, test_y = [X[i] for i in test_idx], [y[i] for i in test_idx]

    # Optimize hyperparameters
    print("Optimizing hyperparameters")
    _objective = partial(
        objective,
        train_val_X=train_val_X,
        train_val_y=train_val_y,
        cfg=cfg,
        train_val_groups=train_val_groups
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=cfg.n_trials, timeout=cfg.timeout)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    # Train final model with best hyperparameters
    best_hidden_dim = study.best_trial.params["hidden_dim"]

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="PROD_DIFF", atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    train_val_dataset = list(zip(data.ReactionDataset(train_val_X, featurizer=featurizer), train_val_y))
    test_dataset = list(zip(data.ReactionDataset(test_X, featurizer=featurizer), test_y))
    train_val_dataloader = DataLoader(train_val_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

    # Construct model
    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=best_hidden_dim)
    pred_head = LinearPredictor(input_dim=mp.output_dim, output_dim=1)

    model = GNN(
        message_passing=mp,
        predictor=pred_head,
        warmup_epochs=cfg.training.warmup_epochs,
        init_lr=cfg.training.init_lr,
        max_lr=cfg.training.max_lr,
        final_lr=cfg.training.final_lr,
    )

    # Callbacks
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        check_on_train_epoch_end=False,
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.2f}.ckpt",
    )

    # Logger
    logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri="file:" + cfg.filepaths.mlruns,
        log_model=True,
    )

    mlflow.set_experiment(experiment_id=logger.experiment_id)

    # Train
    with mlflow.start_run(run_id=logger.run_id):
        flat_resolved_cfg = pd.json_normalize(
            {k: v for k,v in OmegaConf.to_container(cfg, resolve=True).items() if k != 'filepaths'}, # Resolved interpolated values
            sep='/'
        ).to_dict(orient='records')[0]
        # mlflow.log_params(flat_resolved_cfg)
        mlflow.log_params(study.best_trial.params)

        # Train and test
        trainer = L.Trainer(
            max_epochs=cfg.training.max_epochs,
            logger=logger,
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
            ]
        )
        trainer.fit(model=model, train_dataloaders=train_val_dataloader)
        test_metrics = trainer.test(model=model, dataloaders=test_dataloader)

    print("Test metrics:", test_metrics)
    
if __name__ == "__main__":
    main()
