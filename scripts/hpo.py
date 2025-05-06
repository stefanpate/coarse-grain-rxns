import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from chemprop import nn, data, featurizers
from chemprop.data import ReactionDatapoint
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest
from torch.utils.data import DataLoader
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import numpy as np
from functools import partial
import mlflow
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
    sep_aidx_to_bin_label,
)

current_dir = Path(__file__).parent.parent.resolve()

def objective(trial: optuna.trial.Trial, train_val_X: list[ReactionDatapoint], train_val_y: list[np.ndarray], cfg: DictConfig, train_val_groups: list[int] = []) -> float:
    mp_d_h = trial.suggest_int("mp_d_h", cfg.hp_bounds.mp_d_h[0], cfg.hp_bounds.mp_d_h[1], log=True)
    mp_depth = trial.suggest_int("mp_depth", cfg.hp_bounds.mp_depth[0], cfg.hp_bounds.mp_depth[1])
    batch_size = trial.suggest_int("batch_size", cfg.hp_bounds.batch_size[0], cfg.hp_bounds.batch_size[1])
    featurizer_mode = trial.suggest_categorical("featurizer_mode", cfg.hp_bounds.featurizer_mode)
    pred_head_name = trial.suggest_categorical("pred_head_name", cfg.hp_bounds.pred_head_name)
    max_epochs = trial.suggest_int("max_epochs", cfg.hp_bounds.max_epochs[0], cfg.hp_bounds.max_epochs[1])

    if pred_head_name == 'linear':
        pred_head = LinearPredictor(input_dim=mp_d_h, output_dim=1)
    elif pred_head_name == 'ffn':
        pred_head_depth = trial.suggest_int("pred_head_depth", cfg.hp_bounds.pred_head_depth[0], cfg.hp_bounds.pred_head_depth[1])
        pred_head_d_hs = [
            trial.suggest_int("pred_head_d_h", cfg.hp_bounds.pred_head_d_h[0], cfg.hp_bounds.pred_head_d_h[1], log=True)
            for _ in range(pred_head_depth)
        ]
        pred_head = FFNPredictor(input_dim=mp_d_h, output_dim=1, d_hs=pred_head_d_hs)


    metrics = []
    inner_splitter = instantiate(cfg.data.inner_splitter)
    for i, (train_idx, val_idx) in enumerate(inner_splitter.split(train_val_X, train_val_y, groups=train_val_groups)):
        print(f"Inner split {i + 1}/{cfg.data.inner_splitter.n_splits}")
        
        # Featurize
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(
            mode_=featurizer_mode,
            atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2()
        )
        train_X = [train_val_X[j] for j in train_idx]
        train_y = [train_val_y[j] for j in train_idx]
        val_X = [train_val_X[j] for j in val_idx]
        val_y = [train_val_y[j] for j in val_idx]
        
        train_dataset = list(zip(data.ReactionDataset(train_X, featurizer=featurizer), train_y))
        val_dataset = list(zip(data.ReactionDataset(val_X, featurizer=featurizer), val_y))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

        # Construct model
        mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=mp_d_h, depth=mp_depth)
        model = GNN(
            message_passing=mp,
            predictor=pred_head,
            warmup_epochs=cfg.training.warmup_epochs,
            init_lr=cfg.training.init_lr,
            max_lr=cfg.training.max_lr,
            final_lr=cfg.training.final_lr
        )

        logger = MLFlowLogger(
            experiment_name="inner_splits",
            tracking_uri="file:" + cfg.filepaths.mlruns,
            log_model=False,
        )

        # Train
        trainer = L.Trainer(
            max_epochs=max_epochs, 
            logger=logger,
            enable_checkpointing=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        )
        trainer.logger.log_hyperparams(
            dict(
                mp_d_h=mp_d_h,
                mp_depth=mp_depth,
                batch_size=batch_size,
                featurizer_mode=featurizer_mode,
                pred_head_name=pred_head_name,
                pred_head_depth=pred_head_depth if cfg.hp_bounds.pred_head_name == 'ffn' else None,
                pred_head_d_hs=pred_head_d_hs if cfg.hp_bounds.pred_head_name == 'ffn' else None,
                inner_split_idx=i,
            )
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
    study = optuna.create_study(
        direction="minimize",
        sampler= optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(_objective, n_trials=cfg.n_trials, timeout=cfg.timeout)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Val loss: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train final model with best hyperparameters
    best_hps = study.best_trial.params

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=best_hps["featurizer_mode"], atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    train_val_dataset = list(zip(data.ReactionDataset(train_val_X, featurizer=featurizer), train_val_y))
    test_dataset = list(zip(data.ReactionDataset(test_X, featurizer=featurizer), test_y))
    train_val_dataloader = DataLoader(train_val_dataset, batch_size=best_hps['batch_size'], shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

    # Construct model
    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=best_hps['mp_d_h'], depth=best_hps['mp_depth'])
    if best_hps['pred_head_name'] == 'linear':
        pred_head = LinearPredictor(input_dim=best_hps['mp_d_h'], output_dim=1)
    elif best_hps['pred_head_name'] == 'ffn':
        pred_head = FFNPredictor(input_dim=best_hps['mp_d_h'], output_dim=1, d_hs=best_hps['pred_head_d_hs'])

    model = GNN(
        message_passing=mp,
        predictor=pred_head,
        warmup_epochs=cfg.training.warmup_epochs,
        init_lr=cfg.training.init_lr,
        max_lr=cfg.training.max_lr,
        final_lr=cfg.training.final_lr,
    )

    # Logging
    to_log = [
        "data/outer_split_idx",
        "data/split_strategy",
    ]

    logger = MLFlowLogger(
        experiment_name="outer_splits",
        tracking_uri="file:" + cfg.filepaths.mlruns,
        log_model=False,
        tags={"source": "hpo.py"},
    )

    mlflow.set_experiment(experiment_id=logger.experiment_id)

    # Train
    print("Training model")
    with mlflow.start_run(run_id=logger.run_id):
        flat_resolved_cfg = pd.json_normalize(
            {k: v for k,v in OmegaConf.to_container(cfg, resolve=True).items() if k != 'filepaths'}, # Resolved interpolated values
            sep='/'
        ).to_dict(orient='records')[0]
        mlflow.log_params({k: v for k, v in flat_resolved_cfg.items() if k in to_log})
        mlflow.log_params(study.best_trial.params)

        # Train and test
        trainer = L.Trainer(max_epochs=best_hps['max_epochs'], logger=logger)
        trainer.fit(model=model, train_dataloaders=train_val_dataloader)
        trainer.test(model=model, dataloaders=test_dataloader)
    
if __name__ == "__main__":
    main()
