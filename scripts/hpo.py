import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
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
import logging
import sys
from optuna.integration import PyTorchLightningPruningCallback
import numpy as np
from functools import partial
import pickle
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
    sep_aidx_to_bin_label,
)

current_dir = Path(__file__).parent.parent.resolve()
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

def objective(trial: optuna.trial.Trial, train_val_X: list[ReactionDatapoint], train_val_y: list[np.ndarray], cfg: DictConfig, train_val_groups: list[int] = []) -> float:
    special = [
        "model/pred_head_d_h"
    ]

    hyperparams = {}
    for group, hps in cfg.hp_bounds.items():
        for hp_name in hps.keys():
            hp = hps[hp_name]
            combined_key = f"{group}/{hp_name}"
            
            if combined_key in special:
                continue
            elif hp['type'] == "categorical":
                hyperparams[combined_key] = trial.suggest_categorical(combined_key, hp['values'])
            elif hp['type'] == "int":
                hyperparams[combined_key] = trial.suggest_int(combined_key, hp['values'][0], hp['values'][1])
            elif hp['type'] == "int_log":
                hyperparams[combined_key] = trial.suggest_int(combined_key, hp['values'][0], hp['values'][1], log=True)

    # Special hyperparameters ~ compositional
    if hyperparams["model/pred_head_name"] == 'linear':
        pred_head = LinearPredictor(input_dim=hyperparams['model/mp_d_h'], output_dim=1)
    elif hyperparams["model/pred_head_name"] == 'ffn':
        pred_head_d_hs = [
            trial.suggest_int(f"model/pred_head_d_h_{i}", cfg.hp_bounds.model.pred_head_d_h['values'][0], cfg.hp_bounds.model.pred_head_d_h['values'][1], log=True)
            for i in range(hyperparams["model/pred_head_depth"])
        ]
        pred_head = FFNPredictor(input_dim=hyperparams['model/mp_d_h'], output_dim=1, d_hs=pred_head_d_hs)


    metrics = []
    inner_splitter = instantiate(cfg.data.inner_splitter)
    for i, (train_idx, val_idx) in enumerate(inner_splitter.split(train_val_X, train_val_y, groups=train_val_groups)):
        print(f"Inner split {i + 1}/{cfg.data.inner_splitter.n_splits}")
        
        # Featurize
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(
            mode_=hyperparams["model/featurizer_mode"],
            atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2()
        )
        train_X = [train_val_X[j] for j in train_idx]
        train_y = [train_val_y[j] for j in train_idx]
        val_X = [train_val_X[j] for j in val_idx]
        val_y = [train_val_y[j] for j in val_idx]
        
        train_dataset = list(zip(data.ReactionDataset(train_X, featurizer=featurizer), train_y))
        val_dataset = list(zip(data.ReactionDataset(val_X, featurizer=featurizer), val_y))
        train_dataloader = DataLoader(train_dataset, batch_size=hyperparams["training/batch_size"], shuffle=True, collate_fn=collate_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

        # Construct model
        mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=hyperparams["model/mp_d_h"], depth=hyperparams["model/mp_depth"])
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
            max_epochs=hyperparams["training/max_epochs"], 
            logger=logger,
            enable_checkpointing=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
            accelerator="auto",
            devices=1
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
    ).iloc[::50] # TODO: Remove this line to use full dataset
    
    # Prep data
    df["reaction_center"] = df["reaction_center"].apply(rc_to_nest)
    smis = df["am_smarts"].tolist()
    df["binary_label"] = df.apply(lambda x: sep_aidx_to_bin_label(x.am_smarts, x.reaction_center), axis=1) # Convert aidxs to binary labels for block mol
    ys = [elt[0] for elt in df["binary_label"]]
    groups = df["rule_id"].tolist() if cfg.data.split_strategy != "random_split" else None
    X, y = zip(*[(data.ReactionDatapoint.from_smi(smi), y) for smi, y in zip(smis, ys)])

    # Split
    outer_splitter = instantiate(cfg.data.outer_splitter)
    train_val_idx, _ = list(outer_splitter.split(X, y, groups=groups))[cfg.data.outer_split_idx]
    train_val_X, train_val_y = [X[i] for i in train_val_idx], [y[i] for i in train_val_idx]
    train_val_groups = [groups[i] for i in train_val_idx] if groups else None

    # Optimize hyperparameters
    print("Optimizing hyperparameters")
    _objective = partial(
        objective,
        train_val_X=train_val_X,
        train_val_y=train_val_y,
        cfg=cfg,
        train_val_groups=train_val_groups
    )
    sampler_path = Path(cfg.filepaths.hpo_studies) / f"{cfg.study_name}_sampler.pkl"
    
    if sampler_path.exists():
        print(f"Loading sampler from {sampler_path}")
        sampler = pickle.load(open(sampler_path, "rb"))
    else:
        print(f"Creating new sampler seeded with {cfg.seed}")
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(),
        study_name=cfg.study_name,
        storage=f"sqlite:///{cfg.filepaths.hpo_studies}/{cfg.study_name}.db",
        load_if_exists=True
    )
    study.optimize(_objective, n_trials=cfg.n_trials, timeout=cfg.timeout)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Val loss: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open(sampler_path, "wb") as f:
        pickle.dump(study.sampler, f)
    
if __name__ == "__main__":
    main()
