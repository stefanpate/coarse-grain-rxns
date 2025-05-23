import lightning as L
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from chemprop import nn, data, featurizers
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from torch.utils.data import DataLoader
import logging
from ergochemics.mapping import rc_to_nest
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
    sep_aidx_to_bin_label,
    calc_bce_pos_weight
)

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name='train_production')
def main(cfg: DictConfig):
    # Load data
    log.info("Loading & preparing data")
    df = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / "mapped_sprhea_240310_v3_mapped_no_subunits_x_mechinformed_rules.parquet"
    )

    # Prep data
    df["template_aidxs"] = df["template_aidxs"].apply(rc_to_nest)
    smis = df["am_smarts"].tolist()
    df["binary_label"] = df.apply(lambda x: sep_aidx_to_bin_label(x.am_smarts, x.template_aidxs), axis=1) # Convert aidxs to binary labels for block mol
    ys = [elt[0] for elt in df["binary_label"]]
    X, y = zip(*[(data.ReactionDatapoint.from_smi(smi), y) for smi, y in zip(smis, ys)])

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=cfg.model.featurizer_mode, atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    dataset = list(zip(data.ReactionDataset(X, featurizer=featurizer), y))
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_batch)
    
    # Construct model
    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=cfg.model.mp_d_h, depth=cfg.model.mp_depth)
    if cfg.model.pred_head_name == 'linear':
        pred_head = LinearPredictor(input_dim=cfg.model.mp_d_h, output_dim=1)
    elif cfg.model.pred_head_name == 'ffn':
        pred_head = FFNPredictor(input_dim=cfg.model.mp_d_h, output_dim=1, d_hs=cfg.model.pred_head_d_hs)

    model = GNN(
        message_passing=mp,
        predictor=pred_head,
        pos_weight=calc_bce_pos_weight(y, cfg.training.pw_scl),
        warmup_epochs=cfg.training.warmup_epochs,
        init_lr=cfg.training.init_lr,
        max_lr=cfg.training.max_lr,
        final_lr=cfg.training.final_lr,
    )

    # Logging
    logger = MLFlowLogger(
        experiment_name="production",
        tracking_uri="file:" + cfg.filepaths.mlruns,
        log_model=True,
    )

    mlflow.set_experiment(experiment_id=logger.experiment_id)

    # Train
    log.info("Training model")
    with mlflow.start_run(run_id=logger.run_id):
        trainer = L.Trainer(max_epochs=cfg.training.max_epochs, logger=logger, accelerator="auto", devices=1)
        trainer.fit(model=model, train_dataloaders=dataloader)

        # Save and log artifacts & params
        flat_resolved_cfg = pd.json_normalize(
            {k: v for k,v in OmegaConf.to_container(cfg, resolve=True).items() if k != 'filepaths'}, # Resolved interpolated values
            sep='/'
        ).to_dict(orient='records')[0]
        mlflow.log_params(flat_resolved_cfg)

if __name__ == "__main__":
    main()