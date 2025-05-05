import lightning as L
from lightning.pytorch.loggers import MLFlowLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import nn, data, featurizers
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from ergochemics.mapping import rc_to_nest
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
    sep_aidx_to_bin_label,
)

current_dir = Path(__file__).parent.parent.resolve()

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="train")
def main(cfg: DictConfig):
    pass

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

    X, y = zip(*[(data.ReactionDatapoint.from_smi(smi), y) for smi, y in zip(smis, ys)])

    # Split
    train_val_X, test_X, train_val_y, test_y = train_test_split(X, y)
    train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y)

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="PROD_DIFF", atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    train_dataset = list(zip(data.ReactionDataset(train_X, featurizer=featurizer), train_y))
    val_dataset = list(zip(data.ReactionDataset(val_X, featurizer=featurizer), val_y))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

    # Construct model
    print("Constructing model")
    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim)
    pred_head = LinearPredictor(input_dim=mp.output_dim, output_dim=1)
    model = GNN(
        message_passing=mp,
        predictor=pred_head,
        warmup_epochs=cfg.training.warmup_epochs,
        init_lr=cfg.training.init_lr,
        max_lr=cfg.training.max_lr,
        final_lr=cfg.training.final_lr
    )

    logger = MLFlowLogger(
        experiment_name="test",
        tracking_uri="file:" + cfg.filepaths.mlruns,
        log_model=True,
    )

    # Train
    print("Training model")
    trainer = L.Trainer(max_epochs=4, logger=logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    print(trainer.log_dir)
    
if __name__ == "__main__":
    main()
