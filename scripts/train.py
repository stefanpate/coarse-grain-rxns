import lightning as L
from cgr.model import GNN, FFNPredictor, LinearPredictor
from chemprop import nn, data, featurizers
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from ergochemics import rc_to_nest

current_dir = Path(__file__).parent.resolve()

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name="train")
def main(cfg: DictConfig):
    pass

    # Load data
    df = pd.read_parquet(
    Path(cfg.filepaths.raw_data) / "mapped_sprhea_240310_v3_mapped_no_subunits_x_mechanistic_rules.parquet"
    )
    
    # Featurize data
    df["reaction_center"] = df["reaction_center"].apply(rc_to_nest)
    smis = df["am_smarts"].tolist()
    train_data = [data.ReactionDatapoint.from_smi(smi) for smi in smis]
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="PROD_DIFF", atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    train_dataset = data.ReactionDataset(train_data, featurizer=featurizer)
    train_dataloader = data.build_dataloader(train_dataset, shuffle=False)

    # Construct model
    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim)
    pred_head = LinearPredictor(input_dim=mp.output_dim, output_dim=1)
    model = GNN(
        message_passing=mp,
        predictor=,
        metrics=[],
        batch_norm=True,
        warmup_epochs=cfg.training.warmup_epochs,
        init_lr=cfg.training.init_lr,
        max_lr=cfg.training.max_lr,
        final_lr=cfg.training.final_lr
    )

    # Train
    trainer = L.Trainer()
    trainer.fit(model=model, train_dataloaders=train_dataloader)

    # Save model

if __name__ == "__main__":
    main()
