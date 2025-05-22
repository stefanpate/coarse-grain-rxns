import lightning as L
from chemprop import nn, data, featurizers
import hydra
from omegaconf import DictConfig
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
)

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name='infer_mech_labels')
def main(cfg: DictConfig):
    # Load data
    log.info("Loading & preparing data")
    df = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / "mapped_sprhea_240310_v3_mapped_no_subunits_x_rc_plus_0_rules.parquet" # TODO: Update with new data
    )

    # Prep data
    df["template_aidxs"] = df["template_aidxs"].apply(rc_to_nest)
    smis = df["am_smarts"].tolist()
    df["binary_label"] = df.apply(lambda x: sep_aidx_to_bin_label(x.am_smarts, x.template_aidxs), axis=1) # Convert aidxs to binary labels for block mol
    ys = [elt[0] for elt in df["binary_label"]]
    X, y = zip(*[(data.ReactionDatapoint.from_smi(smi), y) for smi, y in zip(smis, ys)])
    rxn_ids = df["rxn_id"].tolist()

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=cfg.model.featurizer_mode, atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    dataset = list(zip(data.ReactionDataset(X, featurizer=featurizer), y))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_batch)
    
    # Construct model
    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=cfg.model.mp_d_h, depth=cfg.model.mp_depth)
    if cfg.model.pred_head_name == 'linear':
        pred_head = LinearPredictor(input_dim=cfg.model.mp_d_h, output_dim=1)
    elif cfg.model.pred_head_name == 'ffn':
        pred_head = FFNPredictor(input_dim=cfg.model.mp_d_h, output_dim=1, d_hs=cfg.model.pred_head_d_hs)

    model = GNN.load_from_checkpoint(
        Path(cfg.filepaths.mlruns) / cfg.model.ckpt,
        message_passing=mp,
        predictor=pred_head,
    )

    # Predict
    log.info("Predicting")
    trainer = L.Trainer(max_epochs=cfg.training.max_epochs, logger=None, accelerator="auto", devices=1)
    probas = trainer.predict(model=model, dataloaders=dataloader)

    # Format preds
    probas = np.vstack([batch.cpu().numpy() for batch in probas])
    aidxs = np.vstack([np.arange(elt.shape[0]).reshape(-1, 1) for elt in y], dtype=np.int32)
    df_rxn_ids = []
    for i in range(len(y)):
        df_rxn_ids.extend([rxn_ids[i]] * y[i].shape[0])
    df_rxn_ids = np.array(df_rxn_ids, dtype=np.int32).reshape(-1, 1)
    y = np.vstack(y)

    pred_df = pd.DataFrame(
        data={
            "rxn_id": df_rxn_ids.flatten(),
            "aidx": aidxs.flatten(),
            "y": y.flatten(),
            "probas": probas.flatten()
        }
    )

    # Save
    pred_df.to_parquet(Path(cfg.filepaths.processed_data) / "mech_probas" / f"{cfg.data.outer_split_idx}.parquet", index=False)

if __name__ == "__main__":
    main()