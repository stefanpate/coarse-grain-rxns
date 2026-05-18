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
from cgr.ml import (
    GNN,
    FFNPredictor,
    LinearPredictor,
    collate_batch,
)

current_dir = Path(__file__).parent.parent.resolve()
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(current_dir / "configs"), config_name='infer_mech_labels')
def main(cfg: DictConfig):
    # Load data
    log.info("Loading & preparing data")
    mapped_rxns_path = Path(cfg.filepaths.mappings) / f"{cfg.mapped_rxns}.parquet"
    df = pd.read_parquet(mapped_rxns_path)

    # TODO: remove. just for debuggin
    from ergochemics.standardize import standardize_reaction
    df['am_smarts'] = df['am_smarts'].apply(standardize_reaction)
    # END TODO

    # Prep data
    smis = df["am_smarts"].tolist()
    X = [data.ReactionDatapoint.from_smi(smi) for smi in smis]
    rxn_ids = df["rxn_id"].tolist()

    # Featurize
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=cfg.model.featurizer_mode, atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    featurized = list(data.ReactionDataset(X, featurizer=featurizer))

    # Drop reactions that chemprop can't collate (1-D bond feature array — e.g. no-bond CGRs)
    keep = [p.mg.E.ndim == 2 for p in featurized]
    n_dropped = sum(1 for k in keep if not k)
    if n_dropped:
        log.warning(f"Dropping {n_dropped} reactions with malformed bond features (no-bond CGR)")
    featurized = [p for p, k in zip(featurized, keep) if k]
    rxn_ids = [rid for rid, k in zip(rxn_ids, keep) if k]

    # Placeholder y vectors (all zeros) sized to each reaction's actual CGR atom count.
    # `trainer.predict` does NOT consume these — they exist only to satisfy
    # collate_batch's (datapoint, label) tuple shape and to give us per-reaction
    # atom counts that line up with the model's prediction tensors downstream.
    y = [np.zeros((p.mg.V.shape[0], 1)) for p in featurized]

    dataset = list(zip(featurized, y))
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
    trainer = L.Trainer(logger=None, accelerator="auto", devices=1)
    probas = trainer.predict(model=model, dataloaders=dataloader)

    # Format preds
    probas = np.vstack([batch.cpu().numpy() for batch in probas])
    aidxs = np.vstack([np.arange(elt.shape[0]).reshape(-1, 1) for elt in y], dtype=np.int32)
    df_rxn_ids = []
    for i in range(len(y)):
        df_rxn_ids.extend([rxn_ids[i]] * y[i].shape[0])
    df_rxn_ids = np.array(df_rxn_ids).reshape(-1, 1)

    pred_df = pd.DataFrame(
        data={
            "rxn_id": df_rxn_ids.flatten(),
            "aidx": aidxs.flatten(),
            "probas": probas.flatten()
        }
    )

    # Save
    prediction_data = mapped_rxns_path.stem
    out_name = f"train_{cfg.data.training_set}_predict_{prediction_data}_split_{cfg.data.outer_split_idx}.parquet"
    pred_df.to_parquet(Path(cfg.filepaths.processed_data) / "mech_probas" / out_name, index=False)

if __name__ == "__main__":
    main()