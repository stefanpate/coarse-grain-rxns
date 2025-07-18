import lightning
from chemprop import nn, data, featurizers
from pathlib import Path
import pandas as pd
import numpy as np
from ergochemics.mapping import rc_to_nest
from torch.utils.data import DataLoader
from hydra import initialize, compose
from hydra.utils import instantiate
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

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(config_name="train")


df = pd.read_parquet(
        Path(cfg.mechinformed_mapped_rxns)
)[:1000]

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

trainer = lightning.Trainer(logger=None, accelerator="auto", devices=1)

skgnn = SklearnGNN(model=model, message_passing=mp, predictor=pred_head, trainer=trainer)

expanded_test_dataset = []
for i, (x, y) in enumerate(test_dataset):
    for j in range(y.shape[0]):
        expanded_test_dataset.append((x, i))

print(skgnn.predict_proba(expanded_test_dataset[:154]).shape)

calibrated_skgnn = CalibratedClassifierCV(FrozenEstimator(skgnn))
calibrated_skgnn.fit(expanded_test_dataset[:154], np.vstack(test_y[:2]).ravel())

print(calibrated_skgnn.predict_proba(expanded_test_dataset[:154]))

import pickle

with open("calibrated_skgnn.pkl", "wb") as f:
    pickle.dump(calibrated_skgnn, f)

with open("calibrated_skgnn.pkl", "rb") as f:
    calibrated_skgnn = pickle.load(f)

print(calibrated_skgnn.predict_proba(expanded_test_dataset[:154]))
print("done")
