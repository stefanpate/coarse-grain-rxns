import torch
from torch import nn, Tensor, optim
from chemprop.nn.message_passing import MessagePassing
from chemprop.schedulers import NoamLR
from chemprop.data import ReactionDatapoint, BatchMolGraph
from typing import Iterable
import lightning
import torch.nn.functional as F
import torcheval.metrics.functional as MF
import numpy as np
from itertools import accumulate
from rdkit import Chem

'''
Model components
'''

class FFNPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_hs: list[int], activation: str = 'ReLU'):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in d_hs:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(getattr(nn, activation)())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
    

class LinearPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
    
class GNN(lightning.LightningModule):
    def __init__(
        self,
        message_passing: MessagePassing,
        predictor: nn.Module,
        pos_weight: float = 1.0,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["message_passing", "predictor"])
        self.predictor = predictor
        self.pos_weight = torch.Tensor([pos_weight]).reshape(1, 1)
        self.message_passing = message_passing
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    def loss_fn(self, logits: Tensor, y: Tensor) -> float:
        return F.binary_cross_entropy_with_logits(
            input=logits,
            target=y,
            pos_weight=self.pos_weight,
        )

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)

        lr_sched = NoamLR(
            opt,
            self.warmup_epochs,
            self.trainer.max_epochs,
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
            self.init_lr,
            self.max_lr,
            self.final_lr,
        )
        lr_sched_config = {
            "scheduler": lr_sched,
            "interval": "step" if isinstance(lr_sched, NoamLR) else "batch",
        }

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}
    
    def forward(self, batch: tuple[BatchMolGraph, Tensor | None]) -> Tensor:
        bmg, _ = batch
        H = self.message_passing(bmg)
        logits = self.predictor(H)
        probas = F.sigmoid(logits)
        return probas
    
    def training_step(self, batch: tuple[BatchMolGraph, Tensor | None], batch_idx: int) -> Tensor:
        bmg, y = batch
        H = self.message_passing(bmg)
        logits = self.predictor(H)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(bmg))
        return loss
    
    def validation_step(self, batch: tuple[BatchMolGraph, Tensor], batch_idx: int) -> Tensor:
        bmg, y = batch
        H = self.message_passing(bmg)
        logits = self.predictor(H)
        val_loss = self.loss_fn(logits, y)
        probas = F.sigmoid(logits).squeeze()
        y = y.squeeze().to(torch.int)
        acc = MF.binary_accuracy(probas, y)
        rec = MF.binary_recall(probas, y)
        prec = MF.binary_precision(probas, y)
        auroc = MF.binary_auroc(probas, y)
        auprc = MF.binary_auprc(probas, y)
        f1 = MF.binary_f1_score(probas, y)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(bmg))
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(bmg))
        self.log("val_recall", rec, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(bmg))
        self.log("val_precision", prec, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(bmg))
        self.log("val_f1", f1, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(bmg))
        self.log("val_auroc", auroc, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(bmg))
        self.log("val_auprc", auprc, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(bmg))

    def test_step(self, batch: tuple[BatchMolGraph, Tensor], batch_idx: int) -> Tensor:
        bmg, y = batch
        H = self.message_passing(bmg)
        logits = self.predictor(H)
        loss = self.loss_fn(logits, y)
        probas = F.sigmoid(logits).squeeze()
        y = y.squeeze().to(torch.int)
        acc = MF.binary_accuracy(probas, y)
        rec = MF.binary_recall(probas, y)
        prec = MF.binary_precision(probas, y)
        auroc = MF.binary_auroc(probas, y)
        auprc = MF.binary_auprc(probas, y)
        self.log("test_loss", loss, prog_bar=True, batch_size=len(bmg))
        self.log("test_acc", acc, prog_bar=True, batch_size=len(bmg))
        self.log("test_recall", rec, prog_bar=True, batch_size=len(bmg))
        self.log("test_precision", prec, prog_bar=True, batch_size=len(bmg))
        self.log("test_auroc", auroc, prog_bar=True, batch_size=len(bmg))
        self.log("test_auprc", auprc, prog_bar=True, batch_size=len(bmg))

'''
Auxiliary
'''

def collate_batch(batch: Iterable[tuple[ReactionDatapoint, np.ndarray]]) -> tuple[BatchMolGraph, Tensor | None]:
    '''
    Custom collate function concatenates datapoints for torch DataLoader
    '''
    points, labels = zip(*batch)
    batch_mol_graph = BatchMolGraph([point.mg for point in points])
    labels = None if labels[0] is None else torch.from_numpy(np.concatenate(labels)).float()
    return batch_mol_graph, labels

def sep_aidx_to_bin_label(smarts: str, aidxs: tuple[tuple[tuple[int]], tuple[tuple[int]]]) -> tuple[np.ndarray, np.ndarray]:
    '''
    Convert atom indices for separate molecules into a binary label based on block molecules
    i.e., all the molecules in a single mol object.

    Args
    ----
    smarts: str
        SMILES string of the reaction
    aidxs: tuple[tuple[tuple[int]], tuple[tuple[int]]]
        Indices of atoms belonging to the positive class for each molecule
        for each side of the reaction
    Returns
    -------
    ys: tuple[np.ndarray, np.ndarray]
        Binary labels for each side of the reaction ordered according to the order
        of atoms on each side of the reaction
    '''
    ys = []
    smiles = [elt.split(".") for elt in smarts.split(">>")]
    for smi_side, aidx_side in zip(smiles, aidxs):
        offsets = [0] + list(accumulate(Chem.MolFromSmiles(smi).GetNumAtoms() for smi in smi_side))
        block_idxs = []
        for i, elt in enumerate(aidx_side):
            for aidx in elt:
                block_idxs.append(aidx + offsets[i])

        y = np.zeros(shape=(offsets[-1], 1))
        y[block_idxs] = 1
        ys.append(y)

    return tuple(ys)

def calc_bce_pos_weight(y: list[np.ndarray], pw_scl: float) -> float:
    '''
    Calculate the positive weight for BCE loss based on the ratio of positive to negative samples
    '''
    npos = sum([np.sum(elt) for elt in y])
    ntot = sum([elt.shape[0] for elt in y])
    nneg = ntot - npos
    pos_weight = (nneg / npos) * pw_scl
    
    return pos_weight