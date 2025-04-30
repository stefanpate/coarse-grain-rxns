import torch
from torch import nn, Tensor, optim
from chemprop.nn.message_passing import MessagePassing
from chemprop.schedulers import NoamLR
import lightning
import torch.nn.functional as F

class FFNPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str = 'relu'):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
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
        metrics: list[str] | None = None,
        batch_norm: bool = True,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4
    ):
        self.predictor = predictor
        self.message_passing = message_passing
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        super().__init__()

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
    
    def training_step(self, batch, batch_idx):
        # TODO
        H = self.message_passing(X)
        logits = self.predictor(H)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss