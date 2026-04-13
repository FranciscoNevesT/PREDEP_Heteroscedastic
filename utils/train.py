from models.estimation import EstimationModel
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from utils.config import TrainConfig
from utils.dataloader import build_dataloader
from utils.model import build_model
from typing import Callable 


def train_one_epoch(
    model: EstimationModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
) -> float:
    """Run one full pass over *loader* and return the mean loss."""
    model.train()
    losses: list[float] = []

    for batch_x, batch_y in loader:
        loss = model.compute_loss(batch_x, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))

def train(
    f: Callable[[float], float],
    cfg: TrainConfig,
) -> EstimationModel:
    """Train a partition model for the given target function *f*."""
    loader, _, _ = build_dataloader(f, cfg)
    model = build_model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
 
    for epoch in range(1, cfg.num_epochs + 1):
        mean_loss = train_one_epoch(model, loader, optimizer)
        if epoch % cfg.log_every == 0:
            print(f"Epoch {epoch:>{len(str(cfg.num_epochs))}}/{cfg.num_epochs}  loss={mean_loss:.4f}")
 
    return model