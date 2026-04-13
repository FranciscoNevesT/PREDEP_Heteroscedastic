    

from collections.abc import Callable
import torch
from torch.utils.data import DataLoader, TensorDataset
from data.syntatic import step_wise
from utils.config import TrainConfig


def build_dataloader(
    f: Callable[[float], float],
    cfg: TrainConfig,
) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    """Generate synthetic data and wrap it in a DataLoader.

    Returns the DataLoader together with the full (x, y) tensors so that
    callers can reuse them for evaluation without re-generating data.
    """
    x_np, y_np = step_wise(cfg.num_points, f, start_std=0.1, end_std=0.5, mode=cfg.mode)

    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).to(cfg.device)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1).to(cfg.device)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    return loader, x, y