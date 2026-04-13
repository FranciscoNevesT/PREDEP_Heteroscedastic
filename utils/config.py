from typing import Callable

import torch


class TrainConfig:
    """Hyper-parameters and training settings."""
    def __init__(self, hidden_dim: int, num_points: int, num_partitions: int, mode : str,
                 num_epochs: int, batch_size: int, learning_rate: float,
                 log_every: int, functions: list[Callable[[float], float]],device: torch.device):
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.num_partitions = num_partitions
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_every = log_every
        self.functions = functions if functions is not None else [lambda x: (4 * x - 2) ** 2]
        self.device = device
        self.mode = mode

