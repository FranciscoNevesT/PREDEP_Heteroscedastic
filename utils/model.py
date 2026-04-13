import torch
import torch.nn.functional as F
import numpy as np
from utils.config import TrainConfig
from models.estimation import EstimationModel

def build_model(cfg: TrainConfig) -> EstimationModel:
    """Instantiate and move the model to the global device."""
    return EstimationModel(
        input_dim=1,
        hidden_dim=cfg.hidden_dim,
        num_partitions=cfg.num_partitions,
        output_dim=1,
        num_gaussians=1,
    ).to(device=cfg.device)


@torch.no_grad()
def get_predictions(
    model: EstimationModel,
    x_test: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, hard partition indices) for *x_test*.

    Both arrays have shape ``(N,)``.
    """
    model.eval()
    x_dev = x_test.to(cfg.device)

    logits = model.partition_model.get_partion_logits(x_dev)
    partition_probs = F.softmax(logits, dim=1)
    partitions = torch.argmax(partition_probs, dim=1).cpu().numpy()
    predictions = model(x_dev).cpu().numpy().squeeze()

    return predictions, partitions