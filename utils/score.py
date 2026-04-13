import torch
from models.estimation import EstimationModel
from utils.config import TrainConfig
import torch.nn.functional as F
from scipy import stats
from utils.model import get_predictions
import numpy as np

def predeptest(
    model: EstimationModel,
    x_test: torch.Tensor,
    cfg: TrainConfig,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic for *model* on the given test data."""
    model.eval()
    predictions, partitions = get_predictions(model, x_test, cfg)
    x_test = x_test.cpu().numpy().squeeze()

    error = x_test - predictions

    num_partitions = cfg.num_partitions
    predep = 0

    for i in range(num_partitions):
        p = error[partitions == i]

        if len(p) < 2:
            continue
        
        std = np.std(p)
        proportion = len(p) / len(x_test)
        predep += proportion / (2 * std * np.sqrt(np.pi)) 

    return float(predep)


def kstest(
    model: EstimationModel,
    x_test: torch.Tensor,
    cfg: TrainConfig,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic for *model* on the given test data."""
    model.eval()
    predictions, partitions = get_predictions(model, x_test, cfg)
    x_test = x_test.cpu().numpy().squeeze()

    error = x_test - predictions

    num_partitions = cfg.num_partitions
    ks_stats = np.zeros((num_partitions, num_partitions))

    for i in range(num_partitions):
        for j in range(i,num_partitions):
            p = error[partitions == i]
            q = error[partitions == j]

            if len(p) == 0 or len(q) == 0:
                ks_stats[i, j] = np.nan  # Handle empty partitions
                continue

            ks_stat = stats.ks_2samp(p, q).pvalue
            ks_stats[i, j] = ks_stat
            ks_stats[j, i] = ks_stat  # Symmetric matrix

    return ks_stats



@torch.no_grad()
def evaluate_model(
    model: EstimationModel,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: TrainConfig,
) -> float:
    """Compute mean squared error of *model* on the given test data."""
    model.eval()
    x_dev = x_test.to(cfg.device)
    y_dev = y_test.to(cfg.device)

    predictions = model(x_dev)
    log_likelihood = model.compute_loss(x_dev, y_dev)
    mse = F.mse_loss(predictions, y_dev).item()

    ks_stats = kstest(model, x_test, cfg)
    predep = predeptest(model, x_test, cfg)

    score = {
        "mse": mse,
        "log_likelihood": log_likelihood.item(),
        "ks_stats": ks_stats,
        "predep": predep,
    }

    return score
