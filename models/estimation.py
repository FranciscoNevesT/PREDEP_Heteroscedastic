import torch 
import torch.nn as nn
import torch.nn.functional as F


class DensityEstimator(nn.Module):
    def __init__(self, input_dim=1, num_gaussians=5):
        super(DensityEstimator, self).__init__()

        self.log_std = nn.Parameter(torch.randn((num_gaussians, input_dim)))  # Learnable log-std for each Gaussian
        self.log_pi = nn.Parameter(torch.zeros(num_gaussians))  # Learnable log-proportions for each Gaussian

    def forward(self, error):
        """
        error: (B, D)
        """
        B, D = error.size()
        K = self.log_std.size(0)

        # --- Std ---
        std = torch.exp(self.log_std)  # (K, D)
        std = std.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)

        # --- Proportions ---
        log_pi = F.log_softmax(self.log_pi, dim=0)  # (K,)
        log_pi = log_pi.unsqueeze(0).unsqueeze(-1)  # (1, K, 1)

        # --- Error ---
        error = error.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)

        # --- Gaussian log-prob ---
        log_prob = -0.5 * (
            (error / std) ** 2 +
            2 * torch.log(std) +
            torch.log(torch.tensor(2 * torch.pi))
        )  # (B, K, D)

        log_prob = log_prob.sum(dim=-1) + log_pi.squeeze(-1)  # Sum over D and add log-proportions -> (B, K)
        
        return torch.logsumexp(log_prob, dim=1) # Log-likelihood for the mixture of Gaussians (B,)
    
class PartitionModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_partitions=4, output_dim=1, num_gaussians=5):
        super(PartitionModel, self).__init__()
        self.num_partitions = num_partitions
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        # Gating netrwork: p(k | x)
        self.partitions = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_partitions)
        )

        # Density estimator for each partition
        self.density_estimators = nn.ModuleList([
            DensityEstimator(input_dim=output_dim, num_gaussians=num_gaussians) for _ in range(num_partitions)
        ])

    def get_partion_logits(self, x):
        """
        Returns logits for each partition (before softmax).
        Shape: (B, K)
        """
        return self.partitions(x)
    
    def forward(self, x, error):
        """
        x: (B, input_dim)
        error: (B, output_dim)
        """
        B = x.size(0)

        # --- Gating ---
        logits = self.get_partion_logits(x)              # (B, K)
        log_pi = F.log_softmax(logits, dim=1)              # (B, K)

        # --- Density Estimation ---
        partition_logs = []
        for k in range(self.num_partitions):
            partition_log = self.density_estimators[k](error)  # Negative log-likelihood for partition k
            partition_logs.append(partition_log)

        partition_log_probs  = torch.stack(partition_logs, dim=1)  # (B, K)
        # --- Mixture log-likelihood ---
        log_mix = log_pi + partition_log_probs
        log_likelihood = torch.logsumexp(log_mix, dim=1)

        return -log_likelihood.mean()

class ExpectedValue_MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(ExpectedValue_MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    

class EstimationModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_partitions=4, output_dim=1, num_gaussians=5):
        super(EstimationModel, self).__init__()
        self.ev_model = ExpectedValue_MLP(input_dim, hidden_dim, output_dim)
        self.partition_model = PartitionModel(input_dim, hidden_dim, num_partitions, output_dim, num_gaussians)

    def forward(self, x):
        ev = self.ev_model(x)  # (B, output_dim)
        return ev

    def compute_loss(self, x, y):
        ev = self.ev_model(x)  # (B, output_dim)
        error = y - ev  # (B, output_dim)
        partition_loss = self.partition_model(x, error)  # Negative log-likelihood loss
        total_loss = partition_loss  # Combine losses
        return total_loss