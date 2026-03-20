import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Partition_MLP(nn.Module):
    def __init__(self, num_partitions, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.num_partitions = num_partitions
        self.output_dim = output_dim

        # Gating network: p(k | x)
        self.partitions = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_partitions)
        )

        # Log-std per partition
        self.log_std = nn.Parameter(torch.zeros(num_partitions, output_dim))


    def get_partition_logits(self, x):
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
        logits = self.get_partition_logits(x)              # (B, K)
        log_pi = F.log_softmax(logits, dim=1)              # (B, K)

        # Expand dims to match output_dim
        log_pi = log_pi.unsqueeze(-1)                      # (B, K, 1)

        # --- Std ---
        std = torch.exp(self.log_std)                      # (K, D)
        std = std.unsqueeze(0).expand(B, -1, -1)           # (B, K, D)

        # --- Error ---
        error = error.unsqueeze(1).expand(-1, self.num_partitions, -1)  # (B, K, D)

        # --- Gaussian log-prob ---
        log_prob = -0.5 * (
            (error / std) ** 2 +
            2 * torch.log(std) +
            math.log(2 * math.pi)
        )  # (B, K, D)

        # If D > 1, sum over dimensions
        log_prob = log_prob.sum(dim=-1, keepdim=True)      # (B, K, 1)

        # --- Mixture log-likelihood ---
        log_mix = log_pi + log_prob                        # (B, K, 1)
        log_likelihood = torch.logsumexp(log_mix, dim=1)   # (B, 1)

        loss = -log_likelihood.mean()

        return loss
    
if __name__ == "__main__":
    model = Partition_MLP(num_partitions=3, input_dim=1, hidden_dim=64, output_dim=1)
    x = torch.randn(10, 1)
    error = torch.randn(10, 1)  # Dummy error input
    loss = model(x, error)
    print("Loss:", loss.item())