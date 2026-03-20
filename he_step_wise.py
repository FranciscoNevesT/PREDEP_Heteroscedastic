from data.syntatic import step_wise
from models.expected_value import ExpectedValue_MLP
from models.partition import Partition_MLP
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn.functional as F

def train_partition_model(f, num_points=10000, num_steps=5, num_epochs=1000, batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Generate step-wise data

    # Initialize model and optimizer
    ev_model = ExpectedValue_MLP(input_dim=1, hidden_dim=64, output_dim=1).to(device)
    partition_model = Partition_MLP(num_partitions=num_steps, input_dim=1, hidden_dim=64, output_dim=1).to(device)
    optimizer = optim.Adam(list(ev_model.parameters()) + list(partition_model.parameters()), lr=0.001)

    # Training loop
    ev_model.train()
    partition_model.train()
    for epoch in range(num_epochs):
        
        x_np, y_np = step_wise(num_points, f, num_steps)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).to(device)
        y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1).to(device)

        y_pred  = ev_model(x)

        error = y - y_pred  # Calculate error

        mse_loss = torch.nn.functional.mse_loss(y_pred, y)  # MSE loss for expected value prediction
        log_likelihood_loss = partition_model(x, error)  # Partition model loss based on error

        optimizer.zero_grad()
        loss = log_likelihood_loss 
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(np.std(error.detach().cpu().numpy()), np.mean(error.detach().cpu().numpy()))
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f} (MSE: {mse_loss.item():.4f}, Log Likelihood: {log_likelihood_loss.item():.4f})')

    return ev_model, partition_model


if __name__ == "__main__":
    fs = [lambda x: x, lambda x: np.sin(2 * np.pi * x), lambda x: np.where(x < 0.5, x, 1 - x)]
    num_points = 10000
    num_steps = 4
    num_epochs = 10000
    for f in fs:
        # Plot the step-wise data
        x_np, y_np = step_wise(num_points, f, num_steps)
        plt.figure(figsize=(10, 6))
        plt.scatter(x_np, y_np, s=10, alpha=0.5)
        plt.title("Step-wise Noise")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        ev_model, partition_model = train_partition_model(f, num_points, num_steps, num_epochs)

        ev_model = ev_model.cpu()
        partition_model = partition_model.cpu()

        # Visualize partition outputs
        x_test = torch.linspace(0, 1, 1000).unsqueeze(1)
        with torch.no_grad():
            partition_outputs = partition_model.get_partition_logits(x_test)
            partition_outputs = F.softmax(partition_outputs, dim=1).cpu().numpy()  # Convert logits to probabilities
            stds = torch.exp(partition_model.log_std).cpu().numpy()
            print("Learned stds for each partition:", stds.squeeze())  # Print learned stds for each partition

        plt.figure(figsize=(10, 6))
        for i in range(num_steps):
            sns.lineplot(x=x_test.squeeze().numpy(), y=partition_outputs[:, i], label=f'Partition {i+1} (std={stds[i, 0]:.2f})')
        plt.title("Partition Outputs")
        plt.xlabel("x")
        plt.ylabel("Partition Probability")
        plt.legend()
        plt.show()