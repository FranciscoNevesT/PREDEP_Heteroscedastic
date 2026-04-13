import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import seaborn as sns

def _build_partition_colormap(num_partitions: int):
    """Return a ``(cmap, norm)`` pair suitable for a discrete partition plot."""
    base = plt.get_cmap("tab10", num_partitions)
    cmap = mcolors.ListedColormap([base(i) for i in range(num_partitions)])
    bounds = np.arange(-0.5, num_partitions + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_partition_predictions(
    x_grid: torch.Tensor,
    predictions_grid: np.ndarray,
    partitions_grid: np.ndarray,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> None:
    """Visualize model predictions with colour-coded partition regions."""
    num_partitions = int(partitions_grid.max()) + 1
    cmap, norm = _build_partition_colormap(num_partitions)

    x_grid = x_grid.cpu().numpy().squeeze()           # shape (N,)

    x_test = x_test.squeeze().cpu().numpy()
    y_test = y_test.squeeze().cpu().numpy() 

    
    y_grid = np.linspace(y_test.min(), y_test.max(), 50)

    # Build 2-D meshes for pcolormesh
    x_mesh = np.repeat(x_grid[:, None], 50, axis=1)
    y_mesh = np.tile(y_grid, (len(x_grid), 1))
    z_mesh = np.repeat(partitions_grid[:, None], 50, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    mesh = ax.pcolormesh(
        x_mesh,
        y_mesh,
        z_mesh,
        shading="auto",
        cmap=cmap,
        norm=norm,
        alpha=0.45,
    )

    sns.scatterplot(x=x_test, y=y_test, alpha=0.6, ax=ax)
    ax.plot(x_grid, predictions_grid, color="black", linewidth=1.5)

    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(num_partitions))
    cbar.set_label("Partition")
    cbar.ax.set_yticklabels([f"P{i}" for i in range(num_partitions)])

    ax.set_title("Partition regions over prediction space")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()