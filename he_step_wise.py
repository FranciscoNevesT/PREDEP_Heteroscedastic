from utils.config import TrainConfig
from utils.train import train
from utils.plot import plot_partition_predictions
import torch
from data.syntatic import step_wise
from utils.model import get_predictions
from utils.score import evaluate_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = TrainConfig(hidden_dim=128, num_points=1_000, num_partitions=3, num_epochs=1000, mode = 'normal',
                      batch_size=64, learning_rate=1e-3, log_every=100, functions=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for f in cfg.functions:
        print(f"\n{'='*50}\nTraining model …\n{'='*50}")
        model = train(f, cfg)

        x_grid = torch.linspace(0, 1, 1_000).unsqueeze(1)
        x_test, y_test = step_wise(10000, f, start_std=0.1, end_std=0.5, mode=cfg.mode)

        x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        predictions_grid, partitions_grid = get_predictions(model, x_grid, cfg=cfg)
        plot_partition_predictions(x_grid, predictions_grid, partitions_grid,
                                   x_test, y_test
                                   )
        
        score = evaluate_model(model, x_test, y_test, cfg)
        print(f"Evaluation score: {score}")


if __name__ == "__main__":
    main()