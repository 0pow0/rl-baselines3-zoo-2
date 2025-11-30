import argparse
import os
import random
from typing import Iterable, Optional

import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a value regressor from enjoy() rollouts")
    parser.add_argument("--data-path", required=True, help=".pt file created by enjoy.py --value-stats-path")
    parser.add_argument("--output-path", default="value_regressor.pt", help="Where to store the trained regressor")
    parser.add_argument(
        "--hidden-dims",
        nargs="*",
        type=int,
        default=[512, 512],
        help="Hidden layer sizes (empty list keeps the model linear)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data used for validation")
    parser.add_argument("--device", default="auto", help="Torch device (cpu, cuda, auto)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-records", type=int, default=0, help="Optional limit on number of records to use")
    parser.add_argument("--no-standardize-features", action="store_true", help="Disable feature standardization")
    parser.add_argument("--no-standardize-target", action="store_true", help="Disable target standardization")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch interval for logging metrics")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


def flatten_observation(obs) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [flatten_observation(obs[key]) for key in sorted(obs.keys())]
        return np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=np.float32)
    arr = np.array(obs, dtype=np.float32)
    return arr.reshape(-1)


class ValueRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        return self.net(inputs).squeeze(-1)


def load_dataset(args: argparse.Namespace) -> tuple[TensorDataset, dict[str, np.ndarray], dict[str, float]]:
    payload = th.load(args.data_path, map_location="cpu", weights_only=False)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("No value records found in data file")
    if args.max_records > 0:
        records = records[: args.max_records]

    features, targets = [], []
    for rec in records:
        obs = rec.get("observation")
        if obs is None:
            continue
        target = rec.get("returns")
        if target is None:
            continue
        flat = flatten_observation(obs)
        if flat.size == 0:
            continue
        features.append(flat)
        targets.append(float(target))

    if len(features) == 0:
        raise ValueError("All records were empty; nothing to train on")

    feature_matrix = np.vstack(features).astype(np.float32)
    target_vector = np.asarray(targets, dtype=np.float32)

    feature_stats = dict(mean=feature_matrix.mean(axis=0), std=feature_matrix.std(axis=0))
    if not args.no_standardize_features:
        feature_std = feature_stats["std"].copy()
        feature_std[feature_std < 1e-6] = 1.0
        feature_matrix = (feature_matrix - feature_stats["mean"]) / feature_std
        feature_stats = dict(mean=feature_stats["mean"], std=feature_std)
    else:
        feature_stats = dict(mean=np.zeros_like(feature_matrix[0]), std=np.ones_like(feature_matrix[0]))

    target_std = float(target_vector.std())
    target_mean = float(target_vector.mean())
    if not args.no_standardize_target and target_std > 1e-6:
        target_vector = (target_vector - target_mean) / target_std
        target_stats = dict(mean=target_mean, std=target_std)
    else:
        target_stats = dict(mean=0.0, std=1.0)

    dataset = TensorDataset(th.from_numpy(feature_matrix), th.from_numpy(target_vector))
    return dataset, feature_stats, target_stats


def split_dataset(dataset: TensorDataset, val_split: float, seed: int) -> tuple[Dataset, Optional[Dataset]]:
    if val_split <= 0.0:
        return dataset, None
    if val_split >= 1.0:
        raise ValueError("val_split must be in (0,1)")
    val_size = int(len(dataset) * val_split)
    if val_size == 0:
        return dataset, None
    train_size = len(dataset) - val_size
    generator = th.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def evaluate(
    model: nn.Module, loader: DataLoader, device: th.device, collect_points: bool = False
) -> tuple[float, float, Optional[th.Tensor], Optional[th.Tensor]]:
    model.eval()
    total_loss = 0.0
    count = 0
    criterion = nn.MSELoss()
    preds_buffer: list[th.Tensor] = []
    targets_buffer: list[th.Tensor] = []
    with th.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            batch = inputs.shape[0]
            total_loss += loss.item() * batch
            count += batch
            preds_buffer.append(preds.detach().cpu())
            targets_buffer.append(targets.detach().cpu())

    mean_loss = total_loss / max(count, 1)
    correlation = 0.0
    preds_vec = targets_vec = None
    if preds_buffer:
        preds_vec = th.cat(preds_buffer)
        targets_vec = th.cat(targets_buffer)
    if count > 1 and preds_vec is not None and targets_vec is not None:
        preds_centered = preds_vec - preds_vec.mean()
        targets_centered = targets_vec - targets_vec.mean()
        numerator = th.sum(preds_centered * targets_centered).item()
        denom = th.sqrt(th.sum(preds_centered**2) * th.sum(targets_centered**2)).item()
        if denom > 0:
            correlation = numerator / denom
    if not collect_points:
        return mean_loss, correlation, None, None
    return mean_loss, correlation, preds_vec, targets_vec


def maybe_plot_validation_figures(
    train_losses: list[float],
    val_losses: list[float],
    val_predictions: Optional[th.Tensor],
    val_targets: Optional[th.Tensor],
    target_stats: dict[str, float],
    output_path: str,
) -> None:
    if len(train_losses) == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping validation plots")
        return

    prefix, _ = os.path.splitext(output_path)
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="train_loss")
    if val_losses:
        plt.plot(epochs[: len(val_losses)], val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_plot_path = f"{prefix}_validation.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved validation loss plot to {loss_plot_path}")

    if val_predictions is None or val_targets is None:
        return
    preds_np = val_predictions.numpy()
    targets_np = val_targets.numpy()
    preds_np = preds_np * target_stats["std"] + target_stats["mean"]
    targets_np = targets_np * target_stats["std"] + target_stats["mean"]
    plt.figure(figsize=(5, 5))
    plt.scatter(targets_np, preds_np, s=8, alpha=0.4)
    line_min = min(targets_np.min(), preds_np.min())
    line_max = max(targets_np.max(), preds_np.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", label="ideal")
    plt.xlabel("True Returns")
    plt.ylabel("Predicted Returns")
    plt.title("Validation Correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    corr_plot_path = f"{prefix}_corr.png"
    plt.tight_layout()
    plt.savefig(corr_plot_path)
    plt.close()
    print(f"Saved correlation figure to {corr_plot_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = th.device(args.device if args.device != "auto" else ("cuda" if th.cuda.is_available() else "cpu"))

    dataset, feature_stats, target_stats = load_dataset(args)
    train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample_input, _ = train_dataset[0]
    input_dim = sample_input.numel()
    model = ValueRegressor(input_dim, args.hidden_dims).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    train_history: list[float] = []
    val_loss_history: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            batch = inputs.shape[0]
            epoch_loss += loss.item() * batch
            seen += batch
        train_loss = epoch_loss / max(seen, 1)
        train_history.append(train_loss)

        val_loss = val_corr = None
        if val_loader is not None:
            val_loss, val_corr, _, _ = evaluate(model, val_loader, device)
            val_loss_history.append(val_loss)

        if epoch % max(args.log_interval, 1) == 0:
            if val_loader is not None and val_loss is not None and val_corr is not None:
                print(
                    f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f}"
                    f" | val_loss={val_loss:.6f} | val_corr={val_corr:.4f}"
                )
            else:
                print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f}")

    metrics = dict(train_loss=train_history[-1] if train_history else 0.0)
    val_preds_tensor: Optional[th.Tensor] = None
    val_targets_tensor: Optional[th.Tensor] = None
    if val_loader is not None:
        final_val_loss, final_val_corr, val_preds_tensor, val_targets_tensor = evaluate(
            model, val_loader, device, collect_points=True
        )
        metrics["val_loss"] = final_val_loss
        metrics["val_corr"] = final_val_corr

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    th.save(
        dict(
            model_state_dict=model.state_dict(),
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            feature_mean=feature_stats["mean"],
            feature_std=np.maximum(feature_stats["std"], 1e-6),
            target_mean=target_stats["mean"],
            target_std=max(target_stats["std"], 1e-6),
            config=vars(args),
            metrics=metrics,
        ),
        args.output_path,
    )
    if val_loader is not None:
        maybe_plot_validation_figures(
            train_history,
            val_loss_history,
            val_preds_tensor,
            val_targets_tensor,
            target_stats,
            args.output_path,
        )
    elif train_history:
        maybe_plot_validation_figures(train_history, [], None, None, target_stats, args.output_path)
    print(f"Saved value regressor to {args.output_path}")


if __name__ == "__main__":
    main()
