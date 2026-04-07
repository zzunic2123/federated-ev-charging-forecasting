"""CLI for training the centralized LSTM baseline."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation import (
    SUPPORTED_FEATURE_COUNT,
    build_prediction_frame,
    build_results_payload,
    compute_regression_metrics,
    configure_run_logger,
    load_experiment_metadata,
    make_run_name,
    save_predictions_csv,
    save_results_payload,
    validate_official_task,
    write_empty_communication_rounds,
)
from ..models import LSTMRegressor
from ..preprocessing.utils import save_json


@dataclass(frozen=True)
class SplitArrays:
    """One centralized dataset split loaded from NPZ."""

    X: np.ndarray
    y: np.ndarray
    t: np.ndarray
    station_ids: np.ndarray


@dataclass(frozen=True)
class CentralizedDatasetBundle:
    """Centralized train, val, and test splits."""

    train: SplitArrays
    val: SplitArrays
    test: SplitArrays


@dataclass(frozen=True)
class TrainingConfig:
    """Runtime configuration for centralized LSTM training."""

    dataset_dir: Path
    experiment_metadata_path: Path
    output_dir: Path
    run_name: str
    epochs: int
    batch_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    patience: int
    grad_clip_norm: float
    seed: int
    device: str

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if self.num_layers == 1 and self.dropout != 0.0:
            raise ValueError("dropout must be 0.0 when num_layers=1.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for centralized LSTM training."""

    parser = argparse.ArgumentParser(
        description="Train the centralized LSTM baseline on canonical experiment NPZ datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/experiments/centralized"),
        help="Directory containing centralized train/val/test NPZ files.",
    )
    parser.add_argument(
        "--experiment-metadata",
        type=Path,
        default=Path("data/processed/experiments/artifacts/experiment_metadata.json"),
        help="Path to official experiment metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/results/centralized/lstm"),
        help="Directory under which the run folder will be created.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit run name.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device selection: auto, cpu, or cuda.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Build validated runtime config from parsed CLI args."""

    return TrainingConfig(
        dataset_dir=args.dataset_dir.resolve(),
        experiment_metadata_path=args.experiment_metadata.resolve(),
        output_dir=args.output_dir.resolve(),
        run_name=args.run_name or make_run_name(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=args.device,
    )


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible centralized runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device_name: str) -> torch.device:
    """Resolve runtime device selection."""

    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device '{device_name}'. Use auto, cpu, or cuda.")


def load_split_npz(
    split_path: Path,
    split_name: str,
    *,
    expected_window_size: int,
    expected_feature_count: int,
) -> SplitArrays:
    """Load and validate one centralized split from NPZ."""

    if not split_path.exists():
        raise FileNotFoundError(f"Missing centralized split file: {split_path}")

    with np.load(split_path) as data:
        required_keys = {"X", "y", "t", "station_ids"}
        if set(data.files) != required_keys:
            raise RuntimeError(
                f"{split_name} split must contain keys {sorted(required_keys)}, got {sorted(data.files)}."
            )
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.float32, copy=False)
        t = data["t"].astype(str)
        station_ids = data["station_ids"].astype(str)

    if X.ndim != 3:
        raise RuntimeError(f"{split_name} X must be 3D, got shape {X.shape}.")
    if y.ndim != 1:
        raise RuntimeError(f"{split_name} y must be 1D, got shape {y.shape}.")
    if X.shape[0] != y.shape[0] or X.shape[0] != t.shape[0] or X.shape[0] != station_ids.shape[0]:
        raise RuntimeError(f"{split_name} arrays must agree on sample count.")
    if X.shape[1] != expected_window_size:
        raise RuntimeError(
            f"{split_name} X has window size {X.shape[1]}, expected {expected_window_size}."
        )
    if X.shape[2] != expected_feature_count:
        raise RuntimeError(
            f"{split_name} X has feature count {X.shape[2]}, expected {expected_feature_count}."
        )
    if X.shape[0] == 0:
        raise RuntimeError(f"{split_name} split is empty.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise RuntimeError(f"{split_name} split contains NaN or Inf values.")

    return SplitArrays(X=X, y=y, t=t, station_ids=station_ids)


def load_centralized_dataset(
    dataset_dir: Path,
    metadata: dict[str, Any],
) -> CentralizedDatasetBundle:
    """Load train, val, and test NPZ splits and validate against metadata."""

    expected_window_size = int(metadata["window_size"])
    expected_feature_count = int(metadata["feature_count"])
    splits = {
        split_name: load_split_npz(
            dataset_dir / f"{split_name}.npz",
            split_name,
            expected_window_size=expected_window_size,
            expected_feature_count=expected_feature_count,
        )
        for split_name in ("train", "val", "test")
    }

    metadata_counts = metadata.get("centralized_counts", {})
    for split_name, split_arrays in splits.items():
        expected_count = int(metadata_counts[split_name])
        actual_count = int(split_arrays.X.shape[0])
        if actual_count != expected_count:
            raise RuntimeError(
                f"{split_name} split count mismatch: expected {expected_count}, got {actual_count}."
            )

    return CentralizedDatasetBundle(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )


def build_dataloader(
    split_arrays: SplitArrays,
    batch_size: int,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a DataLoader for one centralized split."""

    dataset = TensorDataset(
        torch.from_numpy(split_arrays.X),
        torch.from_numpy(split_arrays.y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def train_one_epoch(
    *,
    model: nn.Module,
    split_arrays: SplitArrays,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    grad_clip_norm: float,
) -> float:
    """Train for one epoch and return average training loss."""

    model.train()
    loader = build_dataloader(split_arrays, batch_size, shuffle=True, device=device)
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        y_batch = y_batch.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        predictions = model(X_batch)
        if not torch.isfinite(predictions).all():
            raise RuntimeError("Non-finite predictions encountered during training.")

        loss = loss_fn(predictions, y_batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered during training.")

        loss.backward()
        if grad_clip_norm > 0.0:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size_actual = int(y_batch.shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_samples += batch_size_actual

    return total_loss / total_samples


def evaluate_split(
    *,
    model: nn.Module,
    split_arrays: SplitArrays,
    batch_size: int,
    device: torch.device,
    loss_fn: nn.Module,
    method_name: str,
) -> dict[str, Any]:
    """Evaluate one split and return loss, metrics, and prediction frame."""

    model.eval()
    loader = build_dataloader(split_arrays, batch_size, shuffle=False, device=device)
    total_loss = 0.0
    total_samples = 0
    y_predictions: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y_batch = y_batch.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")

            predictions = model(X_batch)
            if not torch.isfinite(predictions).all():
                raise RuntimeError("Non-finite predictions encountered during evaluation.")

            loss = loss_fn(predictions, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during evaluation.")

            batch_size_actual = int(y_batch.shape[0])
            total_loss += float(loss.item()) * batch_size_actual
            total_samples += batch_size_actual
            y_predictions.append(predictions.cpu().numpy())

    y_pred = np.concatenate(y_predictions, axis=0)
    metrics = compute_regression_metrics(split_arrays.y, y_pred)
    prediction_frame = build_prediction_frame(
        t=split_arrays.t,
        station_ids=split_arrays.station_ids,
        y_true=split_arrays.y,
        y_pred=y_pred,
        method_name=method_name,
    )

    return {
        "loss": total_loss / total_samples,
        "metrics": metrics,
        "prediction_frame": prediction_frame,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def estimate_model_state_bytes(model: nn.Module) -> int:
    """Estimate serialized model state size in bytes from the state dict tensors."""

    state_dict = model.state_dict()
    return int(sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()))


def save_history_csv(history_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Persist epoch history to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(output_path, index=False)


def save_training_config(
    *,
    config: TrainingConfig,
    output_path: Path,
    metadata: dict[str, Any],
    selected_device: torch.device,
) -> None:
    """Persist the centralized training configuration used for the run."""

    payload = {
        "dataset_dir": str(config.dataset_dir),
        "experiment_metadata_path": str(config.experiment_metadata_path),
        "output_dir": str(config.output_dir),
        "run_name": config.run_name,
        "device": str(selected_device),
        "seed": config.seed,
        "model": {
            "input_size": int(metadata["feature_count"]),
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
        },
        "optimization": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "patience": config.patience,
            "grad_clip_norm": config.grad_clip_norm,
        },
        "task": {
            "window_size": int(metadata["window_size"]),
            "horizon": int(metadata["horizon"]),
            "target_column": metadata["target_column"],
            "feature_count": int(metadata["feature_count"]),
        },
    }
    save_json(payload, output_path)


def main() -> None:
    """Train the centralized LSTM baseline and persist all run artifacts."""

    args = parse_args()
    config = build_config_from_args(args)
    metadata = load_experiment_metadata(config.experiment_metadata_path)
    validate_official_task(metadata, expected_feature_count=SUPPORTED_FEATURE_COUNT)
    dataset = load_centralized_dataset(config.dataset_dir, metadata)

    run_dir = config.output_dir / config.run_name
    logger = configure_run_logger(
        name=f"diplomski.training.centralized.{config.run_name}",
        log_path=run_dir / "run.log",
    )
    logger.info("Starting centralized LSTM run: %s", config.run_name)

    selected_device = select_device(config.device)
    set_global_seed(config.seed)
    save_training_config(
        config=config,
        output_path=run_dir / "config.json",
        metadata=metadata,
        selected_device=selected_device,
    )

    model = LSTMRegressor(
        input_size=int(metadata["feature_count"]),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(selected_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    parameter_count = count_trainable_parameters(model)
    model_state_bytes = estimate_model_state_bytes(model)

    logger.info(
        "Device=%s | parameter_count=%d | model_state_bytes=%d",
        selected_device,
        parameter_count,
        model_state_bytes,
    )

    best_model_path = run_dir / "best_model.pt"
    history_rows: list[dict[str, Any]] = []
    best_val_rmse = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            split_arrays=dataset.train,
            batch_size=config.batch_size,
            device=selected_device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_eval = evaluate_split(
            model=model,
            split_arrays=dataset.val,
            batch_size=config.batch_size,
            device=selected_device,
            loss_fn=loss_fn,
            method_name="lstm",
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_eval["loss"]),
                "val_rmse": float(val_eval["metrics"]["rmse"]),
                "val_mae": float(val_eval["metrics"]["mae"]),
                "val_r2": float(val_eval["metrics"]["r2"]),
            }
        )
        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | val_rmse=%.6f | val_mae=%.6f | val_r2=%.6f",
            epoch,
            config.epochs,
            train_loss,
            val_eval["loss"],
            val_eval["metrics"]["rmse"],
            val_eval["metrics"]["mae"],
            val_eval["metrics"]["r2"],
        )

        if float(val_eval["metrics"]["rmse"]) < best_val_rmse:
            best_val_rmse = float(val_eval["metrics"]["rmse"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved new best model at epoch %d with val_rmse=%.6f", epoch, best_val_rmse)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                early_stopped = True
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    if best_epoch is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    best_state = torch.load(best_model_path, map_location=selected_device)
    model.load_state_dict(best_state)

    train_eval = evaluate_split(
        model=model,
        split_arrays=dataset.train,
        batch_size=config.batch_size,
        device=selected_device,
        loss_fn=loss_fn,
        method_name="lstm",
    )
    val_eval = evaluate_split(
        model=model,
        split_arrays=dataset.val,
        batch_size=config.batch_size,
        device=selected_device,
        loss_fn=loss_fn,
        method_name="lstm",
    )
    test_eval = evaluate_split(
        model=model,
        split_arrays=dataset.test,
        batch_size=config.batch_size,
        device=selected_device,
        loss_fn=loss_fn,
        method_name="lstm",
    )

    split_counts = {
        "train": int(dataset.train.X.shape[0]),
        "val": int(dataset.val.X.shape[0]),
        "test": int(dataset.test.X.shape[0]),
    }
    results_payload = build_results_payload(
        run_name=config.run_name,
        stage="baseline",
        scenario="centralized",
        method="lstm",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        epochs_completed=len(history_rows),
        best_epoch=best_epoch,
        target_column=str(metadata["target_column"]),
        window_size=int(metadata["window_size"]),
        horizon=int(metadata["horizon"]),
        experiment_metadata_path=config.experiment_metadata_path,
        valid_station_ids=list(metadata["valid_station_ids"]),
        split_counts=split_counts,
        metrics_by_split={
            "train": train_eval["metrics"],
            "val": val_eval["metrics"],
            "test": test_eval["metrics"],
        },
        training_info={
            "trained": True,
            "early_stopped": early_stopped,
            "best_val_rmse": float(best_val_rmse),
            "parameter_count": parameter_count,
        },
        communication_info={
            "communication_enabled": False,
            "logical_client_count": int(metadata["valid_station_count"]),
            "round_count": 0,
            "messages_up": 0,
            "messages_down": 0,
            "payload_bytes_up": 0,
            "payload_bytes_down": 0,
            "payload_bytes_total": 0,
            "model_state_bytes": model_state_bytes,
        },
        dataset_dir=config.dataset_dir,
        hourly_dir=None,
    )

    save_results_payload(results_payload, run_dir / "results.json")
    save_predictions_csv(test_eval["prediction_frame"], run_dir / "test_predictions.csv")
    write_empty_communication_rounds(run_dir / "communication_rounds.csv")
    save_history_csv(history_rows, run_dir / "history.csv")

    logger.info(
        "Completed centralized LSTM run | best_epoch=%d | test_rmse=%.6f | test_mae=%.6f | test_r2=%.6f",
        best_epoch,
        test_eval["metrics"]["rmse"],
        test_eval["metrics"]["mae"],
        test_eval["metrics"]["r2"],
    )
    logger.info("Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
