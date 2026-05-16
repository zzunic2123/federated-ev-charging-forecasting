"""Shared helpers for experiment evaluation, logging, and artifact writing."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..preprocessing.utils import save_json


SUPPORTED_WINDOW_SIZE = 24
SUPPORTED_HORIZON = 1
SUPPORTED_TARGET_COLUMN = "volume"
SUPPORTED_FEATURE_COUNT = 19

COMMUNICATION_COLUMNS = [
    "round",
    "client_count",
    "messages_up",
    "messages_down",
    "payload_bytes_up",
    "payload_bytes_down",
    "payload_bytes_total",
]
PREDICTION_COLUMNS = [
    "t",
    "station_id",
    "y_true",
    "y_pred",
    "error",
    "abs_error",
    "method_name",
]


def make_run_name() -> str:
    """Create a deterministic UTC timestamp-based run name."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def configure_run_logger(name: str, log_path: Path | None) -> logging.Logger:
    """Configure a logger writing to stderr and optionally to a file."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_experiment_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load experiment metadata from JSON."""

    if not metadata_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def validate_official_task(
    metadata: dict[str, Any],
    *,
    expected_feature_count: int | None = None,
) -> None:
    """Validate the fixed official forecasting task for thesis experiments."""

    window_size = int(metadata.get("window_size", -1))
    horizon = int(metadata.get("horizon", -1))
    target_column = str(metadata.get("target_column", ""))

    if window_size != SUPPORTED_WINDOW_SIZE:
        raise ValueError(
            f"Supported experiments require window_size={SUPPORTED_WINDOW_SIZE}, got {window_size}."
        )
    if horizon != SUPPORTED_HORIZON:
        raise ValueError(
            f"Supported experiments require horizon={SUPPORTED_HORIZON}, got {horizon}."
        )
    if target_column != SUPPORTED_TARGET_COLUMN:
        raise ValueError(
            "Supported experiments require "
            f"target_column='{SUPPORTED_TARGET_COLUMN}', got '{target_column}'."
        )
    if expected_feature_count is not None:
        feature_count = int(metadata.get("feature_count", -1))
        if feature_count != expected_feature_count:
            raise ValueError(
                f"Supported experiments require feature_count={expected_feature_count}, got {feature_count}."
            )


def compute_regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float | int]:
    """Compute RMSE, MAE, and R² for one set of predictions."""

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "Prediction and target arrays must have matching shapes: "
            f"{y_true_array.shape} != {y_pred_array.shape}"
        )
    if y_true_array.ndim != 1:
        raise ValueError("Regression metrics expect 1D prediction vectors.")
    if y_true_array.size == 0:
        raise ValueError("Cannot compute regression metrics on an empty array.")
    if not np.isfinite(y_true_array).all() or not np.isfinite(y_pred_array).all():
        raise ValueError("Regression metrics require finite targets and predictions.")

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "r2": float(r2_score(y_true_array, y_pred_array, force_finite=True)),
        "sample_count": int(y_true_array.shape[0]),
    }


def build_prediction_frame(
    *,
    t: np.ndarray | list[Any],
    station_ids: np.ndarray | list[Any],
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    method_name: str,
) -> pd.DataFrame:
    """Build the canonical prediction CSV dataframe."""

    time_values = np.asarray(t)
    station_id_values = np.asarray(station_ids)
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    lengths = {
        len(time_values),
        len(station_id_values),
        len(y_true_array),
        len(y_pred_array),
    }
    if len(lengths) != 1:
        raise ValueError("Prediction frame inputs must all have the same length.")
    if not np.isfinite(y_true_array).all() or not np.isfinite(y_pred_array).all():
        raise ValueError("Prediction frame inputs must be finite.")

    error = y_pred_array - y_true_array
    frame = pd.DataFrame(
        {
            "t": pd.Series(time_values).astype(str),
            "station_id": pd.Series(station_id_values).astype(str),
            "y_true": y_true_array,
            "y_pred": y_pred_array,
            "error": error,
            "abs_error": np.abs(error),
            "method_name": method_name,
        }
    )
    return frame.loc[:, PREDICTION_COLUMNS]


def save_predictions_csv(frame: pd.DataFrame, output_path: Path) -> None:
    """Persist prediction dataframe with the canonical schema."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, PREDICTION_COLUMNS].to_csv(output_path, index=False)


def write_empty_communication_rounds(output_path: Path) -> None:
    """Write an empty communication rounds CSV using the shared schema."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COMMUNICATION_COLUMNS).to_csv(output_path, index=False)


def build_results_payload(
    *,
    run_name: str,
    stage: str,
    scenario: str,
    method: str,
    generated_at_utc: str,
    epochs_completed: int,
    best_epoch: int | None,
    target_column: str,
    window_size: int,
    horizon: int,
    experiment_metadata_path: Path,
    valid_station_ids: list[str],
    split_counts: dict[str, int],
    metrics_by_split: dict[str, dict[str, float | int]],
    training_info: dict[str, Any],
    communication_info: dict[str, Any],
    dataset_dir: Path | None = None,
    hourly_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the shared results schema for baseline and model experiments."""

    return {
        "run": {
            "run_name": run_name,
            "stage": stage,
            "scenario": scenario,
            "method": method,
            "generated_at_utc": generated_at_utc,
            "epochs_completed": int(epochs_completed),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
        },
        "task": {
            "target_column": target_column,
            "window_size": int(window_size),
            "horizon": int(horizon),
        },
        "data": {
            "dataset_dir": str(dataset_dir.resolve()) if dataset_dir is not None else None,
            "hourly_dir": str(hourly_dir.resolve()) if hourly_dir is not None else None,
            "experiment_metadata_path": str(experiment_metadata_path.resolve()),
            "valid_station_count": len(valid_station_ids),
            "valid_station_ids": valid_station_ids,
            "split_counts": {name: int(count) for name, count in split_counts.items()},
        },
        "training": training_info,
        "metrics": metrics_by_split,
        "communication": communication_info,
    }


def save_results_payload(payload: dict[str, Any], output_path: Path) -> None:
    """Persist a results payload as JSON."""

    save_json(payload, output_path)
