"""CLI for official naive baselines used in the thesis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from ..preprocessing.config import ExperimentConfig
from ..preprocessing.experiment_utils import (
    SplitDataFrame,
    build_station_splits_from_hourly,
    build_windows_by_station,
    clean_split_for_windows,
    discover_hourly_files,
)
BASELINE_METHODS = {
    "last_value": "naive_last_value",
    "seasonal_24h": "naive_seasonal_24h",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for naive baseline execution."""

    parser = argparse.ArgumentParser(
        description="Run official naive baselines for the fixed 24h->1h EV charging forecasting task."
    )
    parser.add_argument(
        "--baseline",
        choices=("all", "last_value", "seasonal_24h"),
        default="all",
        help="Which naive baseline to run.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name reused across selected baselines.",
    )
    parser.add_argument(
        "--hourly-dir",
        type=Path,
        default=Path("data/processed/station_hourly"),
        help="Directory containing station_hourly CSV files.",
    )
    parser.add_argument(
        "--experiment-metadata",
        type=Path,
        default=Path("data/processed/experiments/artifacts/experiment_metadata.json"),
        help="Path to official experiment_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/results"),
        help="Root directory for run outputs.",
    )
    return parser.parse_args()


def resolve_selected_methods(selection: str) -> list[str]:
    """Resolve CLI baseline selection to canonical method names."""

    if selection == "all":
        return [BASELINE_METHODS["last_value"], BASELINE_METHODS["seasonal_24h"]]
    return [BASELINE_METHODS[selection]]


def build_config_from_metadata(metadata: dict[str, Any]) -> ExperimentConfig:
    """Build experiment config from official metadata."""

    split_ratios = metadata.get("split_ratios", {})
    return ExperimentConfig(
        target_column=str(metadata["target_column"]),
        window_size=int(metadata["window_size"]),
        horizon=int(metadata["horizon"]),
        train_ratio=float(split_ratios["train"]),
        val_ratio=float(split_ratios["val"]),
        test_ratio=float(split_ratios["test"]),
    )


def align_and_clean_split(
    split_df: pd.DataFrame,
    feature_columns: list[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Apply the same split-level alignment and cleaning as the official experiment pipeline."""

    aligned = split_df.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    cleaned, _ = clean_split_for_windows(
        split_df=aligned,
        feature_columns=feature_columns,
        target_column=config.target_column,
        time_column=config.time_column,
    )
    return cleaned


def window_sample_count(dataframe: pd.DataFrame, window_size: int, horizon: int) -> int:
    """Return the number of valid window samples available in a cleaned split."""

    return max(len(dataframe) - window_size - horizon + 1, 0)


def build_clean_splits_for_valid_stations(
    station_splits: dict[str, SplitDataFrame],
    valid_station_ids: list[str],
    feature_columns: list[str],
    config: ExperimentConfig,
) -> dict[str, SplitDataFrame]:
    """Build cleaned split dataframes only for officially valid stations."""

    cleaned_splits: dict[str, SplitDataFrame] = {}
    for station_id in valid_station_ids:
        split_frames = station_splits[station_id]
        cleaned_splits[station_id] = SplitDataFrame(
            train=align_and_clean_split(split_frames.train, feature_columns, config),
            val=align_and_clean_split(split_frames.val, feature_columns, config),
            test=align_and_clean_split(split_frames.test, feature_columns, config),
        )
    return cleaned_splits


def build_predictions_for_split(
    cleaned_split: pd.DataFrame,
    station_id: str,
    method_name: str,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Build per-sample predictions for one cleaned split and one baseline."""

    sample_count = window_sample_count(cleaned_split, config.window_size, config.horizon)
    if sample_count == 0:
        return build_prediction_frame(
            t=np.asarray([], dtype=object),
            station_ids=np.asarray([], dtype=object),
            y_true=np.asarray([], dtype=float),
            y_pred=np.asarray([], dtype=float),
            method_name=method_name,
        )

    target_values = cleaned_split.loc[:, config.target_column].to_numpy(dtype=np.float32)
    time_values = cleaned_split.loc[:, config.time_column].to_numpy()

    y_true_values: list[float] = []
    y_pred_values: list[float] = []
    target_times: list[Any] = []
    for start_idx in range(sample_count):
        end_idx = start_idx + config.window_size
        target_idx = end_idx + config.horizon - 1
        y_true = float(target_values[target_idx])

        if method_name == "naive_last_value":
            y_pred = float(target_values[end_idx - 1])
        elif method_name == "naive_seasonal_24h":
            # For the fixed 24h->1h task, the first value in the window is the same hour previous day.
            y_pred = float(target_values[start_idx])
        else:
            raise ValueError(f"Unsupported baseline '{method_name}'.")

        y_true_values.append(y_true)
        y_pred_values.append(y_pred)
        target_times.append(time_values[target_idx])

    station_ids = np.full((sample_count,), station_id, dtype=object)
    return build_prediction_frame(
        t=np.asarray(target_times, dtype=object),
        station_ids=station_ids,
        y_true=np.asarray(y_true_values, dtype=float),
        y_pred=np.asarray(y_pred_values, dtype=float),
        method_name=method_name,
    )


def build_predictions_by_split(
    cleaned_splits: dict[str, SplitDataFrame],
    method_name: str,
    config: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    """Build centralized train/val/test prediction tables across all valid stations."""

    by_split: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for station_id in cleaned_splits:
        split_frames = cleaned_splits[station_id]
        by_split["train"].append(build_predictions_for_split(split_frames.train, station_id, method_name, config))
        by_split["val"].append(build_predictions_for_split(split_frames.val, station_id, method_name, config))
        by_split["test"].append(build_predictions_for_split(split_frames.test, station_id, method_name, config))

    result: dict[str, pd.DataFrame] = {}
    for split_name, frames in by_split.items():
        if not frames:
            result[split_name] = build_prediction_frame(
                t=np.asarray([], dtype=object),
                station_ids=np.asarray([], dtype=object),
                y_true=np.asarray([], dtype=float),
                y_pred=np.asarray([], dtype=float),
                method_name=method_name,
            )
            continue
        result[split_name] = pd.concat(frames, ignore_index=True)
    return result


def validate_sample_universe(
    metadata: dict[str, Any],
    feature_columns: list[str],
    valid_station_ids: list[str],
    predictions_by_split: dict[str, pd.DataFrame],
) -> None:
    """Validate that naive baselines reuse the official experiment sample universe."""

    metadata_feature_columns = metadata.get("feature_columns", [])
    metadata_station_ids = metadata.get("valid_station_ids", [])
    metadata_counts = metadata.get("centralized_counts", {})

    if feature_columns != metadata_feature_columns:
        raise RuntimeError("Feature column set does not match experiment_metadata.json.")
    if valid_station_ids != metadata_station_ids:
        raise RuntimeError("Valid station IDs do not match experiment_metadata.json.")

    actual_counts = {
        split_name: int(frame.shape[0]) for split_name, frame in predictions_by_split.items()
    }
    expected_counts = {split_name: int(metadata_counts[split_name]) for split_name in ("train", "val", "test")}
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Naive baseline sample counts do not match official centralized counts: "
            f"expected={expected_counts}, actual={actual_counts}"
        )


def ensure_finite_predictions(predictions_by_split: dict[str, pd.DataFrame]) -> None:
    """Ensure prediction tables contain only finite numeric values."""

    numeric_columns = ("y_true", "y_pred", "error", "abs_error")
    for split_name, frame in predictions_by_split.items():
        if frame.empty:
            raise RuntimeError(f"Prediction frame for split '{split_name}' is empty.")
        for column in numeric_columns:
            values = frame.loc[:, column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Non-finite values detected in split '{split_name}' column '{column}'."
                )


def run_baseline(
    method_name: str,
    run_name: str,
    generated_at_utc: str,
    hourly_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    cleaned_splits: dict[str, SplitDataFrame],
    feature_columns: list[str],
    valid_station_ids: list[str],
    metadata: dict[str, Any],
    config: ExperimentConfig,
) -> Path:
    """Execute one naive baseline and persist all required artifacts."""

    run_dir = output_dir / "centralized" / method_name / run_name
    logger = configure_run_logger(
        name=f"diplomski.baselines.{method_name}.{run_name}",
        log_path=run_dir / "run.log",
    )
    logger.info("Starting baseline run: method=%s run_name=%s", method_name, run_name)

    predictions_by_split = build_predictions_by_split(cleaned_splits, method_name, config)
    validate_sample_universe(metadata, feature_columns, valid_station_ids, predictions_by_split)
    ensure_finite_predictions(predictions_by_split)

    split_counts = {
        split_name: int(frame.shape[0]) for split_name, frame in predictions_by_split.items()
    }
    metrics_by_split = {
        split_name: compute_regression_metrics(
            frame.loc[:, "y_true"].to_numpy(dtype=float),
            frame.loc[:, "y_pred"].to_numpy(dtype=float),
        )
        for split_name, frame in predictions_by_split.items()
    }

    results_payload = build_results_payload(
        run_name=run_name,
        stage="baseline",
        scenario="centralized",
        method=method_name,
        generated_at_utc=generated_at_utc,
        epochs_completed=0,
        best_epoch=None,
        target_column=config.target_column,
        window_size=config.window_size,
        horizon=config.horizon,
        experiment_metadata_path=metadata_path,
        valid_station_ids=valid_station_ids,
        split_counts=split_counts,
        metrics_by_split=metrics_by_split,
        training_info={
            "trained": False,
            "early_stopped": False,
            "best_val_rmse": float(metrics_by_split["val"]["rmse"]),
            "parameter_count": 0,
        },
        communication_info={
            "communication_enabled": False,
            "logical_client_count": len(valid_station_ids),
            "round_count": 0,
            "messages_up": 0,
            "messages_down": 0,
            "payload_bytes_up": 0,
            "payload_bytes_down": 0,
            "payload_bytes_total": 0,
            "model_state_bytes": 0,
        },
        dataset_dir=None,
        hourly_dir=hourly_dir,
    )

    save_results_payload(results_payload, run_dir / "results.json")
    save_predictions_csv(predictions_by_split["test"], run_dir / "test_predictions.csv")
    write_empty_communication_rounds(run_dir / "communication_rounds.csv")

    logger.info(
        "Saved outputs to %s | train=%d val=%d test=%d | test_rmse=%.6f",
        run_dir,
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
        metrics_by_split["test"]["rmse"],
    )
    return run_dir


def main() -> None:
    """Run official naive baselines."""

    args = parse_args()
    hourly_dir = args.hourly_dir.resolve()
    metadata_path = args.experiment_metadata.resolve()
    output_dir = args.output_dir.resolve()
    run_name = args.run_name or make_run_name()
    generated_at_utc = datetime.now(timezone.utc).isoformat()

    metadata = load_experiment_metadata(metadata_path)
    validate_official_task(metadata, expected_feature_count=SUPPORTED_FEATURE_COUNT)
    config = build_config_from_metadata(metadata)

    bootstrap_logger = configure_run_logger(
        name="diplomski.baselines.bootstrap",
        log_path=None,
    )
    bootstrap_logger.info("Loading official sample universe from %s", metadata_path)

    hourly_files = discover_hourly_files(hourly_dir)
    station_splits, feature_columns, _ = build_station_splits_from_hourly(
        hourly_files=hourly_files,
        config=config,
        logger=bootstrap_logger,
    )
    windows_by_station, _ = build_windows_by_station(
        station_splits=station_splits,
        feature_columns=feature_columns,
        config=config,
        logger=bootstrap_logger,
    )

    valid_station_ids = [
        station_id
        for station_id, payload in windows_by_station.items()
        if payload.X_train.shape[0] > 0
    ]
    if len(valid_station_ids) != int(metadata.get("valid_station_count", -1)):
        raise RuntimeError(
            "Valid station count does not match experiment_metadata.json: "
            f"expected={metadata.get('valid_station_count')}, actual={len(valid_station_ids)}"
        )

    cleaned_splits = build_clean_splits_for_valid_stations(
        station_splits=station_splits,
        valid_station_ids=valid_station_ids,
        feature_columns=feature_columns,
        config=config,
    )

    for method_name in resolve_selected_methods(args.baseline):
        run_dir = run_baseline(
            method_name=method_name,
            run_name=run_name,
            generated_at_utc=generated_at_utc,
            hourly_dir=hourly_dir,
            metadata_path=metadata_path,
            output_dir=output_dir,
            cleaned_splits=cleaned_splits,
            feature_columns=feature_columns,
            valid_station_ids=valid_station_ids,
            metadata=metadata,
            config=config,
        )
        bootstrap_logger.info("Completed %s -> %s", method_name, run_dir)


if __name__ == "__main__":
    main()
