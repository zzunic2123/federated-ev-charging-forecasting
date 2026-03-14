"""Helper functions for the preprocessing pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import PipelinePaths, PreprocessingConfig


ScalerType = StandardScaler | MinMaxScaler


@dataclass
class SplitFrames:
    """Container for train, validation, and test dataframes."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def ensure_output_directories(paths: PipelinePaths) -> None:
    """Create the expected output directory structure."""

    for directory in (
        paths.output_dir,
        paths.station_hourly_dir,
        paths.station_splits_dir,
        paths.centralized_dir,
        paths.artifacts_dir,
        paths.station_scalers_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def discover_station_files(input_dir: Path) -> list[Path]:
    """Return all station CSV files sorted by filename."""

    station_files = sorted(input_dir.glob("*.csv"))
    if not station_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}.")
    return station_files


def build_aggregation_spec(
    columns: Iterable[str],
    config: PreprocessingConfig,
) -> dict[str, Callable[[pd.Series], float] | str]:
    """Build aggregation rules for the available columns."""

    aggregation_spec: dict[str, Callable[[pd.Series], float] | str] = {}
    for column in columns:
        rule = config.aggregation_rules.get(column)
        if rule == "sum":
            aggregation_spec[column] = sum_with_min_count
        elif rule == "mean":
            aggregation_spec[column] = "mean"
    return aggregation_spec


def sum_with_min_count(series: pd.Series) -> float:
    """Sum values while preserving NaN for all-missing windows."""

    return float(series.sum(min_count=1))


def count_missing_values(dataframe: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    """Count missing values per column and omit zero-count columns."""

    missing_counts: dict[str, int] = {}
    for column in columns:
        if column in dataframe.columns:
            missing_count = int(dataframe[column].isna().sum())
            if missing_count:
                missing_counts[column] = missing_count
    return missing_counts


def describe_time_bounds(dataframe: pd.DataFrame, time_column: str) -> dict[str, Any]:
    """Return row count and time bounds for a dataframe."""

    if dataframe.empty or time_column not in dataframe.columns:
        return {"rows": 0, "start": None, "end": None}

    start = dataframe[time_column].min()
    end = dataframe[time_column].max()
    return {
        "rows": int(len(dataframe)),
        "start": start.isoformat() if pd.notna(start) else None,
        "end": end.isoformat() if pd.notna(end) else None,
    }


def load_station_csv(
    csv_path: Path,
    config: PreprocessingConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, validate, and lightly clean a single station CSV."""

    dataframe = pd.read_csv(csv_path)
    if config.time_column not in dataframe.columns:
        raise ValueError(f"Missing required '{config.time_column}' column in {csv_path.name}.")

    available_columns = [column for column in config.expected_columns if column in dataframe.columns]
    missing_expected_columns = [
        column for column in config.expected_columns if column not in dataframe.columns
    ]
    unexpected_columns = [
        column for column in dataframe.columns if column not in config.expected_columns
    ]

    if missing_expected_columns:
        logger.warning(
            "%s is missing expected columns: %s",
            csv_path.name,
            ", ".join(missing_expected_columns),
        )
    if unexpected_columns:
        logger.info(
            "%s has extra columns that will be ignored: %s",
            csv_path.name,
            ", ".join(unexpected_columns),
        )

    dataframe = dataframe.loc[:, available_columns].copy()
    raw_row_count = len(dataframe)
    dataframe[config.time_column] = pd.to_datetime(
        dataframe[config.time_column],
        errors="coerce",
    )

    invalid_time_rows = int(dataframe[config.time_column].isna().sum())
    if invalid_time_rows:
        logger.warning("%s has %d rows with invalid timestamps.", csv_path.name, invalid_time_rows)
    dataframe = dataframe.dropna(subset=[config.time_column]).copy()

    numeric_columns = [column for column in dataframe.columns if column != config.time_column]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    raw_missing_value_counts = count_missing_values(dataframe, numeric_columns)
    fully_missing_numeric_rows = (
        int(dataframe[numeric_columns].isna().all(axis=1).sum()) if numeric_columns else 0
    )
    duplicate_timestamp_rows = int(dataframe[config.time_column].duplicated().sum())

    if duplicate_timestamp_rows:
        logger.warning(
            "%s has %d duplicate timestamps; collapsing them before hourly aggregation.",
            csv_path.name,
            duplicate_timestamp_rows,
        )
        aggregation_spec = build_aggregation_spec(numeric_columns, config)
        dataframe = (
            dataframe.groupby(config.time_column, as_index=False)
            .agg(aggregation_spec)
            .sort_values(config.time_column)
            .reset_index(drop=True)
        )

    dataframe = dataframe.sort_values(config.time_column).reset_index(drop=True)

    metadata = {
        "source_file": str(csv_path),
        "available_columns": available_columns,
        "missing_expected_columns": missing_expected_columns,
        "unexpected_columns_ignored": unexpected_columns,
        "rows_before_cleaning": raw_row_count,
        "rows_after_time_cleaning": int(len(dataframe)),
        "invalid_time_rows_dropped": invalid_time_rows,
        "duplicate_timestamps_collapsed": duplicate_timestamp_rows,
        "fully_missing_numeric_rows": fully_missing_numeric_rows,
        "raw_missing_value_counts": raw_missing_value_counts,
    }
    return dataframe, metadata


def aggregate_to_hourly(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate 5-minute station data to an hourly dataframe."""

    numeric_columns = [column for column in dataframe.columns if column != config.time_column]
    aggregation_spec = build_aggregation_spec(numeric_columns, config)
    hourly = (
        dataframe.set_index(config.time_column)
        .resample(config.resample_frequency)
        .agg(aggregation_spec)
        .reset_index()
        .sort_values(config.time_column)
        .reset_index(drop=True)
    )

    metadata = {
        "rows_after_hourly_aggregation": int(len(hourly)),
        "hourly_missing_counts_before_imputation": count_missing_values(hourly, numeric_columns),
        "hourly_time_bounds": describe_time_bounds(hourly, config.time_column),
    }
    return hourly, metadata


def compute_split_indices(row_count: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """Compute stable chronological split indices."""

    if row_count <= 0:
        return 0, 0
    if row_count == 1:
        return 1, 1
    if row_count == 2:
        return 1, 1

    train_end = int(np.floor(row_count * train_ratio))
    val_size = int(np.floor(row_count * val_ratio))

    train_end = min(max(train_end, 1), row_count - 2)
    remaining_after_train = row_count - train_end
    val_size = min(max(val_size, 1), remaining_after_train - 1)
    val_end = train_end + val_size
    return train_end, val_end


def split_frame_chronologically(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig,
) -> SplitFrames:
    """Split a dataframe into train, validation, and test by time order."""

    dataframe = dataframe.sort_values(config.time_column).reset_index(drop=True)
    train_end, val_end = compute_split_indices(
        row_count=len(dataframe),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    return SplitFrames(
        train=dataframe.iloc[:train_end].copy(),
        val=dataframe.iloc[train_end:val_end].copy(),
        test=dataframe.iloc[val_end:].copy(),
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely and return 0.0 when the denominator is zero or invalid."""

    numerator_values = numerator.to_numpy(dtype=float, copy=False)
    denominator_values = denominator.to_numpy(dtype=float, copy=False)
    result = np.zeros_like(numerator_values, dtype=float)
    valid_mask = (
        np.isfinite(numerator_values)
        & np.isfinite(denominator_values)
        & (denominator_values != 0.0)
    )
    np.divide(
        numerator_values,
        denominator_values,
        out=result,
        where=valid_mask,
    )
    return pd.Series(result, index=numerator.index)


def add_derived_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create occupancy-based features when source columns exist."""

    enriched = dataframe.copy()
    if {"busy", "idle"}.issubset(enriched.columns):
        enriched["occupancy_rate"] = safe_divide(
            enriched["busy"],
            enriched["busy"] + enriched["idle"],
        )
    if {"fast_busy", "fast_idle"}.issubset(enriched.columns):
        enriched["fast_occupancy_rate"] = safe_divide(
            enriched["fast_busy"],
            enriched["fast_busy"] + enriched["fast_idle"],
        )
    if {"slow_busy", "slow_idle"}.issubset(enriched.columns):
        enriched["slow_occupancy_rate"] = safe_divide(
            enriched["slow_busy"],
            enriched["slow_busy"] + enriched["slow_idle"],
        )
    return enriched


def add_time_features(dataframe: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Create calendar and cyclical time features from the timestamp column."""

    enriched = dataframe.copy()
    if time_column not in enriched.columns:
        return enriched

    enriched["hour"] = enriched[time_column].dt.hour.astype("int16")
    enriched["day_of_week"] = enriched[time_column].dt.dayofweek.astype("int16")
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype("int8")
    enriched["hour_sin"] = np.sin(2.0 * np.pi * enriched["hour"] / 24.0)
    enriched["hour_cos"] = np.cos(2.0 * np.pi * enriched["hour"] / 24.0)
    enriched["dow_sin"] = np.sin(2.0 * np.pi * enriched["day_of_week"] / 7.0)
    enriched["dow_cos"] = np.cos(2.0 * np.pi * enriched["day_of_week"] / 7.0)
    return enriched


def reorder_dataset_columns(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Apply a stable column order for saved datasets."""

    ordered_columns = [
        config.time_column,
        config.station_id_column,
        config.target_column,
        *feature_columns,
    ]
    existing_columns = [column for column in ordered_columns if column in dataframe.columns]
    return dataframe.loc[:, existing_columns].sort_values(config.time_column).reset_index(drop=True)


def align_frame_to_feature_columns(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Align a dataframe to a reference feature schema."""

    aligned = dataframe.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return reorder_dataset_columns(aligned, feature_columns, config)


def prepare_split_frame(
    split_frame: pd.DataFrame,
    station_id: str,
    split_name: str,
    config: PreprocessingConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute split-local features, add engineered features, and clean rows."""

    prepared = split_frame.sort_values(config.time_column).reset_index(drop=True).copy()
    base_feature_columns = [
        column
        for column in prepared.columns
        if column not in {config.time_column, config.target_column}
    ]

    if base_feature_columns:
        prepared.loc[:, base_feature_columns] = prepared.loc[:, base_feature_columns].ffill().bfill()

    rows_before_drop = len(prepared)
    required_columns = [config.target_column, *base_feature_columns]
    required_columns = [column for column in required_columns if column in prepared.columns]
    remaining_missing_before_drop = count_missing_values(prepared, required_columns)

    if required_columns:
        prepared = prepared.dropna(subset=required_columns).copy()

    dropped_rows = rows_before_drop - len(prepared)
    if dropped_rows:
        logger.warning(
            "%s/%s dropped %d rows after imputation because required values were still missing.",
            station_id,
            split_name,
            dropped_rows,
        )

    prepared = add_derived_features(prepared)
    prepared = add_time_features(prepared, config.time_column)
    prepared[config.station_id_column] = station_id

    feature_columns = [
        column
        for column in prepared.columns
        if column not in {config.time_column, config.target_column, config.station_id_column}
    ]
    scaled_feature_columns = [
        column
        for column in feature_columns
        if column not in config.non_scalable_feature_columns
    ]

    prepared = reorder_dataset_columns(prepared, feature_columns, config)
    metadata = {
        "rows": int(len(prepared)),
        "dropped_rows": int(dropped_rows),
        "remaining_missing_before_drop": remaining_missing_before_drop,
        "time_bounds": describe_time_bounds(prepared, config.time_column),
        "feature_columns": feature_columns,
        "scaled_feature_columns": scaled_feature_columns,
    }
    return prepared, metadata


def merge_feature_columns(existing_columns: list[str], new_columns: list[str]) -> list[str]:
    """Merge feature lists while preserving first-seen order."""

    merged = list(existing_columns)
    for column in new_columns:
        if column not in merged:
            merged.append(column)
    return merged


def get_scaler(scaler_name: str) -> ScalerType:
    """Instantiate the configured scaler."""

    normalized_name = scaler_name.strip().lower()
    if normalized_name == "standard":
        return StandardScaler()
    if normalized_name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler '{scaler_name}'.")


def scale_split_frames(
    splits: SplitFrames,
    scaled_feature_columns: list[str],
    scaler_name: str,
) -> tuple[SplitFrames, ScalerType | None]:
    """Fit a scaler on train and transform validation/test without leakage."""

    scaled_train = splits.train.copy()
    scaled_val = splits.val.copy()
    scaled_test = splits.test.copy()

    if not scaled_feature_columns:
        return SplitFrames(train=scaled_train, val=scaled_val, test=scaled_test), None
    if scaled_train.empty:
        raise ValueError("Cannot fit a scaler on an empty training split.")

    scaler = get_scaler(scaler_name)
    scaler.fit(scaled_train.loc[:, scaled_feature_columns])
    scaled_train.loc[:, scaled_feature_columns] = scaler.transform(
        scaled_train.loc[:, scaled_feature_columns]
    )
    scaled_val.loc[:, scaled_feature_columns] = scaler.transform(
        scaled_val.loc[:, scaled_feature_columns]
    )
    scaled_test.loc[:, scaled_feature_columns] = scaler.transform(
        scaled_test.loc[:, scaled_feature_columns]
    )
    return SplitFrames(train=scaled_train, val=scaled_val, test=scaled_test), scaler


def combine_split_frames(
    frames: list[pd.DataFrame],
    feature_columns: list[str],
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Concatenate split dataframes and align them to a shared schema."""

    if not frames:
        return pd.DataFrame(
            columns=[
                config.time_column,
                config.station_id_column,
                config.target_column,
                *feature_columns,
            ]
        )

    aligned_frames = [
        align_frame_to_feature_columns(frame, feature_columns, config) for frame in frames
    ]
    combined = pd.concat(aligned_frames, ignore_index=True)
    return reorder_dataset_columns(combined, feature_columns, config)


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def save_scaler(scaler: ScalerType, output_path: Path) -> None:
    """Persist a fitted scaler to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)


def json_default(value: Any) -> Any:
    """Serialize common scientific Python types for JSON output."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    """Persist JSON metadata using a safe serializer."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True, default=json_default)


def create_sliding_windows(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Optional helper for future sequence models."""

    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    feature_matrix = dataframe.loc[:, feature_columns].to_numpy(dtype=float)
    target_vector = dataframe.loc[:, target_column].to_numpy(dtype=float)
    sample_count = len(dataframe) - window_size - horizon + 1
    if sample_count <= 0:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=float),
            np.empty((0,), dtype=float),
        )

    windows = []
    targets = []
    for start_index in range(sample_count):
        end_index = start_index + window_size
        target_index = end_index + horizon - 1
        windows.append(feature_matrix[start_index:end_index])
        targets.append(target_vector[target_index])

    return np.asarray(windows, dtype=float), np.asarray(targets, dtype=float)
