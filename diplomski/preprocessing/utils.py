"""Shared helper functions for preprocessing and experiment dataset steps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import PipelinePaths, PreprocessingConfig


ScalerType = StandardScaler | MinMaxScaler


def ensure_output_directories(paths: PipelinePaths) -> None:
    """Create output directories used by the preprocessing workflow."""

    for directory in (
        paths.output_dir,
        paths.station_hourly_dir,
        paths.artifacts_dir,
        paths.experiments_dir,
        paths.experiments_artifacts_dir,
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
    """Build aggregation rules for available columns."""

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
    dataframe[config.time_column] = pd.to_datetime(dataframe[config.time_column], errors="coerce")

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


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def get_scaler(scaler_name: str) -> ScalerType:
    """Instantiate a scaler by name."""

    normalized_name = scaler_name.strip().lower()
    if normalized_name == "standard":
        return StandardScaler()
    if normalized_name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler '{scaler_name}'.")


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
    """Persist JSON payload."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True, default=json_default)
