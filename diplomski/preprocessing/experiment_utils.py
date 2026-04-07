"""Utilities for building LSTM-ready experiment datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .utils import (
    ScalerType,
    add_derived_features,
    add_time_features,
    get_scaler,
    save_json,
    save_scaler,
)


@dataclass
class SplitDataFrame:
    """Chronological dataframes for train, val, and test."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class WindowSplit:
    """Window tensors for train, val, and test."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    t_train: np.ndarray
    t_val: np.ndarray
    t_test: np.ndarray


def ensure_experiment_directories(config: ExperimentConfig) -> None:
    """Create required output directories for experiment datasets."""

    for directory in (
        config.paths.experiments_dir,
        config.paths.experiments_artifacts_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def configure_experiment_logging(log_path: Path) -> logging.Logger:
    """Configure logger for experiment dataset generation."""

    logger = logging.getLogger("diplomski.preprocessing.experiments")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def discover_hourly_files(hourly_dir: Path) -> list[Path]:
    """Find all hourly station CSV files."""

    files = sorted(hourly_dir.glob("*_hourly.csv"))
    if not files:
        raise FileNotFoundError(f"No hourly CSV files found in {hourly_dir}.")
    return files


def compute_split_indices(row_count: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """Compute robust split indices for short and long sequences."""

    if row_count <= 0:
        return 0, 0
    if row_count == 1:
        return 1, 1
    if row_count == 2:
        return 1, 1

    train_end = int(np.floor(row_count * train_ratio))
    val_size = int(np.floor(row_count * val_ratio))
    train_end = min(max(train_end, 1), row_count - 2)
    remaining = row_count - train_end
    val_size = min(max(val_size, 1), remaining - 1)
    val_end = train_end + val_size
    return train_end, val_end


def clean_split_for_windows(
    split_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    time_column: str,
) -> tuple[pd.DataFrame, int]:
    """Impute feature columns and remove rows that remain invalid."""

    cleaned = split_df.sort_values(time_column).reset_index(drop=True).copy()
    if feature_columns:
        cleaned.loc[:, feature_columns] = cleaned.loc[:, feature_columns].ffill().bfill()
    required_columns = [target_column, *feature_columns]
    required_columns = [column for column in required_columns if column in cleaned.columns]
    before = len(cleaned)
    if required_columns:
        cleaned = cleaned.dropna(subset=required_columns).copy()
    dropped = before - len(cleaned)
    return cleaned, dropped


def create_windows_for_split(
    split_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    time_column: str,
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windows X(t-window+1...t) and target y(t+horizon)."""

    if split_df.empty:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    feature_matrix = split_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    target_vector = split_df.loc[:, target_column].to_numpy(dtype=np.float32)
    time_vector = split_df.loc[:, time_column].to_numpy()

    sample_count = len(split_df) - window_size - horizon + 1
    if sample_count <= 0:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    windows = []
    targets = []
    target_times = []
    for start_idx in range(sample_count):
        end_idx = start_idx + window_size
        target_idx = end_idx + horizon - 1
        windows.append(feature_matrix[start_idx:end_idx])
        targets.append(target_vector[target_idx])
        target_times.append(time_vector[target_idx])

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_times),
    )


def build_station_splits_from_hourly(
    hourly_files: list[Path],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[dict[str, SplitDataFrame], list[str], dict[str, dict[str, int]]]:
    """Load hourly station files and create cleaned chronological splits."""

    station_splits: dict[str, SplitDataFrame] = {}
    station_stats: dict[str, dict[str, int]] = {}
    feature_columns_union: list[str] = []

    for hourly_file in hourly_files:
        station_id = hourly_file.stem.replace("_hourly", "")
        dataframe = pd.read_csv(hourly_file)
        if config.time_column not in dataframe.columns or config.target_column not in dataframe.columns:
            logger.warning(
                "Skipping %s because required columns are missing.",
                hourly_file.name,
            )
            continue

        dataframe[config.time_column] = pd.to_datetime(dataframe[config.time_column], errors="coerce")
        dataframe = dataframe.dropna(subset=[config.time_column]).sort_values(config.time_column).reset_index(drop=True)

        numeric_columns = [c for c in dataframe.columns if c != config.time_column]
        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

        dataframe = add_derived_features(dataframe)
        dataframe = add_time_features(dataframe, config.time_column)

        feature_columns = [
            c
            for c in dataframe.columns
            if c not in {config.time_column, config.target_column}
        ]
        for column in feature_columns:
            if column not in feature_columns_union:
                feature_columns_union.append(column)

        split_train_end, split_val_end = compute_split_indices(
            row_count=len(dataframe),
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
        )
        split_frames = SplitDataFrame(
            train=dataframe.iloc[:split_train_end].copy(),
            val=dataframe.iloc[split_train_end:split_val_end].copy(),
            test=dataframe.iloc[split_val_end:].copy(),
        )
        station_splits[station_id] = split_frames
        station_stats[station_id] = {
            "rows_hourly": int(len(dataframe)),
            "rows_train_raw": int(len(split_frames.train)),
            "rows_val_raw": int(len(split_frames.val)),
            "rows_test_raw": int(len(split_frames.test)),
        }

    if not station_splits:
        raise RuntimeError("No station splits could be built from hourly files.")
    return station_splits, feature_columns_union, station_stats


def build_windows_by_station(
    station_splits: dict[str, SplitDataFrame],
    feature_columns: list[str],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[dict[str, WindowSplit], dict[str, dict[str, int]]]:
    """Build train/val/test windows for each station."""

    windows_by_station: dict[str, WindowSplit] = {}
    window_stats: dict[str, dict[str, int]] = {}

    for station_id, split_frames in station_splits.items():
        split_windows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        split_drop_counts: dict[str, int] = {}
        split_row_counts: dict[str, int] = {}

        for split_name, split_df in (
            ("train", split_frames.train),
            ("val", split_frames.val),
            ("test", split_frames.test),
        ):
            aligned = split_df.copy()
            for column in feature_columns:
                if column not in aligned.columns:
                    aligned[column] = np.nan
            cleaned, dropped_rows = clean_split_for_windows(
                split_df=aligned,
                feature_columns=feature_columns,
                target_column=config.target_column,
                time_column=config.time_column,
            )
            split_drop_counts[split_name] = dropped_rows
            split_row_counts[split_name] = int(len(cleaned))
            split_windows[split_name] = create_windows_for_split(
                split_df=cleaned,
                feature_columns=feature_columns,
                target_column=config.target_column,
                time_column=config.time_column,
                window_size=config.window_size,
                horizon=config.horizon,
            )

        windows_by_station[station_id] = WindowSplit(
            X_train=split_windows["train"][0],
            y_train=split_windows["train"][1],
            X_val=split_windows["val"][0],
            y_val=split_windows["val"][1],
            X_test=split_windows["test"][0],
            y_test=split_windows["test"][1],
            t_train=split_windows["train"][2],
            t_val=split_windows["val"][2],
            t_test=split_windows["test"][2],
        )
        window_stats[station_id] = {
            "rows_train_clean": split_row_counts["train"],
            "rows_val_clean": split_row_counts["val"],
            "rows_test_clean": split_row_counts["test"],
            "rows_dropped_train": split_drop_counts["train"],
            "rows_dropped_val": split_drop_counts["val"],
            "rows_dropped_test": split_drop_counts["test"],
            "windows_train": int(split_windows["train"][0].shape[0]),
            "windows_val": int(split_windows["val"][0].shape[0]),
            "windows_test": int(split_windows["test"][0].shape[0]),
        }
        if split_drop_counts["train"] or split_drop_counts["val"] or split_drop_counts["test"]:
            logger.warning(
                "%s dropped rows during split cleaning: train=%d val=%d test=%d",
                station_id,
                split_drop_counts["train"],
                split_drop_counts["val"],
                split_drop_counts["test"],
            )

    return windows_by_station, window_stats


def concatenate_windows(
    windows_by_station: dict[str, WindowSplit],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate windows from all stations for a given split."""

    x_list = []
    y_list = []
    t_list = []
    station_id_list = []

    for station_id, station_windows in windows_by_station.items():
        X = getattr(station_windows, f"X_{split_name}")
        y = getattr(station_windows, f"y_{split_name}")
        t = getattr(station_windows, f"t_{split_name}")
        if X.shape[0] == 0:
            continue
        x_list.append(X)
        y_list.append(y)
        t_list.append(t)
        station_id_list.append(np.full((X.shape[0],), station_id, dtype=object))

    if not x_list:
        reference_shape = next(iter(windows_by_station.values())).X_train.shape
        return (
            np.empty((0, reference_shape[1], reference_shape[2]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
            np.empty((0,), dtype=object),
        )

    return (
        np.concatenate(x_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(t_list, axis=0),
        np.concatenate(station_id_list, axis=0),
    )


def fit_global_scaler_on_train_union(
    windows_by_station: dict[str, WindowSplit],
    scaler_name: str,
) -> ScalerType:
    """Fit one global feature scaler on union of all train windows."""

    X_train_union, _, _, _ = concatenate_windows(windows_by_station, "train")
    if X_train_union.shape[0] == 0:
        raise RuntimeError("Cannot fit scaler because train union has 0 windows.")

    scaler = get_scaler(scaler_name)
    scaler.fit(X_train_union.reshape(-1, X_train_union.shape[-1]))
    return scaler


def transform_window_tensor(X: np.ndarray, scaler: ScalerType) -> np.ndarray:
    """Transform a 3D tensor [samples, window, features] with a 2D scaler."""

    if X.shape[0] == 0:
        return X
    transformed = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return transformed.astype(np.float32, copy=False)


def apply_scaler_to_all_scenarios(
    scaler: ScalerType,
    windows_by_station: dict[str, WindowSplit],
) -> dict[str, WindowSplit]:
    """Apply the same global scaler to all station split tensors."""

    transformed: dict[str, WindowSplit] = {}
    for station_id, windows in windows_by_station.items():
        transformed[station_id] = WindowSplit(
            X_train=transform_window_tensor(windows.X_train, scaler),
            y_train=windows.y_train,
            X_val=transform_window_tensor(windows.X_val, scaler),
            y_val=windows.y_val,
            X_test=transform_window_tensor(windows.X_test, scaler),
            y_test=windows.y_test,
            t_train=windows.t_train,
            t_val=windows.t_val,
            t_test=windows.t_test,
        )
    return transformed


def build_non_iid_station_partitions(
    windows_by_station: dict[str, WindowSplit],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Build one-client-per-station partitions."""

    partitions: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for station_id, station_windows in windows_by_station.items():
        client_id = f"client_{station_id}"
        partitions[client_id] = {
            "train": (
                station_windows.X_train,
                station_windows.y_train,
                station_windows.t_train,
                np.full((station_windows.X_train.shape[0],), station_id, dtype=object),
            ),
            "val": (
                station_windows.X_val,
                station_windows.y_val,
                station_windows.t_val,
                np.full((station_windows.X_val.shape[0],), station_id, dtype=object),
            ),
            "test": (
                station_windows.X_test,
                station_windows.y_test,
                station_windows.t_test,
                np.full((station_windows.X_test.shape[0],), station_id, dtype=object),
            ),
        }
    return partitions


def split_indices_evenly(sample_count: int, num_clients: int) -> list[np.ndarray]:
    """Split indices into nearly-equal client chunks."""

    base = sample_count // num_clients
    remainder = sample_count % num_clients
    result = []
    start = 0
    for client_idx in range(num_clients):
        count = base + (1 if client_idx < remainder else 0)
        end = start + count
        result.append(np.arange(start, end))
        start = end
    return result


def build_iid_partitions_from_windows(
    windows_by_station: dict[str, WindowSplit],
    num_clients: int,
    seed: int,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Build IID partitions by merging and balanced splitting across virtual clients."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    merged = {
        split_name: concatenate_windows(windows_by_station, split_name)
        for split_name in ("train", "val", "test")
    }

    rng = np.random.default_rng(seed)
    train_size = merged["train"][0].shape[0]
    train_perm = rng.permutation(train_size)
    merged["train"] = tuple(arr[train_perm] for arr in merged["train"])

    partitions: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
        f"client_{idx:03d}": {} for idx in range(num_clients)
    }
    for split_name in ("train", "val", "test"):
        X, y, t, station_ids = merged[split_name]
        split_chunks = split_indices_evenly(X.shape[0], num_clients)
        for client_idx, idx_array in enumerate(split_chunks):
            client_id = f"client_{client_idx:03d}"
            partitions[client_id][split_name] = (
                X[idx_array],
                y[idx_array],
                t[idx_array],
                station_ids[idx_array],
            )
    return partitions


def build_centralized_splits(
    windows_by_station: dict[str, WindowSplit],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Merge all stations into centralized train/val/test tensors."""

    return {
        split_name: concatenate_windows(windows_by_station, split_name)
        for split_name in ("train", "val", "test")
    }


def save_split_npz(
    output_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    station_ids: np.ndarray,
) -> None:
    """Save one split tensor bundle to NPZ."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        t=t.astype("datetime64[ns]").astype(str),
        station_ids=station_ids.astype(str),
    )


def save_centralized(
    centralized: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    base_dir: Path,
) -> None:
    """Save centralized split files."""

    for split_name, (X, y, t, station_ids) in centralized.items():
        save_split_npz(base_dir / "centralized" / f"{split_name}.npz", X, y, t, station_ids)


def save_partitioned_clients(
    partitions: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    scenario_dir: Path,
) -> None:
    """Save split NPZ files per client."""

    for client_id, split_payloads in partitions.items():
        for split_name, (X, y, t, station_ids) in split_payloads.items():
            save_split_npz(scenario_dir / client_id / f"{split_name}.npz", X, y, t, station_ids)


def validate_finite_windows(
    windows_by_station: dict[str, WindowSplit],
) -> tuple[bool, list[str]]:
    """Check that all feature tensors contain finite values only."""

    errors: list[str] = []
    for station_id, windows in windows_by_station.items():
        for split_name, X in (
            ("train", windows.X_train),
            ("val", windows.X_val),
            ("test", windows.X_test),
        ):
            if X.shape[0] == 0:
                continue
            if not np.isfinite(X).all():
                errors.append(f"{station_id}:{split_name} contains NaN/Inf in X")
    return (len(errors) == 0, errors)


def summarize_partition_counts(
    partitions: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]
) -> dict[str, dict[str, int]]:
    """Build counts summary for metadata."""

    summary: dict[str, dict[str, int]] = {}
    for client_id, payloads in partitions.items():
        summary[client_id] = {
            split_name: int(values[0].shape[0]) for split_name, values in payloads.items()
        }
    return summary

