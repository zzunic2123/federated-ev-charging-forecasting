"""CLI entrypoint for preprocessing EV charging station time series data."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, PreprocessingConfig
from .utils import (
    SplitFrames,
    aggregate_to_hourly,
    align_frame_to_feature_columns,
    combine_split_frames,
    describe_time_bounds,
    discover_station_files,
    ensure_output_directories,
    load_station_csv,
    merge_feature_columns,
    prepare_split_frame,
    save_dataframe,
    save_json,
    save_scaler,
    scale_split_frames,
    split_frame_chronologically,
)


@dataclass
class StationProcessingResult:
    """Artifacts produced for a single station."""

    station_id: str
    source_file: Path
    hourly_frame: Any
    prepared_splits: SplitFrames
    scaled_splits: SplitFrames
    feature_columns: list[str]
    scaled_feature_columns: list[str]
    metadata: dict[str, Any]


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure console and file logging."""

    logger = logging.getLogger("diplomski.preprocessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def process_station_file(
    csv_path: Path,
    config: PreprocessingConfig,
    logger: logging.Logger,
) -> StationProcessingResult | None:
    """Run the full preprocessing flow for a single station file."""

    station_id = csv_path.stem
    raw_frame, load_metadata = load_station_csv(csv_path, config, logger)
    if raw_frame.empty:
        logger.warning("%s is empty after initial cleaning and will be skipped.", csv_path.name)
        return None

    hourly_frame, hourly_metadata = aggregate_to_hourly(raw_frame, config)
    if hourly_frame.empty:
        logger.warning("%s produced no hourly rows and will be skipped.", csv_path.name)
        return None

    save_dataframe(
        hourly_frame,
        config.paths.station_hourly_dir / f"{station_id}_hourly.csv",
    )

    split_frames = split_frame_chronologically(hourly_frame, config)
    prepared_train, train_metadata = prepare_split_frame(
        split_frames.train,
        station_id=station_id,
        split_name="train",
        config=config,
        logger=logger,
    )
    prepared_val, val_metadata = prepare_split_frame(
        split_frames.val,
        station_id=station_id,
        split_name="val",
        config=config,
        logger=logger,
    )
    prepared_test, test_metadata = prepare_split_frame(
        split_frames.test,
        station_id=station_id,
        split_name="test",
        config=config,
        logger=logger,
    )

    if prepared_train.empty:
        logger.warning("%s has an empty train split after cleaning and will be skipped.", station_id)
        return None

    feature_columns = train_metadata["feature_columns"]
    scaled_feature_columns = train_metadata["scaled_feature_columns"]
    prepared_splits = SplitFrames(
        train=align_frame_to_feature_columns(prepared_train, feature_columns, config),
        val=align_frame_to_feature_columns(prepared_val, feature_columns, config),
        test=align_frame_to_feature_columns(prepared_test, feature_columns, config),
    )

    scaled_splits, scaler = scale_split_frames(
        prepared_splits,
        scaled_feature_columns=scaled_feature_columns,
        scaler_name=config.scaler_name,
    )

    save_dataframe(
        scaled_splits.train,
        config.paths.station_splits_dir / f"{station_id}_train.csv",
    )
    save_dataframe(
        scaled_splits.val,
        config.paths.station_splits_dir / f"{station_id}_val.csv",
    )
    save_dataframe(
        scaled_splits.test,
        config.paths.station_splits_dir / f"{station_id}_test.csv",
    )

    if scaler is not None:
        save_scaler(
            scaler,
            config.paths.station_scalers_dir / f"{station_id}_scaler.joblib",
        )

    station_metadata = {
        **load_metadata,
        **hourly_metadata,
        "feature_columns": feature_columns,
        "scaled_feature_columns": scaled_feature_columns,
        "split_boundaries": {
            "train": train_metadata["time_bounds"],
            "val": val_metadata["time_bounds"],
            "test": test_metadata["time_bounds"],
        },
        "split_rows": {
            "train": train_metadata["rows"],
            "val": val_metadata["rows"],
            "test": test_metadata["rows"],
        },
        "rows_dropped_after_imputation": {
            "train": train_metadata["dropped_rows"],
            "val": val_metadata["dropped_rows"],
            "test": test_metadata["dropped_rows"],
        },
        "remaining_missing_before_drop": {
            "train": train_metadata["remaining_missing_before_drop"],
            "val": val_metadata["remaining_missing_before_drop"],
            "test": test_metadata["remaining_missing_before_drop"],
        },
    }

    return StationProcessingResult(
        station_id=station_id,
        source_file=csv_path,
        hourly_frame=hourly_frame,
        prepared_splits=prepared_splits,
        scaled_splits=scaled_splits,
        feature_columns=feature_columns,
        scaled_feature_columns=scaled_feature_columns,
        metadata=station_metadata,
    )


def build_metadata(
    config: PreprocessingConfig,
    discovered_station_count: int,
    station_results: list[StationProcessingResult],
    centralized_splits: SplitFrames,
    centralized_feature_columns: list[str],
    centralized_scaled_feature_columns: list[str],
) -> dict[str, Any]:
    """Assemble the metadata payload for JSON export."""

    return {
        "discovered_station_count": discovered_station_count,
        "processed_station_count": len(station_results),
        "target_column": config.target_column,
        "feature_count": len(centralized_feature_columns),
        "feature_columns": centralized_feature_columns,
        "scaled_feature_columns": centralized_scaled_feature_columns,
        "unscaled_feature_columns": [
            column
            for column in centralized_feature_columns
            if column not in centralized_scaled_feature_columns
        ],
        "time_feature_columns": list(config.time_feature_columns),
        "derived_feature_columns": list(config.derived_feature_columns),
        "split_ratios": {
            "train": config.train_ratio,
            "val": config.val_ratio,
            "test": config.test_ratio,
        },
        "scaler_name": config.scaler_name,
        "data_inspection_summary": {
            "actual_csv_count_observed": discovered_station_count,
            "confirmed_column_order": list(config.expected_columns),
            "confirmed_time_format_example": "YYYY-MM-DD HH:MM:SS",
        },
        "centralized_splits": {
            "train": describe_time_bounds(centralized_splits.train, config.time_column),
            "val": describe_time_bounds(centralized_splits.val, config.time_column),
            "test": describe_time_bounds(centralized_splits.test, config.time_column),
        },
        "stations": {
            result.station_id: result.metadata for result in station_results
        },
    }


def main() -> None:
    """Run the preprocessing pipeline end to end."""

    config = DEFAULT_CONFIG
    ensure_output_directories(config.paths)
    logger = configure_logging(config.paths.artifacts_dir / "preprocessing.log")

    station_files = discover_station_files(config.paths.input_dir)
    logger.info("Found %d CSV files in %s", len(station_files), config.paths.input_dir)

    station_results: list[StationProcessingResult] = []
    combined_train_frames = []
    combined_val_frames = []
    combined_test_frames = []
    master_feature_columns: list[str] = []
    master_scaled_feature_columns: list[str] = []
    total_hourly_rows = 0

    for csv_path in station_files:
        try:
            result = process_station_file(csv_path, config, logger)
        except Exception:
            logger.exception("Failed to process %s", csv_path.name)
            continue

        if result is None:
            continue

        station_results.append(result)
        total_hourly_rows += int(len(result.hourly_frame))
        combined_train_frames.append(result.prepared_splits.train)
        combined_val_frames.append(result.prepared_splits.val)
        combined_test_frames.append(result.prepared_splits.test)
        master_feature_columns = merge_feature_columns(
            master_feature_columns,
            result.feature_columns,
        )
        master_scaled_feature_columns = merge_feature_columns(
            master_scaled_feature_columns,
            result.scaled_feature_columns,
        )

    if not station_results:
        raise RuntimeError("No stations were processed successfully.")

    combined_train = combine_split_frames(combined_train_frames, master_feature_columns, config)
    combined_val = combine_split_frames(combined_val_frames, master_feature_columns, config)
    combined_test = combine_split_frames(combined_test_frames, master_feature_columns, config)

    centralized_raw_splits = SplitFrames(
        train=combined_train,
        val=combined_val,
        test=combined_test,
    )
    centralized_scaled_splits, centralized_scaler = scale_split_frames(
        centralized_raw_splits,
        scaled_feature_columns=master_scaled_feature_columns,
        scaler_name=config.scaler_name,
    )

    save_dataframe(
        centralized_scaled_splits.train,
        config.paths.centralized_dir / "train.csv",
    )
    save_dataframe(
        centralized_scaled_splits.val,
        config.paths.centralized_dir / "val.csv",
    )
    save_dataframe(
        centralized_scaled_splits.test,
        config.paths.centralized_dir / "test.csv",
    )

    if centralized_scaler is not None:
        save_scaler(
            centralized_scaler,
            config.paths.artifacts_dir / "centralized_scaler.joblib",
        )

    metadata = build_metadata(
        config=config,
        discovered_station_count=len(station_files),
        station_results=station_results,
        centralized_splits=centralized_scaled_splits,
        centralized_feature_columns=master_feature_columns,
        centralized_scaled_feature_columns=master_scaled_feature_columns,
    )
    save_json(metadata, config.paths.artifacts_dir / "metadata.json")

    logger.info("Successfully processed %d stations.", len(station_results))
    logger.info("Total hourly rows produced: %d", total_hourly_rows)
    logger.info("Outputs saved to %s", config.paths.output_dir)


if __name__ == "__main__":
    main()
