"""CLI entrypoint for hourly station preprocessing."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, PreprocessingConfig
from .utils import (
    aggregate_to_hourly,
    discover_station_files,
    ensure_output_directories,
    load_station_csv,
    save_dataframe,
    save_json,
)


@dataclass
class StationProcessingResult:
    """Artifacts produced for one processed station."""

    station_id: str
    hourly_rows: int
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
    """Load one station CSV, aggregate to hourly, and save it."""

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

    station_metadata = {
        **load_metadata,
        **hourly_metadata,
    }
    return StationProcessingResult(
        station_id=station_id,
        hourly_rows=int(len(hourly_frame)),
        metadata=station_metadata,
    )


def build_metadata(
    config: PreprocessingConfig,
    discovered_station_count: int,
    station_results: list[StationProcessingResult],
) -> dict[str, Any]:
    """Build metadata for hourly preprocessing output."""

    return {
        "discovered_station_count": discovered_station_count,
        "processed_station_count": len(station_results),
        "target_column": config.target_column,
        "resample_frequency": config.resample_frequency,
        "data_inspection_summary": {
            "confirmed_column_order": list(config.expected_columns),
            "confirmed_time_format_example": "YYYY-MM-DD HH:MM:SS",
        },
        "stations": {
            result.station_id: result.metadata for result in station_results
        },
    }


def main() -> None:
    """Run hourly preprocessing end to end."""

    config = DEFAULT_CONFIG
    ensure_output_directories(config.paths)
    logger = configure_logging(config.paths.artifacts_dir / "preprocessing.log")

    station_files = discover_station_files(config.paths.input_dir)
    logger.info("Found %d CSV files in %s", len(station_files), config.paths.input_dir)

    station_results: list[StationProcessingResult] = []
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
        total_hourly_rows += result.hourly_rows

    if not station_results:
        raise RuntimeError("No stations were processed successfully.")

    metadata = build_metadata(
        config=config,
        discovered_station_count=len(station_files),
        station_results=station_results,
    )
    save_json(metadata, config.paths.artifacts_dir / "metadata.json")

    logger.info("Successfully processed %d stations.", len(station_results))
    logger.info("Total hourly rows produced: %d", total_hourly_rows)
    logger.info("Outputs saved to %s", config.paths.output_dir)


if __name__ == "__main__":
    main()
