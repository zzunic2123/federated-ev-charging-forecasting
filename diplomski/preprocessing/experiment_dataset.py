"""CLI for building centralized, IID, and non-IID experiment datasets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .config import DEFAULT_EXPERIMENT_CONFIG, ExperimentConfig
from .experiment_utils import (
    apply_scaler_to_all_scenarios,
    build_centralized_splits,
    build_iid_partitions_from_windows,
    build_non_iid_station_partitions,
    build_station_splits_from_hourly,
    build_windows_by_station,
    configure_experiment_logging,
    discover_hourly_files,
    ensure_experiment_directories,
    fit_global_scaler_on_train_union,
    save_centralized,
    save_partitioned_clients,
    summarize_partition_counts,
    validate_finite_windows,
)
from .utils import save_json, save_scaler


def parse_args() -> argparse.Namespace:
    """Parse CLI options for experiment dataset generation."""

    parser = argparse.ArgumentParser(
        description="Build LSTM-ready centralized, IID, and non-IID datasets from hourly station CSVs."
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_EXPERIMENT_CONFIG.window_size)
    parser.add_argument("--horizon", type=int, default=DEFAULT_EXPERIMENT_CONFIG.horizon)
    parser.add_argument("--iid-num-clients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_EXPERIMENT_CONFIG.seed)
    return parser.parse_args()


def make_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment config from default values and CLI overrides."""

    return ExperimentConfig(
        window_size=args.window_size,
        horizon=args.horizon,
        seed=args.seed,
        iid_num_clients=args.iid_num_clients,
    )


def main() -> None:
    """Run the experiment dataset generation pipeline."""

    args = parse_args()
    config = make_config_from_args(args)
    ensure_experiment_directories(config)

    logger = configure_experiment_logging(
        config.paths.experiments_artifacts_dir / "experiment.log"
    )
    logger.info("Starting experiment dataset generation.")
    logger.info(
        "Config: window_size=%d horizon=%d seed=%d",
        config.window_size,
        config.horizon,
        config.seed,
    )

    hourly_files = discover_hourly_files(config.paths.station_hourly_dir)
    logger.info("Found %d hourly station files.", len(hourly_files))

    station_splits, feature_columns, station_split_stats = build_station_splits_from_hourly(
        hourly_files=hourly_files,
        config=config,
        logger=logger,
    )
    logger.info("Built chronological splits for %d stations.", len(station_splits))

    windows_by_station, station_window_stats = build_windows_by_station(
        station_splits=station_splits,
        feature_columns=feature_columns,
        config=config,
        logger=logger,
    )

    valid_stations = [
        station_id
        for station_id, payload in windows_by_station.items()
        if payload.X_train.shape[0] > 0
    ]
    if not valid_stations:
        raise RuntimeError("No station has enough train windows for experiment generation.")

    filtered_windows = {station_id: windows_by_station[station_id] for station_id in valid_stations}
    iid_num_clients = config.iid_num_clients if config.iid_num_clients is not None else len(valid_stations)
    if iid_num_clients <= 0:
        raise ValueError("iid_num_clients must be positive.")

    logger.info("Stations with non-empty train windows: %d", len(valid_stations))
    logger.info("IID virtual clients: %d", iid_num_clients)

    scaler = fit_global_scaler_on_train_union(
        windows_by_station=filtered_windows,
        scaler_name=config.scaler_name,
    )
    scaled_windows = apply_scaler_to_all_scenarios(scaler=scaler, windows_by_station=filtered_windows)
    finite_ok, finite_errors = validate_finite_windows(scaled_windows)
    if not finite_ok:
        raise RuntimeError("Scaled tensors contain invalid values: " + "; ".join(finite_errors))

    centralized = build_centralized_splits(scaled_windows)
    iid_partitions = build_iid_partitions_from_windows(
        windows_by_station=scaled_windows,
        num_clients=iid_num_clients,
        seed=config.seed,
    )
    non_iid_partitions = build_non_iid_station_partitions(scaled_windows)

    save_centralized(
        centralized=centralized,
        base_dir=config.paths.experiments_dir,
    )
    save_partitioned_clients(
        partitions=iid_partitions,
        scenario_dir=config.paths.experiments_dir / "iid",
    )
    save_partitioned_clients(
        partitions=non_iid_partitions,
        scenario_dir=config.paths.experiments_dir / "non_iid_station",
    )
    save_scaler(
        scaler=scaler,
        output_path=config.paths.experiments_artifacts_dir / "global_window_scaler.joblib",
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_size": config.window_size,
        "horizon": config.horizon,
        "split_ratios": {
            "train": config.train_ratio,
            "val": config.val_ratio,
            "test": config.test_ratio,
        },
        "seed": config.seed,
        "iid_num_clients": iid_num_clients,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "valid_station_count": len(valid_stations),
        "valid_station_ids": valid_stations,
        "station_split_stats": station_split_stats,
        "station_window_stats": station_window_stats,
        "centralized_counts": {
            split_name: int(values[0].shape[0]) for split_name, values in centralized.items()
        },
        "iid_counts": summarize_partition_counts(iid_partitions),
        "non_iid_station_counts": summarize_partition_counts(non_iid_partitions),
        "outputs": {
            "centralized_dir": str(config.paths.experiments_dir / "centralized"),
            "iid_dir": str(config.paths.experiments_dir / "iid"),
            "non_iid_station_dir": str(config.paths.experiments_dir / "non_iid_station"),
            "artifacts_dir": str(config.paths.experiments_artifacts_dir),
        },
        "scaler_name": config.scaler_name,
        "target_column": config.target_column,
    }
    save_json(metadata, config.paths.experiments_artifacts_dir / "experiment_metadata.json")

    logger.info(
        "Saved centralized splits: train=%d val=%d test=%d",
        centralized["train"][0].shape[0],
        centralized["val"][0].shape[0],
        centralized["test"][0].shape[0],
    )
    logger.info("Saved IID partitions for %d clients.", len(iid_partitions))
    logger.info("Saved non-IID station partitions for %d clients.", len(non_iid_partitions))
    logger.info("Artifacts saved to %s", config.paths.experiments_artifacts_dir)


if __name__ == "__main__":
    main()
