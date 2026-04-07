"""Configuration for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    """Filesystem paths used by the preprocessing pipeline."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    station_hourly_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    experiments_dir: Path = field(init=False)
    experiments_artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        output_dir = self.project_root / "data" / "processed"
        object.__setattr__(self, "input_dir", self.project_root / "data" / "cleaned")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "station_hourly_dir", output_dir / "station_hourly")
        object.__setattr__(self, "artifacts_dir", output_dir / "artifacts")
        object.__setattr__(self, "experiments_dir", output_dir / "experiments")
        object.__setattr__(self, "experiments_artifacts_dir", output_dir / "experiments" / "artifacts")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Runtime configuration for preprocessing."""

    paths: PipelinePaths = field(default_factory=PipelinePaths)
    time_column: str = "time"
    target_column: str = "volume"
    expected_columns: tuple[str, ...] = (
        "time",
        "busy",
        "idle",
        "fast_busy",
        "fast_idle",
        "slow_busy",
        "slow_idle",
        "duration",
        "volume",
        "s_price",
        "e_price",
    )
    aggregation_rules: dict[str, str] = field(
        default_factory=lambda: {
            "volume": "sum",
            "duration": "sum",
            "busy": "mean",
            "idle": "mean",
            "fast_busy": "mean",
            "fast_idle": "mean",
            "slow_busy": "mean",
            "slow_idle": "mean",
            "s_price": "mean",
            "e_price": "mean",
        }
    )
    resample_frequency: str = "1h"


DEFAULT_CONFIG = PreprocessingConfig()


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for experiment dataset generation."""

    paths: PipelinePaths = field(default_factory=PipelinePaths)
    time_column: str = "time"
    target_column: str = "volume"
    station_id_column: str = "station_id"
    window_size: int = 24
    horizon: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.05
    test_ratio: float = 0.25
    seed: int = 42
    iid_num_clients: int | None = None
    scaler_name: str = "standard"

    def __post_init__(self) -> None:
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError("Train/val/test ratios must sum to 1.0.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
