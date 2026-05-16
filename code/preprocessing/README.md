# Preprocessing Pipeline

This module now uses a two-step workflow:

1. `cleaned CSV -> station_hourly`
2. `station_hourly -> experiment .npz datasets (centralized, IID, non-IID station-as-client)`

The legacy station-level split CSV workflow is intentionally removed from the primary path.

## Confirmed Input Data

- Source folder: `data/cleaned/`
- Current file count: `51` station CSV files
- Confirmed header:
  `time,busy,idle,s_price,e_price,fast_busy,fast_idle,slow_busy,slow_idle,duration,volume`
- Time format:
  `YYYY-MM-DD HH:MM:SS`

## Step 1: Hourly Preprocessing

Run:

```bash
python -m diplomski.preprocessing.preprocess
```

What it does:

- validates and parses `time`
- sorts data by time
- handles duplicate timestamps
- aggregates 5-minute data to 1-hour data
- saves per-station hourly CSV files
- writes metadata and log

Outputs:

- `data/processed/station_hourly/*.csv`
- `data/processed/artifacts/metadata.json`
- `data/processed/artifacts/preprocessing.log`

## Step 2: Experiment Dataset Generation

Run:

```bash
python -m diplomski.preprocessing.experiment_dataset
```

Optional CLI arguments:

```bash
python -m diplomski.preprocessing.experiment_dataset \
  --window-size 24 \
  --horizon 1 \
  --iid-num-clients 40 \
  --seed 42
```

What it does:

- loads `station_hourly` files as canonical input
- applies chronological train/val/test split
- builds sliding windows `X(t-23...t) -> y(t+1)`
- fits one global scaler on union of train windows only
- applies same scaler to centralized, IID, and non-IID scenarios
- keeps target `y` unscaled

Outputs:

- `data/processed/experiments/centralized/{train,val,test}.npz`
- `data/processed/experiments/iid/client_{id}/{train,val,test}.npz`
- `data/processed/experiments/non_iid_station/client_<station_id>/{train,val,test}.npz`
- `data/processed/experiments/artifacts/global_window_scaler.joblib`
- `data/processed/experiments/artifacts/experiment_metadata.json`
- `data/processed/experiments/artifacts/experiment.log`

## Implementation Notes

- IID shuffling is applied only to train windows.
- Train/val/test temporal boundaries are never mixed.
- Non-IID scenario is natural station-as-client mapping.
