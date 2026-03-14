# Preprocessing Pipeline

This module prepares hourly tabular time-series data for the thesis project on Federated Learning for EV Charging Demand Forecasting.

During the initial inspection of the real input data, the following was confirmed:

- `data/cleaned/` currently contains `51` CSV files
- all CSV files share the same header:
  `time,busy,idle,s_price,e_price,fast_busy,fast_idle,slow_busy,slow_idle,duration,volume`
- `time` uses the format `YYYY-MM-DD HH:MM:SS`
- every inspected file contains at least one block of rows where the timestamp exists but all numeric values are empty

## What The Pipeline Does

For each station, the pipeline:

1. discovers the CSV automatically
2. validates that `time` exists
3. parses `time` to `datetime`
4. sorts by time
5. collapses duplicate timestamps if they appear
6. aggregates 5-minute data to `1h`
7. splits the hourly time series chronologically into `train`, `val`, and `test` using `70/5/25`
8. imputes numeric features within each split using `ffill()` then `bfill()`
9. drops rows only when required values are still missing after imputation
10. creates derived and time-based features
11. fits scalers on train only and applies them to validation and test without leakage
12. saves station-level and centralized outputs to disk

## Features Used By The Pipeline

Target:

- `volume`

Base features:

- all available hourly aggregated numeric columns except `time`
- in practice this means the available subset of:
  `busy`, `idle`, `fast_busy`, `fast_idle`, `slow_busy`, `slow_idle`, `duration`, `volume`, `s_price`, `e_price`

Derived features:

- `occupancy_rate = busy / (busy + idle)` when `busy` and `idle` exist
- `fast_occupancy_rate = fast_busy / (fast_busy + fast_idle)` when `fast_busy` and `fast_idle` exist
- `slow_occupancy_rate = slow_busy / (slow_busy + slow_idle)` when `slow_busy` and `slow_idle` exist

Time features:

- `hour`
- `day_of_week`
- `is_weekend`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`

Scaling:

- continuous numeric features are scaled
- `hour`, `day_of_week`, `is_weekend`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` are not scaled
- target `volume` is kept as the target column and is not scaled

## Aggregation Rules

- `volume` -> `sum`
- `duration` -> `sum`
- `busy`, `idle`, `fast_busy`, `fast_idle`, `slow_busy`, `slow_idle` -> `mean`
- `s_price`, `e_price` -> `mean`

Missing columns from the expected set are skipped and logged.

## Output Structure

The pipeline writes outputs to `data/processed/`:

- `station_hourly/`
  - hourly aggregated station CSV files before scaling
- `station_splits/`
  - `*_train.csv`, `*_val.csv`, `*_test.csv` for each station
  - these files are scaled with station-level scalers
- `centralized/`
  - `train.csv`
  - `val.csv`
  - `test.csv`
  - these files are scaled with one global scaler fitted only on the combined train split
- `artifacts/`
  - `metadata.json`
  - `preprocessing.log`
  - `centralized_scaler.joblib`
  - `station_scalers/*.joblib`

`metadata.json` contains:

- number of discovered and processed stations
- row counts before cleaning and after hourly aggregation for each station
- number of features
- used features
- scaled features
- time boundaries for train, validation, and test splits
- missing-value and dropped-row information

## Running The Pipeline

Required packages:

```bash
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib
```

Run from the project root:

```bash
python -m diplomski.preprocessing.preprocess
```

At the end of execution, the script logs:

- how many CSV files were found
- how many stations were processed successfully
- how many total hourly rows were produced
- where outputs were saved

## Architecture

- `config.py`
  - central configuration for paths, target, aggregation rules, split ratios, and scaler selection
- `utils.py`
  - helper functions for loading, validation, aggregation, feature engineering, split handling, scaling, and artifact persistence
- `preprocess.py`
  - CLI entrypoint that processes all stations, writes outputs, and emits the final summary

The output remains a clean hourly tabular time series so that sliding-window creation for LSTM or Transformer models can be added later without rewriting the preprocessing layer.
