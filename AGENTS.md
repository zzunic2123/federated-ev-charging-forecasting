# AGENTS.md

## Project
Master thesis project: **Primjena federalnog učenja za predviđanje opterećenja u mrežama električnih punionica**

The project compares:
1. centralized learning
2. federated learning with FedAvg
3. federated learning with FedProx
4. optional decentralized FL later

The main goal is to predict future EV charging load and analyze:
- prediction accuracy
- impact of heterogeneous data across charging stations
- communication overhead in federated settings

---

## Paths
- Project root: `/home/zlatko/10.semestar/Diplomski`
- Code directory: `/home/zlatko/10.semestar/Diplomski/code`
- Input data: `/home/zlatko/10.semestar/Diplomski/data/cleaned`

Write all new code inside the `code` directory.

Do not modify raw input files in `data/cleaned`.

---

## Data assumptions
- The input folder contains 50 CSV files.
- Each CSV represents one EV charging station.
- Expected columns may include:
  - `time`
  - `busy`
  - `idle`
  - `fast_busy`
  - `fast_idle`
  - `slow_busy`
  - `slow_idle`
  - `duration`
  - `volume`
  - `s_price`
  - `e_price`

Always inspect real files before implementing logic.
Do not assume all CSV files are perfectly clean or identical.

---

## Main target
- Main prediction target: `volume`
- Treat `volume` as the main proxy for charging load / energy demand.

Secondary analysis may optionally use:
- `busy`
- occupancy-related derived features

---

## Time-series setup
Preferred default setup:
- original data resolution: 5 minutes
- working modeling resolution: 1 hour
- default input window: previous 24 hours
- default prediction horizon: next 1 hour

Use chronological order at all times.
Never use random shuffling for train/validation/test splitting.

---

## Project phases
Work on the project in this order:

1. data inspection
2. preprocessing pipeline
3. centralized baseline
4. federated learning with FedAvg
5. federated learning with FedProx
6. evaluation and comparison
7. communication cost analysis
8. optional decentralized FL extension later

Do not jump to advanced methods before the previous phase is stable and reproducible.

---

## Preprocessing rules
Preprocessing should:
- load all station CSV files
- parse `time` as datetime
- sort by time
- aggregate 5-minute data to hourly data
- create clean tabular time-series per station
- create train/validation/test splits by time
- avoid data leakage
- save processed outputs separately from raw data

Handle missing values carefully.
Log important data quality issues.
Do not silently drop large parts of the dataset.

Fit scalers only on training data.
Apply the same fitted scalers to validation and test data.

---

## Feature rules
Use only features that actually exist in the data.

Preferred feature groups:
1. core operational features:
   - `volume`
   - `busy`
   - `idle`
   - `duration`
   - `s_price`
   - `e_price`

2. derived features when possible:
   - occupancy rate
   - fast occupancy rate
   - slow occupancy rate

3. time features:
   - hour
   - day of week
   - weekend flag
   - cyclical encoding for hour
   - cyclical encoding for day of week

When building models, avoid future leakage.
Only use information available up to time `t` to predict future target values.

---

## Modeling rules
Start with a simple and strong baseline.

Preferred baseline model:
- LSTM for hourly forecasting

Possible later extension:
- GRU
- simple spatio-temporal graph model
- decentralized FL methods

Do not implement unnecessary complexity early.
The first priority is a correct, reproducible baseline.

---

## Federated learning rules
Each charging station should be treated as one client.

Implement and compare:
1. centralized training
2. FedAvg
3. FedProx

FedAvg and FedProx should:
- keep raw data local to each station
- exchange only model parameters or updates
- support heterogeneous local datasets
- allow comparison of convergence and communication cost

FedProx should be implemented as a modification of the local objective, not as a separate unrelated model.

---

## Evaluation rules
Always evaluate with clear and reproducible metrics.

Main metrics:
- RMSE
- MAE
- R²

For federated experiments also track:
- number of communication rounds
- model size
- communication cost
- convergence behavior

All comparisons should be reported for:
- centralized baseline
- FedAvg
- FedProx

Where useful, compare under:
- IID split
- mildly heterogeneous split
- strongly heterogeneous split

---

## Non-IID simulation rules
The thesis must explicitly study the effect of heterogeneity.

Support at least these scenarios:
1. IID split
2. moderate Non-IID split
3. strong Non-IID split

Prefer realistic heterogeneity based on station behavior.
Dirichlet-based partitioning is acceptable when needed.

Document clearly how each split is created.

---

## Code quality rules
Write clean, modular Python code.

Preferred libraries:
- `pathlib`
- `pandas`
- `numpy`
- `scikit-learn`
- `PyTorch`

Use:
- clear function boundaries
- meaningful variable names
- docstrings for important functions
- type hints where useful
- configuration files or constants instead of hardcoded values

Do not hardcode station IDs, filenames, or temporary assumptions unless clearly justified.

---

## Project structure
Prefer this structure inside `code/`:

- `code/preprocessing`
- `code/models`
- `code/training`
- `code/federated`
- `code/evaluation`
- `code/utils`

Keep training, preprocessing, and evaluation clearly separated.

---

## Outputs and artifacts
Save outputs in clearly separated folders.

Expected output types:
- processed datasets
- train/validation/test splits
- scalers and preprocessing artifacts
- trained model checkpoints
- experiment logs
- metrics summaries
- plots and figures

Artifacts should be easy to inspect and reuse.

---

## Reproducibility
Prioritize reproducibility in all tasks.

Always:
- set random seeds where relevant
- keep preprocessing deterministic
- save configs used for experiments
- save metrics and model artifacts
- make scripts runnable from the command line

---

## What to optimize for
Prioritize:
1. correctness
2. clarity
3. reproducibility
4. modularity
5. easy extension for later experiments

Avoid:
- premature optimization
- overly complex abstractions
- fragile one-off scripts
- hidden assumptions about data

---

## Important working style
Before writing final code for any new part of the project:
1. inspect the relevant real data or existing code
2. summarize assumptions
3. implement the simplest correct version
4. keep the design extensible for later federated experiments
