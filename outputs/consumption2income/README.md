# SHS consumption vs income (log–log)

Materials for the Tier A SHS figure of log after-tax income vs log consumption
with weighted hexbin density and smoothed conditional quantiles.

## Contents

- `shs_log_income_consumption.ipynb` — analysis notebook
- `run_shs_log_income_consumption.py` — scripted reproduction of the pooled figure
- `shs_log_inc_cons_quantiles.png` / `_by_year.png` — figures
- `shs_log_inc_cons_quantiles.csv` — binned weighted quantile estimates
- `shs_log_inc_cons_columns.csv` / `.md` — survey columns used in the figures
- `code/survey/` — loader and weighted quantile helpers
- `data/local_paths.yaml` — local PUMF path config (machine-specific)

## Reproduce

1. Ensure SHS Tier A paths in `data/local_paths.yaml` (or env vars `SHS_RY2023_PATH`, `SHS_EDM_PATH`).
2. From this folder: `python run_shs_log_income_consumption.py`

Dollar amounts are CPI-deflated to 2019 CAD before logs.
