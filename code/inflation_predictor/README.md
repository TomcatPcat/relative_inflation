# Personalised inflation predictor

Scores a household's month-over-month personalised inflation from the 2021-vintage monthly OLS models. The coefficient tables and figures live in `outputs/inflation_predictor/`.

Household inputs are calendar year and month, credit-card spend decile (`D1`–`D10`), age group, and mortgage status. The result is predicted `hh_cpi_mom_pct` in percentage points.

Two models are included:

- **Option A** — a separate weighted OLS for each calendar month.
- **Option B** — one pooled model with month fixed effects and household-clustered standard errors.

```python
from pathlib import Path
import sys

sys.path.insert(0, "code/inflation_predictor")
from inflation_predictor import load_predictors

pred_a, pred_b = load_predictors()  # reads outputs/inflation_predictor
```

`inflation_predictor.ipynb` walks through the basket, the two specifications, and example predictions. Open it from this repository so it can find the output folder.

The tables were estimated from Survey of Household Spending microdata. That estimation script is not in the repository because it depends on local survey files. Re-estimation replaces the CSVs in `outputs/inflation_predictor/`; the figures next to them are the Option A R² and decile-coefficient paths, and the Option B month fixed effects and structural coefficients.
