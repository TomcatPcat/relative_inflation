"""
inflation_predictor.py
======================
Predicts a household's month-over-month (MoM) personalised inflation rate
using the monthly OLS regression outputs from sfs_shs_inequality.R.

Two models are available:

  Option A  –  one coefficient set per calendar month (~84 models).
               Call OptionAPredictor.predict(df) after selecting the
               appropriate year/month rows from your input DataFrame.

  Option B  –  one pooled model with month fixed effects (FEs) and
               household-level clustered standard errors.
               Call OptionBPredictor.predict(df); the correct month FE
               is looked up automatically from the year/month columns.

Required CSV files (estimated from SHS microdata; committed under
outputs/inflation_predictor):
  monthly_infl_reg_coefs_A_v21.csv          – per-month coefficients
  monthly_infl_reg_coefs_B_structural_v21.csv – pooled structural coefficients
  monthly_infl_reg_coefs_B_month_fe_v21.csv   – month fixed effects
  monthly_infl_reg_fit_A_v21.csv            – Option A fit statistics (optional)
  monthly_infl_reg_fit_B_v21.csv            – Option B fit statistics (optional)

Input DataFrame schema
----------------------
Column              dtype   Values / notes
──────────────────────────────────────────────────────────────────────────────
year                int     Calendar year, e.g. 2022
month               int     1–12
CCspend_decilegrp   str     "D1" (lowest CC spenders) … "D10"
age_group           str     "0-29", "30-39", "40-54", "55-64", "65-74", "75+"
has_mortgage        str     "No mortgage" | "Mortgage"

Output
------
pd.Series of predicted hh_cpi_mom_pct values (percentage points MoM).
Rows whose year-month has no fitted model (Option A) return NaN.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── Reference categories (must match R factor contrasts) ──────────────────────
_REF_DECILE   = "D1"
_REF_AGE      = "0-29"
_REF_MORTGAGE = "No mortgage"

# Repo layout: code/inflation_predictor/this file, tables in outputs/inflation_predictor
DEFAULT_OUTDIR = Path(__file__).resolve().parents[2] / "outputs" / "inflation_predictor"


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ym_label(year: "pd.Series | int", month: "pd.Series | int") -> "pd.Series | str":
    """Format year + month as 'YYYY-MM' (vectorised or scalar)."""
    y = pd.Series(year, dtype=int) if not isinstance(year, pd.Series) else year.astype(int)
    m = pd.Series(month, dtype=int) if not isinstance(month, pd.Series) else month.astype(int)
    return y.astype(str).str.zfill(4) + "-" + m.astype(str).str.zfill(2)


def _build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the R-equivalent design matrix for the structural predictors
    (everything except month FEs and intercept).

    Returns a float DataFrame with one column per dummy variable.
    Columns are 0/1 floats; the intercept is handled separately.
    Reference levels produce all-zero rows for their variable group.
    """
    rows = len(df)
    out: dict[str, np.ndarray] = {}

    # CCspend_decilegrp: D2 … D10 vs reference D1
    decile_vals = df["CCspend_decilegrp"].astype(str)
    for d in [f"D{i}" for i in range(2, 11)]:
        out[f"CCspend_decilegrp{d}"] = (decile_vals == d).astype(float).values

    # age_group: 30-39, 40-54, 55-64, 65-74, 75+ vs reference 0-29
    age_vals = df["age_group"].astype(str)
    for age in ["30-39", "40-54", "55-64", "65-74", "75+"]:
        out[f"age_group{age}"] = (age_vals == age).astype(float).values

    # has_mortgage: Mortgage vs reference "No mortgage"
    mort_vals = df["has_mortgage"].astype(str)
    out["has_mortgageMortgage"] = (mort_vals == "Mortgage").astype(float).values

    return pd.DataFrame(out, index=df.index)


def _dot_coefs(design: pd.DataFrame,
               intercept: float,
               coef_map: dict[str, float]) -> np.ndarray:
    """
    Compute intercept + Σ coef_i * x_i for every row in design.
    Missing terms contribute 0 (graceful degradation).
    """
    result = np.full(len(design), intercept, dtype=float)
    for col in design.columns:
        c = coef_map.get(col, 0.0)
        if c != 0.0:
            result += c * design[col].values
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Option A — one model per calendar month
# ══════════════════════════════════════════════════════════════════════════════

class OptionAPredictor:
    """
    Applies the coefficient set fitted for the requested calendar month.

    Each month's model is:
        hh_cpi_mom_pct = β₀ + β_decile + β_age + β_mortgage + ε

    Parameters
    ----------
    coef_path : path to monthly_infl_reg_coefs_A_v21.csv
    fit_path  : path to monthly_infl_reg_fit_A_v21.csv (optional, for metadata)
    """

    def __init__(self,
                 coef_path: str | Path,
                 fit_path:  str | Path | None = None):
        raw = pd.read_csv(coef_path)

        # Build dict  "YYYY-MM" → {"term": estimate, ...}
        self._models: dict[str, dict[str, float]] = {
            ym: dict(zip(grp["term"], grp["estimate"]))
            for ym, grp in raw.groupby("ym_label")
        }

        self.fit_stats: pd.DataFrame | None = (
            pd.read_csv(fit_path) if fit_path is not None else None
        )

    # ── public API ────────────────────────────────────────────────────────────

    def available_months(self) -> list[str]:
        """Sorted list of 'YYYY-MM' strings for which a model exists."""
        return sorted(self._models)

    def fit_summary(self, ym_label: str) -> dict | None:
        """Return fit statistics for one month, or None if unavailable."""
        if self.fit_stats is None:
            return None
        row = self.fit_stats[self.fit_stats["ym_label"] == ym_label]
        return row.to_dict("records")[0] if len(row) else None

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict MoM personalised inflation for each row in *df*.

        Rows whose year-month has no fitted model return NaN.

        Parameters
        ----------
        df : DataFrame with columns year, month, CCspend_decilegrp,
             age_group, has_mortgage.

        Returns
        -------
        pd.Series  –  predicted hh_cpi_mom_pct (pp)
        """
        df = df.copy().reset_index(drop=True)
        df["_ym"] = _ym_label(df["year"], df["month"])

        design  = _build_design_matrix(df)
        result  = np.full(len(df), np.nan)

        for ym, idx in df.groupby("_ym").groups.items():
            coef = self._models.get(ym)
            if coef is None:
                continue  # month not modelled → NaN
            intercept = coef.get("(Intercept)", 0.0)
            result[idx] = _dot_coefs(design.iloc[idx], intercept, coef)

        return pd.Series(result, index=df.index, name="pred_hh_cpi_mom_pct_A")

    def predict_with_uncertainty(self,
                                 df: pd.DataFrame,
                                 se_path: str | Path | None = None) -> pd.DataFrame:
        """
        Same as predict() but also returns ±1.96 SE bounds using the
        coefficient standard errors from the same CSV.

        se_path is the coef CSV path (same as coef_path); re-pass if needed.
        Returns a DataFrame with columns: pred, se, lo95, hi95.
        """
        if se_path is None:
            raise ValueError("se_path is required for uncertainty bounds.")
        se_raw = pd.read_csv(se_path)
        se_models: dict[str, dict[str, float]] = {
            ym: dict(zip(grp["term"], grp["std_error"]))
            for ym, grp in se_raw.groupby("ym_label")
        }

        df = df.copy().reset_index(drop=True)
        df["_ym"] = _ym_label(df["year"], df["month"])
        design = _build_design_matrix(df)

        pred_vals = np.full(len(df), np.nan)
        se_vals   = np.full(len(df), np.nan)

        for ym, idx in df.groupby("_ym").groups.items():
            coef = self._models.get(ym)
            se   = se_models.get(ym)
            if coef is None:
                continue
            intercept    = coef.get("(Intercept)", 0.0)
            se_intercept = se.get("(Intercept)", 0.0) if se else 0.0

            pred_vals[idx] = _dot_coefs(design.iloc[idx], intercept, coef)

            # Approximate SE: root-sum-of-squared SEs (ignores covariance)
            se_sq = se_intercept ** 2
            for col in design.columns:
                s = se.get(col, 0.0) if se else 0.0
                se_sq += (s * design.iloc[idx][col].values) ** 2
            se_vals[idx] = np.sqrt(se_sq)

        return pd.DataFrame({
            "pred": pred_vals,
            "se":   se_vals,
            "lo95": pred_vals - 1.96 * se_vals,
            "hi95": pred_vals + 1.96 * se_vals,
        }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Option B — pooled model with month fixed effects
# ══════════════════════════════════════════════════════════════════════════════

class OptionBPredictor:
    """
    Uses the pooled panel model:

        hh_cpi_mom_pct = α_t  +  β_decile + β_age + β_mortgage + ε

    where α_t is the month fixed effect for calendar month t.

    The reference month's FE is absorbed into the intercept; its contribution
    is 0 when the reference month is requested.

    Standard errors are clustered at the household level in the R model.
    The predict_with_uncertainty method propagates the structural SEs only
    (month-FE SE is also available and can be optionally included).

    Parameters
    ----------
    structural_path : monthly_infl_reg_coefs_B_structural_v21.csv
    month_fe_path   : monthly_infl_reg_coefs_B_month_fe_v21.csv
    fit_path        : monthly_infl_reg_fit_B_v21.csv  (optional)
    """

    def __init__(self,
                 structural_path: str | Path,
                 month_fe_path:   str | Path,
                 fit_path:        str | Path | None = None):
        struct = pd.read_csv(structural_path)
        self._structural: dict[str, float] = dict(
            zip(struct["term"], struct["estimate"])
        )
        self._structural_se: dict[str, float] = dict(
            zip(struct["term"], struct["std_error"])
        )

        fe = pd.read_csv(month_fe_path)
        self._month_fe: dict[str, float] = dict(
            zip(fe["ym_label"], fe["estimate"])
        )
        self._month_fe_se: dict[str, float] = dict(
            zip(fe["ym_label"], fe["std_error"])
        )

        # Reference month: not in the FE table; its effect is baked into intercept
        self._ref_month_fe    = 0.0
        self._ref_month_fe_se = 0.0

        self.fit_stats: pd.DataFrame | None = (
            pd.read_csv(fit_path) if fit_path is not None else None
        )

    # ── public API ────────────────────────────────────────────────────────────

    def available_months(self) -> list[str]:
        """
        Months with an explicit FE coefficient (excludes the reference month,
        which is always estimable — its FE contribution is 0).
        """
        return sorted(self._month_fe)

    def month_fe(self, ym_label: str) -> tuple[float, float]:
        """Return (estimate, std_error) for the requested month FE."""
        est = self._month_fe.get(ym_label, self._ref_month_fe)
        se  = self._month_fe_se.get(ym_label, self._ref_month_fe_se)
        return est, se

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict MoM personalised inflation for each row in *df*.

        Prediction = intercept + month_FE(t) + β_decile + β_age + β_mortgage

        Unknown months (outside the estimation window) use FE = 0, which
        means the prediction reduces to the structural component + intercept.

        Parameters
        ----------
        df : DataFrame with columns year, month, CCspend_decilegrp,
             age_group, has_mortgage.

        Returns
        -------
        pd.Series  –  predicted hh_cpi_mom_pct (pp)
        """
        df = df.copy().reset_index(drop=True)
        df["_ym"] = _ym_label(df["year"], df["month"])

        design    = _build_design_matrix(df)
        intercept = self._structural.get("(Intercept)", 0.0)

        # Structural part is the same for all rows
        structural = _dot_coefs(design, intercept, self._structural)

        # Add month FE for each row
        fe_vals = df["_ym"].map(self._month_fe).fillna(self._ref_month_fe).values

        result = structural + fe_vals
        return pd.Series(result, index=df.index, name="pred_hh_cpi_mom_pct_B")

    def predict_with_uncertainty(self,
                                 df: pd.DataFrame,
                                 include_fe_se: bool = True) -> pd.DataFrame:
        """
        Same as predict() with approximate 95 % confidence bounds.

        SE is computed as:
            se_total² = se_structural²  +  (se_fe  if include_fe_se else 0)²

        where se_structural is the root-sum-of-squared SE contributions from
        structural coefficients (diagonal of var-cov, ignoring off-diagonal),
        and se_fe is the clustered SE of the month FE.

        Parameters
        ----------
        df             : input DataFrame (same schema as predict)
        include_fe_se  : add month-FE standard error to total SE (default True)

        Returns
        -------
        DataFrame with columns pred, se_structural, se_fe, se_total, lo95, hi95
        """
        df = df.copy().reset_index(drop=True)
        df["_ym"] = _ym_label(df["year"], df["month"])

        design    = _build_design_matrix(df)
        intercept = self._structural.get("(Intercept)", 0.0)
        structural = _dot_coefs(design, intercept, self._structural)
        fe_vals    = df["_ym"].map(self._month_fe).fillna(self._ref_month_fe).values
        pred_vals  = structural + fe_vals

        # Structural SE (row-wise root-sum-of-squared per-term SEs)
        se_int = self._structural_se.get("(Intercept)", 0.0)
        se_sq  = np.full(len(df), se_int ** 2)
        for col in design.columns:
            s = self._structural_se.get(col, 0.0)
            se_sq += (s * design[col].values) ** 2
        se_structural = np.sqrt(se_sq)

        # Month-FE SE
        se_fe = (
            df["_ym"].map(self._month_fe_se).fillna(self._ref_month_fe_se).values
            if include_fe_se else np.zeros(len(df))
        )

        se_total = np.sqrt(se_structural ** 2 + se_fe ** 2)

        return pd.DataFrame({
            "pred":          pred_vals,
            "se_structural": se_structural,
            "se_fe":         se_fe,
            "se_total":      se_total,
            "lo95":          pred_vals - 1.96 * se_total,
            "hi95":          pred_vals + 1.96 * se_total,
        }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience loader
# ══════════════════════════════════════════════════════════════════════════════

def load_predictors(
    outdir: str | Path = DEFAULT_OUTDIR,
    vintage: str = "v21",
) -> tuple[OptionAPredictor, OptionBPredictor]:
    """
    Load both predictors from *outdir* (the R script's output directory).

    Parameters
    ----------
    outdir  : directory containing the CSV files written by R
    vintage : model vintage label, default "v21" (2021 SHS basket)

    Returns
    -------
    (OptionAPredictor, OptionBPredictor)
    """
    d = Path(outdir)
    a = OptionAPredictor(
        coef_path = d / f"monthly_infl_reg_coefs_A_{vintage}.csv",
        fit_path  = d / f"monthly_infl_reg_fit_A_{vintage}.csv",
    )
    b = OptionBPredictor(
        structural_path = d / f"monthly_infl_reg_coefs_B_structural_{vintage}.csv",
        month_fe_path   = d / f"monthly_infl_reg_coefs_B_month_fe_{vintage}.csv",
        fit_path        = d / f"monthly_infl_reg_fit_B_{vintage}.csv",
    )
    return a, b


# ══════════════════════════════════════════════════════════════════════════════
# Example / smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    outdir  = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_OUTDIR)
    vintage = sys.argv[2] if len(sys.argv) > 2 else "v21"

    print(f"\nLoading regression results from: {outdir!r}  (vintage={vintage})\n")
    pred_a, pred_b = load_predictors(outdir, vintage)

    print(f"Option A: {len(pred_a.available_months())} monthly models  "
          f"({pred_a.available_months()[0]} … {pred_a.available_months()[-1]})")
    print(f"Option B: {len(pred_b.available_months())} month FE coefficients  "
          f"({pred_b.available_months()[0]} … {pred_b.available_months()[-1]})")

    # ── Sample prediction DataFrame ───────────────────────────────────────────
    sample = pd.DataFrame([
        # (year, month, CC decile, age group, mortgage status)
        dict(year=2022, month=6,  CCspend_decilegrp="D1",  age_group="0-29",  has_mortgage="No mortgage"),
        dict(year=2022, month=6,  CCspend_decilegrp="D5",  age_group="40-54", has_mortgage="Mortgage"),
        dict(year=2022, month=6,  CCspend_decilegrp="D10", age_group="65-74", has_mortgage="No mortgage"),
        dict(year=2023, month=3,  CCspend_decilegrp="D3",  age_group="30-39", has_mortgage="Mortgage"),
        dict(year=2021, month=12, CCspend_decilegrp="D7",  age_group="55-64", has_mortgage="No mortgage"),
        dict(year=2019, month=1,  CCspend_decilegrp="D2",  age_group="75+",   has_mortgage="No mortgage"),
    ])

    sample["pred_A"] = pred_a.predict(sample)
    sample["pred_B"] = pred_b.predict(sample)

    print("\n── Point predictions (pp MoM) ──────────────────────────────────────────")
    print(sample.to_string(index=False))

    # ── With uncertainty bounds ───────────────────────────────────────────────
    coef_a_path = Path(outdir) / f"monthly_infl_reg_coefs_A_{vintage}.csv"
    unc_a = pred_a.predict_with_uncertainty(sample, se_path=coef_a_path)
    unc_b = pred_b.predict_with_uncertainty(sample, include_fe_se=True)

    print("\n── Option A: predictions ± 95% CI (approx, diagonal SE only) ──────────")
    print(unc_a.round(4).to_string())

    print("\n── Option B: predictions ± 95% CI (clustered structural + FE SE) ──────")
    print(unc_b.round(4).to_string())

    # ── Sweep across all deciles for June 2022 (Option B) ────────────────────
    sweep = pd.DataFrame([
        dict(year=2022, month=6, CCspend_decilegrp=f"D{d}",
             age_group="40-54", has_mortgage="No mortgage")
        for d in range(1, 11)
    ])
    sweep["pred_A"] = pred_a.predict(sweep)
    sweep["pred_B"] = pred_b.predict(sweep)

    print("\n── June 2022: predicted MoM inflation by CC-spend decile ───────────────")
    print(sweep[["CCspend_decilegrp", "pred_A", "pred_B"]].to_string(index=False))
