"""
SHS 2023: log consumption vs log household income
=================================================
Uses Statistics Canada Survey of Household Spending (SHS) 2023 PUMF diary file:
  - HHTOTINC  : household total income ($)
  - TC001     : total current consumption ($)
  - CCconsumption : TC001 * 0.6
  - WEIGHTD   : diary survey weight

Analysis:
  1. Scatter of log income vs log consumption (consumption on y-axis)
  2. Weighted OLS: log(consumption) ~ log(income)
  3. Weighted local (kernel) regression — MPC allowed to vary by income level
  4. Upper-tail Pareto / power-law fit on consumption; Lorenz coefficient and curve
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# ── Paths (SHS RY2023 documentation + fixed-width PUMF) ─────────────────────
SHS_ROOT = Path(r"C:\Users\thoma\Dropbox\SHS")
RY2023_DIR = SHS_ROOT / "RY2023"
DATA_FILE = RY2023_DIR / "Data" / "TXT" / "PUMF_SHS_2023.txt"
OUT_DIR = SHS_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Column positions from pumf_SHS_2023.dct / SAS layout (1-indexed, inclusive)
COLSPECS = [
    ("caseid", 1, 6, str),
    ("weight", 7, 24, float),
    ("hhtotinc", 66, 76, float),
    ("tc001", 3664, 3674, float),
]
MISSING_NUMERIC = 99_999_999.96


def load_shs_pumf(path: Path) -> pd.DataFrame:
    """Read selected columns from the SHS 2023 fixed-width diary PUMF."""
    usecols = [c[0] for c in COLSPECS]
    colspecs = [(c[1] - 1, c[2]) for c in COLSPECS]
    dtypes = {c[0]: c[3] for c in COLSPECS}

    df = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=usecols,
        dtype=dtypes,
        encoding="latin-1",
    )

    for col in ("weight", "hhtotinc", "tc001"):
        df.loc[df[col] >= MISSING_NUMERIC, col] = np.nan

    df = df.dropna(subset=["weight", "hhtotinc", "tc001"])
    df = df[(df["weight"] > 0) & (df["hhtotinc"] > 0) & (df["tc001"] > 0)]
    df["CCconsumption"] = df["tc001"] * 0.6
    df["log_income"] = np.log(df["hhtotinc"])
    df["log_consumption"] = np.log(df["CCconsumption"])
    return df.reset_index(drop=True)


def weighted_ols(y: np.ndarray, x: np.ndarray, w: np.ndarray):
    """Weighted least squares with intercept."""
    X = sm.add_constant(pd.Series(x, name="log_income"))
    model = sm.WLS(y, X, weights=w)
    return model.fit()


def weighted_local_linear(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    """
    Nadaraya–Watson-style weighted local linear smoother.
    Bandwidth is the Gaussian kernel standard deviation in log-income units.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()

    if bandwidth is None:
        # Rule-of-thumb: ~5% of weighted IQR in x
        order = np.argsort(x)
        x_s, w_s = x[order], w[order]
        cum_w = np.cumsum(w_s)
        q25 = x_s[np.searchsorted(cum_w, 0.25)]
        q75 = x_s[np.searchsorted(cum_w, 0.75)]
        bandwidth = max(0.15, 0.05 * (q75 - q25))

    y_hat = np.empty_like(x_grid, dtype=float)
    for i, x0 in enumerate(x_grid):
        dist = (x - x0) / bandwidth
        kw = w * np.exp(-0.5 * dist**2)
        if kw.sum() <= 0:
            y_hat[i] = np.nan
            continue
        # Local weighted linear regression at x0
        W = np.diag(kw)
        Xloc = np.column_stack([np.ones_like(x), x - x0])
        beta = np.linalg.lstsq(Xloc.T @ W @ Xloc, Xloc.T @ W @ y, rcond=None)[0]
        y_hat[i] = beta[0]
    return y_hat


def weighted_lorenz_curve(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return population share p and Lorenz ordinate L(p) for weighted data."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    w = w / w.sum()
    cum_w = np.cumsum(w)
    cum_xw = np.cumsum(v * w)
    mean_x = cum_xw[-1]
    lorenz = cum_xw / mean_x
    return cum_w, lorenz


def paretocoeff(top_frac: float, top_share: float) -> float:
    """
    Pareto tail exponent from a top-group size and share.
    Same formula as useful_funcs2.R::paretocoeff (top_frac e.g. 0.10 for top 10%).
    """
    if not (0 < top_frac < 1) or not (0 < top_share < 1):
        return np.nan
    return np.log(top_frac) / (np.log(top_frac) - np.log(top_share))


def weighted_top_share(values: np.ndarray, weights: np.ndarray, top_frac: float) -> float:
    """Share of total held by richest units comprising top_frac of weight."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-v)
    v, w = v[order], w[order]
    total = float(np.dot(w, v))
    if total <= 0:
        return np.nan
    target = top_frac * float(w.sum())
    cum = np.cumsum(w)
    j = int(np.searchsorted(cum, target, side="left"))
    j = min(max(j, 0), len(w) - 1)
    return float(np.dot(w[: j + 1], v[: j + 1])) / total


def estimate_pareto_alpha(
    values: np.ndarray, weights: np.ndarray, top_fracs: tuple[float, ...] = (0.10, 0.05, 0.01)
) -> float:
    """Average Pareto exponent across several upper-tail quantile points."""
    alphas = []
    for q in top_fracs:
        share = weighted_top_share(values, weights, q)
        a = paretocoeff(q, share)
        if np.isfinite(a) and a > 1:
            alphas.append(a)
    return float(np.mean(alphas)) if alphas else np.nan


def pareto_lorenz(p: np.ndarray, alpha: float) -> np.ndarray:
    """Lorenz curve ordinates for a Pareto(alpha) distribution (alpha > 1)."""
    return 1.0 - (1.0 - p) ** (1.0 - 1.0 / alpha)


def lorenz_coefficient(alpha: float) -> float:
    """Pareto Lorenz coefficient eta = alpha / (alpha - 1); mean/median ratio."""
    if alpha <= 1:
        return np.nan
    return alpha / (alpha - 1.0)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probs: np.ndarray | float
) -> np.ndarray | float:
    """Weighted quantiles of `values` using normalized weights."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    probs_arr = np.atleast_1d(probs).astype(float)

    order = np.argsort(v)
    v, w = v[order], w[order]
    w = w / w.sum()
    cum_w = np.cumsum(w)

    out = np.empty_like(probs_arr, dtype=float)
    for i, p in enumerate(probs_arr):
        j = int(np.searchsorted(cum_w, p, side="left"))
        j = min(max(j, 0), len(v) - 1)
        out[i] = v[j]
    return out[0] if np.ndim(probs) == 0 else out


def calibrate_pareto_scale(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """
    Scale parameter x_min for a Pareto(alpha) distribution whose mean matches
    the weighted mean of `values`.  For Pareto, E[X] = alpha * x_min / (alpha - 1).
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    mean_x = float(np.dot(w, v) / w.sum())
    return mean_x * (alpha - 1.0) / alpha


def pareto_quantile(p: float | np.ndarray, alpha: float, x_min: float) -> float | np.ndarray:
    """
    Quantile function for Pareto(alpha) with scale x_min.
    p is the population percentile from the bottom (0–1).
    """
    p_arr = np.asarray(p, dtype=float)
    return x_min * (1.0 - p_arr) ** (-1.0 / alpha)


def pareto_quantile_table(
    alpha: float,
    x_min: float,
    percentiles: tuple[float, ...] = (90, 95, 96, 97, 98, 99, 99.5, 99.9),
    values: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Consumption quantiles from mean-calibrated Pareto(alpha), with empirical comparison."""
    probs = np.array(percentiles, dtype=float) / 100.0
    fitted = pareto_quantile(probs, alpha, x_min)

    rows = {
        "percentile": percentiles,
        "population_share_p": probs,
        "pareto_consumption": fitted,
    }
    if values is not None and weights is not None:
        rows["empirical_consumption"] = weighted_quantile(values, weights, probs)

    return pd.DataFrame(rows)


def pareto_conditional_mean(
    p_low: float, p_high: float, alpha: float, x_min: float
) -> float:
    """Mean consumption within population percentile interval (p_low, p_high]."""
    if alpha <= 1:
        return np.nan
    mass = p_high - p_low
    if mass <= 0:
        return np.nan
    term = (1.0 - p_low) ** ((alpha - 1.0) / alpha) - (1.0 - p_high) ** (
        (alpha - 1.0) / alpha
    )
    return (alpha / (alpha - 1.0)) * x_min * term / mass


def weighted_percentile_group_mean(
    values: np.ndarray, weights: np.ndarray, p_low: float, p_high: float
) -> float:
    """Weighted mean of values for households in population percentile (p_low, p_high]."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    w = w / w.sum()
    cum_w = np.cumsum(w)
    cum_w_lag = np.concatenate(([0.0], cum_w[:-1]))
    mask = (cum_w > p_low) & (cum_w_lag < p_high)
    if not mask.any():
        return np.nan
    group_w = w[mask]
    return float(np.dot(group_w, v[mask]) / group_w.sum())


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"SHS PUMF not found: {DATA_FILE}")

    print(f"Loading {DATA_FILE} …")
    df = load_shs_pumf(DATA_FILE)
    print(f"  {len(df):,} households with positive income, consumption, and weight")

    x = df["log_income"].to_numpy()
    y = df["log_consumption"].to_numpy()
    w = df["weight"].to_numpy()
    cons = df["CCconsumption"].to_numpy()

    # ── 1. Weighted OLS ───────────────────────────────────────────────────────
    ols = weighted_ols(y, x, w)
    print("\nWeighted OLS: log(CCconsumption) ~ log(income)")
    print(ols.summary().tables[1])
    print(f"  R-squared: {ols.rsquared:.4f}")

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = ols.params["const"] + ols.params["log_income"] * x_line

    fig, ax = plt.subplots(figsize=(8, 6))
    # Subsample for visibility (preserve weighted representativeness via random draw)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(2500, len(df)), replace=False, p=w / w.sum())
    ax.scatter(
        x[idx],
        y[idx],
        s=8,
        alpha=0.25,
        c="steelblue",
        edgecolors="none",
        label="Households (subsample)",
    )
    ax.plot(x_line, y_line, color="crimson", lw=2.5, label="Weighted OLS fit")
    ax.set_xlabel("Log household income (HHTOTINC)")
    ax.set_ylabel("Log CCconsumption (TC001 x 0.6)")
    ax.set_title("SHS 2023: Log CCconsumption vs log income")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_log_consumption_vs_log_income_ols.png", dpi=150)
    plt.close(fig)

    # ── 2. Flexible local relationship ───────────────────────────────────────
    x_grid = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), 150)
    y_local = weighted_local_linear(x, y, w, x_grid)

    # Unweighted LOWESS on a weighted resample for comparison overlay
    n_resample = min(8000, len(df))
    idx_r = rng.choice(len(df), size=n_resample, replace=True, p=w / w.sum())
    lowess_fit = lowess(y[idx_r], x[idx_r], frac=0.25, return_sorted=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[idx], y[idx], s=8, alpha=0.2, c="steelblue", edgecolors="none")
    ax.plot(x_grid, y_local, color="darkorange", lw=2.5, label="Weighted local linear")
    ax.plot(
        lowess_fit[:, 0],
        lowess_fit[:, 1],
        color="seagreen",
        lw=2,
        ls="--",
        label="LOWESS (weighted resample)",
    )
    ax.plot(x_line, y_line, color="crimson", lw=1.5, ls=":", label="Global WLS")
    ax.set_xlabel("Log household income")
    ax.set_ylabel("Log CCconsumption")
    ax.set_title("SHS 2023: Local vs global income-CCconsumption relationship")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_local_income_consumption_fit.png", dpi=150)
    plt.close(fig)

    # ── 3. Upper-tail Pareto / Lorenz ─────────────────────────────────────────
    p_emp, l_emp = weighted_lorenz_curve(cons, w)
    alpha = estimate_pareto_alpha(cons, w)
    eta = lorenz_coefficient(alpha)

    share10 = weighted_top_share(cons, w, 0.10)
    share5 = weighted_top_share(cons, w, 0.05)
    share1 = weighted_top_share(cons, w, 0.01)

    print("\nUpper-tail CCconsumption (weighted):")
    print(f"  Top 10% share : {share10:.3f}")
    print(f"  Top  5% share : {share5:.3f}")
    print(f"  Top  1% share : {share1:.3f}")
    print(f"  Pareto alpha  : {alpha:.3f}")
    print(f"  Lorenz coeff (eta = alpha/(alpha-1)): {eta:.3f}")

    p_fine = np.linspace(0.001, 0.999, 500)
    l_pareto = pareto_lorenz(p_fine, alpha)

    # Upper-tail panel: p >= 0.70
    tail_mask = p_emp >= 0.70

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(p_emp, l_emp, color="steelblue", lw=2, label="Empirical Lorenz")
    ax.plot(p_fine, l_pareto, color="crimson", lw=2, ls="--", label=f"Pareto fit (α={alpha:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax.set_xlabel("Population share (poorest → richest)")
    ax.set_ylabel("Consumption share")
    ax.set_title("Lorenz curve: CCconsumption")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(p_emp[tail_mask], l_emp[tail_mask], "o-", color="steelblue", ms=3, label="Empirical (upper tail)")
    p_tail = p_fine[p_fine >= 0.70]
    ax.plot(p_tail, pareto_lorenz(p_tail, alpha), color="crimson", lw=2, label=f"Power-law Lorenz (η={eta:.2f})")
    ax.set_xlabel("Population share (upper tail, p ≥ 0.70)")
    ax.set_ylabel("Cumulative consumption share L(p)")
    ax.set_title("Upper-tail Lorenz curve from Pareto / power-law fit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("SHS 2023 weighted CCconsumption inequality", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_lorenz_upper_tail_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save summary table
    summary = pd.DataFrame(
        {
            "statistic": [
                "n_households",
                "ols_intercept",
                "ols_slope_log_income",
                "ols_r_squared",
                "top10_share",
                "top5_share",
                "top1_share",
                "pareto_alpha",
                "lorenz_coefficient_eta",
            ],
            "value": [
                len(df),
                ols.params["const"],
                ols.params["log_income"],
                ols.rsquared,
                share10,
                share5,
                share1,
                alpha,
                eta,
            ],
        }
    )
    summary.to_csv(OUT_DIR / "shs_income_consumption_summary.csv", index=False)

    # ── 4. Pareto-implied consumption quantiles ───────────────────────────────
    x_min = calibrate_pareto_scale(cons, w, alpha)
    pct_levels = (90, 95, 96, 97, 98, 99, 99.5, 99.9)
    quantile_tbl = pareto_quantile_table(alpha, x_min, pct_levels, cons, w)
    quantile_tbl["pareto_consumption_monthly"] = quantile_tbl["pareto_consumption"] / 12.0
    quantile_tbl["empirical_consumption_monthly"] = (
        quantile_tbl["empirical_consumption"] / 12.0
    )
    quantile_tbl.to_csv(OUT_DIR / "shs_pareto_consumption_quantiles.csv", index=False)

    print(f"\nPareto scale x_min (mean-calibrated): ${x_min:,.0f}")
    print("CCconsumption quantiles from fitted Pareto(alpha) vs empirical (weighted):")
    print(
        quantile_tbl.to_string(
            index=False,
            formatters={
                "pareto_consumption": lambda x: f"${x:,.0f}",
                "empirical_consumption": lambda x: f"${x:,.0f}",
                "pareto_consumption_monthly": lambda x: f"${x:,.0f}",
                "empirical_consumption_monthly": lambda x: f"${x:,.0f}",
            },
        )
    )

    # Bar chart: monthly consumption quantiles
    pct_labels = [f"{p:g}" for p in pct_levels]
    x_pos = np.arange(len(pct_labels))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x_pos - bar_w / 2,
        quantile_tbl["pareto_consumption_monthly"],
        bar_w,
        label="Pareto fitted",
        color="crimson",
    )
    ax.bar(
        x_pos + bar_w / 2,
        quantile_tbl["empirical_consumption_monthly"],
        bar_w,
        label="Empirical (weighted)",
        color="steelblue",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pct_labels)
    ax.set_xlabel("Population percentile")
    ax.set_ylabel("Monthly consumption ($)")
    ax.set_title("SHS 2023: Monthly CCconsumption quantiles (annual / 12)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_monthly_consumption_quantiles_bar.png", dpi=150)
    plt.close(fig)

    # ── 5. Average monthly consumption by percentile group ────────────────────
    group_defs = (
        ("90-95", 0.90, 0.95),
        ("95-99", 0.95, 0.99),
        ("99-100", 0.99, 1.00),
    )
    group_rows = []
    for label, p_lo, p_hi in group_defs:
        annual_pareto = pareto_conditional_mean(p_lo, p_hi, alpha, x_min)
        annual_emp = weighted_percentile_group_mean(cons, w, p_lo, p_hi)
        group_rows.append(
            {
                "group": label,
                "p_low": p_lo,
                "p_high": p_hi,
                "pareto_mean_annual": annual_pareto,
                "pareto_mean_monthly": annual_pareto / 12.0,
                "empirical_mean_annual": annual_emp,
                "empirical_mean_monthly": annual_emp / 12.0,
            }
        )
    group_tbl = pd.DataFrame(group_rows)
    group_tbl.to_csv(OUT_DIR / "shs_pareto_consumption_group_means.csv", index=False)

    print("\nAverage CCconsumption by percentile group (annual and monthly):")
    print(
        group_tbl.to_string(
            index=False,
            formatters={
                "pareto_mean_annual": lambda x: f"${x:,.0f}",
                "pareto_mean_monthly": lambda x: f"${x:,.0f}",
                "empirical_mean_annual": lambda x: f"${x:,.0f}",
                "empirical_mean_monthly": lambda x: f"${x:,.0f}",
            },
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(group_tbl))
    bar_w = 0.35
    ax.bar(
        x_pos - bar_w / 2,
        group_tbl["pareto_mean_monthly"],
        bar_w,
        label="Pareto fitted",
        color="crimson",
    )
    ax.bar(
        x_pos + bar_w / 2,
        group_tbl["empirical_mean_monthly"],
        bar_w,
        label="Empirical (weighted)",
        color="steelblue",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_tbl["group"])
    ax.set_xlabel("Population percentile group")
    ax.set_ylabel("Average monthly consumption ($)")
    ax.set_title("SHS 2023: Average monthly CCconsumption by percentile group")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_monthly_consumption_group_means_bar.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures and summary written to {OUT_DIR}")


if __name__ == "__main__":
    main()
