"""Reproduce SHS log income vs log consumption figures.

Expects code/survey/ beside this script (copied helpers), plus data/local_paths.yaml
or environment variables SHS_RY2023_PATH / SHS_EDM_PATH pointing at PUMF roots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))

from survey import load_shs_tier_a, wquantile  # noqa: E402

out = HERE
PRICE_BASE_YEAR = 2019
TITLE = "Log consumption relative to log income"

shs = load_shs_tier_a(price_base_year=PRICE_BASE_YEAR)
work = shs.copy()
y = pd.to_numeric(work["inc_net"], errors="coerce")
c = pd.to_numeric(work["consumption"], errors="coerce")
w = pd.to_numeric(work["weight"], errors="coerce")
ok = (y > 0) & (c > 0) & np.isfinite(y) & np.isfinite(c) & np.isfinite(w) & (w > 0)
work = work.loc[ok].copy()
work["log_inc"] = np.log(y.loc[ok].to_numpy(dtype=float))
work["log_cons"] = np.log(c.loc[ok].to_numpy(dtype=float))
work["weight"] = w.loc[ok].to_numpy(dtype=float)

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
Q_STYLES = {
    0.10: dict(color="#4C72B0", ls=":", lw=1.6, label="p10"),
    0.25: dict(color="#55A868", ls="--", lw=1.7, label="p25"),
    0.50: dict(color="#C44E52", ls="-", lw=2.4, label="p50"),
    0.75: dict(color="#55A868", ls="--", lw=1.7, label="p75"),
    0.90: dict(color="#4C72B0", ls=":", lw=1.6, label="p90"),
}


def conditional_log_quantiles(df, *, n_bins=25, min_bin_n=40):
    x = df["log_inc"].to_numpy(dtype=float)
    yv = df["log_cons"].to_numpy(dtype=float)
    wv = df["weight"].to_numpy(dtype=float)
    order = np.argsort(x, kind="mergesort")
    x_s, y_s, w_s = x[order], yv[order], wv[order]
    edges = np.linspace(0, len(x_s), n_bins + 1, dtype=int)
    rows = []
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        xb, yb, wb = x_s[lo:hi], y_s[lo:hi], w_s[lo:hi]
        if len(xb) < min_bin_n:
            continue
        x_mid = float(np.dot(xb, wb) / wb.sum())
        qvals = np.atleast_1d(wquantile(yb, wb, list(QUANTILES)))
        row = {"bin": b, "n": len(xb), "w_sum": float(wb.sum()), "x_mid": x_mid}
        for p, qv in zip(QUANTILES, qvals):
            row[f"q{int(round(100 * p))}"] = float(qv)
        rows.append(row)
    return pd.DataFrame(rows)


def smooth_curve(x, y, *, n_grid=200, k=3):
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    okm = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[okm], ys[okm]
    uniq_x, inv = np.unique(xs, return_inverse=True)
    uniq_y = np.zeros_like(uniq_x, dtype=float)
    counts = np.zeros_like(uniq_x, dtype=float)
    for i, yi in enumerate(ys):
        uniq_y[inv[i]] += yi
        counts[inv[i]] += 1.0
    uniq_y /= counts
    if len(uniq_x) < 4:
        return uniq_x, uniq_y
    kk = min(k, len(uniq_x) - 1)
    spline = make_interp_spline(uniq_x, uniq_y, k=kk)
    xg = np.linspace(uniq_x.min(), uniq_x.max(), n_grid)
    return xg, spline(xg)


def populated_limits(df, q_lo=0.025, q_hi=0.975, pad=0.08):
    x = df["log_inc"].to_numpy(dtype=float)
    yv = df["log_cons"].to_numpy(dtype=float)
    wv = df["weight"].to_numpy(dtype=float)
    x_lo, x_hi = [float(v) for v in np.atleast_1d(wquantile(x, wv, [q_lo, q_hi]))]
    y_lo, y_hi = [float(v) for v in np.atleast_1d(wquantile(yv, wv, [q_lo, q_hi]))]
    return x_lo - pad, x_hi + pad, y_lo - pad, y_hi + pad


if __name__ == "__main__":
    q_pooled = conditional_log_quantiles(work)
    parts = [q_pooled.assign(sample="pooled")]
    for yr, g in work.groupby("year", sort=True):
        parts.append(
            conditional_log_quantiles(g, n_bins=20, min_bin_n=25).assign(sample=str(int(yr)))
        )
    q_all = pd.concat(parts, ignore_index=True)
    q_all.to_csv(out / "shs_log_inc_cons_quantiles.csv", index=False)

    x0, x1, y0, y1 = populated_limits(work)
    lo, hi = min(x0, y0), max(x1, y1)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    hb = ax.hexbin(
        work["log_inc"],
        work["log_cons"],
        C=work["weight"].to_numpy(dtype=float),
        reduce_C_function=np.sum,
        gridsize=45,
        cmap="Greys",
        mincnt=1,
        linewidths=0.0,
        alpha=0.85,
    )
    ax.plot([lo, hi], [lo, hi], color="0.65", lw=1.0, label="C = Y")
    x_mid = q_pooled["x_mid"].to_numpy(dtype=float)
    for p in QUANTILES:
        col = f"q{int(round(100 * p))}"
        m = (x_mid >= lo) & (x_mid <= hi)
        xs, ys = smooth_curve(x_mid[m], q_pooled[col].to_numpy(dtype=float)[m])
        ax.plot(xs, ys, **Q_STYLES[p])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("log after-tax income")
    ax.set_ylabel("log consumption")
    ax.set_title(TITLE)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("sum of survey weights")
    fig.tight_layout()
    fig.savefig(out / "shs_log_inc_cons_quantiles.png", dpi=140, bbox_inches="tight")
    print("wrote", out / "shs_log_inc_cons_quantiles.png")
