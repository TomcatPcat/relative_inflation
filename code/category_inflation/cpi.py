"""Download and annualize Statistics Canada category CPI (table 18-10-0004)."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .baskets import CPI_PRODUCTS

CPI_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# 2015 is retained so the 2016 year-over-year rate has a prior annual index.
FIRST_YEAR = 2015


def shs_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_dir() -> Path:
    path = shs_root() / "outputs" / "category_inflation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    path = output_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def zip_path() -> Path:
    return raw_dir() / "18100004-eng.zip"


def download_cpi_zip(path: Path | None = None, *, force: bool = False) -> Path:
    """Save the StatCan zip. A browser user agent is required; the bare client is rejected."""
    path = Path(path) if path is not None else zip_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1000 and not force:
        return path
    req = urllib.request.Request(CPI_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp, path.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return path


def load_cpi_monthly(path: Path | None = None) -> pd.DataFrame:
    """Canada, 2002=100 index levels, for the mapped product groups."""
    path = download_cpi_zip(path)
    wanted = set(CPI_PRODUCTS.values())
    label_to_key = {label: key for key, label in CPI_PRODUCTS.items()}
    with zipfile.ZipFile(path) as zf:
        with zf.open("18100004.csv") as fh:
            raw = pd.read_csv(
                fh,
                usecols=["REF_DATE", "GEO", "Products and product groups", "UOM", "VALUE"],
            )
    keep = (
        (raw["GEO"] == "Canada")
        & (raw["UOM"] == "2002=100")
        & (raw["Products and product groups"].isin(wanted))
    )
    monthly = raw.loc[keep, ["REF_DATE", "Products and product groups", "VALUE"]].copy()
    monthly["date"] = pd.to_datetime(monthly["REF_DATE"] + "-01")
    monthly["category"] = monthly["Products and product groups"].map(label_to_key)
    monthly["value"] = pd.to_numeric(monthly["VALUE"], errors="coerce")
    monthly = monthly.dropna(subset=["value", "category"]).sort_values(["category", "date"])
    missing = sorted(set(CPI_PRODUCTS) - set(monthly["category"].unique()))
    if missing:
        raise KeyError(f"CPI extract is missing products: {missing}")
    return monthly[["date", "category", "value"]].reset_index(drop=True)


def annualize_cpi(monthly: pd.DataFrame, *, first_year: int = FIRST_YEAR) -> pd.DataFrame:
    """Calendar-year mean of the monthly index, plus the year-over-year rate of that mean."""
    frame = monthly.copy()
    frame["year"] = frame["date"].dt.year
    frame = frame[frame["year"] >= int(first_year)]
    grouped = frame.groupby(["category", "year"], as_index=False)
    annual = grouped.agg(index=("value", "mean"), n_months=("value", "size"))
    annual = annual.sort_values(["category", "year"])
    annual["yoy"] = annual.groupby("category")["index"].pct_change()
    annual["product"] = annual["category"].map(CPI_PRODUCTS)
    annual["partial_year"] = annual["n_months"] < 12
    cols = ["year", "category", "product", "index", "n_months", "partial_year", "yoy"]
    return annual[cols].reset_index(drop=True)


def cpi_wide(annual: pd.DataFrame) -> pd.DataFrame:
    """Year x category index levels."""
    wide = annual.pivot(index="year", columns="category", values="index").sort_index()
    wide.index = wide.index.astype(int)
    missing = [c for c in CPI_PRODUCTS if c not in wide.columns]
    if missing:
        raise KeyError(f"Annual CPI is missing {missing}")
    if wide.isna().any().any():
        bad = wide.columns[wide.isna().any()].tolist()
        raise ValueError(f"Annual CPI has gaps in {bad}")
    return wide


def cpi_monthly_wide(monthly: pd.DataFrame, *, first_year: int = FIRST_YEAR) -> pd.DataFrame:
    """Month x category index levels, from January of ``first_year``."""
    frame = monthly.loc[monthly["date"].dt.year >= int(first_year)].copy()
    wide = frame.pivot(index="date", columns="category", values="value").sort_index()
    wide.index = pd.DatetimeIndex(wide.index).to_period("M").to_timestamp()
    wide.index.name = "date"
    wide.columns.name = "category"
    missing = [c for c in CPI_PRODUCTS if c not in wide.columns]
    if missing:
        raise KeyError(f"Monthly CPI is missing {missing}")
    expected = pd.period_range(wide.index.min(), wide.index.max(), freq="M").to_timestamp()
    if not wide.index.equals(expected):
        raise ValueError("Monthly CPI has a gap")
    if wide.isna().any().any():
        bad = wide.columns[wide.isna().any()].tolist()
        raise ValueError(f"Monthly CPI has gaps in {bad}")
    return wide


def build_cpi_annual(path: Path | None = None) -> pd.DataFrame:
    monthly = load_cpi_monthly(path)
    annual = annualize_cpi(monthly)
    dest = output_dir() / "cpi_category_annual.csv"
    annual.to_csv(dest, index=False)
    return annual


def build_cpi(path: Path | None = None) -> pd.DataFrame:
    """Write the monthly category index and the annual-mean check, and return the monthly panel."""
    monthly = load_cpi_monthly(path)
    wide = cpi_monthly_wide(monthly)
    long = wide.stack().rename("index").reset_index()
    long["date"] = long["date"].dt.strftime("%Y-%m")
    long["product"] = long["category"].map(CPI_PRODUCTS)
    long[["date", "category", "product", "index"]].to_csv(
        output_dir() / "cpi_category_monthly.csv", index=False
    )
    annualize_cpi(monthly).to_csv(output_dir() / "cpi_category_annual.csv", index=False)
    return wide
