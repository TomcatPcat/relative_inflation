"""CPI deflation to a price base year and income numeraire for model units."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .weighted import wmean

PathLike = Union[str, Path]

DEFAULT_PRICE_BASE_YEAR = 2019
DEFAULT_NUMERAIRE_YEAR = 2019

# Macro panel written by scripts/pull_macro_returns.py
_DEFAULT_MACRO_MONTHLY = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "macro_returns"
    / "macro_returns_monthly.csv"
)

# SFS dollar columns (components + identity + incomes)
SFS_DOLLAR_COLS: tuple[str, ...] = (
    "w_bank_acc",
    "w_bonds",
    "w_mut_funds",
    "w_oth_inv",
    "w_oth_nonfin",
    "w_hous_princ_res_val",
    "w_hous_oth_real_est",
    "w_stocks",
    "w_vehicles",
    "w_tfsa",
    "w_rrsp",
    "w_rrif",
    "w_oth_ret_funds",
    "w_bus_val",
    "w_pens_employer_val_gc",
    "w_pens_employer_val_tm",
    "w_pens_employer",
    "w_tot_incl_pens_gc",
    "w_tot_incl_pens_tm",
    "w_networth_gc",
    "w_networth_tm",
    "debt_mort_princ_res_val",
    "debt_mort_oth_real_est_val",
    "debt_stud_loans_val",
    "debt_cc_installment_val",
    "debt_loc_val",
    "debt_oth_val",
    "debt_vehicles_val",
    "debt_tot_val",
    "housing_assets_gross",
    "housing_debt",
    "housing_net",
    "financial_assets_gross",
    "financial_debt",
    "financial_net",
    "nw",
    "auto_assets",
    "auto_debt",
    "auto_net",
    "nw_with_auto",
    "inc_gross",
    "inc_net",
    "inc_transfers",
)

SHS_DOLLAR_COLS: tuple[str, ...] = (
    "hhtotinc",
    "consumption",
    "taxes",
    "income_at",
    "inc_net",
    "rent",
    "mortgage_payment",
)


def load_cpi_monthly(
    path: Optional[PathLike] = None,
    *,
    allow_download: bool = True,
) -> pd.Series:
    """
    Canada all-items CPI, monthly.

    Prefer ``output/macro_returns/macro_returns_monthly.csv`` (same series as
    notebook 04 / ``pull_macro_returns.load_cpi``). Optionally download StatCan
    table 18-10-0004 if the cache is missing.
    """
    path = Path(path) if path is not None else _DEFAULT_MACRO_MONTHLY
    if path.is_file():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "cpi" not in df.columns:
            raise KeyError(f"No 'cpi' column in {path}")
        s = df["cpi"].astype(float).dropna().sort_index()
        s.name = "cpi"
        return s

    if not allow_download:
        raise FileNotFoundError(
            f"CPI cache not found at {path}; run scripts/pull_macro_returns.py "
            "or pass allow_download=True"
        )

    # Lazy import to avoid pulling urllib machinery unless needed
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.pull_macro_returns import load_cpi

    s = load_cpi()
    s.name = "cpi"
    return s


def annual_cpi(
    monthly: Optional[pd.Series] = None,
    *,
    path: Optional[PathLike] = None,
    allow_download: bool = True,
) -> pd.Series:
    """Calendar-year mean of monthly CPI (index)."""
    if monthly is None:
        monthly = load_cpi_monthly(path, allow_download=allow_download)
    ann = monthly.groupby(monthly.index.year).mean()
    ann.index = ann.index.astype(int)
    ann.index.name = "year"
    ann.name = "cpi"
    return ann.sort_index()


def cpi_deflator_to_base(
    years: Union[pd.Series, Sequence[int], np.ndarray],
    *,
    base_year: int = DEFAULT_PRICE_BASE_YEAR,
    annual: Optional[pd.Series] = None,
    path: Optional[PathLike] = None,
) -> pd.Series:
    """
    Factor ``CPI_base / CPI_y`` so that ``nominal * factor`` is in ``base_year`` dollars.

    Missing CPI years raise for the base; unmatched survey years get NaN factors.
    """
    if annual is None:
        annual = annual_cpi(path=path)
    if int(base_year) not in annual.index:
        raise KeyError(f"CPI missing base year {base_year}")
    base = float(annual.loc[int(base_year)])
    if isinstance(years, pd.Series):
        y = pd.to_numeric(years, errors="coerce")
        idx = years.index
    else:
        y = pd.Series(pd.to_numeric(pd.Series(list(years)), errors="coerce"))
        idx = y.index
    cpi_map = {int(k): float(v) for k, v in annual.items()}
    cpi_y = y.map(lambda yy: cpi_map.get(int(yy), np.nan) if pd.notna(yy) else np.nan)
    return pd.Series(base / cpi_y.to_numpy(dtype=float), index=idx, name="cpi_factor")


def deflate_dollar_columns(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    year_col: str = "year",
    base_year: int = DEFAULT_PRICE_BASE_YEAR,
    annual_cpi_series: Optional[pd.Series] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Multiply listed dollar columns by the year-specific CPI factor to ``base_year``."""
    out = df if inplace else df.copy()
    if year_col not in out.columns:
        raise KeyError(f"deflate_dollar_columns requires '{year_col}'")
    factor = cpi_deflator_to_base(
        out[year_col], base_year=base_year, annual=annual_cpi_series
    )
    # Align to out.index
    factor = pd.Series(factor.to_numpy(), index=out.index)
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce") * factor
    out.attrs["price_base_year"] = int(base_year)
    out.attrs["dollars"] = f"{base_year}_CAD"
    return out


@dataclass
class IncomeNumeraire:
    """
    Scale survey / model quantities by mean household net income in a reference year.

    Model units: ``x_model = x_cad2019 / mean_inc_net``.
    Dollars: ``x_cad2019 = x_model * mean_inc_net``.
    """

    mean_inc_net: float
    year: int = DEFAULT_NUMERAIRE_YEAR
    price_base_year: int = DEFAULT_PRICE_BASE_YEAR
    income_col: str = "inc_net"
    n: int = 0

    def to_model(self, x: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        if self.mean_inc_net <= 0 or not np.isfinite(self.mean_inc_net):
            raise ValueError(f"invalid mean_inc_net={self.mean_inc_net}")
        return x / self.mean_inc_net

    def to_dollars(self, x: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        return x * self.mean_inc_net

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: PathLike) -> "IncomeNumeraire":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


def income_numeraire_from_sfs(
    df: pd.DataFrame,
    *,
    year: int = DEFAULT_NUMERAIRE_YEAR,
    income_col: str = "inc_net",
    positive_only: bool = True,
    price_base_year: int = DEFAULT_PRICE_BASE_YEAR,
) -> IncomeNumeraire:
    """
    Weighted mean of ``income_col`` in ``year`` (expects dollars already in
    ``price_base_year`` CAD if CPI deflation was applied).
    """
    if "year" not in df.columns:
        raise KeyError("income_numeraire_from_sfs requires 'year'")
    if income_col not in df.columns:
        raise KeyError(f"missing income column {income_col}")
    if "weight" not in df.columns:
        raise KeyError("missing weight")

    sub = df[df["year"] == int(year)].copy()
    y = pd.to_numeric(sub[income_col], errors="coerce")
    w = pd.to_numeric(sub["weight"], errors="coerce")
    ok = y.notna() & w.notna() & (w > 0)
    if positive_only:
        ok = ok & (y > 0)
    if not ok.any():
        raise ValueError(f"No usable {income_col} rows for year={year}")
    mu = float(wmean(y[ok].to_numpy(), w[ok].to_numpy()))
    return IncomeNumeraire(
        mean_inc_net=mu,
        year=int(year),
        price_base_year=int(price_base_year),
        income_col=income_col,
        n=int(ok.sum()),
    )


def scale_columns_to_model_units(
    df: pd.DataFrame,
    cols: Sequence[str],
    numeraire: IncomeNumeraire,
    *,
    inplace: bool = False,
    suffix_cad: Optional[str] = "_cad2019",
) -> pd.DataFrame:
    """
    Divide ``cols`` by the income numeraire.

    If ``suffix_cad`` is set, keep a copy of the pre-scale 2019$ values under
    ``{col}{suffix_cad}`` when that name is free.
    """
    out = df if inplace else df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if suffix_cad:
            cad_name = f"{c}{suffix_cad}"
            if cad_name not in out.columns:
                out[cad_name] = out[c]
        out[c] = numeraire.to_model(pd.to_numeric(out[c], errors="coerce"))
    out.attrs["income_numeraire"] = numeraire.to_dict()
    out.attrs["units"] = "mean_inc_net"
    return out
