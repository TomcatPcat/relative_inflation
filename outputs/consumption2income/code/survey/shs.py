"""SHS (Survey of Household Spending) loaders and C/Y moments."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .paths import load_local_paths, require_dir, require_file
from .weighted import wmean, wquantile, wshare

# 1-indexed inclusive positions from StatCan reading cards / CA_costsinflation.
_COLS_2023 = [
    ("CASEID", 1, 6),
    ("WEIGHTD", 7, 24),
    ("PROV", 25, 26),
    ("RP_AGEGP", 36, 37),
    ("TENURE", 47, 47),
    ("HHTOTINC", 66, 76),
    ("SH004", 3301, 3311),
    ("SH011", 3323, 3333),  # Mortgage paid on owned principal residence
    ("TC001", 3664, 3674),
    ("TX010", 3994, 4004),
]

# 2017 diary: clothing=CL001→CL030 slot; taxes=TX001→TX010 slot.
_COLS_2017 = [
    ("CASEID", 1, 6),
    ("WEIGHTD", 7, 24),
    ("PROV", 25, 26),
    ("RP_AGEGP", 37, 38),
    ("TENURE", 49, 49),
    ("HHTOTINC", 236, 246),
    ("SH004", 3647, 3657),
    ("SH011", 3669, 3679),  # Mortgage paid on the principal residence
    ("TC001", 3988, 3998),
    ("TX010", 4362, 4372),
]

_MISSING_NUMERIC = 99_999_999.96

_TIER_A_YEARS = (2017, 2019, 2023)

# Reference-person age group (RP_AGEGP) labels used in SHS Tier A PUMF.
SHS_AGE_BAND_LABELS = {
    1: "<30",
    2: "30-39",
    3: "40-54",
    4: "55-64",
    5: "65-74",
    6: "75+",
}
SHS_AGE_BAND_ORDER = ("<30", "30-39", "40-54", "55-64", "65-74", "75+")


def _read_fwf(path: Path, cols: Sequence[tuple]) -> pd.DataFrame:
    names = [c[0] for c in cols]
    # pandas colspecs are 0-indexed half-open [start, end)
    colspecs = [(c[1] - 1, c[2]) for c in cols]
    df = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=names,
        encoding="latin-1",
        dtype=str,
    )
    return df


def _to_num(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
    v = v.where(v < _MISSING_NUMERIC, np.nan)
    return v


def _shs_age_band(agegp: pd.Series) -> pd.Series:
    g = pd.to_numeric(agegp.astype(str).str.strip(), errors="coerce")
    return g.map(SHS_AGE_BAND_LABELS)


def _harmonize_tier_a(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "year": int(year),
            "tier": "A",
            "caseid": df["CASEID"].astype(str).str.strip(),
            "weight": _to_num(df["WEIGHTD"]),
            "hhtotinc": _to_num(df["HHTOTINC"]),
            "consumption": _to_num(df["TC001"]),
            "taxes": _to_num(df["TX010"]) if "TX010" in df.columns else np.nan,
        }
    )
    if "PROV" in df.columns:
        out["prov"] = df["PROV"].astype(str).str.strip()
    if "TENURE" in df.columns:
        out["tenure"] = df["TENURE"].astype(str).str.strip()
        # Tier A: 1=owned with mortgage, 2=owned free & clear, 3=rented
        out["renter"] = out["tenure"].isin({"3", "3.0"})
        out["mortgage_holder"] = out["tenure"].isin({"1", "1.0"})
    if "RP_AGEGP" in df.columns:
        out["rp_agegp"] = df["RP_AGEGP"].astype(str).str.strip()
        out["age_band"] = _shs_age_band(df["RP_AGEGP"])
    if "SH004" in df.columns:
        out["rent"] = _to_num(df["SH004"])
    if "SH011" in df.columns:
        out["mortgage_payment"] = _to_num(df["SH011"])
    out["income_at"] = out["hhtotinc"] - out["taxes"]
    # Prefer after-tax; fall back to total income if taxes missing
    out["inc_net"] = out["income_at"].where(out["income_at"].notna(), out["hhtotinc"])
    out = out[out["weight"].notna() & (out["weight"] > 0)].copy()
    return out.reset_index(drop=True)


def _deflate_shs(
    df: pd.DataFrame,
    *,
    price_base_year: Optional[int],
) -> pd.DataFrame:
    if price_base_year is None:
        return df
    from .units import SHS_DOLLAR_COLS, deflate_dollar_columns

    return deflate_dollar_columns(
        df, SHS_DOLLAR_COLS, base_year=int(price_base_year), inplace=False
    )


def load_shs_2023(
    ry2023: Optional[Union[str, Path]] = None,
    *,
    price_base_year: Optional[int] = 2019,
) -> pd.DataFrame:
    root = Path(ry2023) if ry2023 is not None else load_local_paths()["shs_ry2023"]
    root = require_dir(Path(root), "SHS RY2023")
    path = require_file(root / "Data" / "TXT" / "PUMF_SHS_2023.txt", "SHS 2023 PUMF")
    return _deflate_shs(_harmonize_tier_a(_read_fwf(path, _COLS_2023), 2023), price_base_year=price_base_year)


def load_shs_2019(
    edm: Optional[Union[str, Path]] = None,
    *,
    price_base_year: Optional[int] = 2019,
) -> pd.DataFrame:
    root = Path(edm) if edm is not None else load_local_paths()["shs_edm"]
    root = require_dir(Path(root), "SHS EDM")
    path = require_file(root / "2019" / "shs_2019.csv", "SHS 2019 CSV")
    raw = pd.read_csv(path, low_memory=False)
    upper = {str(c).strip().upper(): c for c in raw.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in raw.columns:
                return n
            if n.upper() in upper:
                return upper[n.upper()]
        return None

    frame = pd.DataFrame()
    for std, cands in [
        ("CASEID", ("CASEID", "CaseID")),
        ("WEIGHTD", ("WEIGHTD", "WeightD")),
        ("PROV", ("PROV", "Prov")),
        ("RP_AGEGP", ("RP_AGEGP", "RP_AgeGrp")),
        ("TENURE", ("TENURE", "Tenure")),
        ("HHTOTINC", ("HHTOTINC", "HH_TotInc", "HHTotInc")),
        ("SH004", ("SH004",)),
        ("SH011", ("SH011",)),
        ("TC001", ("TC001",)),
        ("TX010", ("TX010", "TX001")),
    ]:
        src = pick(*cands)
        frame[std] = raw[src] if src is not None else np.nan
    return _deflate_shs(_harmonize_tier_a(frame, 2019), price_base_year=price_base_year)


def load_shs_2017(
    edm: Optional[Union[str, Path]] = None,
    *,
    price_base_year: Optional[int] = 2019,
) -> pd.DataFrame:
    root = Path(edm) if edm is not None else load_local_paths()["shs_edm"]
    root = require_dir(Path(root), "SHS EDM")
    zip_path = root / "2017" / "data" / "RAW.zip"
    member = "RAW/pumf_shs2017_diary_flatfile.txt"
    if not zip_path.is_file():
        raise FileNotFoundError(f"SHS 2017 RAW.zip not found at {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            # Temporary extract to bytes via pandas from a NamedTemporaryFile
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp.write(fh.read())
                tmp_path = Path(tmp.name)
    try:
        df = _read_fwf(tmp_path, _COLS_2017)
    finally:
        tmp_path.unlink(missing_ok=True)
    return _deflate_shs(_harmonize_tier_a(df, 2017), price_base_year=price_base_year)


def load_shs_tier_a(
    *,
    ry2023: Optional[Union[str, Path]] = None,
    edm: Optional[Union[str, Path]] = None,
    years: Sequence[int] = _TIER_A_YEARS,
    price_base_year: Optional[int] = 2019,
) -> pd.DataFrame:
    """Load post-redesign SHS years (2017, 2019, 2023). Skips missing sources."""
    paths = load_local_paths()
    ry = ry2023 if ry2023 is not None else paths["shs_ry2023"]
    ed = edm if edm is not None else paths["shs_edm"]
    pieces = []
    loaders = {
        2017: lambda: load_shs_2017(ed, price_base_year=price_base_year),
        2019: lambda: load_shs_2019(ed, price_base_year=price_base_year),
        2023: lambda: load_shs_2023(ry, price_base_year=price_base_year),
    }
    for y in years:
        if y not in loaders:
            continue
        try:
            pieces.append(loaders[y]())
        except FileNotFoundError:
            continue
    if not pieces:
        raise FileNotFoundError("No SHS Tier A years could be loaded")
    return pd.concat(pieces, ignore_index=True)


def load_shs_tier_b(
    path: Optional[Union[str, Path]] = None,
    *,
    price_base_year: Optional[int] = 2019,
) -> pd.DataFrame:
    """Load pre-redesign consolidated SHS (1997–2009)."""
    if path is None:
        path = load_local_paths()["shs_consolidated"]
    path = require_file(Path(path), "SHS consolidated")
    raw = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(raw["survey_year"], errors="coerce"),
            "tier": "B",
            "caseid": raw["case_id"].astype(str) if "case_id" in raw.columns else np.arange(len(raw)),
            "weight": pd.to_numeric(raw["weights"], errors="coerce"),
            "consumption": pd.to_numeric(raw["consumption"], errors="coerce"),
            "inc_net": pd.to_numeric(raw["net_income"], errors="coerce"),
            "hhtotinc": pd.to_numeric(raw["gross_income"], errors="coerce")
            if "gross_income" in raw.columns
            else np.nan,
            "taxes": pd.to_numeric(raw["personal_taxes"], errors="coerce")
            if "personal_taxes" in raw.columns
            else np.nan,
        }
    )
    out["income_at"] = out["inc_net"]
    out = out[out["weight"].notna() & (out["weight"] > 0)].copy()
    out = out[out["year"].notna()].copy()
    out["year"] = out["year"].astype(int)
    out = out.reset_index(drop=True)
    return _deflate_shs(out, price_base_year=price_base_year)


def load_shs_all(
    *,
    include_tier_a: bool = True,
    include_tier_b: bool = True,
    price_base_year: Optional[int] = 2019,
) -> pd.DataFrame:
    """Concatenate Tier A and Tier B with a ``tier`` column (do not splice trends)."""
    frames = []
    if include_tier_a:
        try:
            frames.append(load_shs_tier_a(price_base_year=price_base_year))
        except FileNotFoundError:
            pass
    if include_tier_b:
        try:
            frames.append(load_shs_tier_b(price_base_year=price_base_year))
        except FileNotFoundError:
            pass
    if not frames:
        raise FileNotFoundError("No SHS data could be loaded")
    out = pd.concat(frames, ignore_index=True)
    if price_base_year is not None:
        out.attrs["price_base_year"] = int(price_base_year)
        out.attrs["dollars"] = f"{int(price_base_year)}_CAD"
    return out


def shs_cy_moments(
    df: pd.DataFrame,
    by: Union[str, Sequence[str], None] = "year",
) -> pd.DataFrame:
    """
    Aggregate consumption-to-net-income ratio: ``sum(w C) / sum(w Y)`` for Y > 0.

    Uses ``consumption`` and ``inc_net`` (after-tax + transfers).
    """
    for c in ("weight", "consumption", "inc_net"):
        if c not in df.columns:
            raise KeyError(f"shs_cy_moments missing {c}")

    if by is None:
        groups: Iterable[tuple] = [(None, df)]
        group_keys: list[str] = []
    else:
        keys = [by] if isinstance(by, str) else list(by)
        group_keys = keys
        groups = df.groupby(keys, sort=True, dropna=False)

    rows = []
    for key, g in groups:
        w = g["weight"].to_numpy(dtype=float)
        c = g["consumption"].to_numpy(dtype=float)
        y = g["inc_net"].to_numpy(dtype=float)
        pos = np.isfinite(y) & (y > 0) & np.isfinite(c) & np.isfinite(w) & (w > 0)
        row = {
            "n": int(pos.sum()),
            "n_raw": int(len(g)),
            "cy_ratio": wshare(c[pos], y[pos], w[pos]),
            "mean_consumption": wmean(c[pos], w[pos]) if pos.any() else float("nan"),
            "mean_inc_net": wmean(y[pos], w[pos]) if pos.any() else float("nan"),
        }
        if "tier" in g.columns:
            tiers = sorted({str(t) for t in g["tier"].dropna().unique()})
            row["tier"] = ",".join(tiers) if tiers else None
        if group_keys:
            if len(group_keys) == 1:
                row[group_keys[0]] = key[0] if isinstance(key, tuple) else key
            else:
                vals = key if isinstance(key, tuple) else (key,)
                for k, v in zip(group_keys, vals):
                    row[k] = v
        rows.append(row)

    front = group_keys + (["tier"] if any("tier" in r for r in rows) else [])
    cols = front + ["n", "n_raw", "cy_ratio", "mean_consumption", "mean_inc_net"]
    # de-dupe while preserving order
    seen: set[str] = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    frame = pd.DataFrame(rows)
    return frame[[c for c in cols if c in frame.columns]]


def _weighted_decile_labels(x: np.ndarray, w: np.ndarray, n_groups: int = 10) -> np.ndarray:
    """Assign 1..n_groups by weighted rank of ``x`` (ties broken by order)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=int)
    ranks[order] = np.arange(x.size)
    # Mid-rank cumulative weight for each observation
    w_ord = w[order]
    cum = np.cumsum(w_ord)
    total = float(cum[-1]) if cum.size else 0.0
    if total <= 0:
        return np.full(x.size, np.nan)
    # Use start-of-bin cumulative weight so the heaviest obs can reach group n
    start = cum - w_ord
    frac = start / total
    labels_ord = np.floor(frac * n_groups).astype(int) + 1
    labels_ord = np.clip(labels_ord, 1, n_groups)
    out = np.empty(x.size, dtype=int)
    out[order] = labels_ord
    return out


def prepare_shs_renters(
    df: pd.DataFrame,
    *,
    min_inc_net: float = 1.0,
    require_positive_rent: bool = False,
) -> pd.DataFrame:
    """
    Restrict to Tier-A-style renters with usable rent and after-tax income.

    Ratio ``rent_share = rent / inc_net`` (after-tax household income).
    """
    need = ("weight", "rent", "inc_net", "renter")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"prepare_shs_renters missing {missing}")
    out = df.copy()
    rent = pd.to_numeric(out["rent"], errors="coerce")
    y = pd.to_numeric(out["inc_net"], errors="coerce")
    w = pd.to_numeric(out["weight"], errors="coerce")
    ok = (
        out["renter"].fillna(False).astype(bool)
        & np.isfinite(rent)
        & np.isfinite(y)
        & (y >= min_inc_net)
        & np.isfinite(w)
        & (w > 0)
    )
    if require_positive_rent:
        ok = ok & (rent > 0)
    else:
        ok = ok & (rent >= 0)
    out = out.loc[ok].copy()
    out["rent"] = rent.loc[ok]
    out["inc_net"] = y.loc[ok]
    out["weight"] = w.loc[ok]
    out["rent_share"] = out["rent"] / out["inc_net"]
    return out.reset_index(drop=True)


def shs_renter_rent_share_percentiles(
    df: pd.DataFrame,
    *,
    by: str = "inc_decile",
    probs: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    n_income_groups: int = 10,
    prepare: bool = True,
) -> pd.DataFrame:
    """
    Weighted percentiles of rent / after-tax income for renter households.

    ``by``:
      - ``inc_decile``: within-renter weighted after-tax income deciles (D01…D10)
      - ``age_band``: reference-person age bands
      - ``year``: survey year
      - ``all``: single pooled row
    """
    work = prepare_shs_renters(df) if prepare else df
    if work.empty:
        raise ValueError("no renter rows with usable rent and income")

    if by == "inc_decile":
        labels = _weighted_decile_labels(
            work["inc_net"].to_numpy(dtype=float),
            work["weight"].to_numpy(dtype=float),
            n_groups=n_income_groups,
        )
        work = work.copy()
        work["inc_decile"] = [f"D{int(k):02d}" for k in labels]
        group_col = "inc_decile"
        order = [f"D{i:02d}" for i in range(1, n_income_groups + 1)]
    elif by == "age_band":
        group_col = "age_band"
        order = list(SHS_AGE_BAND_ORDER)
        if group_col not in work.columns:
            raise KeyError("age_band missing; reload SHS Tier A with RP_AGEGP")
    elif by == "year":
        group_col = "year"
        order = sorted(int(y) for y in work["year"].dropna().unique())
    elif by == "all":
        group_col = None
        order = ["all"]
    else:
        raise ValueError("by must be 'inc_decile', 'age_band', 'year', or 'all'")

    rows = []
    if group_col is None:
        groups = [("all", work)]
    else:
        groups = list(work.groupby(group_col, sort=False, dropna=False))

    for key, g in groups:
        share = g["rent_share"].to_numpy(dtype=float)
        w = g["weight"].to_numpy(dtype=float)
        row = {
            "by": by if by != "all" else "all",
            "group": key if group_col is None else key,
            "n": int(len(g)),
            "mean_rent_share": wmean(share, w),
            "mean_rent": wmean(g["rent"].to_numpy(dtype=float), w),
            "mean_inc_net": wmean(g["inc_net"].to_numpy(dtype=float), w),
            "agg_rent_share": wshare(
                g["rent"].to_numpy(dtype=float),
                g["inc_net"].to_numpy(dtype=float),
                w,
            ),
        }
        for p in probs:
            lab = f"p{int(round(100 * float(p))):02d}"
            row[lab] = float(wquantile(share, w, float(p)))
        rows.append(row)

    out = pd.DataFrame(rows)
    # Stable group order
    if order and "group" in out.columns:
        cat = pd.Categorical(out["group"], categories=order, ordered=True)
        out = out.assign(group=cat).sort_values("group").reset_index(drop=True)
        out["group"] = out["group"].astype(str)
    return out


def prepare_shs_mortgage_holders(
    df: pd.DataFrame,
    *,
    min_consumption: float = 1.0,
    require_positive_mortgage: bool = True,
) -> pd.DataFrame:
    """
    Restrict to Tier-A mortgage holders with usable payment and expenditure.

    Ratio ``mortgage_exp_share = mortgage_payment / consumption`` where
    ``mortgage_payment`` is SH011 (mortgage paid on owned principal residence)
    and ``consumption`` is TC001 (total consumption expenditure).
    """
    need = ("weight", "mortgage_payment", "consumption", "mortgage_holder")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"prepare_shs_mortgage_holders missing {missing}")
    out = df.copy()
    mort = pd.to_numeric(out["mortgage_payment"], errors="coerce")
    c = pd.to_numeric(out["consumption"], errors="coerce")
    w = pd.to_numeric(out["weight"], errors="coerce")
    ok = (
        out["mortgage_holder"].fillna(False).astype(bool)
        & np.isfinite(mort)
        & np.isfinite(c)
        & (c >= min_consumption)
        & np.isfinite(w)
        & (w > 0)
    )
    if require_positive_mortgage:
        ok = ok & (mort > 0)
    else:
        ok = ok & (mort >= 0)
    out = out.loc[ok].copy()
    out["mortgage_payment"] = mort.loc[ok]
    out["consumption"] = c.loc[ok]
    out["weight"] = w.loc[ok]
    if "inc_net" in out.columns:
        out["inc_net"] = pd.to_numeric(out["inc_net"], errors="coerce")
    out["mortgage_exp_share"] = out["mortgage_payment"] / out["consumption"]
    return out.reset_index(drop=True)


def shs_mortgage_expenditure_share_percentiles(
    df: pd.DataFrame,
    *,
    by: str = "inc_decile",
    probs: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    n_income_groups: int = 10,
    prepare: bool = True,
) -> pd.DataFrame:
    """
    Weighted percentiles of mortgage payment / total expenditure.

    ``by``:
      - ``inc_decile``: within-mortgage-holder weighted after-tax income deciles
      - ``age_band``: reference-person age bands
      - ``year``: survey year
      - ``all``: single pooled row
    """
    work = prepare_shs_mortgage_holders(df) if prepare else df
    if work.empty:
        raise ValueError("no mortgage-holder rows with usable payment and expenditure")

    if by == "inc_decile":
        if "inc_net" not in work.columns:
            raise KeyError("inc_net missing for income-decile breakdown")
        y = work["inc_net"].to_numpy(dtype=float)
        w = work["weight"].to_numpy(dtype=float)
        finite = np.isfinite(y)
        if not finite.any():
            raise ValueError("no finite inc_net for mortgage-holder income deciles")
        labels = np.full(work.shape[0], np.nan)
        labels[finite] = _weighted_decile_labels(y[finite], w[finite], n_groups=n_income_groups)
        work = work.copy()
        work["inc_decile"] = [
            f"D{int(k):02d}" if np.isfinite(k) else None for k in labels
        ]
        work = work[work["inc_decile"].notna()].copy()
        group_col = "inc_decile"
        order = [f"D{i:02d}" for i in range(1, n_income_groups + 1)]
    elif by == "age_band":
        group_col = "age_band"
        order = list(SHS_AGE_BAND_ORDER)
        if group_col not in work.columns:
            raise KeyError("age_band missing; reload SHS Tier A with RP_AGEGP")
    elif by == "year":
        group_col = "year"
        order = sorted(int(y) for y in work["year"].dropna().unique())
    elif by == "all":
        group_col = None
        order = ["all"]
    else:
        raise ValueError("by must be 'inc_decile', 'age_band', 'year', or 'all'")

    rows = []
    if group_col is None:
        groups = [("all", work)]
    else:
        groups = list(work.groupby(group_col, sort=False, dropna=False))

    for key, g in groups:
        share = g["mortgage_exp_share"].to_numpy(dtype=float)
        w = g["weight"].to_numpy(dtype=float)
        row = {
            "by": by if by != "all" else "all",
            "group": key if group_col is None else key,
            "n": int(len(g)),
            "mean_mortgage_exp_share": wmean(share, w),
            "mean_mortgage_payment": wmean(g["mortgage_payment"].to_numpy(dtype=float), w),
            "mean_consumption": wmean(g["consumption"].to_numpy(dtype=float), w),
            "agg_mortgage_exp_share": wshare(
                g["mortgage_payment"].to_numpy(dtype=float),
                g["consumption"].to_numpy(dtype=float),
                w,
            ),
        }
        if "inc_net" in g.columns:
            y = g["inc_net"].to_numpy(dtype=float)
            y_ok = np.isfinite(y)
            row["mean_inc_net"] = wmean(y[y_ok], w[y_ok]) if y_ok.any() else float("nan")
        for p in probs:
            lab = f"p{int(round(100 * float(p))):02d}"
            row[lab] = float(wquantile(share, w, float(p)))
        rows.append(row)

    out = pd.DataFrame(rows)
    if order and "group" in out.columns:
        cat = pd.Categorical(out["group"], categories=order, ordered=True)
        out = out.assign(group=cat).sort_values("group").reset_index(drop=True)
        out["group"] = out["group"].astype(str)
    return out
