"""Load nominal SHS category dollars for 2017, 2019, and 2023.

Dollar fields are left in survey-year nominal dollars. The all-items deflator
used elsewhere in this project is not applied.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .baskets import (
    AGE_LABELS,
    CC_BAND_TARGETS,
    CC_INCOME_BANDS,
    CC_LOCATION,
    CC_MAX_LEVEL,
    CC_SCALE,
    MAPPED_CATEGORIES,
    TENURE_LABELS,
)
from .cpi import output_dir, shs_root

MISSING_NUMERIC = 99_999_999.96

# 1-indexed inclusive positions from the reading cards.
_COLS_2023: tuple[tuple[str, int, int], ...] = (
    ("caseid", 1, 6),
    ("weight", 7, 24),
    ("prov", 25, 26),
    ("rp_agegp", 36, 37),
    ("tenure_code", 47, 47),
    ("hhtotinc", 66, 76),
    ("cl", 210, 220),  # CL030
    ("ed002", 364, 374),
    ("fd001", 408, 418),
    ("gc001", 2135, 2145),
    ("hc001", 2146, 2156),
    ("hf001", 2366, 2376),
    ("ho001", 2432, 2442),
    ("me001", 2586, 2596),
    ("pc001", 2674, 2684),
    ("re001", 2729, 2739),
    ("ro001", 3147, 3157),
    ("sh001", 3268, 3278),
    ("sh003", 3290, 3300),
    ("sh010", 3312, 3322),
    ("sh011", 3323, 3333),
    ("sh030", 3367, 3377),
    ("ta", 3620, 3630),  # TA018
    ("tc001", 3664, 3674),
    ("tr001", 3730, 3740),
)

_COLS_2017: tuple[tuple[str, int, int], ...] = (
    ("caseid", 1, 6),
    ("weight", 7, 24),
    ("prov", 25, 26),
    ("rp_agegp", 37, 38),
    ("tenure_code", 49, 49),
    ("hhtotinc", 236, 246),
    ("cl", 347, 357),  # CL001
    ("ed002", 556, 566),
    ("fd001", 666, 676),
    ("gc001", 2393, 2403),
    ("hc001", 2404, 2414),
    ("hf001", 2635, 2645),
    ("ho001", 2701, 2711),
    ("me001", 2877, 2887),
    ("pc001", 2965, 2975),
    ("re001", 3020, 3030),
    ("ro001", 3449, 3459),
    ("sh001", 3614, 3624),
    ("sh003", 3636, 3646),
    ("sh010", 3658, 3668),
    ("sh011", 3669, 3679),
    ("sh030", 3724, 3734),
    ("ta", 3922, 3932),  # TA001
    ("tc001", 3988, 3998),
    ("tr001", 4054, 4064),
)

_DOLLAR_COLS: tuple[str, ...] = (
    "hhtotinc",
    "cl",
    "ed002",
    "fd001",
    "gc001",
    "hc001",
    "hf001",
    "ho001",
    "me001",
    "pc001",
    "re001",
    "ro001",
    "sh001",
    "sh003",
    "sh010",
    "sh011",
    "sh030",
    "ta",
    "tc001",
    "tr001",
)

_RENAME_2019: dict[str, str] = {
    "CaseID": "caseid",
    "WeightD": "weight",
    "Prov": "prov",
    "RP_AgeGrp": "rp_agegp",
    "Tenure": "tenure_code",
    "HH_TotInc": "hhtotinc",
    "CL030": "cl",
    "ED002": "ed002",
    "FD001": "fd001",
    "GC001": "gc001",
    "HC001": "hc001",
    "HF001": "hf001",
    "HO001": "ho001",
    "ME001": "me001",
    "PC001": "pc001",
    "RE001": "re001",
    "RO001": "ro001",
    "SH001": "sh001",
    "SH003": "sh003",
    "SH010": "sh010",
    "SH011": "sh011",
    "SH030": "sh030",
    "TA018": "ta",
    "TC001": "tc001",
    "TR001": "tr001",
}


def _edm_root() -> Path:
    return Path(r"C:\Users\thoma\Dropbox\dataSHS\3508_SHS_EDM")


def _read_fwf(path: Path, cols: tuple[tuple[str, int, int], ...]) -> pd.DataFrame:
    names = [c[0] for c in cols]
    colspecs = [(c[1] - 1, c[2]) for c in cols]
    return pd.read_fwf(
        path,
        colspecs=colspecs,
        names=names,
        encoding="latin-1",
        dtype=str,
    )


def _to_num(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    return values.where(values < MISSING_NUMERIC)


def _strict_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].sum(axis=1, min_count=len(cols))


def _weighted_groups(values: np.ndarray, weight: np.ndarray, n_groups: int) -> np.ndarray:
    """1..n_groups by weighted rank. Non-positive values stay missing."""
    labels = np.full(values.size, np.nan)
    ok = np.isfinite(values) & np.isfinite(weight) & (weight > 0) & (values > 0)
    if int(ok.sum()) < n_groups:
        return labels
    xx = values[ok]
    ww = weight[ok]
    order = np.argsort(xx, kind="mergesort")
    w_ord = ww[order]
    cum = np.cumsum(w_ord)
    total = float(cum[-1])
    if total <= 0:
        return labels
    start = cum - w_ord
    lab_ord = np.clip(np.floor(start / total * n_groups).astype(int) + 1, 1, n_groups)
    out = np.empty(xx.size, dtype=float)
    out[order] = lab_ord
    labels[np.flatnonzero(ok)] = out
    return labels


def _attach_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    age_code = pd.to_numeric(out["rp_agegp"], errors="coerce")
    tenure_code = pd.to_numeric(out["tenure_code"], errors="coerce")
    out["age_band"] = age_code.map(lambda v: AGE_LABELS.get(int(v)) if pd.notna(v) else np.nan)
    out["tenure"] = tenure_code.map(lambda v: TENURE_LABELS.get(int(v)) if pd.notna(v) else np.nan)
    pieces = []
    for _, g in out.groupby("year", sort=True):
        income_labels = _weighted_groups(
            g["hhtotinc"].to_numpy(dtype=float),
            g["weight"].to_numpy(dtype=float),
            n_groups=5,
        )
        cons_labels = _weighted_groups(
            g["tc001"].to_numpy(dtype=float),
            g["weight"].to_numpy(dtype=float),
            n_groups=10,
        )
        g = g.copy()
        g["inc_q"] = pd.Series(income_labels, index=g.index).map(
            lambda v: f"Q{int(v)}" if pd.notna(v) else np.nan
        )
        g["cons_d"] = pd.Series(cons_labels, index=g.index).map(
            lambda v: f"D{int(v):02d}" if pd.notna(v) else np.nan
        )
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    both = out["inc_q"].notna() & out["tenure"].notna()
    out["inc_q_x_tenure"] = np.where(both, out["inc_q"] + " | " + out["tenure"], np.nan)
    return out


def _attach_baskets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["food"] = out["fd001"]
    out["shelter"] = out["sh001"]
    out["household"] = _strict_sum(out, ["ho001", "hf001"])
    out["clothing"] = out["cl"]
    out["transportation"] = out["tr001"]
    out["health_personal"] = _strict_sum(out, ["hc001", "pc001"])
    out["recreation_education"] = _strict_sum(out, ["re001", "ed002", "ro001"])
    out["alcohol_tobacco"] = out["ta"]
    out["residual"] = out["tc001"] - _strict_sum(out, list(MAPPED_CATEGORIES))
    out["rent"] = out["sh003"]
    out["owned"] = out["sh010"]
    out["water_fuel"] = out["sh030"]
    out["other_shelter"] = out["sh001"] - _strict_sum(out, ["sh003", "sh010", "sh030"])
    weight_ok = out["weight"].notna() & (out["weight"] > 0)
    tc_ok = out["tc001"].notna() & (out["tc001"] > 0)
    mapped_ok = out[list(MAPPED_CATEGORIES)].notna().all(axis=1)
    out["complete_primary"] = weight_ok & tc_ok & mapped_ok & out["residual"].notna()
    shelter_ok = out[["rent", "owned", "water_fuel", "other_shelter"]].notna().all(axis=1)
    out["complete_shelter"] = out["complete_primary"] & shelter_ok
    income = out["hhtotinc"].to_numpy(dtype=float)
    prop = np.full(len(out), np.nan)
    ok = np.isfinite(income) & (income >= 0)
    z = (income[ok] - CC_LOCATION) / CC_SCALE
    prop[ok] = CC_MAX_LEVEL / (1.0 + np.exp(-z))
    out["imputed_cc_prop"] = prop
    out["cc_consumption"] = out["imputed_cc_prop"] * out["tc001"]
    return out


def _finalize(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    out = raw.copy()
    out["year"] = int(year)
    out["caseid"] = out["caseid"].astype(str).str.strip()
    out["prov"] = out["prov"].astype(str).str.strip()
    out["weight"] = _to_num(out["weight"])
    for col in _DOLLAR_COLS:
        out[col] = _to_num(out[col])
    out = out[out["weight"].notna() & (out["weight"] > 0)].copy()
    out = _attach_baskets(out)
    out = _attach_groups(out)
    return out.reset_index(drop=True)


def load_shs_2023(root: Path | None = None) -> pd.DataFrame:
    root = Path(root) if root is not None else shs_root() / "RY2023"
    path = root / "Data" / "TXT" / "PUMF_SHS_2023.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _finalize(_read_fwf(path, _COLS_2023), 2023)


def load_shs_2019(edm: Path | None = None) -> pd.DataFrame:
    edm = Path(edm) if edm is not None else _edm_root()
    path = edm / "2019" / "shs_2019.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, usecols=list(_RENAME_2019), low_memory=False)
    raw = raw.rename(columns=_RENAME_2019)
    for col in raw.columns:
        if col != "caseid":
            raw[col] = raw[col].astype(str)
    return _finalize(raw, 2019)


def load_shs_2017(edm: Path | None = None) -> pd.DataFrame:
    edm = Path(edm) if edm is not None else _edm_root()
    zip_path = edm / "2017" / "data" / "RAW.zip"
    member = "RAW/pumf_shs2017_diary_flatfile.txt"
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        payload = fh.read()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        raw = _read_fwf(tmp_path, _COLS_2017)
    finally:
        tmp_path.unlink(missing_ok=True)
    return _finalize(raw, 2017)


def load_shs_categories() -> pd.DataFrame:
    frames = [load_shs_2017(), load_shs_2019(), load_shs_2023()]
    return pd.concat(frames, ignore_index=True)


def _wshare(numer: pd.Series, denom: pd.Series, weight: pd.Series) -> float:
    ok = numer.notna() & denom.notna() & weight.notna() & (weight > 0) & (denom > 0)
    if not ok.any():
        return float("nan")
    w = weight[ok].to_numpy(dtype=float)
    d = denom[ok].to_numpy(dtype=float)
    n = numer[ok].to_numpy(dtype=float)
    return float(np.sum(w * n) / np.sum(w * d))


def adding_up_report(df: pd.DataFrame) -> pd.DataFrame:
    """Check that the mapped categories plus residual rebuild TC001, and describe the gap.

    Residual is defined as TC001 minus the mapped sum, so mapped + residual equals TC001
    on complete rows. The informative check is whether that residual lines up with
    miscellaneous plus games of chance, and whether the shelter parts rebuild SH001.
    """
    rows = []
    for year, g in df.groupby("year"):
        primary = g[g["complete_primary"]]
        shelter = g[g["complete_shelter"]]
        both = primary["me001"].notna() & primary["gc001"].notna()
        me_gc = primary.loc[both, "me001"] + primary.loc[both, "gc001"]
        rows.append(
            {
                "year": int(year),
                "n_households": int(len(g)),
                "n_complete_primary": int(len(primary)),
                "n_complete_shelter": int(shelter.shape[0]),
                "residual_share": _wshare(primary["residual"], primary["tc001"], primary["weight"]),
                "me_gc_share": _wshare(me_gc, primary.loc[both, "tc001"], primary.loc[both, "weight"]),
                "residual_minus_me_gc_share": _wshare(
                    primary.loc[both, "residual"] - me_gc,
                    primary.loc[both, "tc001"],
                    primary.loc[both, "weight"],
                ),
                "other_shelter_share_of_tc": _wshare(
                    shelter["other_shelter"], shelter["tc001"], shelter["weight"]
                ),
                "other_shelter_share_of_shelter": _wshare(
                    shelter["other_shelter"], shelter["shelter"], shelter["weight"]
                ),
                "share_hh_residual_below_minus_1": float((primary["residual"] < -1).mean())
                if len(primary)
                else float("nan"),
                "share_hh_other_shelter_below_minus_1": float((shelter["other_shelter"] < -1).mean())
                if len(shelter)
                else float("nan"),
                "max_abs_dollar_gap_vs_identity": float(
                    (primary[list(MAPPED_CATEGORIES) + ["residual"]].sum(axis=1) - primary["tc001"])
                    .abs()
                    .max()
                )
                if len(primary)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _wmean(values: pd.Series, weight: pd.Series) -> float:
    ok = values.notna() & weight.notna() & (weight > 0)
    if not ok.any():
        return float("nan")
    w = weight[ok].to_numpy(dtype=float)
    v = values[ok].to_numpy(dtype=float)
    return float(np.sum(w * v) / np.sum(w))


def cc_band_check(df: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
    """Weighted mean imputed card share by the 2023 income bands used to fit the curve."""
    sub = df[df["year"] == int(year)].copy()
    rows = []
    income = sub["hhtotinc"]
    for name, low, high in CC_INCOME_BANDS:
        mask = income.notna() & (income >= low) & (income < high) & sub["imputed_cc_prop"].notna()
        g = sub.loc[mask]
        rows.append(
            {
                "year": int(year),
                "income_band": name,
                "target": CC_BAND_TARGETS[name],
                "weighted_mean_prop": _wmean(g["imputed_cc_prop"], g["weight"]),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def cc_spend_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean card propensity and card consumption by year and group."""
    specs: tuple[tuple[str, str | None], ...] = (
        ("all", None),
        ("cons_d", "cons_d"),
        ("inc_q", "inc_q"),
    )
    frames = []
    for grouping, col in specs:
        piece = df.dropna(subset=["imputed_cc_prop", "cc_consumption", "tc001"]).copy()
        piece = piece[piece["weight"] > 0]
        if col is None:
            piece["group"] = "all"
        else:
            piece = piece.dropna(subset=[col])
            piece["group"] = piece[col].astype(str)
        rows = []
        for (year, group), g in piece.groupby(["year", "group"], sort=False):
            rows.append(
                {
                    "grouping": grouping,
                    "group": group,
                    "year": int(year),
                    "n": int(len(g)),
                    "weighted_mean_prop": _wmean(g["imputed_cc_prop"], g["weight"]),
                    "weighted_mean_cc_consumption": _wmean(g["cc_consumption"], g["weight"]),
                    "weighted_mean_tc001": _wmean(g["tc001"], g["weight"]),
                }
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def save_household_extract(df: pd.DataFrame) -> Path:
    dest = output_dir() / "shs_household_categories.parquet"
    df.to_parquet(dest, compression="zstd", index=False)
    return dest
