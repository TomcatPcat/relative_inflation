"""Expenditure-weighted basket shares and group price indexes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baskets import (
    AGE_ORDER,
    BASKETS,
    CATEGORY_CPI,
    CONS_DECILE_ORDER,
    GROUPING_ORDER,
    TENURE_ORDER,
    TENURE_PRETTY,
)
PUBLISH_FROM = 2016
BASE_YEAR = 2019
SURVEY_YEARS = (2017, 2019, 2023)


def _complete_col(basket: str) -> str:
    if basket == "primary":
        return "complete_primary"
    if basket == "shelter_split":
        return "complete_shelter"
    raise KeyError(basket)


def expenditure_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Survey-weighted expenditure shares within survey year and group.

    The weight on a category is the sum of weighted dollars over the sum of
    weighted total consumption, not the average of household shares.
    """
    frames: list[pd.DataFrame] = []
    specs: tuple[tuple[str, list[str] | None], ...] = (
        ("all", None),
        ("inc_q", ["inc_q"]),
        ("cons_d", ["cons_d"]),
        ("age_band", ["age_band"]),
        ("tenure", ["tenure"]),
        ("inc_q_x_tenure", ["inc_q_x_tenure"]),
    )
    for basket, categories in BASKETS.items():
        work = df.loc[df[_complete_col(basket)]].copy()
        for grouping, cols in specs:
            piece = work
            keys = ["year"]
            if cols is None:
                piece = piece.copy()
                piece["group"] = "all"
            else:
                piece = piece.dropna(subset=cols).copy()
                piece["group"] = piece[cols[0]].astype(str)
            keys = ["year", "group"]
            piece["_w"] = piece["weight"].to_numpy(dtype=float)
            value_cols = list(categories) + ["tc001"]
            for col in value_cols:
                piece[f"_wx_{col}"] = piece["_w"] * piece[col].to_numpy(dtype=float)
            wx_cols = [f"_wx_{col}" for col in value_cols]
            summed = piece.groupby(keys, dropna=False)[wx_cols + ["_w"]].sum()
            counts = piece.groupby(keys, dropna=False).size().rename("n")
            summed = summed.join(counts)
            total = summed["_wx_tc001"].to_numpy(dtype=float)
            long_rows = []
            for category in categories:
                dollars = summed[f"_wx_{category}"]
                share = dollars / total
                block = pd.DataFrame(
                    {
                        "basket": basket,
                        "grouping": grouping,
                        "group": summed.index.get_level_values("group"),
                        "survey_year": summed.index.get_level_values("year").astype(int),
                        "category": category,
                        "share": share.to_numpy(dtype=float),
                        "w_expenditure": dollars.to_numpy(dtype=float),
                        "w_tc001": total,
                        "n": summed["n"].to_numpy(dtype=int),
                        "weight_sum": summed["_w"].to_numpy(dtype=float),
                    }
                )
                long_rows.append(block)
            frames.append(pd.concat(long_rows, ignore_index=True))
    out = pd.concat(frames, ignore_index=True)
    sums = out.groupby(["basket", "grouping", "group", "survey_year"])["share"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-8):
        bad = sums[~np.isclose(sums, 1.0, atol=1e-8)]
        raise AssertionError(f"Shares do not sum to 1:\n{bad.head(10)}")
    return out


def _weights_lookup(shares: pd.DataFrame, basket: str) -> dict[tuple, pd.Series]:
    """Map (grouping, group, survey_year) to a category-weight Series."""
    sub = shares[shares["basket"] == basket]
    lookup: dict[tuple, pd.Series] = {}
    for (grouping, group, year), g in sub.groupby(["grouping", "group", "survey_year"], sort=False):
        weights = g.set_index("category")["share"]
        lookup[(grouping, str(group), int(year))] = weights
    return lookup


def _index_from_weights(weights: pd.Series, cpi: pd.DataFrame, categories: tuple[str, ...]) -> pd.Series:
    """Lowe index, 2019 = 100, using fixed expenditure weights."""
    if int(BASE_YEAR) not in cpi.index:
        raise KeyError(f"CPI is missing base year {BASE_YEAR}")
    rel = pd.DataFrame(index=cpi.index)
    for category in categories:
        key = CATEGORY_CPI[category]
        rel[category] = cpi[key] / float(cpi.loc[BASE_YEAR, key])
    w = weights.reindex(list(categories)).astype(float)
    if w.isna().any():
        raise ValueError(f"Missing weights for {w[w.isna()].index.tolist()}")
    level = rel.mul(w, axis=1).sum(axis=1) * 100.0
    level.name = "index"
    return level


def weight_source_for_year(year: int) -> str:
    """Label the basket used in a calendar year on the updated path.

    Shares are the 2017 survey at and before 2017, the 2023 survey at and
    after 2023, and a straight line between the neighbouring surveys otherwise.
    """
    year = int(year)
    if year <= 2017:
        return "2017"
    if year < 2019:
        return "interp_2017_2019"
    if year == 2019:
        return "2019"
    if year < 2023:
        return "interp_2019_2023"
    return "2023"


def interpolate_weights(
    weights_by_year: dict[int, pd.Series],
    year: int,
    categories: tuple[str, ...],
) -> pd.Series:
    """Category shares for one calendar year. Endpoints match the survey waves."""
    year = int(year)
    names = list(categories)
    for survey_year in SURVEY_YEARS:
        if survey_year not in weights_by_year:
            raise KeyError(f"No SHS basket for share year {survey_year}")
    if year <= 2017:
        blended = weights_by_year[2017].reindex(names).astype(float)
    elif year >= 2023:
        blended = weights_by_year[2023].reindex(names).astype(float)
    else:
        if year < 2019:
            y0, y1 = 2017, 2019
        else:
            y0, y1 = 2019, 2023
        span = (year - y0) / (y1 - y0)
        early = weights_by_year[y0].reindex(names).astype(float)
        late = weights_by_year[y1].reindex(names).astype(float)
        blended = early * (1.0 - span) + late * span
    if blended.isna().any():
        raise ValueError(f"Missing weights for {blended[blended.isna()].index.tolist()}")
    if not np.isclose(float(blended.sum()), 1.0, atol=1e-8):
        raise AssertionError(f"Interpolated shares in {year} sum to {float(blended.sum())}")
    blended.name = "share"
    return blended


def _chain_updated(
    weights_by_year: dict[int, pd.Series],
    cpi: pd.DataFrame,
    categories: tuple[str, ...],
) -> pd.DataFrame:
    """Chain annual basket-weighted price relatives and rebase so 2019 = 100.

    The growth into calendar year t uses that year's interpolated SHS basket.
    The year before the published window is kept as an anchor so the first
    published year has a year-over-year rate. Callers drop that anchor.
    """
    years = sorted(int(y) for y in cpi.index if int(y) >= PUBLISH_FROM)
    anchor = years[0] - 1
    if anchor not in cpi.index:
        raise KeyError(f"CPI missing lag year {anchor}")
    levels: dict[int, float] = {anchor: 1.0}
    records = [
        {
            "year": anchor,
            "share_year": 2017,
            "weight_source": weight_source_for_year(anchor),
            "level_raw": 1.0,
        }
    ]
    for year in years:
        source = weight_source_for_year(year)
        weights = interpolate_weights(weights_by_year, year, categories)
        growth = 0.0
        for category in categories:
            key = CATEGORY_CPI[category]
            growth += float(weights[category]) * float(cpi.loc[year, key] / cpi.loc[year - 1, key])
        levels[year] = levels[year - 1] * growth
        records.append(
            {
                "year": year,
                "share_year": int(source) if source.isdigit() else pd.NA,
                "weight_source": source,
                "level_raw": levels[year],
            }
        )
    frame = pd.DataFrame(records)
    base = float(frame.loc[frame["year"] == BASE_YEAR, "level_raw"].iloc[0])
    frame["index"] = frame["level_raw"] / base * 100.0
    return frame


def _decorate(frame: pd.DataFrame, cpi: pd.DataFrame, n_months: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    all_items = cpi["all_items"]
    base = float(all_items.loc[BASE_YEAR])
    out["all_items_index"] = out["year"].map(lambda y: float(all_items.loc[int(y)] / base * 100.0))
    out["cum_from_2019"] = out["index"] / 100.0 - 1.0
    out["all_items_cum_from_2019"] = out["all_items_index"] / 100.0 - 1.0
    out["cum_gap"] = out["cum_from_2019"] - out["all_items_cum_from_2019"]
    out["n_months"] = out["year"].map(n_months)
    out["partial_year"] = out["n_months"] < 12
    return out


def build_indexes(shares: pd.DataFrame, cpi: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Fixed-basket Lowe indexes and a chained updated-basket index, from 2016 onward."""
    n_months = (
        annual.loc[annual["category"] == "all_items", ["year", "n_months"]]
        .drop_duplicates()
        .set_index("year")["n_months"]
    )
    rows: list[pd.DataFrame] = []
    for basket, categories in BASKETS.items():
        lookup = _weights_lookup(shares, basket)
        groups = sorted({(g, grp) for (g, grp, _y) in lookup})
        for grouping, group in groups:
            vintages = {}
            for year in (2017, 2019, 2023):
                key = (grouping, group, year)
                if key not in lookup:
                    continue
                vintages[year] = lookup[key]
            for share_year, weights in vintages.items():
                level = _index_from_weights(weights, cpi, categories)
                block = level.rename("index").reset_index()
                if "year" not in block.columns:
                    block = block.rename(columns={block.columns[0]: "year"})
                block["basket"] = basket
                block["path"] = "fixed"
                block["grouping"] = grouping
                block["group"] = group
                block["share_year"] = int(share_year)
                block["weight_source"] = str(int(share_year))
                rows.append(block)
            if all(year in vintages for year in SURVEY_YEARS):
                chained = _chain_updated(vintages, cpi, categories)
                chained["basket"] = basket
                chained["path"] = "updated"
                chained["grouping"] = grouping
                chained["group"] = group
                rows.append(
                    chained[
                        [
                            "year",
                            "index",
                            "basket",
                            "path",
                            "grouping",
                            "group",
                            "share_year",
                            "weight_source",
                        ]
                    ]
                )
    raw = pd.concat(rows, ignore_index=True)
    # Keep the lag year so the first published YoY is defined, then drop it.
    decorated = _decorate(raw, cpi, n_months)
    # Fixed paths: pct_change must be within share_year. Updated: within the path.
    # _decorate used one rule. Recompute YoY separately to keep both correct.
    decorated = _recompute_yoy(decorated)
    out = decorated[decorated["year"] >= PUBLISH_FROM].copy()
    out["grouping"] = pd.Categorical(out["grouping"], categories=list(GROUPING_ORDER), ordered=True)
    # Fixed vintages stay together. The updated path is one chronological series,
    # including years whose basket is a blend and so has no single survey year.
    out["_vintage"] = out["share_year"]
    out.loc[out["path"] == "updated", "_vintage"] = 0
    out = out.sort_values(
        ["basket", "path", "grouping", "group", "_vintage", "year"]
    ).drop(columns="_vintage").reset_index(drop=True)
    _validate_indexes(out)
    return out


def _recompute_yoy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pieces = []
    for path, g in out.groupby("path", sort=False):
        if path == "fixed":
            keys = ["basket", "grouping", "group", "share_year"]
            g = g.sort_values(keys + ["year"])
        else:
            keys = ["basket", "grouping", "group"]
            g = g.sort_values(keys + ["year"])
        g = g.copy()
        g["inflation"] = g.groupby(keys, sort=False)["index"].pct_change()
        g["all_items_inflation"] = g.groupby(keys, sort=False)["all_items_index"].pct_change()
        g["inflation_gap"] = g["inflation"] - g["all_items_inflation"]
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def _validate_indexes(indexes: pd.DataFrame) -> None:
    fixed_base = indexes[(indexes["path"] == "fixed") & (indexes["year"] == BASE_YEAR)]
    if not np.allclose(fixed_base["index"].to_numpy(dtype=float), 100.0, atol=1e-6):
        raise AssertionError("Fixed-basket index is not 100 in 2019")
    updated_base = indexes[(indexes["path"] == "updated") & (indexes["year"] == BASE_YEAR)]
    if updated_base.empty or not np.allclose(updated_base["index"].to_numpy(dtype=float), 100.0, atol=1e-6):
        raise AssertionError("Updated-basket index is not 100 in 2019")
    if indexes["inflation"].isna().any():
        missing = indexes.loc[indexes["inflation"].isna(), ["path", "grouping", "group", "year"]].head()
        raise AssertionError(f"Inflation rate missing for some published years:\n{missing}")


def basket_change(shares: pd.DataFrame) -> pd.DataFrame:
    """Category shares at each SHS wave, and the percentage-point change between waves."""
    sub = shares[shares["survey_year"].isin(SURVEY_YEARS)].copy()
    wide = sub.pivot(
        index=["basket", "grouping", "group", "category"],
        columns="survey_year",
        values="share",
    )
    missing = [year for year in SURVEY_YEARS if year not in wide.columns]
    if missing:
        raise KeyError(f"Share file is missing survey years {missing}")
    wide = wide.rename(columns={2017: "share_2017", 2019: "share_2019", 2023: "share_2023"})
    wide = wide.reset_index()
    if wide[["share_2017", "share_2019", "share_2023"]].isna().any().any():
        raise AssertionError("A group is missing a category share in one survey wave")
    wide["pp_2017_2019"] = (wide["share_2019"] - wide["share_2017"]) * 100.0
    wide["pp_2019_2023"] = (wide["share_2023"] - wide["share_2019"]) * 100.0
    wide["pp_2017_2023"] = (wide["share_2023"] - wide["share_2017"]) * 100.0
    wide["grouping"] = pd.Categorical(wide["grouping"], categories=list(GROUPING_ORDER), ordered=True)
    return wide.sort_values(["basket", "grouping", "group", "category"]).reset_index(drop=True)


def basket_shift(change: pd.DataFrame) -> pd.DataFrame:
    """Share of the basket that moved between waves, in percentage points.

    ``shift_pp`` is ``0.5 * sum(|change|)``. Two percentage points means two
    percent of consumption moved from one category to another.
    """
    intervals = (
        ("2017_2019", "pp_2017_2019"),
        ("2019_2023", "pp_2019_2023"),
        ("2017_2023", "pp_2017_2023"),
    )
    rows = []
    keys = ["basket", "grouping", "group"]
    for key, g in change.groupby(keys, sort=False, observed=False):
        basket, grouping, group = key
        for interval, col in intervals:
            rows.append(
                {
                    "basket": basket,
                    "grouping": grouping,
                    "group": group,
                    "interval": interval,
                    "shift_pp": 0.5 * float(g[col].abs().sum()),
                }
            )
    out = pd.DataFrame(rows)
    out["grouping"] = pd.Categorical(out["grouping"], categories=list(GROUPING_ORDER), ordered=True)
    interval_order = ["2017_2019", "2019_2023", "2017_2023"]
    out["interval"] = pd.Categorical(out["interval"], categories=interval_order, ordered=True)
    return out.sort_values(["basket", "grouping", "group", "interval"]).reset_index(drop=True)


def interpolated_shares(shares: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Yearly category shares. Survey years are exact; other years are linear blends."""
    rows: list[dict] = []
    calendar = [int(year) for year in years]
    for basket, categories in BASKETS.items():
        lookup = _weights_lookup(shares, basket)
        groups = sorted({(g, grp) for (g, grp, _y) in lookup})
        for grouping, group in groups:
            vintages = {}
            complete = True
            for year in SURVEY_YEARS:
                key = (grouping, str(group), year)
                if key not in lookup:
                    complete = False
                    break
                vintages[year] = lookup[key]
            if not complete:
                continue
            for year in calendar:
                weights = interpolate_weights(vintages, year, categories)
                source = weight_source_for_year(year)
                for category in categories:
                    rows.append(
                        {
                            "basket": basket,
                            "grouping": grouping,
                            "group": group,
                            "year": year,
                            "category": category,
                            "share": float(weights[category]),
                            "weight_source": source,
                        }
                    )
    out = pd.DataFrame(rows)
    totals = out.groupby(["basket", "grouping", "group", "year"], sort=False)["share"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1e-8):
        raise AssertionError("Interpolated yearly shares do not sum to 1")
    out["grouping"] = pd.Categorical(out["grouping"], categories=list(GROUPING_ORDER), ordered=True)
    return out.sort_values(["basket", "grouping", "group", "year", "category"]).reset_index(drop=True)


def relative_inflation_table(
    households: pd.DataFrame,
    cpi: pd.DataFrame,
    annual: pd.DataFrame,
) -> pd.DataFrame:
    """Updated primary-basket inflation for age, mortgage status, and consumption decile.

    Deciles are the within-year ranks of all households. A cell is included when
    it has a basket in 2017, 2019, and 2023, so the yearly series can be chained.
    ``relative_inflation`` is that cell's inflation rate, the same object as
    ``inflation`` on the updated path. ``cum_rel_inflation`` is the percent
    change in that cell's price index since 2019.
    """
    categories = BASKETS["primary"]
    work = households.loc[households["complete_primary"]].dropna(
        subset=["age_band", "tenure", "cons_d"]
    ).copy()
    work["_w"] = work["weight"].to_numpy(dtype=float)
    lookup: dict[tuple, pd.Series] = {}
    grouped = work.groupby(["year", "age_band", "tenure", "cons_d"], sort=False)
    for (year, age, tenure, decile), g in grouped:
        weight = g["_w"].to_numpy(dtype=float)
        total = float(np.sum(weight * g["tc001"].to_numpy(dtype=float)))
        if total <= 0:
            continue
        shares = {
            category: float(np.sum(weight * g[category].to_numpy(dtype=float)) / total)
            for category in categories
        }
        series = pd.Series(shares, dtype=float)
        if not np.isclose(float(series.sum()), 1.0, atol=1e-8):
            raise AssertionError(
                f"Shares for {age}, {tenure}, {decile}, {year} sum to {float(series.sum())}"
            )
        lookup[(str(age), str(tenure), str(decile), int(year))] = series

    n_months = (
        annual.loc[annual["category"] == "all_items", ["year", "n_months"]]
        .drop_duplicates()
        .set_index("year")["n_months"]
    )
    rows: list[pd.DataFrame] = []
    cells = sorted({(age, tenure, decile) for age, tenure, decile, _year in lookup})
    for age, tenure, decile in cells:
        vintages = {}
        complete = True
        for year in SURVEY_YEARS:
            key = (age, tenure, decile, year)
            if key not in lookup:
                complete = False
                break
            vintages[year] = lookup[key]
        if not complete:
            continue
        chained = _chain_updated(vintages, cpi, categories)
        chained["age_group"] = age
        chained["mortgage_status"] = TENURE_PRETTY.get(tenure, tenure)
        chained["consumption_decile"] = decile
        chained["basket"] = "primary"
        chained["path"] = "updated"
        chained["grouping"] = "age_tenure_cons_d"
        chained["group"] = f"{age}|{tenure}|{decile}"
        rows.append(chained)
    if not rows:
        raise AssertionError("No age, tenure, and decile cell has a basket in all three survey years")
    raw = pd.concat(rows, ignore_index=True)
    decorated = _recompute_yoy(_decorate(raw, cpi, n_months))
    out = decorated[decorated["year"] >= PUBLISH_FROM].copy()
    if out["inflation"].isna().any():
        raise AssertionError("Relative inflation is missing for a published year")
    out["relative_inflation"] = out["inflation"]
    out["cum_rel_inflation"] = out["cum_from_2019"]
    out["age_group"] = pd.Categorical(out["age_group"], categories=list(AGE_ORDER), ordered=True)
    out["mortgage_status"] = pd.Categorical(
        out["mortgage_status"],
        categories=[TENURE_PRETTY[code] for code in TENURE_ORDER],
        ordered=True,
    )
    out["consumption_decile"] = pd.Categorical(
        out["consumption_decile"], categories=list(CONS_DECILE_ORDER), ordered=True
    )
    columns = [
        "age_group",
        "mortgage_status",
        "consumption_decile",
        "year",
        "relative_inflation",
        "cum_rel_inflation",
        "partial_year",
    ]
    return out.sort_values(columns[:4]).reset_index(drop=True)[columns]


def decile_differentials(indexes: pd.DataFrame) -> pd.DataFrame:
    """Updated primary-basket gaps for consumption deciles.

    ``cum_gap`` is versus all-items. ``cum_gap_vs_all_households`` subtracts the
    all-household SHS basket. A ``D10-D01`` row is the top-minus-bottom spread.
    """
    base = indexes[
        (indexes["basket"] == "primary")
        & (indexes["path"] == "updated")
        & (indexes["grouping"].isin(["cons_d", "all"]))
    ].copy()
    base["group"] = base["group"].astype(str)
    rows = []
    for year, g in base.groupby("year"):
        all_row = g[(g["grouping"] == "all") & (g["group"] == "all")]
        if all_row.empty:
            continue
        all_gap = float(all_row["cum_gap"].iloc[0])
        all_inf = float(all_row["inflation"].iloc[0])
        dec = g[g["grouping"] == "cons_d"].set_index("group")
        for group, row in dec.iterrows():
            rows.append(
                {
                    "year": int(year),
                    "group": group,
                    "inflation": float(row["inflation"]),
                    "cum_gap": float(row["cum_gap"]),
                    "inflation_vs_all_households": float(row["inflation"]) - all_inf,
                    "cum_gap_vs_all_households": float(row["cum_gap"]) - all_gap,
                    "partial_year": bool(row["partial_year"]),
                }
            )
        if "D10" in dec.index and "D01" in dec.index:
            rows.append(
                {
                    "year": int(year),
                    "group": "D10-D01",
                    "inflation": float(dec.loc["D10", "inflation"] - dec.loc["D01", "inflation"]),
                    "cum_gap": float(dec.loc["D10", "cum_gap"] - dec.loc["D01", "cum_gap"]),
                    "inflation_vs_all_households": float(
                        dec.loc["D10", "inflation"] - dec.loc["D01", "inflation"]
                    ),
                    "cum_gap_vs_all_households": float(
                        dec.loc["D10", "cum_gap"] - dec.loc["D01", "cum_gap"]
                    ),
                    "partial_year": bool(dec.loc["D01", "partial_year"]),
                }
            )
    out = pd.DataFrame(rows)
    order = [f"D{i:02d}" for i in range(1, 11)] + ["D10-D01"]
    out["group"] = pd.Categorical(out["group"], categories=order, ordered=True)
    return out.sort_values(["year", "group"]).reset_index(drop=True)
