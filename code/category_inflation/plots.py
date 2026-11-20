"""Figures for SHS basket shares and group inflation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import time

import matplotlib.pyplot as plt
import pandas as pd

from .baskets import (
    AGE_ORDER,
    CATEGORY_LABELS,
    CONS_DECILE_ORDER,
    PRIMARY_CATEGORIES,
    TENURE_ORDER,
    TENURE_PRETTY,
)
from .indexes import basket_change, basket_shift

_STACK_COLORS = {
    "food": "#4C78A8",
    "shelter": "#F58518",
    "household": "#54A24B",
    "clothing": "#E45756",
    "transportation": "#72B7B2",
    "health_personal": "#B279A2",
    "recreation_education": "#EECA3B",
    "alcohol_tobacco": "#9D755D",
    "residual": "#BAB0AC",
}

_QUINTILE_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#08519c", "#08306b"]
_TENURE_COLORS = {
    "owner_mortgage": "#F58518",
    "owner_free_clear": "#4C78A8",
    "renter": "#E45756",
}


def _save_fig(fig, path: Path, **kwargs) -> None:
    """Write a PNG, retrying if Dropbox briefly locks the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(6):
        try:
            fig.savefig(path, dpi=120, **kwargs)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.4)
    if last_error is not None:
        raise last_error


def latest_month(frame: pd.DataFrame) -> str:
    """Latest YYYY-MM in a monthly index or cell table."""
    if "date" not in frame.columns or frame.empty:
        raise ValueError("No monthly date in the index file")
    return str(pd.to_datetime(frame["date"]).max().strftime("%Y-%m"))


def _as_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values)


def _ordered_groups(grouping: str, groups: list[str]) -> list[str]:
    if grouping == "inc_q":
        order = [f"Q{i}" for i in range(1, 6)]
    elif grouping == "cons_d":
        order = list(CONS_DECILE_ORDER)
    elif grouping == "age_band":
        order = list(AGE_ORDER)
    elif grouping == "tenure":
        order = list(TENURE_ORDER)
    else:
        order = sorted(groups)
    return [g for g in order if g in groups]


def _pretty(grouping: str, group: str) -> str:
    if grouping == "tenure":
        return TENURE_PRETTY.get(group, group)
    if grouping == "all":
        return "All households"
    return group


def _share_wide(
    shares: pd.DataFrame,
    *,
    basket: str,
    grouping: str,
    survey_year: int,
    categories: tuple[str, ...],
) -> pd.DataFrame:
    sub = shares[
        (shares["basket"] == basket)
        & (shares["grouping"] == grouping)
        & (shares["survey_year"] == int(survey_year))
    ]
    wide = sub.pivot(index="group", columns="category", values="share")
    wide = wide.reindex(index=_ordered_groups(grouping, list(wide.index)))
    return wide.reindex(columns=list(categories))


def _stacked_shares(wide: pd.DataFrame, title: str, path: Path, figsize: tuple[float, float] = (9.5, 5.2)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    bottom = None
    x = range(len(wide.index))
    for category in wide.columns:
        values = wide[category].to_numpy(dtype=float) * 100.0
        ax.bar(
            list(x),
            values,
            bottom=0 if bottom is None else bottom,
            color=_STACK_COLORS.get(category, "#999999"),
            width=0.72,
            label=CATEGORY_LABELS.get(category, category),
        )
        bottom = values if bottom is None else bottom + values
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(wide.index))
    ax.set_ylabel("Share of consumption (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _line_by_group(
    indexes: pd.DataFrame,
    *,
    basket: str,
    path_name: str,
    grouping: str,
    value: str,
    ylabel: str,
    title: str,
    dest: Path,
    scale: float = 100.0,
    zero_line: bool = False,
    figsize: tuple[float, float] = (9.5, 5.2),
) -> None:
    sub = indexes[
        (indexes["basket"] == basket)
        & (indexes["path"] == path_name)
        & (indexes["grouping"] == grouping)
    ].copy()
    if path_name == "fixed":
        raise ValueError("Line charts are for the updated path")
    groups = _ordered_groups(grouping, list(sub["group"].unique()))
    fig, ax = plt.subplots(figsize=figsize)
    if grouping == "inc_q":
        colors = dict(zip(groups, _QUINTILE_COLORS))
    elif grouping == "cons_d":
        cmap = plt.get_cmap("viridis")
        colors = {group: cmap(i / max(len(groups) - 1, 1)) for i, group in enumerate(groups)}
    elif grouping == "tenure":
        colors = _TENURE_COLORS
    else:
        colors = {}
    for i, group in enumerate(groups):
        g = sub[sub["group"] == group].sort_values("date")
        ax.plot(
            _as_dates(g["date"]),
            g[value] * scale,
            label=_pretty(grouping, group),
            color=colors.get(group),
            lw=2,
        )
    if value == "inflation":
        ref = sub[sub["group"] == groups[0]].sort_values("date")
        ax.plot(
            _as_dates(ref["date"]),
            ref["all_items_inflation"] * scale,
            color="black",
            lw=1.6,
            ls="--",
            label="All-items CPI",
        )
    if zero_line:
        ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Month")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    _save_fig(fig, dest)
    plt.close(fig)


def _fixed_vintage_bars(indexes: pd.DataFrame, month: str, dest: Path) -> None:
    sub = indexes[
        (indexes["basket"] == "primary")
        & (indexes["path"] == "fixed")
        & (indexes["grouping"] == "inc_q")
        & (indexes["date"] == month)
    ].copy()
    vintages = [2017, 2019, 2023]
    groups = _ordered_groups("inc_q", list(sub["group"].unique()))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = 0.24
    x = list(range(len(groups)))
    for i, vintage in enumerate(vintages):
        g = sub[sub["share_year"] == vintage].set_index("group").reindex(groups)
        ax.bar(
            [v + (i - 1) * width for v in x],
            g["cum_gap"] * 100.0,
            width=width,
            label=f"{vintage} basket",
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Cumulative gap vs all-items (pp)")
    ax.set_title(f"Income-quintile gap from 2019 to {month}, three fixed baskets")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, dest)
    plt.close(fig)


def _tenure_shelter_compare(indexes: pd.DataFrame, dest: Path) -> None:
    sub = indexes[
        (indexes["path"] == "updated")
        & (indexes["grouping"] == "tenure")
        & (indexes["basket"].isin(["primary", "shelter_split"]))
    ].copy()
    groups = _ordered_groups("tenure", list(sub["group"].unique()))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    for ax, basket, title in zip(
        axes,
        ("primary", "shelter_split"),
        ("Single shelter index", "Shelter split"),
    ):
        for group in groups:
            g = sub[(sub["basket"] == basket) & (sub["group"] == group)].sort_values("date")
            ax.plot(
                _as_dates(g["date"]),
                g["cum_gap"] * 100.0,
                color=_TENURE_COLORS[group],
                lw=2,
                label=_pretty("tenure", group),
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Month")
    axes[0].set_ylabel("Cumulative gap vs all-items (pp)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Updated basket, by tenure", y=1.02)
    fig.tight_layout()
    _save_fig(fig, dest, bbox_inches="tight")
    plt.close(fig)


def _shelter_by_income_tenure(shares: pd.DataFrame, survey_year: int, dest: Path) -> None:
    sub = shares[
        (shares["basket"] == "primary")
        & (shares["grouping"] == "inc_q_x_tenure")
        & (shares["survey_year"] == int(survey_year))
        & (shares["category"] == "shelter")
    ].copy()
    parts = sub["group"].str.split(" | ", n=1, regex=False, expand=True)
    sub["inc_q"] = parts[0]
    sub["tenure"] = parts[1]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    quintiles = [f"Q{i}" for i in range(1, 6)]
    width = 0.24
    x = list(range(len(quintiles)))
    for i, tenure in enumerate(TENURE_ORDER):
        g = sub[sub["tenure"] == tenure].set_index("inc_q").reindex(quintiles)
        ax.bar(
            [v + (i - 1) * width for v in x],
            g["share"] * 100.0,
            width=width,
            color=_TENURE_COLORS[tenure],
            label=TENURE_PRETTY[tenure],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(quintiles)
    ax.set_ylabel("Shelter share of consumption (%)")
    ax.set_title(f"Shelter share by income quintile and tenure, {survey_year}")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, dest)
    plt.close(fig)


def _decile_gap_bars(indexes: pd.DataFrame, month: str, dest: Path) -> None:
    sub = indexes[
        (indexes["basket"] == "primary")
        & (indexes["path"] == "updated")
        & (indexes["grouping"] == "cons_d")
        & (indexes["date"] == month)
    ].copy()
    groups = _ordered_groups("cons_d", list(sub["group"].astype(str).unique()))
    g = sub.set_index("group").reindex(groups)
    all_row = indexes[
        (indexes["basket"] == "primary")
        & (indexes["path"] == "updated")
        & (indexes["grouping"] == "all")
        & (indexes["group"] == "all")
        & (indexes["date"] == month)
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = [plt.get_cmap("viridis")(i / 9) for i in range(len(groups))]
    ax.bar(groups, g["cum_gap"].to_numpy(dtype=float) * 100.0, color=colors)
    if not all_row.empty:
        ax.axhline(
            float(all_row["cum_gap"].iloc[0]) * 100.0,
            color="black",
            ls="--",
            lw=1.4,
            label="All households",
        )
        ax.legend(frameon=False)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative gap vs all-items (pp)")
    ax.set_title(f"Consumption-decile gap from 2019 to {month}")
    fig.tight_layout()
    _save_fig(fig, dest)
    plt.close(fig)


_SHELTER_COMPONENT_COLORS = {
    "rent": "#E45756",
    "owned": "#B279A2",
    "water_fuel": "#54A24B",
    "shelter": "#F58518",
}


_PRODUCT_PANELS = (
    (
        (
            "food",
            "shelter",
            "household",
            "clothing",
            "transportation",
            "health_personal",
            "recreation_education",
            "alcohol_tobacco",
        ),
        "Basket products",
        _STACK_COLORS,
    ),
    (
        ("rent", "owned", "water_fuel", "shelter"),
        "Shelter components",
        _SHELTER_COMPONENT_COLORS,
    ),
)


def _plot_products(
    frame: pd.DataFrame,
    *,
    value: str,
    ylabel: str,
    title: str,
    dest: Path,
    zero_line: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 8.4), sharex=True)
    for ax, (categories, panel_title, colors) in zip(axes, _PRODUCT_PANELS):
        for category in categories:
            g = frame[frame["category"] == category].sort_values("date")
            ax.plot(
                _as_dates(g["date"]),
                g[value] * 100.0,
                color=colors[category],
                lw=1.4,
                label=CATEGORY_LABELS[category],
            )
        ref = frame[frame["category"] == "all_items"].sort_values("date")
        ax.plot(
            _as_dates(ref["date"]),
            ref[value] * 100.0,
            color="black",
            lw=1.6,
            ls="--",
            label="All-items CPI",
        )
        if zero_line:
            ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    axes[-1].set_xlabel("Month")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    _save_fig(fig, dest, bbox_inches="tight")
    plt.close(fig)


def _product_month_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    """Twelve-month product inflation and the change from each product's 2019 average."""
    long = monthly.stack().rename("index").reset_index()
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["category", "date"])
    long["yoy"] = long.groupby("category", sort=False)["index"].pct_change(12)
    base = long.loc[long["date"].dt.year == 2019].groupby("category")["index"].mean()
    long["cum"] = long["index"] / long["category"].map(base) - 1.0
    published = long[long["date"] >= "2016-01-01"].copy()
    published["date"] = published["date"].dt.strftime("%Y-%m")
    return published


def _product_inflation(monthly: pd.DataFrame, dest: Path) -> None:
    """Twelve-month CPI inflation for the products that price the baskets."""
    frame = _product_month_frame(monthly).dropna(subset=["yoy"])
    _plot_products(
        frame,
        value="yoy",
        ylabel="Twelve-month inflation (%)",
        title="CPI inflation by product",
        dest=dest,
    )


def _product_cumulative(monthly: pd.DataFrame, dest: Path) -> None:
    """Percent change in each product's monthly CPI from its 2019 average."""
    frame = _product_month_frame(monthly)
    frame = frame[frame["date"] >= "2019-01"].dropna(subset=["cum"])
    _plot_products(
        frame,
        value="cum",
        ylabel="Cumulative inflation since 2019 (%)",
        title="Cumulative CPI inflation by product since 2019",
        dest=dest,
        zero_line=True,
    )


def _wave_shares(shares: pd.DataFrame, dest: Path) -> None:
    """All-household primary shares at the three SHS waves."""
    sub = shares[
        (shares["basket"] == "primary")
        & (shares["grouping"] == "all")
        & (shares["survey_year"].isin([2017, 2019, 2023]))
    ]
    categories = list(PRIMARY_CATEGORIES)
    colors = {2017: "#9ecae1", 2019: "#3182bd", 2023: "#08519c"}
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    x = list(range(len(categories)))
    width = 0.24
    for i, year in enumerate((2017, 2019, 2023)):
        g = sub[sub["survey_year"] == year].set_index("category").reindex(categories)
        ax.bar(
            [v + (i - 1) * width for v in x],
            g["share"].to_numpy(dtype=float) * 100.0,
            width=width,
            color=colors[year],
            label=str(year),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories], rotation=30, ha="right")
    ax.set_ylabel("Share of consumption (%)")
    ax.set_title("All-household consumption shares by SHS wave")
    ax.legend(frameon=False, title="Survey year")
    fig.tight_layout()
    _save_fig(fig, dest)
    plt.close(fig)


def _basket_shift_bars(shift: pd.DataFrame, grouping: str, title: str, dest: Path, figsize: tuple[float, float]) -> None:
    sub = shift[
        (shift["basket"] == "primary")
        & (shift["grouping"] == grouping)
        & (shift["interval"].isin(["2017_2019", "2019_2023"]))
    ].copy()
    groups = _ordered_groups(grouping, [str(g) for g in sub["group"].unique()])
    labels = {"2017_2019": "2017–2019", "2019_2023": "2019–2023"}
    colors = {"2017_2019": "#9ecae1", "2019_2023": "#08519c"}
    fig, ax = plt.subplots(figsize=figsize)
    x = list(range(len(groups)))
    width = 0.36
    for i, interval in enumerate(("2017_2019", "2019_2023")):
        g = sub[sub["interval"] == interval].set_index("group")
        g.index = g.index.astype(str)
        ax.bar(
            [v + (i - 0.5) * width for v in x],
            [float(g.loc[group, "shift_pp"]) for group in groups],
            width=width,
            color=colors[interval],
            label=labels[interval],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty(grouping, group) for group in groups])
    ax.set_ylabel("Share of the basket that moved (pp)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, dest)
    plt.close(fig)


def _cell_grid(table: pd.DataFrame):
    ages = [age for age in AGE_ORDER if age in set(table["age_group"].astype(str))]
    tenures = [TENURE_PRETTY[code] for code in TENURE_ORDER if TENURE_PRETTY[code] in set(table["mortgage_status"].astype(str))]
    deciles = [decile for decile in CONS_DECILE_ORDER if decile in set(table["consumption_decile"].astype(str))]
    cmap = plt.get_cmap("viridis")
    colors = {decile: cmap(i / max(len(deciles) - 1, 1)) for i, decile in enumerate(deciles)}
    return ages, tenures, deciles, colors


def _relative_line_panels(
    table: pd.DataFrame,
    *,
    value: str,
    ylabel: str,
    title: str,
    dest: Path,
    from_year: int | None = None,
    zero_line: bool = False,
) -> None:
    frame = table.copy()
    if from_year is not None:
        frame = frame[frame["date"] >= f"{int(from_year):04d}-01"]
    ages, tenures, deciles, colors = _cell_grid(frame)
    fig, axes = plt.subplots(len(ages), len(tenures), figsize=(12.6, 15.2), sharex=True, sharey=True)
    for i, age in enumerate(ages):
        for j, tenure in enumerate(tenures):
            ax = axes[i, j]
            sub = frame[(frame["age_group"].astype(str) == age) & (frame["mortgage_status"].astype(str) == tenure)]
            for decile in deciles:
                g = sub[sub["consumption_decile"].astype(str) == decile].sort_values("date")
                if g.empty:
                    continue
                ax.plot(_as_dates(g["date"]), g[value] * 100.0, color=colors[decile], lw=1.2, label=decile)
            if zero_line:
                ax.axhline(0, color="black", lw=0.6)
            if i == 0:
                ax.set_title(tenure)
            if j == 0:
                ax.set_ylabel(age)
    handles = [
        plt.Line2D([0], [0], color=colors[decile], lw=1.6, label=decile) for decile in deciles
    ]
    for ax in axes[-1, :]:
        ax.set_xlabel("Month")
    fig.suptitle(title)
    fig.supylabel(ylabel)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.97))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(deciles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    _save_fig(fig, dest, bbox_inches="tight")
    plt.close(fig)


def _relative_cum_bars(table: pd.DataFrame, dest: Path) -> None:
    end = latest_month(table)
    frame = table[table["date"] == end].copy()
    ages, tenures, deciles, colors = _cell_grid(frame)
    fig, axes = plt.subplots(len(ages), len(tenures), figsize=(12.6, 15.2), sharey=True)
    for i, age in enumerate(ages):
        for j, tenure in enumerate(tenures):
            ax = axes[i, j]
            sub = frame[(frame["age_group"].astype(str) == age) & (frame["mortgage_status"].astype(str) == tenure)]
            sub = sub.set_index(sub["consumption_decile"].astype(str))
            xs = [k for k, decile in enumerate(deciles) if decile in sub.index]
            heights = [float(sub.loc[decile, "cum_rel_inflation"]) * 100.0 for decile in deciles if decile in sub.index]
            ax.bar(xs, heights, color=[colors[deciles[k]] for k in xs])
            ax.set_xticks(list(range(len(deciles))))
            ax.set_xticklabels(deciles, rotation=90, fontsize=7)
            if i == 0:
                ax.set_title(tenure)
            if j == 0:
                ax.set_ylabel(age)
    fig.suptitle(f"Cumulative inflation since 2019, by {end}")
    fig.supylabel("Cumulative inflation since 2019 (%)")
    fig.tight_layout(rect=(0.02, 0.0, 1, 0.97))
    _save_fig(fig, dest, bbox_inches="tight")
    plt.close(fig)


def make_relative_figures(table: pd.DataFrame, dest: Path) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _relative_line_panels(
        table,
        value="relative_inflation",
        ylabel="Twelve-month inflation (%)",
        title="Inflation by age, mortgage status, and consumption decile",
        dest=dest / "relative_inflation_panels.png",
    )
    _relative_line_panels(
        table,
        value="cum_rel_inflation",
        ylabel="Cumulative inflation since 2019 (%)",
        title="Cumulative inflation since 2019 by age, mortgage status, and consumption decile",
        dest=dest / "cum_rel_inflation_panels.png",
        from_year=2019,
        zero_line=True,
    )
    _relative_cum_bars(table, dest / "cum_rel_inflation_bars.png")


def make_figures(
    shares: pd.DataFrame,
    indexes: pd.DataFrame,
    dest: Path,
    monthly: pd.DataFrame,
) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for grouping, name, title_group, figsize in (
        ("inc_q", "income", "income quintile", (9.5, 5.2)),
        ("tenure", "tenure", "tenure", (9.5, 5.2)),
        ("age_band", "age", "age", (9.5, 5.2)),
        ("cons_d", "consumption_decile", "consumption decile", (11.0, 5.6)),
    ):
        wide = _share_wide(
            shares,
            basket="primary",
            grouping=grouping,
            survey_year=2019,
            categories=PRIMARY_CATEGORIES,
        )
        wide.index = [_pretty(grouping, g) for g in wide.index]
        _stacked_shares(
            wide,
            f"Consumption shares by {title_group}, SHS 2019",
            dest / f"shares_{name}_2019.png",
            figsize=figsize,
        )
        _line_by_group(
            indexes,
            basket="primary",
            path_name="updated",
            grouping=grouping,
            value="inflation",
            ylabel="Twelve-month inflation (%)",
            title=f"Basket inflation by {title_group}",
            dest=dest / f"inflation_{name}_updated.png",
            figsize=figsize,
        )
        _line_by_group(
            indexes,
            basket="primary",
            path_name="updated",
            grouping=grouping,
            value="cum_gap",
            ylabel="Cumulative gap vs all-items (pp)",
            title=f"Cumulative gap vs all-items since 2019, by {title_group}",
            dest=dest / f"cumgap_{name}_updated.png",
            zero_line=True,
            figsize=figsize,
        )
    end = latest_month(indexes)
    _product_inflation(monthly, dest / "inflation_by_product.png")
    _product_cumulative(monthly, dest / "cum_inflation_by_product.png")
    _fixed_vintage_bars(indexes, end, dest / "cumgap_income_fixed_vintages.png")
    _tenure_shelter_compare(indexes, dest / "cumgap_tenure_shelter_split.png")
    _shelter_by_income_tenure(shares, 2019, dest / "shelter_share_income_tenure_2019.png")
    _decile_gap_bars(indexes, end, dest / "cumgap_consumption_decile_bar.png")
    _wave_shares(shares, dest / "shares_all_waves.png")
    shift = basket_shift(basket_change(shares))
    for grouping, name, title, figsize in (
        ("inc_q", "income", "Basket shift by income quintile", (9.5, 5.2)),
        ("cons_d", "consumption_decile", "Basket shift by consumption decile", (11.0, 5.6)),
        ("age_band", "age", "Basket shift by age", (9.5, 5.2)),
        ("tenure", "tenure", "Basket shift by tenure", (9.5, 5.2)),
    ):
        _basket_shift_bars(shift, grouping, title, dest / f"basket_shift_{name}.png", figsize)
