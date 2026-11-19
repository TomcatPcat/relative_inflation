"""Build category CPI, SHS baskets, and group inflation rates.

Run from anywhere:

    python code/category_inflation/run_category_inflation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow `python path/to/run_category_inflation.py` without an installed package.
_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from category_inflation.baskets import concordance_rows
from category_inflation.cpi import build_cpi_annual, cpi_wide, output_dir
from category_inflation.indexes import (
    basket_change,
    basket_shift,
    build_indexes,
    decile_differentials,
    expenditure_shares,
    interpolated_shares,
    relative_inflation_table,
)
from category_inflation.plots import latest_full_year, make_figures, make_relative_figures
from category_inflation.shs_extract import (
    adding_up_report,
    cc_band_check,
    cc_spend_summary,
    load_shs_categories,
    save_household_extract,
)


def main() -> dict[str, pd.DataFrame]:
    out = output_dir()
    print("Building annual category CPI...")
    annual = build_cpi_annual()
    prices = cpi_wide(annual)
    print(
        f"  CPI years {int(prices.index.min())}-{int(prices.index.max())}, "
        f"partial years: {sorted(annual.loc[annual['partial_year'], 'year'].unique().tolist())}"
    )

    print("Loading SHS 2017, 2019, 2023 (nominal dollars)...")
    households = load_shs_categories()
    extract_path = save_household_extract(households)
    counts = households.groupby("year").size()
    print("  households:", ", ".join(f"{int(y)}={int(n)}" for y, n in counts.items()))
    print("  extract:", extract_path)

    check = adding_up_report(households)
    check_path = out / "adding_up_by_year.csv"
    check.to_csv(check_path, index=False)
    print(check.to_string(index=False))
    identity = check["max_abs_dollar_gap_vs_identity"].max()
    if identity > 0.05:
        raise AssertionError(f"Mapped categories plus residual do not rebuild TC001 (max gap {identity})")

    concordance = pd.DataFrame(concordance_rows())
    concordance_path = out / "concordance.csv"
    concordance.to_csv(concordance_path, index=False)

    print("Computing expenditure shares and group indexes...")
    shares = expenditure_shares(households)
    shares.to_csv(out / "shares_by_group.csv", index=False)
    change = basket_change(shares)
    change.to_csv(out / "basket_change.csv", index=False)
    shift = basket_shift(change)
    shift.to_csv(out / "basket_shift.csv", index=False)
    yearly = interpolated_shares(shares, [int(year) for year in prices.index if int(year) >= 2016])
    yearly.to_csv(out / "shares_interpolated.csv", index=False)
    indexes = build_indexes(shares, prices, annual)
    indexes.to_csv(out / "group_inflation.csv", index=False)
    deciles = decile_differentials(indexes)
    deciles.to_csv(out / "decile_differentials.csv", index=False)
    relative = relative_inflation_table(households, prices, annual)
    relative.to_csv(out / "relative_inflation.csv", index=False)
    print(
        f"Relative inflation cells: {relative.groupby(['age_group', 'mortgage_status', 'consumption_decile'], observed=True).ngroups}"
    )

    bands = cc_band_check(households)
    bands.to_csv(out / "cc_income_band_check.csv", index=False)
    cc_summary = cc_spend_summary(households)
    cc_summary.to_csv(out / "cc_spend_by_group.csv", index=False)
    print("2023 imputed card share by income band:")
    print(bands.to_string(index=False))

    fig_dir = out / "figures"
    make_figures(shares, indexes, fig_dir, annual)
    make_relative_figures(relative, fig_dir)
    _print_headline(shares, indexes, deciles)
    print("Wrote", out)
    return {
        "annual": annual,
        "households": households,
        "adding_up": check,
        "concordance": concordance,
        "shares": shares,
        "basket_change": change,
        "basket_shift": shift,
        "shares_interpolated": yearly,
        "indexes": indexes,
        "deciles": deciles,
        "relative_inflation": relative,
        "cc_bands": bands,
        "cc_summary": cc_summary,
    }


def _print_headline(shares: pd.DataFrame, indexes: pd.DataFrame, deciles: pd.DataFrame) -> None:
    end = latest_full_year(indexes)
    focus = indexes[
        (indexes["basket"] == "primary")
        & (indexes["path"] == "updated")
        & (indexes["year"] == end)
        & (indexes["grouping"].isin(["inc_q", "cons_d", "tenure", "age_band", "all"]))
    ][["grouping", "group", "inflation", "cum_gap"]]
    focus = focus.sort_values(["grouping", "group"])
    print(f"Updated primary basket at {end} (inflation and cumulative gap vs all-items):")
    print(focus.to_string(index=False))
    shelter = shares[
        (shares["basket"] == "primary")
        & (shares["grouping"] == "all")
        & (shares["survey_year"] == 2019)
    ][["category", "share"]]
    print("All-household shares, 2019:")
    print(shelter.to_string(index=False))
    end_rows = deciles[(deciles["year"] == end) & (deciles["group"].astype(str).isin(["D01", "D10", "D10-D01"]))]
    print(f"Consumption-decile gaps at {end}:")
    print(end_rows.to_string(index=False))


if __name__ == "__main__":
    main()
