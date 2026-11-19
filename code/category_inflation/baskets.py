"""SHS expenditure baskets and their CPI counterparts.

Category dollars are nominal within each survey year. Residual spending is
priced with the all-items CPI. The shelter split replaces the single shelter
weight; other shelter is priced with the shelter index.
"""

from __future__ import annotations

# Internal CPI key -> Statistics Canada "Products and product groups" label
# in table 18-10-0004, Canada, UOM 2002=100.
CPI_PRODUCTS: dict[str, str] = {
    "all_items": "All-items",
    "food": "Food",
    "shelter": "Shelter",
    "rent": "Rented accommodation",
    "owned": "Owned accommodation",
    "water_fuel": "Water, fuel and electricity",
    "household": "Household operations, furnishings and equipment",
    "clothing": "Clothing and footwear",
    "transportation": "Transportation",
    "health_personal": "Health and personal care",
    "recreation_education": "Recreation, education and reading",
    "alcohol_tobacco": "Alcoholic beverages, tobacco products and recreational cannabis",
}

# Basket category -> CPI key used to price it.
CATEGORY_CPI: dict[str, str] = {
    "food": "food",
    "shelter": "shelter",
    "household": "household",
    "clothing": "clothing",
    "transportation": "transportation",
    "health_personal": "health_personal",
    "recreation_education": "recreation_education",
    "alcohol_tobacco": "alcohol_tobacco",
    "residual": "all_items",
    "rent": "rent",
    "owned": "owned",
    "water_fuel": "water_fuel",
    "other_shelter": "shelter",
}

CATEGORY_LABELS: dict[str, str] = {
    "food": "Food",
    "shelter": "Shelter",
    "household": "Household operations and furnishings",
    "clothing": "Clothing and footwear",
    "transportation": "Transportation",
    "health_personal": "Health and personal care",
    "recreation_education": "Recreation, education and reading",
    "alcohol_tobacco": "Alcohol, tobacco and cannabis",
    "residual": "Residual (all-items CPI)",
    "rent": "Rented accommodation",
    "owned": "Owned accommodation",
    "water_fuel": "Water, fuel and electricity",
    "other_shelter": "Other shelter",
}

# Order used in tables and stacked bars.
PRIMARY_CATEGORIES: tuple[str, ...] = (
    "food",
    "shelter",
    "household",
    "clothing",
    "transportation",
    "health_personal",
    "recreation_education",
    "alcohol_tobacco",
    "residual",
)

SHELTER_CATEGORIES: tuple[str, ...] = (
    "food",
    "rent",
    "owned",
    "water_fuel",
    "other_shelter",
    "household",
    "clothing",
    "transportation",
    "health_personal",
    "recreation_education",
    "alcohol_tobacco",
    "residual",
)

# Mapped parents that partition current consumption, before the residual.
MAPPED_CATEGORIES: tuple[str, ...] = tuple(
    c for c in PRIMARY_CATEGORIES if c != "residual"
)

BASKETS: dict[str, tuple[str, ...]] = {
    "primary": PRIMARY_CATEGORIES,
    "shelter_split": SHELTER_CATEGORIES,
}

def concordance_rows() -> list[dict[str, str]]:
    """Primary basket, then the shelter split (shared categories repeated)."""
    shelter_only = {row["category"] for row in _SHELTER_SPLIT_ROWS}
    rows = [dict(row) for row in _PRIMARY_ROWS]
    for row in _PRIMARY_ROWS:
        if row["category"] in shelter_only or row["category"] == "shelter":
            continue
        copied = dict(row)
        copied["basket"] = "shelter_split"
        copied["notes"] = "Same SHS weight as the primary basket. " + row["notes"]
        rows.append(copied)
    rows.extend(dict(row) for row in _SHELTER_SPLIT_ROWS)
    return rows


# Concordance rows: basket, category, cpi product, SHS sources, notes.
_PRIMARY_ROWS: tuple[dict[str, str], ...] = (
    {
        "basket": "primary",
        "category": "food",
        "cpi_product": CPI_PRODUCTS["food"],
        "shs_sources": "FD001",
        "notes": "Food expenditures.",
    },
    {
        "basket": "primary",
        "category": "shelter",
        "cpi_product": CPI_PRODUCTS["shelter"],
        "shs_sources": "SH001",
        "notes": "Shelter total, including rented and owned accommodation and utilities.",
    },
    {
        "basket": "primary",
        "category": "household",
        "cpi_product": CPI_PRODUCTS["household"],
        "shs_sources": "HO001 + HF001",
        "notes": "Household operations plus furnishings and equipment. Child care stays inside HO001.",
    },
    {
        "basket": "primary",
        "category": "clothing",
        "cpi_product": CPI_PRODUCTS["clothing"],
        "shs_sources": "CL030 (2017: CL001)",
        "notes": "Clothing and accessories. The 2017 diary names the aggregate CL001.",
    },
    {
        "basket": "primary",
        "category": "transportation",
        "cpi_product": CPI_PRODUCTS["transportation"],
        "shs_sources": "TR001",
        "notes": "Transportation.",
    },
    {
        "basket": "primary",
        "category": "health_personal",
        "cpi_product": CPI_PRODUCTS["health_personal"],
        "shs_sources": "HC001 + PC001",
        "notes": "Out-of-pocket health care plus personal care.",
    },
    {
        "basket": "primary",
        "category": "recreation_education",
        "cpi_product": CPI_PRODUCTS["recreation_education"],
        "shs_sources": "RE001 + ED002 + RO001",
        "notes": "Recreation, education, and reading materials.",
    },
    {
        "basket": "primary",
        "category": "alcohol_tobacco",
        "cpi_product": CPI_PRODUCTS["alcohol_tobacco"],
        "shs_sources": "TA018 (2017: TA001)",
        "notes": "Alcohol, tobacco, and cannabis. The 2017 aggregate TA001 has no cannabis line.",
    },
    {
        "basket": "primary",
        "category": "residual",
        "cpi_product": CPI_PRODUCTS["all_items"],
        "shs_sources": "TC001 - mapped sum",
        "notes": "Priced at all-items CPI. Absorbs miscellaneous (ME001), games of chance (GC001), and any classification gap.",
    },
)

_SHELTER_SPLIT_ROWS: tuple[dict[str, str], ...] = (
    {
        "basket": "shelter_split",
        "category": "rent",
        "cpi_product": CPI_PRODUCTS["rent"],
        "shs_sources": "SH003",
        "notes": "Replaces SH001. Rented principal residence.",
    },
    {
        "basket": "shelter_split",
        "category": "owned",
        "cpi_product": CPI_PRODUCTS["owned"],
        "shs_sources": "SH010",
        "notes": (
            "Replaces SH001. CPI owned accommodation is mortgage interest, replacement cost, "
            "property taxes, insurance, and maintenance. SH010 is the survey's owned-principal-residence "
            "spending, which includes mortgage payments (SH011). Mortgage principal is not priced on its own."
        ),
    },
    {
        "basket": "shelter_split",
        "category": "water_fuel",
        "cpi_product": CPI_PRODUCTS["water_fuel"],
        "shs_sources": "SH030",
        "notes": "Water, fuel and electricity for the principal accommodation.",
    },
    {
        "basket": "shelter_split",
        "category": "other_shelter",
        "cpi_product": CPI_PRODUCTS["shelter"],
        "shs_sources": "SH001 - SH003 - SH010 - SH030",
        "notes": "Remainder of shelter, priced at the shelter index.",
    },
)

AGE_LABELS: dict[int, str] = {
    1: "<30",
    2: "30-39",
    3: "40-54",
    4: "55-64",
    5: "65-74",
    6: "75+",
}
AGE_ORDER: tuple[str, ...] = ("<30", "30-39", "40-54", "55-64", "65-74", "75+")

TENURE_LABELS: dict[int, str] = {
    1: "owner_mortgage",
    2: "owner_free_clear",
    3: "renter",
}
TENURE_ORDER: tuple[str, ...] = ("owner_mortgage", "owner_free_clear", "renter")
TENURE_PRETTY: dict[str, str] = {
    "owner_mortgage": "Owner with mortgage",
    "owner_free_clear": "Owner free and clear",
    "renter": "Renter",
}

GROUPING_ORDER: tuple[str, ...] = (
    "all",
    "inc_q",
    "cons_d",
    "age_band",
    "tenure",
    "inc_q_x_tenure",
)

CONS_DECILE_ORDER: tuple[str, ...] = tuple(f"D{i:02d}" for i in range(1, 11))

# Frozen 2023 logistic fit from code/logistic_cc_fit.ipynb.
# imputed_cc_prop = ceiling / (1 + exp(-(income - location) / scale))
CC_LOCATION = 4500.0
CC_SCALE = 170_676.25
CC_MAX_LEVEL = 0.846
CC_INCOME_BANDS: tuple[tuple[str, float, float], ...] = (
    ("under_45k", 0.0, 45_000.0),
    ("45k_to_85k", 45_000.0, 85_000.0),
    ("above_85k", 85_000.0, float("inf")),
)
CC_BAND_TARGETS: dict[str, float] = {
    "under_45k": 0.45,
    "45k_to_85k": 0.50,
    "above_85k": 0.60,
}
