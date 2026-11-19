# Survey columns used in log income–consumption figures

Source: SHS Tier A PUMF via `load_shs_tier_a(price_base_year=2019)`.
Dollar fields are CPI-deflated to **2019 CAD** before taking logs.

| Role | Harmonized | PUMF source | Notes |
|---|---|---|---|
| weight | `weight` | WEIGHTD | Survey household weight; used in hexbin density (sum of weights) and wquantile |
| gross_income | `hhtotinc` | HHTOTINC | Total household income before taxes |
| taxes | `taxes` | TX010 (2019/2023); TX001/TX010 slot in 2017 loader | Personal taxes; subtracted from HHTOTINC |
| after_tax_income | `inc_net` | derived: HHTOTINC - TX010 (fallback HHTOTINC if taxes missing) | x-axis is log(inc_net); CPI-deflated to 2019 CAD before log |
| consumption | `consumption` | TC001 | Total consumption expenditure; y-axis is log(consumption); 2019 CAD before log |
| survey_year | `year` | wave indicator (loader) | Used for by-year panels; not plotted on axes |

Derived analysis columns: `log_inc = log(inc_net)`, `log_cons = log(consumption)`
for rows with strictly positive income, consumption, and weight.
