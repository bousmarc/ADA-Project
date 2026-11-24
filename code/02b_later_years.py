"""
code/02b_later_years.py
-----------------------
Stage 2b: outcomes of LATER years, for the out-of-time test of the 2022 results
(notebook 07). Only the outcomes change year; every explanatory variable stays
2022 (communes_2022.csv).

Input  : data/raw/statistik-dbst-np-mit-belastung-2023.xlsx          (ESTV, payers, 2023)
         data/raw/statistik-dbst-np-mit-und-ohne-belastung-2023.xlsx  (ESTV, all filers, 2023)
         data/raw/population_2023.csv                                 (FSO)
         data/raw/social_assistance_2023.csv, social_assistance_2024.csv (FSO)
         data/raw/foreigners_2023.csv, foreigners_2024.csv            (FSO)
Output : data/processed/outcomes_later_years.csv   (2,115 communes, 2025 list)

Decisions:
  - Tax 2023 is on the 2023 commune list: moved to the 2025 list with the same
    merger code as 2022 (counts added). Tax per resident = tax / population 2023.
  - Social assistance 2023 and 2024 are already on the 2025 list. 2024 contains
    FSO estimates for about half of the communes (flagged, `sa_estimate_2024`).
  - Social assistance 2025 is NOT used: Geneva and Zug are missing entirely and
    other cantons partly, so it is not comparable.

Run from the repository folder:   python code/02b_later_years.py   (a few seconds)
"""

import numpy as np
import pandas as pd

import helpers
from helpers import PROCESSED, SECRET, read_key_figures, read_map_explorer, to_numbers

communes_2025 = helpers.communes_2025()
to_new = helpers.merger_map(communes_2025)


# ---------------------------------------------------------------------------
# 1. Federal income tax 2023 (ESTV key figures, 2023 commune list)
# ---------------------------------------------------------------------------
payers = read_key_figures("statistik-dbst-np-mit-belastung-2023.xlsx")
filers = read_key_figures("statistik-dbst-np-mit-und-ohne-belastung-2023.xlsx")
tax = (filers[["code", "n"]].rename(columns={"n": "filers_2023"})
       .merge(payers[["code", "n", "tax"]].rename(columns={"n": "taxpayers_2023", "tax": "tax_total_2023"}),
              on="code"))
tax = helpers.harmonise(tax, ["filers_2023", "taxpayers_2023", "tax_total_2023"], "tax 2023",
                        to_new, communes_2025)


# ---------------------------------------------------------------------------
# 2. Population 2023 (FSO, 2025 commune list)
# ---------------------------------------------------------------------------
pop = read_map_explorer("population_2023.csv")
pop = pop.rename(columns={"Population résidante permanente 2023": "population_2023"})
pop = to_numbers(pop[["code", "population_2023"]], ["population_2023"])


# ---------------------------------------------------------------------------
# 3. Social assistance 2023 and 2024 (FSO, 2025 commune list)
# ---------------------------------------------------------------------------
def read_sa(year):
    sa = read_map_explorer(f"social_assistance_{year}.csv")
    estimate = f"Taux d'aide sociale {year} - estimation (*=oui)"
    sa = sa.rename(columns={f"Nombre de bénéficiaires de l'aide sociale {year}": f"sa_recipients_{year}",
                            f"Taux d'aide sociale {year}": f"sa_rate_{year}"})
    sa[f"sa_secret_{year}"] = (sa[f"sa_rate_{year}"] == SECRET).astype(int)
    sa[f"sa_estimate_{year}"] = (sa[estimate].fillna("").str.strip() == "*").astype(int)
    sa = to_numbers(sa, [f"sa_recipients_{year}", f"sa_rate_{year}"])
    return sa[["code", f"sa_recipients_{year}", f"sa_rate_{year}", f"sa_secret_{year}", f"sa_estimate_{year}"]]


sa = read_sa(2023).merge(read_sa(2024), on="code")


# ---------------------------------------------------------------------------
# 3b. Foreign residents 2023 and 2024 (FSO, 2025 commune list), for the analysis
#     of changes since 2022. From 2023, persons with protection status S (Ukraine)
#     who have been in Switzerland for 12 months count as permanent residents.
# ---------------------------------------------------------------------------
def read_foreigners(year):
    fr = read_map_explorer(f"foreigners_{year}.csv")
    fr = fr.rename(columns={f"Population résidante permanente étrangère {year}": f"foreigners_{year}",
                            f"Part de la population résidante permanente étrangère {year}": f"foreign_share_{year}"})
    return to_numbers(fr[["code", f"foreigners_{year}", f"foreign_share_{year}"]],
                      [f"foreigners_{year}", f"foreign_share_{year}"])


foreign = read_foreigners(2023).merge(read_foreigners(2024), on="code")


# ---------------------------------------------------------------------------
# 4. Merge, new variables, checks
# ---------------------------------------------------------------------------
for name, table in [("tax", tax), ("population", pop), ("social assistance", sa), ("foreigners", foreign)]:
    assert set(table["code"]) == communes_2025, f"{name} file does not match the 2025 list"

out = (pd.DataFrame({"code": sorted(communes_2025)})
       .merge(pop, on="code").merge(tax, on="code").merge(sa, on="code").merge(foreign, on="code"))
out["tax_per_capita_2023"] = out["tax_total_2023"] / out["population_2023"]

print(f"Communes: {len(out)} | population 2023: {out['population_2023'].sum():,.0f} | "
      f"tax 2023: CHF {out['tax_total_2023'].sum() / 1e6:,.0f} m")
for year in [2023, 2024]:
    print(f"SA {year}: reported {out[f'sa_rate_{year}'].notna().sum()}, "
          f"secret {out[f'sa_secret_{year}'].sum()}, FSO estimates {out[f'sa_estimate_{year}'].sum()}")

assert out["code"].is_unique and len(out) == len(communes_2025)
assert out["population_2023"].notna().all() and 8.8e6 < out["population_2023"].sum() < 9.1e6
assert out["tax_total_2023"].notna().all() and (out["tax_total_2023"] > 0).all()
assert (out["filers_2023"] >= out["taxpayers_2023"]).all(), "more payers than filers"
for year in [2023, 2024]:
    assert out[f"sa_rate_{year}"].dropna().between(0, 100).all(), f"SA rate {year} outside 0-100 %"
    assert out[f"foreign_share_{year}"].dropna().between(0, 100).all(), f"foreign share {year} outside 0-100 %"
# foreign_share_2023 should equal foreigners / population 2023 (up to rounding to 0.1)
gap = (100 * out["foreigners_2023"] / out["population_2023"] - out["foreign_share_2023"]).abs()
assert gap.dropna().max() < 0.06, "foreign_share_2023 inconsistent with foreigners / population"
print(f"Foreign residents: 2023 {out['foreigners_2023'].sum():,.0f}, 2024 {out['foreigners_2024'].sum():,.0f}")

out.to_csv(PROCESSED / "outcomes_later_years.csv", index=False, encoding="utf-8-sig")
print(f"Saved -> {PROCESSED / 'outcomes_later_years.csv'}")
