"""
code/02c_panel.py
-----------------
Stage 2c: a commune x year PANEL 2011-2022, for the within-commune analysis
(notebook 08). Same variables as the 2022 cross-section, one row per commune and year.

Input  : data/raw/population_{year}.csv, sa_{year}.csv      (FSO, already on the 2025 list)
         data/raw/jobs_{year}.csv                           (FSO, 2024 list)
         data/raw/estv_2011_2021/statistik-dbst-np-kennzahlen-{mit,ohne}-{year}-de.xlsx  (ESTV)
         data/raw/statistik-dbst-np-kennzahlen-{mit,ohne}-2022.xlsx                      (ESTV)
         data/raw/commune_mutations_2011_2025.xlsx          (FSO mutation register)
         data/processed/communes_2022.csv                   (canton, urban/rural, language)
Output : data/processed/panel_2011_2022.csv   (2,115 communes x 12 years)

Decisions:
  - Population and social assistance: the FSO already gives every year on the
    2025 commune list, so nothing has to be merged.
  - Tax: each ESTV year is on that year's commune list (2,551 communes in 2011,
    2,172 in 2021). Moved to the 2025 list with the mutation register 2011-2025;
    counts (taxpayers, CHF) are added for merged communes.
  - ESTV gives the tax in CHF MILLIONS up to 2019 and in CHF from 2020.
  - Time-invariant variables (canton, urban/rural, language) come from the 2022 dataset.
  - Stops at 2022: 2024-2025 social assistance contains FSO estimates, and 2022
    is the main year of the project.

Run from the repository folder:   python code/02c_panel.py   (about 10 seconds)
"""

import numpy as np
import pandas as pd

import helpers
from helpers import PROCESSED, RAW, read_key_figures, read_map_explorer, to_numbers

YEARS = range(2011, 2023)
communes_2025 = helpers.communes_2025()
# The register from 2011 is needed: the tax files of 2011-2021 use old commune numbers
to_new = helpers.merger_map(communes_2025, "commune_mutations_2011_2025.xlsx")


def harmonise(df, count_columns, name):
    """Move a table to the 2025 list (see helpers.harmonise)."""
    return helpers.harmonise(df, count_columns, name, to_new, communes_2025)

    # ---------------------------------------------------------------------------
# 1. Federal income tax (ESTV key figures, one commune list per year)
#    "mit"  = taxpayers who pay federal tax -> number of taxpayers, tax in CHF
#    "ohne" = all filers, also those who pay nothing -> number of filers
# ---------------------------------------------------------------------------
def read_old_key_figures(year, kind):
    """Read an ESTV key-figures file of 2011-2021 (older layout than 2022:
    short column names, sheet 'Gemeinden - Communes' or 'Gemeinden')."""
    path = RAW / "estv_2011_2021" / f"statistik-dbst-np-kennzahlen-{kind}-{year}-de.xlsx"
    book = pd.ExcelFile(path)
    sheet = [s for s in book.sheet_names if s.startswith("Gemeinden")][0]
    df = book.parse(sheet)
    df = df[df["Einheit"] == "Total"]             # all households (also drops footnote rows)
    tax = pd.to_numeric(df["snpstbetr"])
    if year <= 2019:
        tax = tax * 1e6                           # CHF millions -> CHF
    return pd.DataFrame({"code": df["gdenr"].astype(int),
                         "n": pd.to_numeric(df["stpf"]),
                         "tax": tax})


def read_tax(year):
    """Filers, taxpayers and tax of one year, on the 2025 commune list."""
    if year == 2022:
        payers = read_key_figures("statistik-dbst-np-kennzahlen-mit-2022.xlsx")
        filers = read_key_figures("statistik-dbst-np-kennzahlen-ohne-2022.xlsx")
    else:
        payers = read_old_key_figures(year, "mit")
        filers = read_old_key_figures(year, "ohne")
    tax = (filers[["code", "n"]].rename(columns={"n": "filers"})
           .merge(payers[["code", "n", "tax"]].rename(columns={"n": "taxpayers", "tax": "tax_total"}),
                  on="code", validate="1:1"))
    tax = harmonise(tax, ["filers", "taxpayers", "tax_total"], f"tax {year}")
    tax["year"] = year
    return tax


tax = pd.concat([read_tax(year) for year in YEARS], ignore_index=True)
print("Federal income tax, CHF billion:",
      (tax.groupby("year")["tax_total"].sum() / 1e9).round(2).to_dict())

      # ---------------------------------------------------------------------------
# 2. Population, foreign share, age structure, density (2025 list in every year)
# ---------------------------------------------------------------------------
def read_population(year):
    pop = read_map_explorer(f"population_{year}.csv")
    pop = pop.rename(columns={
        f"Population résidante permanente {year}": "population",
        f"Population résidante permanente étrangère {year}": "foreigners",
        f"Part de la population résidante permanente étrangère {year}": "foreign_share",
        f"Part de personnes de moins de 20 ans {year}": "share_under20",
        f"Part de personnes des 65 ans ou plus {year}": "share_65plus",
        f"Densité de la population (surface totale) {year}": "density",
    })
    numbers = ["population", "foreigners", "foreign_share", "share_under20", "share_65plus", "density"]
    pop = to_numbers(pop[["code"] + numbers], numbers)
    assert set(pop["code"]) == communes_2025, f"population {year}: not the 2025 list"
    pop["year"] = year
    return pop


# ---------------------------------------------------------------------------
# 3. Social assistance (2025 list in every year; hidden values -> NaN)
# ---------------------------------------------------------------------------
def read_sa(year):
    sa = read_map_explorer(f"sa_{year}.csv")
    sa = sa.rename(columns={f"Nombre de bénéficiaires de l'aide sociale {year}": "sa_recipients",
                            f"Taux d'aide sociale {year}": "sa_rate"})
    sa = to_numbers(sa[["code", "sa_recipients", "sa_rate"]], ["sa_recipients", "sa_rate"])
    assert set(sa["code"]) == communes_2025, f"social assistance {year}: not the 2025 list"
    sa["year"] = year
    return sa


# ---------------------------------------------------------------------------
# 4. Jobs by sector (2024 list -> 2025 list); same rule as in 02_clean_merge:
#    a hidden sector counts as 0, but if all three are hidden the total is unknown
# ---------------------------------------------------------------------------
def read_jobs(year):
    jobs = read_map_explorer(f"jobs_{year}.csv")
    sectors = [f"Emplois du secteur {s} {year}" for s in ("primaire", "secondaire", "tertiaire")]
    jobs = to_numbers(jobs, sectors)
    jobs["jobs_total"] = jobs[sectors].sum(axis=1, min_count=1)
    jobs = harmonise(jobs[["code", "jobs_total"]], ["jobs_total"], f"jobs {year}")
    jobs["year"] = year
    return jobs


pop = pd.concat([read_population(year) for year in YEARS], ignore_index=True)
sa = pd.concat([read_sa(year) for year in YEARS], ignore_index=True)
jobs = pd.concat([read_jobs(year) for year in YEARS], ignore_index=True)

# ---------------------------------------------------------------------------
# 5. Merge into one long table: one row per commune and year
# ---------------------------------------------------------------------------
static = pd.read_csv(PROCESSED / "communes_2022.csv", encoding="utf-8-sig",
                     usecols=["code", "name", "canton", "major_region", "urban_rural", "language"])

panel = (pop.merge(sa, on=["code", "year"], how="left", validate="1:1")
            .merge(jobs, on=["code", "year"], how="left", validate="1:1")
            .merge(tax, on=["code", "year"], how="left", validate="1:1")
            .merge(static, on="code", how="left", validate="m:1"))

panel["tax_per_capita"] = panel["tax_total"] / panel["population"]     # CHF per resident
panel["log_tax"] = np.log(panel["tax_per_capita"])
panel["jobs_per_capita"] = panel["jobs_total"] / panel["population"]
panel["filers_per_capita"] = panel["filers"] / panel["population"]

first = ["code", "name", "year", "canton", "major_region", "urban_rural", "language"]
panel = panel[first + [c for c in panel.columns if c not in first]]
panel = panel.sort_values(["code", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Checks
# ---------------------------------------------------------------------------
n_years = len(YEARS)
assert len(panel) == len(communes_2025) * n_years, "not one row per commune and year"
assert not panel.duplicated(["code", "year"]).any(), "duplicate commune-year"
assert panel["canton"].notna().all(), "commune without canton"
totals = panel.groupby("year")["population"].sum()
assert totals.between(7.8e6, 9.1e6).all(), f"implausible population totals: {totals.to_dict()}"
for col in ["foreign_share", "share_under20", "share_65plus", "sa_rate"]:
    assert panel[col].dropna().between(0, 100).all(), f"{col} outside 0-100"
assert (panel["tax_total"] > 0).all(), "a commune-year without tax"
assert (panel["taxpayers"] <= panel["filers"]).all(), "more taxpayers than filers"

# The 2022 rows must reproduce the cross-section (same sources, built separately)
cross = pd.read_csv(PROCESSED / "communes_2022.csv", encoding="utf-8-sig").set_index("code")
p22 = panel[panel["year"] == 2022].set_index("code").loc[cross.index]
for col in ["population", "foreign_share", "tax_total", "filers", "taxpayers"]:
    assert np.allclose(p22[col], cross[col], equal_nan=True), f"2022 {col} differs from communes_2022.csv"
same_sa = np.isclose(p22["sa_rate"], cross["sa_rate"]) | (p22["sa_rate"].isna() & cross["sa_rate"].isna())
print(f"2022 rows reproduce communes_2022.csv (SA rate identical in {same_sa.sum()} of {len(cross)} communes)")


# ---------------------------------------------------------------------------
# 7. Missing values per year, then save
# ---------------------------------------------------------------------------
missing = panel.groupby("year")[["foreign_share", "sa_rate", "jobs_total", "tax_total"]].agg(
    lambda s: s.isna().sum())
print("\nMissing values per year:\n", missing.to_string())

panel.to_csv(PROCESSED / "panel_2011_2022.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {len(panel):,} rows ({len(communes_2025):,} communes x {n_years} years), "
      f"{panel.shape[1]} columns")