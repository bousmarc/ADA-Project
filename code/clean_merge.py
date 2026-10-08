"""
01_clean_merge.py
-----------------
Stage 1 of the project: read the 6 raw files, clean them, and merge them
into ONE table with one row per Swiss commune (data for 2022).

Input  : data/raw/  (the 6 files downloaded from SFSO and ESTV)
Output : data/processed/communes_2022.csv

Run from the project folder (ada-project):   python code/01_clean_merge.py

Written with the help of Claude (Anthropic); see the paper's appendix.
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Commune mergers: translate old commune numbers to the 2025 list
#
# The files do not use the same list of communes:
#   population 06.04.2025, typology 01.01.2025, social assistance and jobs
#   01.01.2024, tax 2022. Communes merged in between. The SFSO list of
#   mutations (01.01.2022 - 06.04.2025) says which old commune went into which
#   new one. We use it to move every file onto the population file's list.
# ---------------------------------------------------------------------------
mut = pd.read_excel(RAW / "Communes_mutées.xlsx", sheet_name="Données",
                    header=None, skiprows=2)
mut = mut[[3, 7]].dropna()
mut.columns = ["old", "new"]
mut = mut.astype(int)
mut = mut[mut["old"] != mut["new"]]          # a commune that keeps its number maps to itself

# Only translate communes that NO LONGER EXIST in 2025. The list also contains
# small border swaps between two communes that both still exist (e.g. Kloten and
# Nürensdorf exchanged land in 2024); those are not mergers and are ignored.
communes_2025 = set(pd.read_csv(RAW / "population_2022.csv", sep=";", skiprows=2,
                                encoding="utf-8-sig")["Code"])
mut = mut[~mut["old"].isin(communes_2025)]
to_new = dict(zip(mut["old"], mut["new"]))


def to_2025(code):
    """Follow the chain of mergers until the commune number no longer changes.
    (A commune can merge twice, e.g. in 2022 and again in 2024.)"""
    for _ in range(10):                       # safety limit: never loop forever
        if code not in to_new:
            break
        code = to_new[code]
    return code


def harmonise(df, count_columns):
    """Put a table on the 2025 commune list.
    Merged communes become one row; their counts (people, CHF, jobs) are ADDED.
    Rates must NOT be added: they are recomputed from the counts afterwards."""
    df = df.copy()
    df["code"] = df["code"].astype(int).map(to_2025)
    # min_count=1: if every part of a merged commune is missing, stay missing
    return df.groupby("code", as_index=False)[count_columns].sum(min_count=1)


# ---------------------------------------------------------------------------
# Helper: the SFSO Map Explorer CSVs all look the same
#   - 2 title lines on top (skipped)
#   - separator ";"
#   - hidden values written as text, e.g. "N/A - secret statistique"
# ---------------------------------------------------------------------------
def read_map_explorer(filename):
    df = pd.read_csv(RAW / filename, sep=";", skiprows=2, encoding="utf-8-sig")
    df = df.rename(columns={"Code": "code", "Libellé": "name"})
    # turn every data column into numbers; text such as "secret" becomes NaN
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 1. Population, foreign share, age structure, density  (2025 commune list)
#    This file defines the commune list we keep: everything else is matched to it.
# ---------------------------------------------------------------------------
pop = read_map_explorer("population_2022.csv")
pop = pop.rename(columns={
    "Population résidante permanente 2022": "population",
    "Population résidante permanente étrangère 2022": "foreigners",
    "Part de la population résidante permanente étrangère 2022": "foreign_share",
    "Part de personnes de moins de 20 ans 2022": "share_under20",
    "Part de personnes des 65 ans ou plus 2022": "share_65plus",
    "Densité de la population (surface totale) 2022": "density",
})
pop = pop[["code", "name", "population", "foreigners", "foreign_share",
           "share_under20", "share_65plus", "density"]]


# ---------------------------------------------------------------------------
# 2. Social assistance  (2024 commune list)
#    Small communes are hidden for privacy -> NaN. We keep them as missing.
# ---------------------------------------------------------------------------
sa = read_map_explorer("social_assistance_2022.csv")
sa = sa.rename(columns={
    "Nombre de bénéficiaires de l'aide sociale 2022": "sa_recipients",
    "Taux d'aide sociale 2022": "sa_rate",
})
sa = sa[["code", "sa_recipients", "sa_rate"]]
sa["parts"] = 1                       # how many 2024 communes end up in one 2025 commune
# Move to the 2025 list, adding recipients of merged communes.
# The official rate is kept where nothing merged; for merged communes it is
# recomputed later as recipients / population (rates cannot be added).
sa = harmonise(sa, ["sa_recipients", "sa_rate", "parts"])


# ---------------------------------------------------------------------------
# 3. Jobs by sector  (2024 commune list)
#    Total jobs = primary + secondary + tertiary.
#    A missing sector value is very small (hidden for privacy) -> counted as 0.
# ---------------------------------------------------------------------------
jobs = read_map_explorer("jobs_2022.csv")
jobs = jobs.rename(columns={
    "Emplois du secteur primaire 2022": "jobs_primary",
    "Emplois du secteur secondaire 2022": "jobs_secondary",
    "Emplois du secteur tertiaire 2022": "jobs_tertiary",
})
sectors = ["jobs_primary", "jobs_secondary", "jobs_tertiary"]
jobs["jobs_total"] = jobs[sectors].fillna(0).sum(axis=1)
jobs = jobs[["code"] + sectors + ["jobs_total"]]
jobs = harmonise(jobs, sectors + ["jobs_total"])


# ---------------------------------------------------------------------------
# 4. Federal income tax  (ESTV, 2022 commune list)
#    Sheet 113 / 213 = tax revenue in CHF; last column = total over income classes.
#    Normal cases + special cases = total revenue from individuals.
# ---------------------------------------------------------------------------
def read_tax(filename, sheet):
    df = pd.read_excel(RAW / filename, sheet_name=sheet, header=None, skiprows=3)
    df = df[[2, 13]]                       # column C = commune number, column N = Total
    df.columns = ["code", "tax"]
    df = df.dropna(subset=["code"])
    df = df[df["code"] != 10000]           # 10000 = canton / Switzerland total rows
    df["code"] = df["code"].astype(int)
    df["tax"] = pd.to_numeric(df["tax"], errors="coerce")
    return df

tax_normal = read_tax("dbst_2022_normal.xlsx", "113").rename(columns={"tax": "tax_normal"})
tax_special = read_tax("dbst_2022_special.xlsx", "213").rename(columns={"tax": "tax_special"})
tax = tax_normal.merge(tax_special, on="code", how="outer")
tax["tax_total"] = tax[["tax_normal", "tax_special"]].fillna(0).sum(axis=1)
# The tax file uses the 2022 commune list: add up the tax of merged communes.
tax = harmonise(tax, ["tax_normal", "tax_special", "tax_total"])


# ---------------------------------------------------------------------------
# 5. Typology: canton, major region, urban/rural, language region  (2025 list)
#    The file stores codes; we translate them into readable labels.
# ---------------------------------------------------------------------------
typ = pd.read_excel(RAW / "commune_typology_2025.xlsx", sheet_name="Daten",
                    header=None, skiprows=3)
typ = typ[[0, 3, 6, 7, 10]]
typ.columns = ["code", "canton", "major_region", "urban_rural", "language"]
typ = typ.dropna(subset=["code"])
typ["code"] = typ["code"].astype(int).map(to_2025)
# Labels cannot be added up: if communes merged between January and April 2025,
# keep the label of the first part (they share canton and almost always type).
typ = typ.drop_duplicates(subset="code", keep="first")

typ["urban_rural"] = typ["urban_rural"].map({1: "urban", 2: "intermediate", 3: "rural"})
typ["language"] = typ["language"].map({1: "German", 2: "French", 3: "Italian", 4: "Romansh"})
typ["major_region"] = typ["major_region"].map({
    1: "Lake Geneva", 2: "Espace Mittelland", 3: "Northwestern CH", 4: "Zurich",
    5: "Eastern CH", 6: "Central CH", 7: "Ticino"})


# ---------------------------------------------------------------------------
# 6. Merge everything on the commune number, starting from the population list.
#    how="left" keeps every commune of the population file; if a commune
#    (usually a recent merger) is absent from another file, its values are NaN.
# ---------------------------------------------------------------------------
df = (pop.merge(typ, on="code", how="left")
         .merge(sa, on="code", how="left")
         .merge(jobs, on="code", how="left")
         .merge(tax, on="code", how="left"))


# ---------------------------------------------------------------------------
# 7. New variables
# ---------------------------------------------------------------------------
merged = df["parts"] > 1
df.loc[merged, "sa_rate"] = 100 * df.loc[merged, "sa_recipients"] / df.loc[merged, "population"]
df = df.drop(columns="parts")
df["tax_per_capita"] = df["tax_total"] / df["population"]     # CHF per resident
df["jobs_per_capita"] = df["jobs_total"] / df["population"]   # job-centre indicator


# ---------------------------------------------------------------------------
# 8. Report: what matched, what is missing  (goes into the paper's data section)
# ---------------------------------------------------------------------------
print(f"Communes in final table      : {len(df)}")
for col in ["foreign_share", "language", "sa_rate", "jobs_total", "tax_total"]:
    n_miss = df[col].isna().sum()
    pop_share = df.loc[df[col].isna(), "population"].sum() / df["population"].sum()
    print(f"Missing {col:<15}: {n_miss:4d} communes  ({pop_share:.1%} of population)")

# ---------------------------------------------------------------------------
# 9. Checks: stop with an error if something is clearly wrong
# ---------------------------------------------------------------------------
assert df["code"].is_unique, "a commune appears twice"
assert 8.7e6 < df["population"].sum() < 9.0e6, "Swiss population 2022 should be about 8.8 million"
for col in ["foreign_share", "share_under20", "share_65plus", "sa_rate"]:
    assert df[col].dropna().between(0, 100).all(), f"{col} outside 0-100 %"
assert (df["tax_total"].dropna() > 0).all(), "a commune with zero or negative tax"
print("All checks passed.")

df.to_csv(OUT / "communes_2022.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved -> {OUT / 'communes_2022.csv'}")
