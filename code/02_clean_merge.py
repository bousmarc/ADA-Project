"""
code/02_clean_merge.py
----------------------
Stage 2 of the pipeline: read the raw files, clean them and merge them into ONE
table with one row per Swiss commune (data for 2022, commune list of 6 April 2025).

Input  : data/raw/ (SFSO, ESTV, FSO permit table) and data/processed/commune_geo.csv
         (from code/01_geo_features.py, which must run first)
Output : data/processed/communes_2022.csv   (2,115 communes x 51 columns)

Main decisions (details in the comments of each section and in the paper):
  - Every file is moved to the 2025 commune list with the SFSO mutation register;
    counts of merged communes are added, rates are recomputed, never added.
  - Hidden ("secret") values stay missing. Social assistance: `sa_status` says why
    a value is missing; a merged commune with a partly hidden SA value is missing.
  - Federal income tax and taxpayers come from the ESTV key figures (complete at
    commune level); the income-class sheets give only the share of tax paid by
    incomes above CHF 200k. Tax at source is NOT in the ESTV data.
  - Tax split: tax per resident = filers per resident x payers per filer x tax
    per payer. These parts, `tax_share_top`, median income and the income Gini
    are OUTCOMES or descriptions, never predictors (built from the outcome).
  - Foreign residents by permit (B, C, L, asylum F+N+S, international officials)
    from a STATPOP table sent by the FSO on request.
  - Many asserts stop the script if a file changes layout or a check fails.

Run from the repository folder:   python code/02_clean_merge.py   (under a minute)

Written with the help of Claude (Anthropic); see the paper's appendix.
"""

import numpy as np
import pandas as pd

import helpers
from helpers import PROCESSED, RAW, SECRET, check_header, read_map_explorer, to_numbers

PROCESSED.mkdir(parents=True, exist_ok=True)

# The reference list (2,115 communes) and the merger mapping old -> 2025 number
communes_2025 = helpers.communes_2025()
to_new = helpers.merger_map(communes_2025)


def to_2025(code):
    """Commune number on the 2025 list (follows chains of mergers)."""
    return helpers.to_2025(code, to_new)


def harmonise(df, count_columns, name, strict_columns=()):
    """Move a table to the 2025 list (see helpers.harmonise)."""
    return helpers.harmonise(df, count_columns, name, to_new, communes_2025, strict_columns)


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
pop_numbers = ["population", "foreigners", "foreign_share",
               "share_under20", "share_65plus", "density"]
pop = to_numbers(pop[["code", "name"] + pop_numbers], pop_numbers)


# ---------------------------------------------------------------------------
# 2. Social assistance  (2024 commune list)
#    Two kinds of missing value in the SFSO file:
#      "secret"  = hidden for privacy (few recipients in a small commune)
#      "blank"   = no value at all (meaning not documented in the file)
#    We keep track of which one, because they may not mean the same thing.
# ---------------------------------------------------------------------------
sa = read_map_explorer("social_assistance_2022.csv")
sa = sa.rename(columns={
    "Nombre de bénéficiaires de l'aide sociale 2022": "sa_recipients",
    "Taux d'aide sociale 2022": "sa_rate",
})
sa["n_secret"] = (sa["sa_recipients"] == SECRET).astype(int)
sa = to_numbers(sa, ["sa_recipients", "sa_rate"])
sa["parts"] = 1                                        # 2024 communes in one 2025 commune
sa["parts_reported"] = sa["sa_recipients"].notna().astype(int)
sa = sa[["code", "sa_recipients", "sa_rate", "parts", "parts_reported", "n_secret"]]
# Move to the 2025 list, adding recipients of merged communes.
# The official rate is kept where nothing merged; for merged communes it is
# recomputed later as recipients / population (rates cannot be added).
sa = harmonise(sa, ["sa_recipients", "sa_rate", "parts", "parts_reported", "n_secret"],
               "social assistance")


# ---------------------------------------------------------------------------
# 3. Jobs by sector  (2024 commune list)
#    Total jobs = primary + secondary + tertiary.
#    A hidden sector is very small (privacy) -> counted as 0 in the total,
#    but if ALL sectors are hidden we do not know the total -> NaN.
# ---------------------------------------------------------------------------
jobs = read_map_explorer("jobs_2022.csv")
jobs = jobs.rename(columns={
    "Emplois du secteur primaire 2022": "jobs_primary",
    "Emplois du secteur secondaire 2022": "jobs_secondary",
    "Emplois du secteur tertiaire 2022": "jobs_tertiary",
})
sectors = ["jobs_primary", "jobs_secondary", "jobs_tertiary"]
jobs = to_numbers(jobs, sectors)
jobs["jobs_hidden_sectors"] = jobs[sectors].isna().sum(axis=1)
jobs["jobs_total"] = jobs[sectors].sum(axis=1, min_count=1)
jobs = jobs[["code"] + sectors + ["jobs_total", "jobs_hidden_sectors"]]
jobs = harmonise(jobs, sectors + ["jobs_total", "jobs_hidden_sectors"], "jobs")


# ---------------------------------------------------------------------------
# 4. Federal income tax by income class  (ESTV, 2022 commune list)
#    Used ONLY for the share of tax paid by incomes above CHF 200k. The main tax
#    total and taxpayer count come from the key figures (section 4c): ESTV hides
#    small commune cells in these sheets, so their commune rows miss some
#    special cases (445 taxpayers, about CHF 20m nationally).
#    Normal cases + special cases = total revenue from individuals.
#    Every sheet has the same layout: column C = commune number, columns E-M =
#    9 classes of net income (CHF), column N = Total. Small cells are hidden ("-").
#    Sheets 111 / 211 = number of taxpayers, 113 / 213 = tax revenue in CHF.
#    "Taxpayer" = a tax unit: a married couple files jointly and counts as ONE.
# ---------------------------------------------------------------------------
LOW_CLASSES = list(range(4, 10))           # income classes up to CHF 200,000


def read_tax(filename, sheet, name):
    """Return commune code, Total, and the tax of the classes up to CHF 200k."""
    df = pd.read_excel(RAW / filename, sheet_name=sheet, header=None)
    check_header(df.iloc[2], 2, "Gemeinde ID")
    check_header(df.iloc[2], 4, "1 - 30'000")
    check_header(df.iloc[2], 9, "100'001 - 200'000")
    check_header(df.iloc[2], 13, "Total")
    df = df.iloc[3:]
    df = df[df[2].notna() & (df[2] != 10000)]   # 10000 = canton / Switzerland total rows
    out = pd.DataFrame({"code": df[2].astype(int),
                        name: pd.to_numeric(df[13], errors="coerce")})
    # Up to 200k: missing if ANY of these classes is hidden (sum would be too low).
    # The Total itself is never hidden, so "above 200k" = Total - up to 200k.
    low = df[LOW_CLASSES].apply(pd.to_numeric, errors="coerce")
    out[name + "_upto200k"] = low.sum(axis=1, min_count=len(LOW_CLASSES))
    return out


def add_normal_special(normal, special, name):
    """Normal + special cases; missing only if both are missing."""
    both = normal.merge(special, on="code", how="outer", suffixes=("_normal", "_special"))
    both[name] = both[[name + "_normal", name + "_special"]].sum(axis=1, min_count=1)
    # "up to 200k" must be known in BOTH files, otherwise it is unknown
    both[name + "_upto200k"] = (both[name + "_upto200k_normal"]
                                + both[name + "_upto200k_special"])
    return both


# Tax revenue (CHF)
tax = add_normal_special(read_tax("dbst_2022_normal.xlsx", "113", "tax"),
                         read_tax("dbst_2022_special.xlsx", "213", "tax"), "tax")
tax = tax.rename(columns={"tax": "tax_classes"})          # total of these sheets
tax = tax[["code", "tax_classes", "tax_upto200k"]]

# Number of taxpayers in these sheets (only to compare with the key figures)
payers = add_normal_special(read_tax("dbst_2022_normal.xlsx", "111", "taxpayers"),
                            read_tax("dbst_2022_special.xlsx", "211", "taxpayers"),
                            "taxpayers")
tax = tax.merge(payers[["code", "taxpayers"]].rename(columns={"taxpayers": "taxpayers_classes"}),
                on="code", how="outer")

# The tax file uses the 2022 commune list: add up the counts of merged communes.
tax = harmonise(tax, ["tax_classes", "tax_upto200k", "taxpayers_classes"],
                "tax", strict_columns=["tax_upto200k"])


# ---------------------------------------------------------------------------
# 4b. Foreign residents by permit  (STATPOP 2022, 2022 commune list)
#     File sent by the FSO (info.dem@bfs.admin.ch) on request, Oct 2026.
#     Only the PERMANENT resident population (columns C-Q) is used: the same
#     definition as `population` and `foreigners` above.
# ---------------------------------------------------------------------------
PERMIT_FILE = "STATPOP_2022_Gemeinden_Anwesenheitsbewilligung_detailliert_alleBevtypen.xlsx"
perm = pd.read_excel(RAW / PERMIT_FILE, sheet_name="STATPOP_2022", header=None)
check_header(perm.iloc[0], 2, "Ständige Wohnbevölkerung")       # permanent residents
check_header(perm.iloc[0], 17, "Nichtständige")                  # next block starts here
permit_columns = {2: "Total", 3: "Schweizer", 4: "(B)", 5: "(C)", 6: "(Ci)", 7: "(F)",
                  8: "(G)", 9: "(L)", 10: "(N)", 11: "(S)", 12: "Meldepflichtige",
                  13: "Diplomat", 14: "Internationaler Funktionär ohne",
                  15: "Nicht zugeteilt", 16: "Ohne Angabe"}
for col, text in permit_columns.items():
    check_header(perm.iloc[4], col, text)

perm = perm[perm[0].astype(str).str.strip().str.isdigit()]       # commune rows only
perm = pd.DataFrame({
    "code": perm[0].astype(int),
    "statpop_total": perm[2],
    "swiss": perm[3],
    "permit_B": perm[4],
    "permit_C": perm[5],
    "permit_L": perm[9],
    # asylum: provisionally admitted (F), asylum seekers (N), protection status (S, Ukraine)
    "permit_asylum": perm[7] + perm[10] + perm[11],
    # diplomats and international officials: exempt from Swiss income tax
    "permit_intl": perm[13] + perm[14],
    # Ci (working family members of diplomats), G (always 0 here: cross-border
    # workers do not live in Switzerland), persons to be registered, not assigned
    "permit_other": perm[6] + perm[8] + perm[12] + perm[15] + perm[16],
}).astype(int)
permit_groups = ["permit_B", "permit_C", "permit_L", "permit_asylum", "permit_intl", "permit_other"]
# Swiss + every permit group = total: nobody is lost or counted twice
assert (perm["swiss"] + perm[permit_groups].sum(axis=1) == perm["statpop_total"]).all(), \
    "permit groups do not add up to the total"
perm = harmonise(perm, ["statpop_total", "swiss"] + permit_groups, "permits")


# ---------------------------------------------------------------------------
# 4c. ESTV key figures for individuals, tax period 2022  (2022 commune list)
#     Two versions of the same table: "ohne" = all filers, with AND without
#     federal tax to pay; "mit" = only those who pay. One row per commune and
#     household type; we keep the rows "T" = all households.
#     "mit" gives the MAIN tax variables: tax_total and taxpayers (complete at
#     commune level; their national sums equal ESTV's published totals).
#     Like the other ESTV data: tax at source is NOT included, except residents
#     taxed at source with a definitive ordinary assessment (income > CHF 120k
#     or on request), whom ESTV counts with the ordinary income data
#     (ESTV, NFA instructions, annex 2, section 2.1.2).
# ---------------------------------------------------------------------------
def read_key_figures(filename):
    df = pd.read_excel(RAW / filename, sheet_name="Gemeinden - Communes", header=None)
    for col, text in [(2, "Gemeindenummer"), (4, "Einheit ID"), (6, "stpf"), (7, "snpstbetr"),
                      (12, "median_steink"), (13, "gini_steink")]:
        check_header(df.iloc[0], col, text)
    df = df.iloc[1:]
    df = df[df[4] == "T"]                          # T = all households (not married / single)
    return pd.DataFrame({"code": df[2].astype(int),
                         "n": pd.to_numeric(df[6]),             # stpf: number of taxpayers
                         "tax": pd.to_numeric(df[7]),           # snpstbetr: federal tax, CHF
                         "median": pd.to_numeric(df[12]),       # median taxable income, CHF
                         "gini": pd.to_numeric(df[13])})        # Gini of taxable income


kf_all = read_key_figures("statistik-dbst-np-kennzahlen-ohne-2022.xlsx")   # all filers
kf_pay = read_key_figures("statistik-dbst-np-kennzahlen-mit-2022.xlsx")    # payers only

# Counts can be added for merged communes ...
key = (kf_all[["code", "n"]].rename(columns={"n": "filers"})
       .merge(kf_pay[["code", "n", "tax"]].rename(columns={"n": "taxpayers", "tax": "tax_total"}),
              on="code"))
key = harmonise(key, ["filers", "taxpayers", "tax_total"], "key figures")
# ... but medians and Ginis cannot: keep them only where nothing merged.
kf_all["code_2025"] = kf_all["code"].map(to_2025)
single = kf_all.groupby("code_2025")["code"].transform("size") == 1
income = (kf_all.loc[single, ["code_2025", "median", "gini"]]
          .rename(columns={"code_2025": "code", "median": "median_taxable_income",
                           "gini": "gini_taxable_income"}))
key = key.merge(income, on="code", how="left")


# ---------------------------------------------------------------------------
# 5. Typology: canton, major region, urban/rural, language region  (2025 list)
#    The file stores codes; we translate them into readable labels.
# ---------------------------------------------------------------------------
typ = pd.read_excel(RAW / "commune_typology_2025.xlsx", sheet_name="Daten", header=None)
for col, text in [(0, "BFS Gde-nummer"), (3, "Kanton"), (6, "Grossregionen"),
                  (7, "Stadt/Land"), (10, "Sprachgebiete")]:
    check_header(typ.iloc[1], col, text)
typ = typ.iloc[3:][[0, 3, 6, 7, 10]]
typ.columns = ["code", "canton", "major_region", "urban_rural", "language"]
typ = typ.dropna(subset=["code"])
# the header rows make Excel store the codes as text ("4"): turn them into numbers
for col in ["code", "major_region", "urban_rural", "language"]:
    typ[col] = pd.to_numeric(typ[col]).astype(int)
typ["code"] = typ["code"].map(to_2025)
# Labels cannot be added up: if communes merged between January and April 2025,
# keep the label of the first part (they share canton and almost always type).
n_dup = typ["code"].duplicated().sum()
typ = typ.drop_duplicates(subset="code", keep="first")
assert set(typ["code"]) == communes_2025, "typology does not cover the 2025 list"

typ["urban_rural"] = typ["urban_rural"].map({1: "urban", 2: "intermediate", 3: "rural"})
typ["language"] = typ["language"].map({1: "German", 2: "French", 3: "Italian", 4: "Romansh"})
typ["major_region"] = typ["major_region"].map({
    1: "Lake Geneva", 2: "Espace Mittelland", 3: "Northwestern CH", 4: "Zurich",
    5: "Eastern CH", 6: "Central CH", 7: "Ticino"})
assert typ.notna().all().all(), "a typology code has no label"


# ---------------------------------------------------------------------------
# 5b. Geography  (already on the 2025 list; built by code/01_geo_features.py
#     because reading the maps is slow and needs geopandas)
# ---------------------------------------------------------------------------
geo = pd.read_csv(PROCESSED / "commune_geo.csv", encoding="utf-8-sig")
assert set(geo["code"]) == communes_2025, "commune_geo.csv does not match the 2025 list"


# ---------------------------------------------------------------------------
# 6. Merge everything on the commune number, starting from the population list.
#    Thanks to the checks in harmonise(), every file covers exactly this list.
# ---------------------------------------------------------------------------
df = (pop.merge(typ, on="code", how="left")
         .merge(sa, on="code", how="left")
         .merge(jobs, on="code", how="left")
         .merge(tax, on="code", how="left")
         .merge(perm, on="code", how="left")
         .merge(geo, on="code", how="left")
         .merge(key, on="code", how="left"))


# ---------------------------------------------------------------------------
# 7. Social assistance for merged communes, and status of every SA value
# ---------------------------------------------------------------------------
merged = df["parts"] > 1
complete = df["parts_reported"] == df["parts"]
partly_hidden = merged & (df["parts_reported"] > 0) & ~complete

# Merged and fully visible: recompute the rate from the counts.
recompute = merged & complete
df.loc[recompute, "sa_rate"] = 100 * df.loc[recompute, "sa_recipients"] / df.loc[recompute, "population"]
# Merged but only partly visible: the sum would undercount recipients -> unknown.
df.loc[partly_hidden, ["sa_recipients", "sa_rate"]] = float("nan")

df["sa_status"] = "reported"
df.loc[df["sa_rate"].isna() & (df["n_secret"] > 0), "sa_status"] = "secret"
df.loc[df["sa_rate"].isna() & (df["n_secret"] == 0), "sa_status"] = "blank"
df.loc[partly_hidden, "sa_status"] = "partly hidden (merger)"
df = df.drop(columns=["parts", "parts_reported", "n_secret"])


# ---------------------------------------------------------------------------
# 8. New variables
# ---------------------------------------------------------------------------
df["tax_per_capita"] = df["tax_total"] / df["population"]     # CHF per resident
df["jobs_per_capita"] = df["jobs_total"] / df["population"]   # job-centre indicator

# Tax split: tax per resident = (taxpayers / residents) x (tax / taxpayer).
# In logs the two parts ADD UP, so OLS coefficients on the two parts add up too.
df["taxpayers_per_capita"] = df["taxpayers"] / df["population"]   # extensive margin
df["tax_per_taxpayer"] = df["tax_total"] / df["taxpayers"]        # intensive margin (CHF)

# Share of federal income tax paid by taxpayers with net income above CHF 200k (%).
# Computed within the income-class sheets (numerator and denominator from the
# same source), so the hidden special cases cannot distort it.
df["tax_share_top"] = 100 * (df["tax_classes"] - df["tax_upto200k"]) / df["tax_classes"]
df = df.drop(columns="tax_upto200k")

# Permit shares, in % of the permanent resident population (FSO total, so that
# share_swiss + all permit shares = 100 exactly)
for group in ["permit_B", "permit_C", "permit_L", "permit_asylum", "permit_intl", "permit_other"]:
    df["share_" + group.removeprefix("permit_")] = 100 * df[group] / df["statpop_total"]

# Three-part tax split (all three in logs add up to log tax per resident):
#   filers per resident  : who files an ordinary return at all (people taxed at
#                          source do not; children are counted in their parents' return)
#   payers per filer     : how many filers earn enough to pay federal tax
#   tax per payer        : how much each payer pays (= tax_per_taxpayer above)
df["filers_per_capita"] = df["filers"] / df["population"]
df["payers_per_filer"] = df["taxpayers"] / df["filers"]


# ---------------------------------------------------------------------------
# 9. Report: what is missing  (goes into the paper's data section)
# ---------------------------------------------------------------------------
print(f"Communes in final table : {len(df)}")
print(f"Typology rows merged in 2025 (label of first part kept): {n_dup}")
print("\nMissing values (communes, share of Swiss population):")
for col in df.columns[df.isna().any()]:
    n_miss = df[col].isna().sum()
    pop_share = df.loc[df[col].isna(), "population"].sum() / df["population"].sum()
    print(f"  {col:<20}: {n_miss:4d}  ({pop_share:.1%})")
print("\nSocial assistance status:")
print(df["sa_status"].value_counts().to_string())

# The FSO permit file and the Map Explorer population file both come from
# STATPOP 2022, but a few communes differ slightly (land swaps between communes).
diff_pop = df["statpop_total"] != df["population"]
diff_for = (df["statpop_total"] - df["swiss"]) != df["foreigners"]
print(f"\nPermit file vs population file: total differs in {diff_pop.sum()} communes "
      f"(max {(df['statpop_total'] - df['population']).abs().max()} persons); "
      f"foreigners differ in {diff_for.sum()} communes "
      f"({df['foreigners'].isna().sum()} of them hidden in the population file)")

# Key figures (main source) vs income-class sheets: the class sheets hide small
# commune cells, so they should be equal or slightly LOWER, never higher.
gap_payers = df["taxpayers"] - df["taxpayers_classes"]
gap_tax = df["tax_total"] - df["tax_classes"]
print(f"\nKey figures vs income-class sheets: taxpayers higher in {(gap_payers > 0).sum()} communes "
      f"(national gap {gap_payers.sum():+,.0f}, {gap_payers.sum() / df['taxpayers'].sum():+.3%}); "
      f"tax higher in {(gap_tax > 1).sum()} communes "
      f"(national gap {gap_tax.sum() / df['tax_total'].sum():+.3%})")

# ---------------------------------------------------------------------------
# 10. Checks: stop with an error if something is clearly wrong
# ---------------------------------------------------------------------------
assert df["code"].is_unique, "a commune appears twice"
assert len(df) == len(communes_2025), "commune count changed during the merge"
assert 8.7e6 < df["population"].sum() < 9.0e6, "Swiss population 2022 should be about 8.8 million"
for col in ["foreign_share", "share_under20", "share_65plus", "sa_rate"]:
    assert df[col].dropna().between(0, 100).all(), f"{col} outside 0-100 %"
assert df["tax_total"].notna().all(), "a commune without tax data"
assert (df["tax_total"] > 0).all(), "a commune with zero or negative tax"
# foreign_share should equal foreigners / population (up to rounding to 0.1)
gap = (100 * df["foreigners"] / df["population"] - df["foreign_share"]).abs()
assert gap.dropna().max() < 0.06, "foreign_share inconsistent with foreigners / population"
# tax split is exact: the two parts multiply back to tax per resident
assert np.isclose(df["taxpayers_per_capita"] * df["tax_per_taxpayer"], df["tax_per_capita"]).all(), \
    "tax split does not multiply back to tax per resident"
assert (df["taxpayers"] > 0).all(), "a commune without taxpayers"
assert df["tax_share_top"].dropna().between(0, 100).all(), "tax_share_top outside 0-100 %"
# permit file: covers every commune and is close to the population file
assert df["statpop_total"].notna().all(), "a commune without permit data"
assert ((df["statpop_total"] - df["population"]).abs() / df["population"]).max() < 0.05, \
    "permit file total far from the population file"
# key figures: every commune has filers; payers cannot outnumber filers
assert (df["filers"] > 0).all(), "a commune without filers"
assert (df["payers_per_filer"] <= 1).all(), "more payers than filers in a commune"
# national totals equal ESTV's published totals (confirmed by ESTV, Nov 2026)
assert df["taxpayers"].sum() == 4_060_583, "taxpayers differ from ESTV's national total"
# the class sheets can only MISS cases (hidden cells), never have more
assert (gap_payers >= 0).all() and (gap_tax >= -1).all(), "class sheets exceed the key figures"
assert abs(gap_tax.sum()) / df["tax_total"].sum() < 0.01, "tax totals disagree by more than 1%"
assert np.isclose(df["filers_per_capita"] * df["payers_per_filer"] * df["tax_per_taxpayer"],
                  df["tax_per_capita"]).all(), "three-part split does not multiply back"
print("\nAll checks passed.")

df = df.drop(columns=["tax_classes", "taxpayers_classes"])   # used only for checks / top share
df.to_csv(PROCESSED / "communes_2022.csv", index=False, encoding="utf-8-sig")
print(f"Saved -> {PROCESSED / 'communes_2022.csv'}")
