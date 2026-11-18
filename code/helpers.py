"""
code/helpers.py
---------------
Shared code for every stage of the project (Requirement 4.9: code used more
than once is imported from here, never copied).

Contents
  1. Paths and settings   ROOT, RAW, PROCESSED, FIGURES, results_dir(), SEED
  2. Reading raw files    check_header(), read_map_explorer(), to_numbers()
  3. Commune mergers      communes_2025(), merger_map(), to_2025(), harmonise()
  4. Plot style           COLORS, LABELS, SEQUENTIAL, DIVERGING, save_figure()

How to import it
  - from a script in code/:          import helpers
  - from a notebook in notebooks/:   import sys; sys.path.append("../code"); import helpers
All paths below are built from this file's location, so they work from anywhere.

Written with the help of Claude (Anthropic); see the paper's appendix.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# 1. Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]      # the ada-project folder
RAW = ROOT / "data" / "raw"                     # original downloads, never edited
PROCESSED = ROOT / "data" / "processed"         # outputs of the data stages
FIGURES = ROOT / "figures"                      # figures used in the paper
SEED = 42                                       # random_state everywhere

MUTATIONS_FILE = "commune_mutations_2022_2025.xlsx"   # SFSO mutation register


def results_dir(stage):
    """Return Results/<stage>/ and create it if needed (works on a fresh clone)."""
    path = ROOT / "Results" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 2. Reading raw files
# ---------------------------------------------------------------------------
SECRET = "N/A - secret statistique"             # how the SFSO writes hidden values


def check_header(header_row, column, expected):
    """Stop if a column read by its position is not the one we think it is
    (protects against a new download with a different layout)."""
    found = str(header_row[column])
    assert expected in found, f"column {column}: expected '{expected}', found '{found}'"


def read_map_explorer(filename):
    """Read an SFSO Map Explorer CSV: 2 title lines, separator ';', all as text
    (so that hidden values like "N/A - secret statistique" can still be seen)."""
    df = pd.read_csv(RAW / filename, sep=";", skiprows=2, encoding="utf-8-sig", dtype=str)
    df = df.rename(columns={"Code": "code", "Libellé": "name"})
    df["code"] = df["code"].astype(int)
    return df


def to_numbers(df, columns):
    """Turn text columns into numbers; "secret" and blanks become NaN."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 3. Commune mergers: put every file on the commune list of 6 April 2025
# ---------------------------------------------------------------------------
def communes_2025():
    """The reference list: the 2,115 communes of the population file."""
    return set(pd.read_csv(RAW / "population_2022.csv", sep=";", skiprows=2,
                           encoding="utf-8-sig")["Code"])


def merger_map(communes, filename=MUTATIONS_FILE):
    """Dictionary old commune number -> new number, from the SFSO mutation register
    (01.01.2022 - 06.04.2025). Only communes that NO LONGER EXIST in 2025 are
    translated; land swaps between two surviving communes (e.g. Kloten and
    Nürensdorf, 2024) are not mergers and are ignored."""
    mut = pd.read_excel(RAW / filename, sheet_name="Données", header=None)
    check_header(mut.iloc[1], 3, "Numéro de la commune")   # predecessor
    check_header(mut.iloc[1], 7, "Numéro de la commune")   # successor
    mut = mut.iloc[2:][[3, 7]].dropna().astype(int)
    mut.columns = ["old", "new"]
    mut = mut[(mut["old"] != mut["new"]) & ~mut["old"].isin(communes)]
    assert mut["old"].is_unique, "an old commune is mapped to two new communes"
    return dict(zip(mut["old"], mut["new"]))


def to_2025(code, to_new):
    """Follow the chain of mergers until the number no longer changes
    (a commune can merge twice, e.g. in 2022 and again in 2024)."""
    for _ in range(10):                         # safety limit: never loop forever
        if code not in to_new:
            break
        code = to_new[code]
    return code


def harmonise(df, count_columns, name, to_new, communes, strict_columns=()):
    """Put a table on the 2025 commune list.
    Merged communes become one row; their COUNTS (people, CHF, jobs) are added.
    Rates must never be added: recompute them from the counts afterwards.
    strict_columns: counts that must be missing if ANY part is missing
    (otherwise a merged commune would get an undercounted total)."""
    df = df.copy()
    df["code"] = df["code"].astype(int).map(lambda c: to_2025(c, to_new))
    # every commune must land on the list, and every listed commune must be there
    extra = set(df["code"]) - communes
    absent = communes - set(df["code"])
    assert not extra, f"{name}: codes not on the 2025 list: {sorted(extra)[:10]}"
    assert not absent, f"{name}: 2025 communes missing: {sorted(absent)[:10]}"
    # min_count=1: if every part of a merged commune is missing, stay missing
    out = df.groupby("code", as_index=False)[count_columns].sum(min_count=1)
    for col in strict_columns:
        any_missing = df[col].isna().groupby(df["code"]).any()
        out.loc[out["code"].map(any_missing), col] = np.nan
    return out


# ---------------------------------------------------------------------------
# 4. Plot style: the same colours and labels in every figure
#    Colours come from a colour-blind-safe reference palette and were checked
#    with a palette validator:
#      - urban_rural (3 colours): distinguishable in ANY chart, scatter plots included
#      - language (4 colours): fine in bar charts and boxplots; in a scatter plot,
#        draw one panel per language instead (4 colours on one cloud are too close)
#    The aqua colour is light: always add a legend or labels next to it.
# ---------------------------------------------------------------------------
COLORS = {
    "urban_rural": {"urban": "#2a78d6", "intermediate": "#eb6834", "rural": "#1baf7a"},
    "language": {"German": "#2a78d6", "French": "#eb6834",
                 "Italian": "#1baf7a", "Romansh": "#eda100"},
}
ORDER = {                                       # fixed order in legends and axes
    "urban_rural": ["urban", "intermediate", "rural"],
    "language": ["German", "French", "Italian", "Romansh"],
}

# Magnitude on a map (e.g. foreign share): one hue, light = low, dark = high
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
# Signed values (e.g. model errors): blue = below, grey = zero, red = above
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#e34948"])
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e1e0d9"}

# Readable axis titles: never show raw column names in a figure
LABELS = {
    "population": "Population",
    "foreign_share": "Foreign residents (% of population)",
    "share_B": "B permit holders (% of population)",
    "share_C": "C permit holders (% of population)",
    "share_L": "L permit holders (% of population)",
    "share_asylum": "Asylum-related permits F, N, S (% of population)",
    "share_intl": "International officials (% of population)",
    "share_under20": "Residents under 20 (%)",
    "share_65plus": "Residents aged 65+ (%)",
    "density": "Population density (residents per km²)",
    "jobs_per_capita": "Jobs per resident",
    "tax_per_capita": "Federal income tax per resident (CHF)",
    "taxpayers_per_capita": "Taxpayers per resident",
    "tax_per_taxpayer": "Federal income tax per taxpayer (CHF)",
    "tax_share_top": "Share of tax paid by incomes > CHF 200k (%)",
    "sa_rate": "Social assistance rate (%)",
    "border_km": "Distance to the national border (km)",
    "urban_rural": "Urban/rural type",
    "language": "Language region",
}


def save_figure(fig, path):
    """Save a figure as PNG at 300 dpi, trimmed, and create the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
