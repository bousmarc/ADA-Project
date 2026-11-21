# Foreign residents and fiscal indicators across Swiss communes

Advanced Data Analytics, HEC Lausanne, autumn 2026. Author: Marc Boustany.

**What it does.** Across the 2,115 Swiss communes (2022), the project asks whether the share of foreign residents, and *which* foreigners (permit B, C, L, asylum, international officials), is associated with federal income tax per resident and the social assistance rate, after controls. It compares a linear model with flexible machine-learning models and looks for commune types with clustering. The results are associations, not causal effects.

## Install
Python 3.12 (tested in the course's Nuvolos workspace).
```bash
pip install -r requirements.txt
```

## Run, in this order (from the repository folder)
| Step | File | Produces | Time |
|---|---|---|---|
| 1 | `python code/01_geo_features.py` | `data/processed/commune_geo.csv` (position, distance to border) | ~2 min |
| 2 | `python code/02_clean_merge.py` | `data/processed/communes_2022.csv` (2,115 communes × 51 columns) | < 1 min |
| 3 | `notebooks/03_descriptive_stats.ipynb` | tables, figures and LaTeX tables in `Results/descriptive/` | < 1 min |
| 4 | `notebooks/04_regression_ols.ipynb` | regression tables, figures and LaTeX tables in `Results/regressions/` | ~1 min |
| 5 | `notebooks/05_machine_learning.ipynb` | `Results/ml/` | (to come) |
| 6 | `notebooks/06_clustering.ipynb` | `Results/clustering/` | (to come) |

Each stage reads only what the previous one saved. Notebooks run top to bottom from a fresh kernel. `Results/` is created by the code; `figures/` holds the figures used in the paper. Shared functions are in `code/helpers.py`. Hardware: a standard laptop / Nuvolos; no GPU.

## Data
All data refer to 2022, at commune level, on the commune list of 6 April 2025.

| File in `data/raw/` | Source | Licence / use |
|---|---|---|
| `population_2022.csv`, `social_assistance_2022.csv`, `jobs_2022.csv` | Swiss Federal Statistical Office (FSO), Map Explorer / Statistical Atlas | OPEN-BY-ASK: non-commercial use, source must be cited |
| `statistik-dbst-np-kennzahlen-mit-2022.xlsx`, `statistik-dbst-np-kennzahlen-ohne-2022.xlsx` | Federal Tax Administration (ESTV), federal direct tax, key figures for individuals, tax period 2022 (payers only / all filers): main source for tax and taxpayers | reproduction allowed with source ("© ESTV") |
| `dbst_2022_normal.xlsx`, `dbst_2022_special.xlsx` | ESTV, federal direct tax by income class, tax period 2022: share of tax paid by incomes above CHF 200k | reproduction allowed with source ("© ESTV") |
| `commune_typology_2025.xlsx` | FSO, spatial classifications (Raumgliederungen) | FSO terms of use, source must be cited |
| `commune_mutations_2022_2025.xlsx` | FSO, register of commune mutations | FSO terms of use, source must be cited |
| `swissBOUNDARIES3D_1_5_LV95_LN02.gdb` | Federal Office of Topography swisstopo, swissBOUNDARIES3D | open government data, source must be cited ("© swisstopo") |
| `STATPOP_2022_Gemeinden_Anwesenheitsbewilligung_detailliert_alleBevtypen.xlsx` | FSO, STATPOP 2022, permanent residents by residence permit and commune; not online, sent on request by the FSO information centre, Demography and Migration section (info.dem@bfs.admin.ch), 2026 | published here with the FSO's agreement; source must be cited |

## Use of AI tools
Claude (Anthropic) helped write the data-preparation code and the shared helpers; this is marked at the top of each file and declared in the paper's appendix.
