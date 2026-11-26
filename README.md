# Foreign residents and fiscal indicators across Swiss communes

Advanced Data Analytics, HEC Lausanne, autumn 2026. Author: Marc Boustany.

**What it does.** Across the 2,115 Swiss communes (2022), the project asks whether the share of foreign residents, and *which* foreigners (permit B, C, L, asylum, international officials), is associated with federal income tax per resident and the social assistance rate, after controls. It compares a linear model with flexible machine-learning models and looks for commune types with clustering. The results are associations, not causal effects.

## Install
Python 3.12 (tested in the course's Nuvolos workspace).
```bash
pip install -r requirements.txt
```

## Run
Everything, from the raw data to every table and figure, with one command from the repository folder (about 30 minutes on a laptop; about 15 minutes without notebook 05b, which can be switched off at the top of `run_all.py`):
```bash
python run_all.py
```
It runs the steps below in this order; each notebook is run from top to bottom in a fresh kernel and saved with its outputs. Switches at the top of `run_all.py` skip steps that have already run. Each step can also be run on its own:

| Step | File | Produces | Time |
|---|---|---|---|
| 1 | `python code/01_geo_features.py` | `data/processed/commune_geo.csv` (position, distance to border) | ~2 min |
| 2 | `python code/02_clean_merge.py` | `data/processed/communes_2022.csv` (2,115 communes × 58 columns) | < 1 min |
| 2b | `python code/02b_later_years.py` | `data/processed/outcomes_later_years.csv` (tax 2023, social assistance 2023–2024, for the out-of-time test) | a few seconds |
| 2c | `python code/02c_panel.py` | `data/processed/panel_2011_2022.csv` (2,115 communes × 12 years: population, social assistance, jobs, federal tax) | ~10 seconds |
| 3 | `notebooks/03_descriptive_stats.ipynb` | tables, figures and LaTeX tables in `Results/descriptive/` | < 1 min |
| 4 | `notebooks/04_regression_ols.ipynb` | regression tables, figures and LaTeX tables in `Results/regressions/` | ~1 min |
| 4b | `notebooks/04b_specification_curve.ipynb` | specification curve: the two key results under every combination of controls, sample, weights and outcome, in `Results/specification_curve/` | < 1 min |
| 4c | `notebooks/04c_missing_sa.ipynb` | which communes have a hidden SA rate (logistic regression, random forest, boosting), and the SA result reweighted by inverse probability weighting, in `Results/missing_sa/` | < 1 min |
| 5 | `notebooks/05_machine_learning.ipynb` | model comparison, figures and LaTeX tables in `Results/ml/` | ~10 min |
| 5b | `notebooks/05b_tuning.ipynb` | every tuning option tried (number of trees, full grids, narrow vs wide search, canton-fold tuning) in `Results/tuning/` | ~15 min |
| 6 | `notebooks/06_clustering.ipynb` | PCA, commune types, figures and LaTeX tables in `Results/clustering/` | < 1 min |
| 7 | `notebooks/07_later_years.ipynb` | out-of-time test of the 2022 results in `Results/out_of_time/` | ~2 min |
| 8 | `notebooks/08_panel.ipynb` | panel 2011–2022: changes within communes (commune and canton × year fixed effects, pyfixest), long difference, robustness, slopes by commune type and tax split, in `Results/panel/` | < 1 min |

Each stage reads only what the previous one saved. Notebooks run top to bottom from a fresh kernel. `Results/` is created by the code and holds every table and figure, including those in the paper. Shared functions are in `code/helpers.py`. Hardware: a standard laptop / Nuvolos; no GPU.

## Data
The main data refer to 2022, at commune level, on the commune list of 6 April 2025. Files for 2023–2025 are used only for an out-of-time test of the 2022 results; files for 2011–2021 only for the panel (notebook 08).

| File in `data/raw/` | Source | Licence / use |
|---|---|---|
| `population_2022.csv`, `social_assistance_2022.csv`, `jobs_2022.csv`, `nationality_2022.csv` (German and French nationals, % of foreign residents), `housing_residents_2022.csv` (residents of single-family houses, %) | Swiss Federal Statistical Office (FSO), Map Explorer / Statistical Atlas | OPEN-BY-ASK: non-commercial use, source must be cited |
| `social_assistance_2023.csv`, `_2024.csv`, `_2025.csv`, `population_2023.csv`, `foreigners_2023.csv`, `foreigners_2024.csv` (out-of-time test and changes within communes); `social_assistance_2022_check.csv` (2022 re-downloaded in 2026, to check for revisions) | FSO, Map Explorer | OPEN-BY-ASK: non-commercial use, source must be cited |
| `statistik-dbst-np-mit-belastung-2023.xlsx`, `statistik-dbst-np-mit-und-ohne-belastung-2023.xlsx` (out-of-time test) | ESTV, federal direct tax, key figures for individuals, tax period 2023 (published 5 November 2026) | reproduction allowed with source ("© ESTV") |
| `statistik-dbst-np-kennzahlen-mit-2022.xlsx`, `statistik-dbst-np-kennzahlen-ohne-2022.xlsx` | Federal Tax Administration (ESTV), federal direct tax, key figures for individuals, tax period 2022 (payers only / all filers): main source for tax and taxpayers | reproduction allowed with source ("© ESTV") |
| `tax_burden_2022_single_100k.xlsx`, `tax_burden_2022_single_200k.xlsx` | ESTV tax calculator, tax burden statistics 2022 (income tax, single, no children, no church tax, gross income CHF 100,000 / 200,000, all communes); exported from swisstaxcalculator.estv.admin.ch | reproduction allowed with source ("© ESTV") |
| `dbst_2022_normal.xlsx`, `dbst_2022_special.xlsx` | ESTV, federal direct tax by income class, tax period 2022: share of tax paid by incomes above CHF 200k | reproduction allowed with source ("© ESTV") |
| `commune_typology_2025.xlsx` | FSO, spatial classifications (Raumgliederungen) | FSO terms of use, source must be cited |
| `commune_mutations_2022_2025.xlsx` | FSO, register of commune mutations | FSO terms of use, source must be cited |
| `population_2011.csv` … `population_2025.csv`, `sa_2011.csv` … `sa_2025.csv`, `jobs_2011.csv` … `jobs_2024.csv` (panel; population and social assistance already on the 2025 commune list, jobs on the 2024 list) | FSO, Map Explorer | OPEN-BY-ASK: non-commercial use, source must be cited |
| `estv_2011_2021/statistik-dbst-np-kennzahlen-{mit,ohne}-{year}-de.xlsx` (panel; from www.estv.admin.ch, "Statistiken zur direkten Bundessteuer"; tax in CHF millions up to 2019) | ESTV, federal direct tax, key figures for individuals, tax periods 2011–2021 | reproduction allowed with source ("© ESTV") |
| `commune_mutations_2011_2025.xlsx` (panel: old commune numbers of the 2011–2021 tax files) | FSO, register of commune mutations, 1 January 2011 – 6 April 2025 | FSO terms of use, source must be cited |
| `swissBOUNDARIES3D_1_5_LV95_LN02.gdb` | Federal Office of Topography swisstopo, swissBOUNDARIES3D | open government data, source must be cited ("© swisstopo") |
| `STATPOP_2022_Gemeinden_Anwesenheitsbewilligung_detailliert_alleBevtypen.xlsx` | FSO, STATPOP 2022, permanent residents by residence permit and commune; not online, sent on request by the FSO information centre, Demography and Migration section (info.dem@bfs.admin.ch), 2026 | published here with the FSO's agreement; source must be cited |

## Use of AI tools
Claude (Anthropic) helped write the data-preparation code and the shared helpers; this is marked at the top of each file and declared in the paper's appendix.
