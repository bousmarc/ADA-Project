"""
run_all.py
----------
Runs the whole project, from the raw data to every table and figure in Results/.

Run from the repository folder:   python run_all.py     (about 1 hour in total; about 20
                                                         minutes without notebook 05b)

Steps (in this order, because each one reads what the previous ones saved):
  1. code/01_geo_features.py      coordinates and distance to the border (~2 min, needs geopandas)
  2. code/02_clean_merge.py       the 2022 commune dataset (< 1 min)
     code/02b_later_years.py      outcomes of 2023-2024 for the out-of-time test (seconds)
  3. notebooks/03 to 07, and 05b  each notebook is run from top to bottom in a fresh
                                  kernel, and saved with its outputs (tables and figures
                                  also go to Results/<stage>/)

The switches below skip steps that have already run, for a quicker rerun. A notebook
reads only files saved by earlier steps, so steps can be skipped once they have run once.
"""

import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent

RUN_GEO = True          # step 1 (slow, needs geopandas); data/processed/commune_geo.csv is then reused
RUN_DATA = True         # step 2
RUN_NOTEBOOKS = True    # step 3
NOTEBOOKS = [           # remove a notebook from this list to skip it
    "03_descriptive_stats.ipynb",
    "04_regression_ols.ipynb",
    "05_machine_learning.ipynb",       # about 15 minutes
    "05b_tuning.ipynb",                # about 35 minutes: every tuning option tried (documents notebook 05's choices)
    "06_clustering.ipynb",
    "07_later_years.ipynb",
]


def run_script(relative_path):
    """Run a Python script with the same interpreter, from the repository folder."""
    print(f"\n=== {relative_path}")
    subprocess.run([sys.executable, str(ROOT / relative_path)], cwd=ROOT, check=True)


def run_notebook(name):
    """Run a notebook from top to bottom in a fresh kernel and save it with its outputs.
    It runs inside notebooks/, so its relative paths ("../code") work as in Jupyter."""
    print(f"\n=== notebooks/{name}")
    path = ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    client.execute()                    # stops with the failing cell if a cell raises an error
    nbformat.write(notebook, path)


start = time.perf_counter()
if RUN_GEO:
    run_script("code/01_geo_features.py")
if RUN_DATA:
    run_script("code/02_clean_merge.py")
    run_script("code/02b_later_years.py")
if RUN_NOTEBOOKS:
    for name in NOTEBOOKS:
        step_start = time.perf_counter()
        run_notebook(name)
        print(f"    done in {(time.perf_counter() - step_start) / 60:.1f} min")
print(f"\nEverything ran in {(time.perf_counter() - start) / 60:.1f} minutes. Outputs are in Results/.")
