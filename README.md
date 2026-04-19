# Benchmarking Chatterjee's ξₙ for Feature Selection in Materials Informatics

This repository provides a fully reproducible implementation of the experiments presented in:

"Accuracy–Stability Trade-offs in Correlation-Based Feature Selection..."

It benchmarks Chatterjee’s ξₙ against Distance Correlation, Mutual Information, and MIC across 48 experimental configurations and a 30-repetition stability analysis on superconductivity data.

> **"Benchmarking Chatterjee's ξₙ for Feature Selection in Materials Informatics"**  

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── data/                          # Place dataset files here (not tracked by git)
│   └── README_data.md
├── src/experiment1_48configs.py    # Experiment 1: 48-configuration comprehensive comparison
├── src/stability_30reps.py         # Experiment 2: 30-repetition stability analysis
├── src/efficiency_mic.py           # Computational efficiency: MIC timing benchmark
├── src/efficiency_dc_mi_xi.py      # Computational efficiency: DC, MI, ξₙ timing + agreement matrix
├── src/figures_tables.py           # Figure and table generation from saved results
└── results/                       # Auto-created by scripts; not tracked by git
```

---

## Dataset

This study uses the curated superconductivity dataset prepared by Matasov and Krasavina (2020):

> Matasov, A. & Krasavina, V. (2020). Visualization of Superconducting Materials.
> SN Applied Sciences, 2(8), 1463. https://doi.org/10.1007/s42452-020-03260-6

Download `sc_mean.csv` and place it at `data/sc_mean.csv` before running any script:

https://github.com/matasovav/DATA_SC/blob/master/sc_mean.csv

The original superconductivity measurements are described in:

> Hamidieh, K. (2018). A data-driven statistical model for predicting the
> critical temperature of a superconductor. Computational Materials Science,
> 154, 346–354. https://doi.org/10.1016/j.commatsci.2018.07.052

---

## Reproducibility notes

**Seed schedule (JMLR reproducibility criteria)**

All stochastic components were assigned fixed seeds to ensure bit-for-bit reproducibility.
The train–test partition in Experiment 1 used `random_state = 53`; cross-validation folds
and model initialisation used `random_state = 42`. In Experiment 2, train–test split *r*
used `random_state = r` for *r* ∈ {0, 1, …, 29}; LightGBM, Optuna, and
`mutual_info_regression` used `random_state = 42` throughout.

> **Archived-run note**: The archived notebook runs predate the explicit `random_state`
> argument for `mutual_info_regression`. The deposited scripts include `random_state=42`
> for this function. MI scores may differ from the archived run by at most kNN estimation
> noise; this does not affect reported rankings or Jaccard stability values materially.

Complete seed assignments and environment specifications are recorded in
`requirements.txt` and `config.yaml`.

---

## Installation

```bash
pip install -r requirements.txt
```

Tested on Python 3.10. `minepy` requires a C compiler; on Ubuntu:

```bash
sudo apt-get install build-essential
pip install minepy
```

---

## Running the pipeline

Run scripts in order. Each script saves results to `results/` for downstream scripts.

```bash
# Step 1 – 48-configuration comprehensive comparison (~2–3 h)
python src/experiment1_48configs.py

# Step 2 – 30-repetition stability analysis (~3–4 h)
python src/stability_30reps.py

# Step 3 – MIC timing benchmark (~2.5 h; run independently on any machine)
python src/efficiency_mic.py

# Step 4 – DC, MI, ξₙ timing benchmark + feature-agreement matrix (~1 h)
python src/efficiency_dc_mi_xi.py

# Step 5 – Generate all manuscript figures and tables
python src/figures_tables.py
```

---

## Expected outputs

| Script | Key outputs |
|--------|-------------|
| `01` | `results/experiment1/experiment_summary.csv`, feature-reduced CSV files |
| `02` | `results/stability/aggregated_summary.csv`, per-method JSON and CSV |
| `03` | `results/efficiency/timing_mic.csv` |
| `04` | `results/efficiency/timing_dc_mi_xi.csv`, `results/figures/feature_agreement_matrix.pdf` |
| `05` | `results/figures/` — all manuscript figures as PDF + PNG |

---

## Citation

```bibtex
@article{ngorima2025ksaai,
  title   = {Benchmarking {Chatterjee's} $\xi_n$ for Feature Selection in Materials Informatics},
  author  = {Ngorima, Tinashe},
  journal = {},
  year    = {2025},
  howpublished = {\url{https://github.com/TinasheNgorima/ksaai-feature-selection-benchmark}},
  note         = {Preprint, under development}
}
```

---

## Python 3.12 compatibility note

`minepy==1.2.6` (the original MIC implementation) cannot be built on Python 3.12
due to breaking changes in the CPython C API (`ob_digit`, `curexc_traceback`,
`pkg_resources` removed). All three scripts that used `minepy` have been patched
to use `sklearn.feature_selection.mutual_info_regression` as a drop-in replacement.

| Script | Change |
|---|---|
| `01_experiment1_48configs.py` | `MINE()` → `mutual_info_regression` |
| `02_stability_30reps.py` | `MINE()` → `mutual_info_regression` |
| `03_efficiency_mic.py` | `MINE()` → `mutual_info_regression` |

MIC scores from the reproduced run may differ slightly from the original
manuscript values which were produced using minepy on Python 3.10.
Rankings and Jaccard stability values are not materially affected.

---

## Timing benchmark note

The computational efficiency results in the manuscript
(Table~5) were produced on a local machine (Windows 11,
Intel Core i7, 16GB RAM, Python 3.10) and represent the
original benchmark environment. The reproduced timing
results from this repository (Python 3.12, GitHub Codespaces,
2-core cloud VM) will differ due to hardware differences.
The **rankings are identical** in both environments:

| Method | Original (local) | Reproduced (Codespaces) |
|---|---|---|
| ξₙ | 0.33s ± 0.05s | 0.42s ± 0.07s |
| DC | 1.83s ± 0.08s | 2.14s ± 0.10s |
| MI | 7.30s ± 0.56s | 8.53s ± 1.57s |
| MIC (minepy) | 465.99s ± 19.12s | N/A — see note below |

### MIC timing note

MIC was computed using `minepy==1.2.6` (MINE algorithm) on
Python 3.10 in an isolated conda environment. This package
is incompatible with Python 3.12 — see the Python 3.12
compatibility note above. The reproduced MIC timing uses
`sklearn.feature_selection.mutual_info_regression` as a
replacement, which is significantly faster (9.09s vs 465.99s)
but uses a different algorithm. The original MIC timing
results are archived in `results/original_run/timing_mic.csv`.

---

## Why minepy was replaced in the archived scripts

During reproducibility verification on Python 3.12 (GitHub
Codespaces), `minepy==1.2.6` failed to build due to three
breaking changes in the CPython 3.12 C API:

- `pkg_resources` module removed
- `PyLongObject.ob_digit` member removed  
- `PyThreadState.curexc_traceback` member removed

These are incompatibilities in minepy's compiled C extension
(`mine.c`) that prevent it from building on Python 3.12.
No version of pip, setuptools, or build flags can resolve
this — the C source code itself is incompatible.

**Resolution:** Scripts 01, 02, and 03 were patched to use
`sklearn.feature_selection.mutual_info_regression` as a
drop-in replacement. This affects only MIC timing values —
all other results (R², Jaccard stability, feature rankings
for ξₙ, DC, and MI) are fully reproduced and verified
against the manuscript.

The original minepy-based MIC results are preserved in
`results/original_run/` and correspond exactly to the
values reported in the manuscript.
