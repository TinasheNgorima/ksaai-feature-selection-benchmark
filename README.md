# Benchmarking Chatterjee's ξₙ for Feature Selection in Materials Informatics

This repository provides a fully reproducible implementation of the experiments presented in:  
&gt; "Accuracy–Stability Trade-offs in Correlation-Based Feature Selection: A Benchmark of Chatterjee's ξₙ, Distance Correlation, Mutual Information, and the Maximal Information Coefficient"

It benchmarks Chatterjee's ξₙ against Distance Correlation, Mutual Information, and MIC across 48 experimental configurations and a 30-repetition stability analysis on superconductivity data.

## Quick start

Minimal reproduction (Experiment 1 only, ~2–3 hours):

```markdown
```bash
pip install -r requirements.txt
python src/experiment1_48configs.py
```

Results will be saved to `results/experiment1/`.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── data/                          # Place dataset files here (not tracked by git)
│   └── README_data.md
├── src/
│   ├── experiment1_48configs.py   # Experiment 1: 48-configuration comprehensive comparison
│   ├── stability_30reps.py        # Experiment 2: 30-repetition stability analysis
│   ├── efficiency_mic.py          # Computational efficiency: MIC timing benchmark
│   ├── efficiency_dc_mi_xi.py     # Computational efficiency: DC, MI, ξₙ timing + agreement matrix
│   └── figures_tables.py          # Figure and table generation from saved results
└── results/                       # Auto-created by scripts; not tracked by git
```

---

## Dataset

This study uses the curated superconductivity dataset prepared by Matasov and Krasavina (2020):

> Matasov, A. & Krasavina, V. (2020). Visualization of Superconducting Materials.
> SN Applied Sciences, 2(8), 1463. https://doi.org/10.1007/s42452-020-03260-6

Download sc_mean.csv and place it at data/sc_mean.csv before running any script:

https://github.com/matasovav/DATA_SC/blob/master/sc_mean.csv

The original superconductivity measurements are described in:

> Hamidieh, K. (2018). A data-driven statistical model for predicting the
> critical temperature of a superconductor. Computational Materials Science,
> 154, 346–354. https://doi.org/10.1016/j.commatsci.2018.07.052

---

## Reproducibility notes

**Seed schedule (JMLR reproducibility criteria)**

All stochastic components were assigned fixed seeds to ensure bit-for-bit reproducibility.
The train–test partition in Experiment 1 used random_state = 53; cross-validation folds
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

**Tested on:** Python 3.10+

**System dependencies:** `minepy` requires a C compiler. On Ubuntu/Debian:

```bash
sudo apt-get install build-essential
pip install minepy
```

---

## Running the pipeline

Run scripts in order. Each script saves results to `results/` for downstream steps.

### Step 1 – 48-configuration comprehensive comparison (~2–3 h)
```bash
python src/experiment1_48configs.py
```

### Step 2 – 30-repetition stability analysis (~3–4 h)
```bash
python src/stability_30reps.py
```

### Step 3 – MIC timing benchmark (~2.5 h; runs independently)
```bash
python src/efficiency_mic.py
```

### Step 4 – DC, MI, ξₙ timing benchmark + feature agreement matrix (~1 h)
```bash
python src/efficiency_dc_mi_xi.py
```

### Step 5 – Generate all manuscript figures and tables
```bash
python src/figures_tables.py
```

---

## Expected outputs

| Script                     | Key outputs                                                                              |
|---------------------------|-------------------------------------------------------------------------------------------|
| `experiment1_48configs.py` | `results/experiment1/experiment_summary.csv`, feature-reduced datasets                   |
| `stability_30reps.py`      | `results/stability/aggregated_summary.csv`, per-method JSON and CSV                      |
| `efficiency_mic.py`        | `results/efficiency/timing_mic.csv`                                                       |
| `efficiency_dc_mi_xi.py`   | `results/efficiency/timing_dc_mi_xi.csv`, `results/figures/feature_agreement_matrix.pdf` |
| `figures_tables.py`        | `results/figures/` — all manuscript figures (PDF + PNG)                                  |

---

## Reproducibility checklist

| Item                  | Status | Evidence                                               |
|----------------------|--------|--------------------------------------------------------|
| Fixed random seeds   | ✅     | `config.yaml`; all scripts use explicit `random_state` |
| Data split isolation | ✅     | Feature selection fit on training folds only           |
| No data leakage      | ✅     | Test sets held out for final evaluation                |
| Environment specified| ✅     | `requirements.txt`, `config.yaml`                      |
| Results archived     | ✅     | `results/` directory preserved after each run          |

---

## Citation

```bibtex
@dataset{ngorima2025ksaai,
  author  = {Ngorima, Tinashe},
  title   = {Accuracy–Stability Trade-offs in Correlation-Based Feature
Selection: A Benchmark of Chatterjee’s ξn, Distance
Correlation, Mutual Information, and the Maximal
Information Coefficient},
  year    = {2025},
  doi     = {10.xxxx/zenodo.xxxxxx},
  url     = {https://doi.org/10.xxxx/zenodo.xxxxxx}
}
```

---

## MIC implementation note

**Summary:**  
MIC was originally computed using `minepy==1.2.6` (Python 3.10). Due to C API incompatibilities with Python 3.12, reproduced scripts use `sklearn.feature_selection.mutual_info_regression` as a proxy. Original `minepy`-based results are archived in `results/original_run/`.

**Impact:**
- **Not affected:** Rankings, Jaccard stability values, feature selection agreement  
- **Affected:** Absolute MIC timing values (`sklearn` MI is ~50× faster than the MINE algorithm)

**Technical details:**  
`minepy` fails to build on Python 3.12 due to removed C API members (`ob_digit`, `curexc_traceback`, `pkg_resources`). No build flags resolve this issue. Scripts were therefore patched to use `sklearn` as a drop-in replacement for reproducibility.

---

### Timing comparison

| Method       | Original (Python 3.10, local) | Reproduced (Python 3.12, cloud VM) |
|--------------|-------------------------------|------------------------------------|
| ξₙ           | 0.33s ± 0.05s                 | 0.42s ± 0.07s                      |
| DC           | 1.83s ± 0.08s                 | 2.14s ± 0.10s                      |
| MI           | 7.30s ± 0.56s                 | 8.53s ± 1.57s                      |
| MIC (minepy) | 465.99s ± 19.12s              | N/A — replaced with `sklearn` MI (9.09s) |

---

