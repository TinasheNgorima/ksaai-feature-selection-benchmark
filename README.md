# Evaluating Chatterjee's ξₙ for Feature Selection: Accuracy and Stability Against Modern Dependence Measures

This repository provides a fully reproducible implementation of the experiments presented in:

> "Evaluating Chatterjee's ξₙ for Feature Selection: Accuracy and Stability Against Modern Dependence Measures"

It benchmarks Chatterjee's ξₙ against Distance Correlation, Mutual Information, and MIC across 48 experimental configurations and a 30-repetition stability analysis on superconductivity data.

## Quick start

Minimal reproduction (Experiment 1 only, ~2–3 hours):

```bash
pip install -r requirements.txt
python src/experiments/experiment1_48configs.py
```

Results will be saved to `results/experiment1/`.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── CITATION.bib
├── LICENSE
├── data/                          # Place dataset files here (not tracked by git)
│   └── README_data.md
├── src/
│   ├── experiments/
│      ├── experiment1_48configs.py   # Experiment 1: 48-configuration comprehensive comparison
│      ├── stability_30reps.py        # Experiment 2: 30-repetition stability analysis
│      ├── cohens_d_bootstrap.py      # Paired Cohen's d_z + BCa bootstrap CIs
│      ├── efficiency_mic.py          # Computational efficiency: MIC timing benchmark
│      ├── efficiency_dc_mi_xi.py     # Computational efficiency: DC, MI, ξₙ timing + agreement matrix
│      └── figures_tables.py          # Figure and table generation from saved results   
└── results/                           # Auto-created by scripts; not tracked by git
```

## Dataset

This study uses the curated superconductivity dataset prepared by Matasov and Krasavina (2020):

> Matasov, A. & Krasavina, V. (2020). Visualization of Superconducting Materials. *SN Applied Sciences*, 2(8), 1463. https://doi.org/10.1007/s42452-020-03260-6

Download `sc_mean.csv` and place it at `data/sc_mean.csv` before running any script:
https://github.com/matasovav/DATA_SC/blob/master/sc_mean.csv

The original superconductivity measurements are described in:

> Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. *Computational Materials Science*, 154, 346–354. https://doi.org/10.1016/j.commatsci.2018.07.052

## Reproducibility notes

### Seed schedule

All stochastic components were assigned fixed seeds. The train–test partition in Experiment 1 used `random_state = 53`; cross-validation folds and model initialisation used `random_state = 42`. In Experiment 2, the train–test split for repetition *r* used `random_state = r` for *r* ∈ {0, 1, …, 29}; LightGBM, Optuna, and `mutual_info_regression` used `random_state = 42` throughout.

Complete seed assignments and environment specifications are recorded in `requirements.txt` and `config.yaml`.

## Installation

Tested on Python 3.10+.

`minepy` requires a C compiler. On Ubuntu/Debian:

```bash
sudo apt-get install build-essential
pip install -r requirements.txt
```

## Running the pipeline

Run scripts in order. Each script saves results to `results/` for downstream steps.

**Step 1 – 48-configuration comprehensive comparison (~2–3 h)**
```bash
python src/experiments/experiment1_48configs.py
```

**Step 2 – 30-repetition stability analysis (~3–4 h)**
```bash
python src/experiments/stability_30reps.py
```

**Step 3 – Paired effect sizes and bootstrap confidence intervals**
```bash
python src/experiments/cohens_d_bootstrap.py
```

**Step 4 – MIC timing benchmark**
```bash
python src/experiments/efficiency_mic.py
```

**Step 5 – DC, MI, ξₙ timing benchmark + feature agreement matrix (~1 h)**
```bash
python src/experiments/efficiency_dc_mi_xi.py
```

**Step 6 – Generate all manuscript figures and tables**
```bash
python src/experiments/figures_tables.py
```

## Expected outputs

| Script | Key outputs |
|---|---|
| `experiment1_48configs.py` | `results/experiment1/experiment_summary.csv`, feature-reduced datasets |
| `stability_30reps.py` | `results/stability/aggregated_summary.csv`, per-method `repetitions.csv`, `feature_sets.json` |
| `cohens_d_bootstrap.py` | `results/stability/bootstrap_ci_delta_r2.csv`, `cohens_d_matrix_r2.csv`, `cohens_d_matrix_mae.csv`, `cohens_d_pairwise.csv` |
| `efficiency_mic.py` | `results/efficiency/timing_mic.csv` |
| `efficiency_dc_mi_xi.py` | `results/efficiency/timing_dc_mi_xi.csv`, `results/figures/feature_agreement_matrix.pdf` |
| `figures_tables.py` | `results/figures/` — all manuscript figures (PDF + PNG) |

## Reproducibility checklist

| Item | Status | Evidence |
|---|---|---|
| Fixed random seeds | ✅ | `config.yaml`; all scripts use explicit `random_state` |
| Data split isolation | ✅ | Feature selection fit on training folds only |
| No data leakage | ✅ | Test sets held out for final evaluation |
| Environment specified | ✅ | `requirements.txt`, `config.yaml` |
| Results archived | ✅ | `results/` directory preserved after each run |

## MIC implementation note

**Scoring.** MIC feature scores are computed with `minepy==1.2.6` (MINE algorithm) in an isolated Python 3.10 subprocess (`stability_30reps.py`). All reported MIC rankings, Jaccard stability values, and feature-selection agreement derive from this implementation.

**Timing.** The MIC timing benchmark (`efficiency_mic.py`) uses `sklearn.feature_selection.mutual_info_regression` as a fast proxy rather than the MINE algorithm. The value reported in `timing_mic.csv` is therefore a proxy measurement and is not directly comparable to the MINE computation cost. The MINE algorithm is substantially slower than the sklearn estimator.

**Scope of impact.** This proxy affects the absolute MIC *timing* value only. It does not affect MIC scores, rankings, Jaccard stability, or feature-selection agreement, which are computed with `minepy`.

## Citation

```bibtex
@misc{ngorima2026ksaai,
  author       = {Ngorima, Tinashe},
  title        = {Evaluating Chatterjee's $\xi_n$ for Feature Selection: Accuracy and Stability Against Modern Dependence Measures},
  year         = {2026},
  howpublished = {\url{https://doi.org/10.5281/zenodo.19675804}},
  note         = {Zenodo. \texttt{doi:10.5281/zenodo.19675804}}
}
```
