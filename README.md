# Benchmarking Chatterjee's ξₙ for Feature Selection in Materials Informatics

Reproducible code for the manuscript:

> **"Benchmarking Chatterjee's ξₙ for Feature Selection in Materials Informatics"**  
> *Journal of Machine Learning Research (submitted)*

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── data/                          # Place dataset files here (not tracked by git)
│   └── README_data.md
├── 01_experiment1_48configs.py    # Experiment 1: 48-configuration comprehensive comparison
├── 02_stability_30reps.py         # Experiment 2: 30-repetition stability analysis
├── 03_efficiency_mic.py           # Computational efficiency: MIC timing benchmark
├── 04_efficiency_dc_mi_xi.py      # Computational efficiency: DC, MI, ξₙ timing + agreement matrix
├── 05_figures_tables.py           # Figure and table generation from saved results
└── results/                       # Auto-created by scripts; not tracked by git
```

---

## Dataset

The superconductivity dataset is publicly available from the UCI Machine Learning Repository:

- Hamidieh, K. (2018). *Superconductivity Data*. UCI ML Repository.  
  <https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data>

Download `train.csv` and place the two required files in `data/`:

| File | Description |
|------|-------------|
| `data/superconductor_features.csv` | 81 physicochemical descriptors (n = 15,542) |
| `data/critic_temp_df.csv` | Target column `critical_temp` (n = 15,542) |

Alternatively, use `data/sc_mean.csv` (the merged file with `critical_temp` and `material`
columns included), which is the format used in notebooks 03 and 04.

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
python 01_experiment1_48configs.py

# Step 2 – 30-repetition stability analysis (~3–4 h)
python 02_stability_30reps.py

# Step 3 – MIC timing benchmark (~2.5 h; run independently on any machine)
python 03_efficiency_mic.py

# Step 4 – DC, MI, ξₙ timing benchmark + feature-agreement matrix (~1 h)
python 04_efficiency_dc_mi_xi.py

# Step 5 – Generate all manuscript figures and tables
python 05_figures_tables.py
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
  howpublished = {\url{https://github.com/YourUsername/ksaai-feature-selection-benchmark}},
  note         = {Preprint, under development}
}
```
