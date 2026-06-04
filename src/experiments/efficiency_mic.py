"""
03_efficiency_mic.py
====================
Computational efficiency benchmark for MIC (Maximal Information Coefficient)
on the superconductivity dataset (n = 15,542, p = 81).

MIC is timed separately from the other methods (04_efficiency_dc_mi_xi.py)
because of its substantially longer runtime (~465 s per pass).

Seed schedule
-------------
  MIC (MINE) is deterministic; no random_state is required.

Outputs (written to results/efficiency/)
-----------------------------------------
  timing_mic.csv    — one row: method, mean_s, std_s, mean_s_per_feature
"""

# =============================================================================
# Imports
# =============================================================================
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_selection import mutual_info_regression as _mir

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
N_REPETITIONS = 30
DATA_FILE     = Path("data/sc_mean.csv")
RESULTS_DIR   = Path("results/efficiency")

# =============================================================================
# MIC scoring
# =============================================================================

def score_mic_timed(X: np.ndarray, y: np.ndarray) -> float:
    """Time one full pass of MIC over all 81 features. Returns elapsed seconds."""
    def compute_mic(x, y): return _mir(x.reshape(-1,1), y.reshape(-1), random_state=42)[0]
    t0 = time.perf_counter()
    for j in range(X.shape[1]):
        compute_mic(X[:, j], y)
    return time.perf_counter() - t0


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    df = pd.read_csv(DATA_FILE)
    y = df["critical_temp"].astype(np.float32).values
    X_df = df.drop(columns=["critical_temp", "material"], errors="ignore")
    X = X_df.astype(np.float32).values
    assert X.shape[1] == 81, f"Expected 81 features, got {X.shape[1]}"
    return X, y


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPUTATIONAL EFFICIENCY — MIC Feature Selection")
    print("=" * 70)

    X, y = load_data()
    print(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"Repetitions: {N_REPETITIONS}\n")

    times = []
    for rep in tqdm(range(N_REPETITIONS), desc="MIC timing"):
        elapsed = score_mic_timed(X, y)
        times.append(elapsed)
        print(f"  Rep {rep + 1:02d}/{N_REPETITIONS}: {elapsed:.2f} s  "
              f"({elapsed / X.shape[1]:.4f} s/feature)")

    mean_t = float(np.mean(times))
    std_t  = float(np.std(times))

    print("\n" + "=" * 70)
    print(f"MIC FINAL: {mean_t:.3f} s ± {std_t:.3f} s per full ranking pass")
    print(f"           {mean_t / X.shape[1]:.5f} s/feature")
    print(f"           {mean_t / 60:.2f} min per pass  "
          f"({mean_t * N_REPETITIONS / 60:.1f} min total)")
    print("=" * 70)

    result_df = pd.DataFrame([{
        "method"              : "MIC",
        "n_repetitions"       : N_REPETITIONS,
        "mean_total_s"        : mean_t,
        "std_total_s"         : std_t,
        "mean_s_per_feature"  : mean_t / X.shape[1],
        "cv_pct"              : std_t / mean_t * 100,
    }])
    out = RESULTS_DIR / "timing_mic.csv"
    result_df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    return result_df


if __name__ == "__main__":
    main()
