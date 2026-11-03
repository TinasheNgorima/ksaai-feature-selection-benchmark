"""
efficiency_mic.py
=================
Timing benchmark for MIC using the genuine MINE algorithm
(minepy==1.2.6, MINE(alpha=0.6, c=15)), the same configuration that scores MIC
in experiment1_48configs.py and stability_30reps.py.

Protocol (identical to efficiency_dc_mi_xi.py for DC, MI and xi_n)
-------------------------------------------------------------------
  - full dataset, n = 15,542, p = 81, loaded as float32
  - one timed pass = scoring all 81 features against critical_temp
  - the float64 conversion minepy needs is inside the timed region,
    as for DC in efficiency_dc_mi_xi.py
  - 30 repetitions, mean and population std (ddof = 0)

Earlier versions of this script timed sklearn's k-NN mutual information
estimator as a proxy. Those values are not MIC timings.

Checkpointing
-------------
Each finished repetition is appended to results/efficiency/timing_mic_reps.csv.
If the run stops (screen lock, reboot), rerun the same command and it resumes
from the last finished repetition. Use --fresh to discard earlier repetitions.

Usage (repository root, Python 3.10 environment with minepy)
-------------------------------------------------------------
  conda activate ngorima_mic
  python src/experiments/efficiency_mic.py --pilot     # time 3 features, print ETA
  python src/experiments/efficiency_mic.py             # 30 repetitions
  python src/experiments/efficiency_mic.py --reps 5    # shorter run

Outputs (results/efficiency/)
-----------------------------
  timing_mic_reps.csv        one row per repetition (checkpoint file)
  timing_mic.csv             summary row, same columns as timing_dc_mi_xi.csv
  mic_scores_fulldata.csv    MIC scores from the last pass (traceability)
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import minepy
    from minepy import MINE
except ImportError:
    sys.exit("minepy is not installed in this environment. "
             "Run: conda activate ngorima_mic")

DATA_FILE   = Path("data/sc_mean.csv")
RESULTS_DIR = Path("results/efficiency")
REPS_FILE   = RESULTS_DIR / "timing_mic_reps.csv"
ALPHA, C    = 0.6, 15


def load_data():
    if not DATA_FILE.exists():
        sys.exit(f"{DATA_FILE} not found. Copy sc_mean.csv into data/ "
                 "(the data folder is not tracked by git).")
    df = pd.read_csv(DATA_FILE)
    y = df["critical_temp"].astype(np.float32).values
    X_df = df.drop(columns=["critical_temp", "material"], errors="ignore").astype(np.float32)
    assert X_df.shape[1] == 81, f"Expected 81 features, got {X_df.shape[1]}"
    return X_df, y


def mine_pass(X_df, y, n_features=None):
    """One timed MINE pass. Returns (elapsed seconds, scores)."""
    cols = list(X_df.columns)[:n_features] if n_features else list(X_df.columns)
    t0 = time.perf_counter()
    X64 = X_df[cols].values.astype(np.float64)
    y64 = y.astype(np.float64)
    mine = MINE(alpha=ALPHA, c=C)
    scores = np.zeros(len(cols))
    for j in range(len(cols)):
        mine.compute_score(X64[:, j], y64)
        scores[j] = mine.mic()
    return time.perf_counter() - t0, pd.Series(scores, index=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--pilot", action="store_true",
                    help="time 3 features only and print the estimated run time")
    ap.add_argument("--fresh", action="store_true",
                    help="discard earlier repetitions and start again")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_df, y = load_data()
    n, p = X_df.shape
    print(f"Dataset: {n:,} samples x {p} features | minepy {minepy.__version__} | "
          f"Python {platform.python_version()}")

    if args.pilot:
        t, _ = mine_pass(X_df, y, n_features=3)
        per_pass = t / 3 * p
        print(f"Pilot: {t:.1f} s for 3 features -> about {per_pass:.0f} s per pass")
        print(f"Estimated time: 5 reps {per_pass * 5 / 60:.0f} min, "
              f"30 reps {per_pass * 30 / 3600:.1f} h")
        return

    if args.fresh and REPS_FILE.exists():
        REPS_FILE.unlink()
    done = pd.read_csv(REPS_FILE) if REPS_FILE.exists() else pd.DataFrame(columns=["rep", "seconds"])
    if len(done):
        print(f"Resuming: {len(done)} repetition(s) already recorded in {REPS_FILE}")

    scores = None
    for rep in range(len(done), args.reps):
        t, scores = mine_pass(X_df, y)
        pd.DataFrame([{"rep": rep, "seconds": t}]).to_csv(
            REPS_FILE, mode="a", header=not REPS_FILE.exists(), index=False)
        print(f"  Rep {rep + 1:02d}/{args.reps}: {t:.1f} s  ({t / p:.3f} s/feature)", flush=True)

    times = pd.read_csv(REPS_FILE)["seconds"].values[:args.reps]
    if len(times) < args.reps:
        sys.exit(f"Only {len(times)} of {args.reps} repetitions recorded.")
    mean_t, std_t = float(times.mean()), float(times.std())

    pd.DataFrame([{
        "method"             : "MIC",
        "n_repetitions"      : len(times),
        "mean_total_s"       : mean_t,
        "std_total_s"        : std_t,
        "mean_s_per_feature" : mean_t / p,
        "cv_pct"             : std_t / mean_t * 100,
        "implementation"     : f"minepy {minepy.__version__} MINE(alpha={ALPHA}, c={C})",
        "python_version"     : platform.python_version(),
        "processor"          : platform.processor(),
    }]).to_csv(RESULTS_DIR / "timing_mic.csv", index=False)

    if scores is not None:
        scores.rename("MIC Score").rename_axis("Feature").reset_index().sort_values(
            "MIC Score", ascending=False).to_csv(
            RESULTS_DIR / "mic_scores_fulldata.csv", index=False)

    print(f"\nMIC (MINE): {mean_t:.1f} s +/- {std_t:.1f} s per pass over {len(times)} reps")
    print(f"Saved {RESULTS_DIR / 'timing_mic.csv'}")


if __name__ == "__main__":
    main()
