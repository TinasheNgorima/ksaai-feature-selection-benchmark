"""
04_efficiency_dc_mi_xi.py
=========================
Computational efficiency benchmark for DC, MI, and ξₙ, and generation of the
feature-agreement matrix figure (Figure 1, main text).

Seed schedule
-------------
  mutual_info_regression : random_state = 42
  DC and ξₙ are deterministic; no random_state required.

Outputs (written to results/efficiency/ and results/figures/)
--------------------------------------------------------------
  results/efficiency/timing_dc_mi_xi.csv
  results/efficiency/timing_all.csv          (merged with mic if available)
  results/figures/feature_agreement_matrix.pdf
  results/figures/feature_agreement_matrix.png
  results/figures/full_feature_agreement_matrix.pdf
"""

# =============================================================================
# Imports
# =============================================================================
import time
import warnings
from itertools import combinations
from pathlib import Path

import dcor
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
SEED_MI       = 42
N_REPETITIONS = 30
THRESHOLD_PCT = 0.20    # 20% retention → 16 features from 81
DATA_FILE     = Path("data/sc_mean.csv")
RESULTS_DIR   = Path("results/efficiency")
FIGURES_DIR   = Path("results/figures")

# JMLR figure style (single-column: 6.50 in wide)
JMLR_RC = {
    "font.family"     : "serif",
    "font.serif"      : ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size"       : 10,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9,
    "figure.dpi"      : 300,
    "savefig.dpi"     : 300,
    "savefig.bbox"    : "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth"  : 0.8,
    "text.color"      : "#000000",
    "axes.labelcolor" : "#000000",
}

# =============================================================================
# Feature-selection scoring functions
# =============================================================================

def xicor(x: np.ndarray, y: np.ndarray, ties: bool = True) -> float:
    n = len(x)
    y_sorted = y[np.argsort(x)]
    r = rankdata(y_sorted, method="average")
    if ties:
        l = rankdata(y_sorted, method="max")
        return float(1 - np.sum(np.abs(r[:-1] - r[1:])) / (2 * np.sum(l * (n - l) / n)))
    return float(1 - 3 * np.sum(np.abs(r[:-1] - r[1:])) / (n ** 2 - 1))


def score_xi(X_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    scores = {}
    for col in tqdm(X_df.columns, desc="ξₙ", leave=False):
        try:
            scores[col] = xicor(X_df[col].values, y)
        except Exception:
            scores[col] = 0.0
    return pd.DataFrame(list(scores.items()), columns=["Feature", "Xi Score"])


def score_mi(X_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    mi = mutual_info_regression(X_df.values, y, random_state=SEED_MI)
    return pd.DataFrame({"Feature": X_df.columns, "MI Score": mi})


def score_dc(X_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    X64 = X_df.astype(np.float64)
    y64 = y.astype(np.float64)
    scores = {}
    for col in tqdm(X64.columns, desc="DC", leave=False):
        try:
            scores[col] = dcor.distance_correlation(X64[col].values, y64)
        except Exception:
            scores[col] = 0.0
    return pd.DataFrame(list(scores.items()), columns=["Feature", "DC Score"])


# =============================================================================
# Timing benchmark
# =============================================================================

def time_method(name: str, fn, X_df: pd.DataFrame, y: np.ndarray, n_reps: int) -> dict:
    print(f"\n{'='*70}\nMETHOD: {name.upper()}\n{'='*70}")
    rep_times = []
    for rep in range(n_reps):
        t0 = time.perf_counter()
        fn(X_df, y)
        elapsed = time.perf_counter() - t0
        rep_times.append(elapsed)
        print(f"  Rep {rep + 1:02d}/{n_reps}: {elapsed:.2f} s")
    arr = np.array(rep_times)
    return {
        "method"          : name,
        "n_repetitions"   : n_reps,
        "mean_total_s"    : float(arr.mean()),
        "std_total_s"     : float(arr.std()),
        "mean_s_per_feat" : float(arr.mean() / X_df.shape[1]),
        "cv_pct"          : float(arr.std() / arr.mean() * 100),
    }


# =============================================================================
# Feature-agreement matrix figure
# =============================================================================

def plot_agreement_matrix(
    df_dc: pd.DataFrame,
    df_mi: pd.DataFrame,
    df_mic: pd.DataFrame,
    df_xi: pd.DataFrame,
    n_features_total: int,
    out_prefix: Path,
    show_all: bool = False,
):
    """
    Binary feature-agreement matrix (Figure 1 and Appendix A).

    Parameters
    ----------
    show_all : if True, include features selected by only one method (Appendix A).
               if False, show only features with agreement ≥ 2 (main-text Figure 1).
    """
    n_top = int(np.ceil(THRESHOLD_PCT * n_features_total))

    dc_feat  = set(df_dc.sort_values("DC Score",  ascending=False).head(n_top)["Feature"])
    mi_feat  = set(df_mi.sort_values("MI Score",  ascending=False).head(n_top)["Feature"])
    mic_feat = set(df_mic.sort_values("MIC Score", ascending=False).head(n_top)["Feature"])
    xi_feat  = set(df_xi.sort_values("Xi Score",  ascending=False).head(n_top)["Feature"])

    all_features = sorted(dc_feat | mi_feat | mic_feat | xi_feat)
    methods = ["DC", "MI", "MIC", r"$\xi_n$"]

    pres = pd.DataFrame({
        "Feature": all_features,
        "DC"     : [1 if f in dc_feat  else 0 for f in all_features],
        "MI"     : [1 if f in mi_feat  else 0 for f in all_features],
        "MIC"    : [1 if f in mic_feat else 0 for f in all_features],
        "Xi"     : [1 if f in xi_feat  else 0 for f in all_features],
    })
    pres["Agreement"] = pres[["DC", "MI", "MIC", "Xi"]].sum(axis=1)
    pres = pres.sort_values(["Agreement", "Feature"], ascending=[False, True]).reset_index(drop=True)

    if not show_all:
        pres = pres[pres["Agreement"] >= 2].reset_index(drop=True)

    # Save matrix as CSV
    pres.to_csv(out_prefix.parent / "feature_presence_matrix.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update(JMLR_RC)
    fig_height = max(3.5, len(pres) * 0.28)
    fig, ax = plt.subplots(figsize=(6.50, fig_height))

    matrix = pres[["DC", "MI", "MIC", "Xi"]].values
    ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=0, vmax=1)

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_yticks(np.arange(len(pres)))
    ax.set_yticklabels(pres["Feature"], fontsize=9)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)

    # Minor grid (cell boundaries)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pres), 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Agreement-level separators
    for level in [4, 3, 2]:
        idx = pres[pres["Agreement"] == level].index
        if len(idx) > 0:
            ax.axhline(idx.max() + 0.5, color="black", linewidth=0.8,
                       linestyle="--" if (not show_all and level == 2) else "-")

    # Right axis: agreement count
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks(np.arange(len(pres)))
    ax_r.set_yticklabels(pres["Agreement"].astype(int), fontsize=9)
    ax_r.set_ylabel("Agreement level (out of 4)", fontsize=11)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}")
    plt.close(fig)
    print(f"Saved {out_prefix}.pdf / .png")


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    df = pd.read_csv(DATA_FILE)
    y = df["critical_temp"].astype(np.float32).values
    X_df = df.drop(columns=["critical_temp", "material"], errors="ignore").astype(np.float32)
    assert X_df.shape[1] == 81, f"Expected 81 features, got {X_df.shape[1]}"
    print(f"Dataset: {X_df.shape[0]:,} samples × {X_df.shape[1]} features")
    return X_df, y


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPUTATIONAL EFFICIENCY — DC, MI, ξₙ")
    print(f"Dataset: {DATA_FILE}  |  Repetitions: {N_REPETITIONS}")
    print("=" * 70)

    X_df, y = load_data()

    # ── Timing benchmark ──────────────────────────────────────────────────────
    benchmarks = [
        ("xi", score_xi),
        ("mi", score_mi),
        ("dc", score_dc),
    ]
    timing_rows = []
    for name, fn in benchmarks:
        row = time_method(name, fn, X_df, y, N_REPETITIONS)
        timing_rows.append(row)
        print(f"\n  {name.upper()}: {row['mean_total_s']:.3f} s ± {row['std_total_s']:.3f} s"
              f"  ({row['mean_s_per_feat']:.5f} s/feature)")

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(RESULTS_DIR / "timing_dc_mi_xi.csv", index=False)
    print(f"\nTiming results saved to {RESULTS_DIR / 'timing_dc_mi_xi.csv'}")

    # Merge with MIC timing if available
    mic_path = RESULTS_DIR / "timing_mic.csv"
    if mic_path.exists():
        mic_df = pd.read_csv(mic_path)
        all_timing = pd.concat([timing_df, mic_df], ignore_index=True)
        all_timing.to_csv(RESULTS_DIR / "timing_all.csv", index=False)
        print(f"Merged timing (all methods) → {RESULTS_DIR / 'timing_all.csv'}")

        # Print speedup analysis
        dc_time = all_timing.loc[all_timing["method"] == "dc", "mean_total_s"].values[0]
        print("\n" + "─" * 50)
        print("SPEEDUP vs. DC:")
        for _, r in all_timing.iterrows():
            spd = dc_time / r["mean_total_s"]
            print(f"  {r['method'].upper():3s}: {spd:.2f}×")

    # ── One final (deterministic) scoring run for the figure ──────────────────
    print("\nComputing final feature scores for agreement matrix figure …")
    df_dc  = score_dc(X_df, y)
    df_mi  = score_mi(X_df, y)
    df_xi  = score_xi(X_df, y)

    # Save correlation scores
    scores_dir = RESULTS_DIR / "feature_scores"
    scores_dir.mkdir(exist_ok=True)
    df_dc.to_csv(scores_dir / "dc_scores.csv",  index=False)
    df_mi.to_csv(scores_dir / "mi_scores.csv",  index=False)
    df_xi.to_csv(scores_dir / "xi_scores.csv",  index=False)

    # MIC scores must come from 03_efficiency_mic.py or be loaded if already computed.
    # For figure generation we attempt to load; skip if absent.
    mic_scores_path = scores_dir / "mic_scores.csv"
    if mic_scores_path.exists():
        df_mic = pd.read_csv(mic_scores_path)
        print(f"Loaded MIC scores from {mic_scores_path}")
    else:
        print(
            "[WARNING] MIC scores not found at results/efficiency/feature_scores/mic_scores.csv\n"
            "          Run 03_efficiency_mic.py first, or copy the file manually.\n"
            "          Skipping agreement matrix figure."
        )
        return

    # Main-text figure (agreement ≥ 2)
    plot_agreement_matrix(
        df_dc, df_mi, df_mic, df_xi,
        n_features_total=X_df.shape[1],
        out_prefix=FIGURES_DIR / "feature_agreement_matrix",
        show_all=False,
    )

    # Appendix figure (all 31 features)
    plot_agreement_matrix(
        df_dc, df_mi, df_mic, df_xi,
        n_features_total=X_df.shape[1],
        out_prefix=FIGURES_DIR / "full_feature_agreement_matrix",
        show_all=True,
    )


if __name__ == "__main__":
    main()
