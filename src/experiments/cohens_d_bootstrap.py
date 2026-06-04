"""
03_cohens_d_bootstrap.py
Computes paired Cohen's d_z and BCa bootstrap CIs from Experiment 2 outputs.

Reads:  results/stability/{METHOD}/repetitions.csv
Writes:
  results/stability/cohens_d_pairwise.csv      → prose paragraph
  results/stability/cohens_d_matrix_r2.csv     → Figure 1 panel b
  results/stability/cohens_d_matrix_mae.csv    → Figure 1 panel b
  results/stability/bootstrap_ci_delta_r2.csv  → Table 5 (tab:bootstrap_cv)

Definition:
    d_z = mean(diff) / std(diff, ddof=1)   [Lakens 2013, within-subjects]
    n   = 30 paired splits
    Sign: d_z > 0 means row method outperforms column method on R².
          Direction is flipped for MAE so d_z > 0 = row method is better.

BCa CIs:
    B = 10,000 resamples, seed 20260512
    Applied to: d_z, mean ΔR², mean R²
"""

from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── paths ───────────────────────────────────────────────────────────────────
STABILITY_DIR = Path("results/stability")
METHODS       = ["DC", "MI", "MIC", "XI"]
B             = 10_000
SEED          = 20260512
RNG           = np.random.default_rng(SEED)


# ── BCa bootstrap ────────────────────────────────────────────────────────────
def bca_ci(data: np.ndarray, statistic, alpha: float = 0.05,
           B: int = B, rng=None):
    """
    Bias-corrected and accelerated (BCa) bootstrap CI.
    Returns (lower, upper) rounded to 4 decimal places.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n     = len(data)
    theta = statistic(data)

    boot = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(B)
    ])

    # bias correction z0
    prop_less = np.mean(boot < theta)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = norm.ppf(prop_less)

    # acceleration a (jackknife)
    jack   = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jmean  = np.mean(jack)
    num    = np.sum((jmean - jack) ** 3)
    denom  = 6.0 * (np.sum((jmean - jack) ** 2) ** 1.5)
    a      = num / denom if denom != 0 else 0.0

    def adj(z):
        denom2 = 1.0 - a * (z0 + z)
        if abs(denom2) < 1e-12:
            return z
        return norm.cdf(z0 + (z0 + z) / denom2)

    lo = np.percentile(boot, 100.0 * adj(norm.ppf(alpha / 2)))
    hi = np.percentile(boot, 100.0 * adj(norm.ppf(1 - alpha / 2)))
    return round(float(lo), 4), round(float(hi), 4)


def dz(diff: np.ndarray) -> float:
    """Cohen's d_z = mean(diff) / std(diff, ddof=1)."""
    s = np.std(diff, ddof=1)
    return float(np.mean(diff) / s) if s > 0 else 0.0


# ── load per-method repetition files ────────────────────────────────────────
def load_reps():
    data = {}
    for method in METHODS:
        path = STABILITY_DIR / method / "repetitions.csv"
        df   = pd.read_csv(path).sort_values("rep").reset_index(drop=True)
        assert len(df) == 30, f"{method}: expected 30 rows, got {len(df)}"
        data[method] = df
    return data


# ── pairwise Cohen's d_z ─────────────────────────────────────────────────────
def compute_pairwise(data: dict):
    rows = []
    pairs = list(permutations(METHODS, 2))   # ordered: (A, B) means A vs B

    for m_a, m_b in pairs:
        r2_a   = data[m_a]["test_r2"].values
        r2_b   = data[m_b]["test_r2"].values
        mae_a  = data[m_a]["test_mae"].values
        mae_b  = data[m_b]["test_mae"].values

        diff_r2  = r2_a  - r2_b          # positive = A better on R²
        diff_mae = mae_b - mae_a          # positive = A better on MAE (lower is better)

        dz_r2  = dz(diff_r2)
        dz_mae = dz(diff_mae)

        lo_r2,  hi_r2  = bca_ci(diff_r2,  dz,  rng=RNG)
        lo_mae, hi_mae = bca_ci(diff_mae, dz,  rng=RNG)

        rows.append({
            "method_a":      m_a,
            "method_b":      m_b,
            "metric":        "R2",
            "mean_diff":     round(float(np.mean(diff_r2)), 5),
            "std_diff":      round(float(np.std(diff_r2, ddof=1)), 5),
            "dz":            round(dz_r2, 4),
            "bca_lo":        lo_r2,
            "bca_hi":        hi_r2,
            "n_pairs":       30,
        })
        rows.append({
            "method_a":      m_a,
            "method_b":      m_b,
            "metric":        "MAE",
            "mean_diff":     round(float(np.mean(diff_mae)), 5),
            "std_diff":      round(float(np.std(diff_mae, ddof=1)), 5),
            "dz":            round(dz_mae, 4),
            "bca_lo":        lo_mae,
            "bca_hi":        hi_mae,
            "n_pairs":       30,
        })

    return pd.DataFrame(rows)


def pairwise_to_matrix(pairwise_df: pd.DataFrame, metric: str):
    """Convert long-format pairwise df to 4×4 matrix for heatmap."""
    sub = pairwise_df[pairwise_df["metric"] == metric]
    mat = pd.DataFrame(np.nan, index=METHODS, columns=METHODS)
    for _, row in sub.iterrows():
        mat.loc[row["method_a"], row["method_b"]] = row["dz"]
    return mat


# ── bootstrap CIs on ΔR² per method  ────────────────────────────────────────
def compute_bootstrap_delta_r2(data: dict):
    """
    For each method: BCa CI on mean ΔR² = mean(train_r2 - test_r2).
    Also BCa CI on mean test R² itself.
    Writes bootstrap_ci_delta_r2.csv → Table 5 values.
    """
    rows = []
    for method in METHODS:
        df     = data[method]
        delta  = df["delta_r2"].values          # already computed in script 02
        r2     = df["test_r2"].values

        mean_d = float(np.mean(delta))
        std_d  = float(np.std(delta, ddof=1))
        lo_d, hi_d = bca_ci(delta, np.mean, rng=RNG)

        mean_r2_val = float(np.mean(r2))
        lo_r2, hi_r2 = bca_ci(r2, np.mean, rng=RNG)

        rows.append({
            "method":          method,
            "delta_r2_mean":   round(mean_d, 4),
            "delta_r2_std":    round(std_d,  4),
            "delta_r2_bca_lo": lo_d,
            "delta_r2_bca_hi": hi_d,
            "r2_mean":         round(mean_r2_val, 4),
            "r2_bca_lo":       lo_r2,
            "r2_bca_hi":       hi_r2,
        })

    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    data = load_reps()

    # 1. pairwise Cohen's d_z
    pairwise_df = compute_pairwise(data)
    pairwise_df.to_csv(STABILITY_DIR / "cohens_d_pairwise.csv", index=False)
    print("cohens_d_pairwise.csv written.")

    # 2. d_z matrices (for Figure 1 panel b)
    mat_r2  = pairwise_to_matrix(pairwise_df, "R2")
    mat_mae = pairwise_to_matrix(pairwise_df, "MAE")
    mat_r2.to_csv(STABILITY_DIR  / "cohens_d_matrix_r2.csv")
    mat_mae.to_csv(STABILITY_DIR / "cohens_d_matrix_mae.csv")
    print("cohens_d_matrix_r2.csv and cohens_d_matrix_mae.csv written.")

    # 3. bootstrap CIs on ΔR² (for Table 5)
    bca_df = compute_bootstrap_delta_r2(data)
    bca_df.to_csv(STABILITY_DIR / "bootstrap_ci_delta_r2.csv", index=False)
    print("\nbootstrap_ci_delta_r2.csv written:")
    print(bca_df[["method", "delta_r2_mean", "delta_r2_std",
                  "delta_r2_bca_lo", "delta_r2_bca_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
