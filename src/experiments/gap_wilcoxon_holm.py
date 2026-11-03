"""
gap_wilcoxon_holm.py
Paired two-sided Wilcoxon signed-rank tests on the generalization gap
(Delta R2 = train R2 - test R2) over the 30 shared Experiment 2 splits,
with Holm correction over all six method pairs.

Input : results/stability/<METHOD>/repetitions.csv  (or a flat folder of
        repetitions files passed with --files)
Output: results/stability/gap_wilcoxon_holm.csv
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METHODS = ["DC", "MI", "MIC", "XI"]
STAB = Path("results/stability")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", help="repetitions.csv files (any order)")
    args = ap.parse_args()
    files = args.files or [STAB / m / "repetitions.csv" for m in METHODS]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    gap = df.pivot(index="rep", columns="method", values="delta_r2")[METHODS]

    rows = []
    for a, b in combinations(METHODS, 2):
        d = gap[a] - gap[b]
        res = wilcoxon(gap[a], gap[b], alternative="two-sided")
        rows.append(dict(method_a=a, method_b=b, n=len(d),
                         mean_diff=d.mean(), n_a_larger=int((d > 0).sum()),
                         n_b_larger=int((d < 0).sum()),
                         W=res.statistic, p_raw=res.pvalue))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"])
    STAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(STAB / "gap_wilcoxon_holm.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
