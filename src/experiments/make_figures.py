r"""
make_figures.py
===============
Regenerate every manuscript figure from saved pipeline results.

  fig1_stat_tests.pdf               Figure 1  (Nemenyi p-values, Exp. 1 | paired d_z, Exp. 2)
  fig3_pareto.pdf                   Figure 2  (mean test R^2 vs CoV(R^2), Exp. 2)
  feature_agreement_matrix.pdf      Figure 3  (descriptors selected by >= 2 methods)
  full_feature_agreement_matrix.pdf Figure 4  (all descriptors selected by >= 1 method)

Inputs (repository layout)
--------------------------
  results/experiment1/nemenyi_matrix_r2.csv
  results/experiment1/friedman_results.json
  results/experiment1/feature_scores/{dc,mi,mic,xi}_scores.csv
  results/stability/cohens_d_matrix_r2.csv
  results/stability/aggregated_summary.csv

Outputs
-------
  results/figures_tables/<name>.pdf and .png, with the PDFs also copied to the
  repository root for \includegraphics.

Run from the repository root:
  python make_figures.py
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
R      = Path("results")
EXP1   = R / "experiment1"
STAB   = R / "stability"
SCORES = EXP1 / "feature_scores"
OUT    = R / "figures_tables"
OUT.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
METHODS      = ["DC", "MI", "MIC", "XI"]
METHOD_LABEL = {"DC": "DC", "MI": "MI", "MIC": "MIC", "XI": r"$\xi_n$"}
METHOD_COLOR = {"DC": "#d62728", "MI": "#1f77b4", "MIC": "#9467bd", "XI": "#2ca02c"}

N_FEATURES = 81
THRESHOLD  = 0.20     # retention level for the agreement figures
# Same floor rule as config.yaml (n_top_features: 16) and stability_30reps.py
N_SELECT   = int(np.floor(N_FEATURES * THRESHOLD))

FONT_BODY, FONT_TICK, FONT_SMALL = 10, 9, 8
FIG_W_COL, FIG_W_HALF = 6.0, 3.1

plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        FONT_BODY,
    "axes.titlesize":   FONT_BODY,
    "axes.labelsize":   FONT_BODY,
    "xtick.labelsize":  FONT_TICK,
    "ytick.labelsize":  FONT_TICK,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
})


# ── helpers ──────────────────────────────────────────────────────────────────
def load_csv(path, **kw):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, **kw)


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    shutil.copy(OUT / f"{name}.pdf", Path(f"{name}.pdf"))
    print(f"  wrote {OUT / name}.pdf (copied to repo root)")


def friedman_r2(friedman):
    """Friedman R^2 result from either JSON layout the pipeline has written."""
    if "r2" in friedman:  # experiment1_48configs.py layout
        return friedman["r2"]["statistic"], friedman["r2"]["p_value"]
    return friedman.get("chi2_r2", np.nan), friedman.get("p_r2", np.nan)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: Nemenyi p-values (Exp. 1) and paired Cohen's d_z (Exp. 2)
# ════════════════════════════════════════════════════════════════════════════
def fig1_stat_tests():
    nem = load_csv(EXP1 / "nemenyi_matrix_r2.csv", index_col=0)
    dz  = load_csv(STAB / "cohens_d_matrix_r2.csv", index_col=0)
    chi2_val, p_val = friedman_r2(load_json(EXP1 / "friedman_results.json"))

    labels = [METHOD_LABEL[m] for m in METHODS]
    k = len(METHODS)
    lower = np.tril(np.ones((k, k), dtype=bool), k=-1)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_COL * 1.35, FIG_W_HALF * 1.35))

    # panel (a): Nemenyi adjusted p, three discrete bins matching the caption
    ax = axes[0]
    nem_arr = nem.loc[METHODS, METHODS].values.astype(float)
    cmap_a = ListedColormap(["#67000d", "#ef3b2c", "#d9d9d9"])
    cmap_a.set_bad("white")
    norm_a = BoundaryNorm([0.0, 0.01, 0.05, 1.0], cmap_a.N)
    im_a = ax.imshow(np.where(lower, nem_arr, np.nan), cmap=cmap_a, norm=norm_a)
    for i in range(k):
        for j in range(k):
            if lower[i, j]:
                v = nem_arr[i, j]
                ax.text(j, i, "<0.001" if v < 0.001 else f"{v:.3f}",
                        ha="center", va="center", fontsize=FONT_SMALL,
                        color="white" if v < 0.05 else "black")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    p_txt = r"$p<0.0001$" if p_val < 1e-4 else rf"$p={p_val:.4f}$"
    ax.set_title("(a) Nemenyi adjusted $p$, Exp. 1\n"
                 rf"Friedman $\chi^2={chi2_val:.1f}$, " + p_txt,
                 fontsize=FONT_SMALL + 1)
    cb = fig.colorbar(im_a, ax=ax, shrink=0.75, ticks=[0.005, 0.03, 0.525])
    cb.ax.set_yticklabels([r"$<0.01$", r"$0.01$–$0.05$", r"$\geq 0.05$"],
                          fontsize=FONT_SMALL)

    # panel (b): paired d_z, column method minus row method
    ax = axes[1]
    dz_arr = dz.loc[METHODS, METHODS].values.astype(float)
    dz_plot = np.where(lower, -dz_arr, np.nan)
    vmax = np.nanmax(np.abs(dz_arr))
    cmap_b = plt.cm.Blues.copy()
    cmap_b.set_bad("white")
    im_b = ax.imshow(dz_plot, cmap=cmap_b, vmin=0, vmax=vmax)
    for i in range(k):
        for j in range(k):
            if lower[i, j]:
                v = dz_plot[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FONT_SMALL,
                        color="white" if v > vmax * 0.5 else "black")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_title("(b) Paired $d_z$ on test $R^2$, Exp. 2\n"
                 "column method minus row method", fontsize=FONT_SMALL + 1)
    fig.colorbar(im_b, ax=ax, shrink=0.75, label=r"$d_z$")

    plt.tight_layout()
    save(fig, "fig1_stat_tests")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: accuracy-variability plane (Exp. 2)
# ════════════════════════════════════════════════════════════════════════════
def fig2_accuracy_variability():
    df = load_csv(STAB / "aggregated_summary.csv").set_index("method")

    fig, ax = plt.subplots(figsize=(FIG_W_HALF * 1.5, FIG_W_HALF * 1.2))
    for m in METHODS:
        row = df.loc[m]
        ax.errorbar(row["r2_mean"], row["cov_r2_pct"], xerr=row["r2_std"],
                    fmt="o", color=METHOD_COLOR[m], markersize=7, capsize=3,
                    label=METHOD_LABEL[m])
        ax.annotate(METHOD_LABEL[m], (row["r2_mean"], row["cov_r2_pct"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=FONT_SMALL)

    ax.set_xlabel(r"Mean test $R^2$")
    ax.set_ylabel(r"CoV($R^2$), %")
    ax.set_title("Accuracy and variability over 30 repetitions\n"
                 "(LightGBM, 20% retention)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=FONT_SMALL, loc="center left")
    plt.tight_layout()
    save(fig, "fig3_pareto")   # filename kept for \includegraphics{fig3_pareto}


# ════════════════════════════════════════════════════════════════════════════
# Figures 3 and 4: feature selection agreement matrices
# ════════════════════════════════════════════════════════════════════════════
def load_selected(method):
    # experiment1_48configs.py writes DC_scores.csv etc.; accept either case
    path = SCORES / f"{method}_scores.csv"
    if not path.exists():
        path = SCORES / f"{method.lower()}_scores.csv"
    df = load_csv(path)
    feat_col, score_col = df.columns[0], df.columns[1]
    return set(df.nlargest(N_SELECT, score_col)[feat_col])


def draw_agreement(features, selected, counts, n_multi, title, name):
    n = len(features)
    fig, ax = plt.subplots(figsize=(5.0, max(3.0, n * 0.18 + 0.8)))

    for r, f in enumerate(features):
        for c, m in enumerate(METHODS):
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1, edgecolor="white", lw=0.5,
                facecolor="#2166ac" if f in selected[m] else "#f7f7f7"))

    if n > n_multi:   # divider between >=2-method and single-method rows
        ax.axhline(n_multi - 0.5, color="#e08214", lw=1.5, ls="--")

    ax.set_xlim(-0.5, len(METHODS) - 0.5)
    ax.set_ylim(n - 0.5, -0.5)            # first row at the top
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS])
    ax.set_yticks(range(n))
    ax.set_yticklabels(features, fontsize=FONT_SMALL)
    ax.set_xlabel("Method")
    ax.set_ylabel("Feature")
    ax.set_title(title, pad=6)

    ax2 = ax.twinx()
    ax2.set_ylim(n - 0.5, -0.5)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([str(counts[f]) for f in features], fontsize=FONT_SMALL)
    ax2.set_ylabel("Agreement\n(no. methods)", fontsize=FONT_SMALL)

    plt.tight_layout()
    save(fig, name)


def fig3_fig4_agreement():
    selected = {m: load_selected(m) for m in METHODS}
    union = set().union(*selected.values())
    counts = {f: sum(f in selected[m] for m in METHODS) for f in union}

    multi = sorted((f for f in union if counts[f] >= 2),
                   key=lambda f: (-counts[f], f))
    single = sorted((f for f in union if counts[f] == 1),
                    key=lambda f: ([m for m in METHODS if f in selected[m]][0], f))
    xi_only = selected["XI"] - selected["DC"] - selected["MI"] - selected["MIC"]

    print(f"  {N_SELECT} features per method | union {len(union)} | "
          f">=2 methods {len(multi)} | single {len(single)} | "
          f"xi-exclusive {len(xi_only)}")

    draw_agreement(multi, selected, counts, len(multi),
                   f"Descriptors selected by two or more methods "
                   f"({len(multi)} of {len(union)}, 20% retention, $p={N_FEATURES}$)",
                   "feature_agreement_matrix")
    draw_agreement(multi + single, selected, counts, len(multi),
                   f"All descriptors selected by at least one method "
                   f"({len(union)}, 20% retention, $p={N_FEATURES}$)",
                   "full_feature_agreement_matrix")


# ════════════════════════════════════════════════════════════════════════════
def main():
    print("Figure 1: statistical comparisons")
    fig1_stat_tests()
    print("Figure 2: accuracy-variability plane")
    fig2_accuracy_variability()
    print("Figures 3 and 4: feature agreement matrices")
    fig3_fig4_agreement()
    print(f"\nDone. Upload the four PDFs from the repo root to Overleaf.")


if __name__ == "__main__":
    main()
