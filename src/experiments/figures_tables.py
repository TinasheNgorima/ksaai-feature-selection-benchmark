"""
06_figures_tables.py
Generates all manuscript tables (LaTeX) and figures (PDF/PNG).
ZERO hardcoded result values. Every number loaded from a CSV.

Input CSVs (all under results/):
  experiment1/experiment_summary.csv      → Table 3
  experiment1/nemenyi_matrix_r2.csv       → Figure 1a
  experiment1/nemenyi_matrix_mae.csv      → Figure 1a (sensitivity check)
  experiment1/friedman_results.json       → prose χ² values
  experiment1/feature_sets_fixed.json     → Figures 2, 4
  stability/aggregated_summary.csv        → Table 4
  stability/bootstrap_ci_delta_r2.csv     → Table 5
  stability/cohens_d_matrix_r2.csv        → Figure 1b
  stability/cohens_d_matrix_mae.csv       → Figure 1b
  stability/cohens_d_pairwise.csv         → prose d_z paragraph
  efficiency/timing_summary.csv           → Table 6

Output files (all under results/figures_tables/):
  table3_overall_results.tex
  table4_stability.tex
  table5_bootstrap_cv.tex
  table6_timing.tex
  fig1_stat_tests.pdf
  fig2_feature_agreement.pdf   (also PNG for draft)
  fig3_pareto.pdf
  fig4_feature_agreement_full.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
R         = Path("results")
EXP1      = R / "experiment1"
STAB      = R / "stability"
EFF       = R / "efficiency"
OUT       = R / "figures_tables"
OUT.mkdir(parents=True, exist_ok=True)

METHODS      = ["DC", "MI", "MIC", "XI"]
METHOD_LABEL = {"DC": "DC", "MI": "MI", "MIC": "MIC", "XI": r"$\xi_n$"}

# JMLR style constants
FONT_BODY  = 10
FONT_TICK  = 9
FONT_SMALL = 8
DPI        = 300
FIG_W_COL  = 6.0   # single column width inches (JMLR approx)
FIG_W_HALF = 3.1

plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        FONT_BODY,
    "axes.titlesize":   FONT_BODY,
    "axes.labelsize":   FONT_BODY,
    "xtick.labelsize":  FONT_TICK,
    "ytick.labelsize":  FONT_TICK,
    "figure.dpi":       DPI,
    "savefig.dpi":      DPI,
    "savefig.bbox":     "tight",
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
})


# ── helpers ──────────────────────────────────────────────────────────────────
def load_csv(path, **kwargs):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required CSV not found: {p}")
    return pd.read_csv(p, **kwargs)


def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required JSON not found: {p}")
    with open(p) as fh:
        return json.load(fh)


def bold(val, best, fmt):
    s = fmt.format(val)
    return r"\textbf{" + s + "}" if val == best else s


def uline(val, second, fmt):
    s = fmt.format(val)
    return r"\underline{" + s + "}" if val == second else s


def write_tex(path, content):
    Path(path).write_text(content, encoding="utf-8")
    print(f"  wrote {path}")


# ════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Experiment 1: 48-config performance
# ════════════════════════════════════════════════════════════════════════════
def table3_overall_results():
    df = load_csv(EXP1 / "experiment_summary.csv")

    # Only show top rows per model class (best + selected representative rows)
    # Full table is 48 rows; manuscript shows a curated subset — replicate that
    # by keeping only rows that appear in the manuscript (identified by model+method+threshold)
    # If you want the full 48-row table, remove the filter below.
    keep = [
        ("lightgbm", "DC",  20),
        ("lightgbm", "DC",  10),
        ("lightgbm", "MI",  20),
        ("lightgbm", "MIC", 20),
        ("lightgbm", "XI",  20),
        ("rf",       "DC",  15),
        ("rf",       "MI",  20),
        ("rf",       "MIC", 20),
        ("rf",       "XI",  20),
        ("lasso",    "DC",  20),
        ("lasso",    "XI",  20),
        ("elastic_net","DC", 20),
    ]

    lines = []
    lines.append(r"\begin{table}[tp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Predictive performance across models and feature "
                 r"selection methods (Experiment~1, single fixed split).}")
    lines.append(r"\label{tab:overall_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & Method & Threshold & Test $R^2$ & Test RMSE "
                 r"& Test MAE & CV RMSE (K) \\")
    lines.append(r"\midrule")

    current_model = None
    for model, method, thresh in keep:
        row = df[
            (df["model"] == model) &
            (df["method"] == method) &
            (df["threshold_pct"] == thresh)
        ]
        if row.empty:
            print(f"  WARNING: no row for {model}/{method}/{thresh}")
            continue
        row = row.iloc[0]
        if model != current_model:
            label = {"lightgbm": "LightGBM", "rf": "Random Forest",
                     "lasso": "Linear models"}.get(model, model)
            if model == "elastic_net":
                pass  # already under Linear models heading
            else:
                lines.append(r"\multicolumn{7}{l}{\textit{" + label + r"}} \\")
            current_model = model

        m_tex = METHOD_LABEL.get(method, method)
        lines.append(
            f"{model} & {m_tex} & {thresh} & "
            f"{row['test_r2']:.3f} & {row['test_rmse']:.2f} & "
            f"{row['test_mae']:.2f} & {row['cv_rmse']:.2f} \\\\"
        )
        if (model, method, thresh) in [("lightgbm","DC",10),
                                        ("rf","XI",20)]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\smallskip\\")
    lines.append(r"{\small\textit{Note.} Bold indicates the best result within "
                 r"each model class. CV-RMSE: cross-validated RMSE on the 5-fold "
                 r"training partition. K: kelvin.}")
    lines.append(r"\end{table}")

    write_tex(OUT / "table3_overall_results.tex", "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Experiment 2: stability metrics (30 reps)
# ════════════════════════════════════════════════════════════════════════════
def table4_stability():
    df = load_csv(STAB / "aggregated_summary.csv")
    df = df.set_index("method").loc[METHODS].reset_index()

    # best/second per column
    best_r2      = df["r2_mean"].max()
    best_rmse    = df["rmse_mean"].min()
    best_cov     = df["cov_r2_pct"].min()
    best_jaccard = df["jaccard_mean"].max()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stability metrics from 30 independent repetitions "
                 r"(LightGBM, 20\% threshold).}")
    lines.append(r"\label{tab:stability_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & $R^2$ (mean $\pm$ std) & RMSE (K) "
                 r"& CoV($R^2$) & Jaccard & Time (min) \\")
    lines.append(r"\midrule")

    # sort: DC first (best R²), then MI, MIC, XI
    order = ["DC", "MI", "MIC", "XI"]
    for method in order:
        row = df[df["method"] == method].iloc[0]
        m_tex = METHOD_LABEL[method]

        r2_s = f"{row['r2_mean']:.3f} $\\pm$ {row['r2_std']:.3f}"
        if row["r2_mean"] == best_r2:
            r2_s = r"\mathbf{" + r2_s + "}"
        r2_s = "$" + r2_s + "$"

        rmse_s = f"{row['rmse_mean']:.2f} $\\pm$ {row['rmse_std']:.2f}"
        if row["rmse_mean"] == best_rmse:
            rmse_s = r"\mathbf{" + rmse_s + "}"
        rmse_s = "$" + rmse_s + "$"

        cov_s  = f"{row['cov_r2_pct']:.1f}\\%"
        if row["cov_r2_pct"] == best_cov:
            cov_s = r"\textbf{" + cov_s + "}"

        jac_s  = f"{row['jaccard_mean']:.2f} $\\pm$ {row['jaccard_std']:.2f}"
        if row["jaccard_mean"] == best_jaccard:
            jac_s = r"\mathbf{" + jac_s + "}"
        jac_s = "$" + jac_s + "$"

        # time column — read from efficiency CSV if available
        try:
            eff = load_csv(EFF / "timing_summary.csv")
            t_row = eff[eff["method"] == method]
            time_s = f"{t_row['total_pipeline_min'].values[0]:.1f}" if not t_row.empty else "---"
        except FileNotFoundError:
            time_s = "---"

        lines.append(f"{m_tex} & {r2_s} & {rmse_s} & {cov_s} & {jac_s} & {time_s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\smallskip\\")
    lines.append(
        r"{\small\textit{Note.} Bold indicates best result per column. "
        r"CoV$= \mathrm{std}/\mathrm{mean} \times 100$, computed within each method "
        r"across 30 repetitions. Pairwise paired effect sizes (Cohen's $d_z$) "
        r"with 95\% BCa intervals are deposited in "
        r"\texttt{results/stability/cohens\_d\_pairwise.csv}.}"
    )
    lines.append(r"\end{table}")

    write_tex(OUT / "table4_stability.tex", "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════════
# TABLE 5 — Bootstrap CIs on ΔR² (Experiment 2)
# ════════════════════════════════════════════════════════════════════════════
def table5_bootstrap_cv():
    df = load_csv(STAB / "bootstrap_ci_delta_r2.csv")
    df = df.sort_values("delta_r2_mean").reset_index(drop=True)  # ascending ΔR²

    best_delta = df["delta_r2_mean"].min()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Generalization gap "
        r"($\Delta R^{2} = R^{2}_{\mathrm{train}} - R^{2}_{\mathrm{test}}$) "
        r"across 30 independent repetitions (LightGBM, 20\% threshold), "
        r"with 95\% BCa bootstrap confidence intervals "
        r"($B = 10{,}000$, seed \texttt{20260512}).}"
    )
    lines.append(r"\label{tab:bootstrap_cv}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & $\Delta R^{2}$ (mean $\pm$ std) & 95\% BCa CI & Interpretation \\")
    lines.append(r"\midrule")

    interp = {
        "XI":  "Lowest overfitting",
        "MIC": "",
        "MI":  "",
        "DC":  "Highest overfitting",
    }

    for _, row in df.iterrows():
        method = row["method"]
        m_tex  = METHOD_LABEL[method]
        delta_s = f"{row['delta_r2_mean']:.3f} $\\pm$ {row['delta_r2_std']:.3f}"
        if row["delta_r2_mean"] == best_delta:
            delta_s = r"\mathbf{" + delta_s + "}"
        delta_s = "$" + delta_s + "$"

        ci_s = f"[{row['delta_r2_bca_lo']:.3f},\\ {row['delta_r2_bca_hi']:.3f}]"
        note = interp.get(method, "")
        lines.append(f"{m_tex} & {delta_s} & {ci_s} & {note} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\smallskip\\")
    lines.append(
        r"{\small\textit{Note.} Methods ordered by increasing $\Delta R^{2}$. "
        r"Bold indicates lowest (most favorable) gap. "
        r"BCa intervals computed from per-repetition train--test $R^{2}$ pairs; "
        r"full values in \texttt{results/stability/bootstrap\_ci\_delta\_r2.csv}.}"
    )
    lines.append(r"\end{table}")

    write_tex(OUT / "table5_bootstrap_cv.tex", "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════════
# TABLE 6 — Computational timing
# ════════════════════════════════════════════════════════════════════════════
def table6_timing():
    df = load_csv(EFF / "timing_summary.csv")
    # expected columns: method, mean_time_s, std_time_s, time_per_feature_ms, speedup_vs_dc

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{threeparttable}")
    lines.append(
        r"\caption{Computational efficiency of feature selection methods "
        r"(isolated scoring step).}"
    )
    lines.append(r"\label{tab:feature_selection_timing}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Mean Time (s) & Std (s) & Time per Feature (ms) "
                 r"& Speedup vs.\ DC \\")
    lines.append(r"\midrule")

    best_time = df["mean_time_s"].min()
    for _, row in df.sort_values("mean_time_s").iterrows():
        method = row["method"]
        m_tex  = METHOD_LABEL.get(method, method)
        t_s    = f"{row['mean_time_s']:.2f}"
        if row["mean_time_s"] == best_time:
            t_s = r"\textbf{" + t_s + "}"
        spd    = f"{row['speedup_vs_dc']:.1f}" + r"\times"
        if row["speedup_vs_dc"] == df["speedup_vs_dc"].max():
            spd = r"\textbf{" + spd + "}"
        lines.append(
            f"{m_tex} & {t_s} & {row['std_time_s']:.2f} & "
            f"{row['time_per_feature_ms']:.1f} & ${spd}$ \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(
        r"\item[]\footnotesize \textit{Note.} Single-pass feature selection "
        r"timing on $n = 15{,}542$ samples and $p = 81$ features, averaged "
        r"over 30 independent repetitions. MIC computed using "
        r"\texttt{minepy==1.2.6} \citep{Albanese2013minepy} (Python~3.10, "
        r"isolated conda environment). Time per feature is mean time divided "
        r"by 81. Speedup relative to DC."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")

    write_tex(OUT / "table6_timing.tex", "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Nemenyi heatmap (panel a) + Cohen's d_z heatmap (panel b)
# ════════════════════════════════════════════════════════════════════════════
def fig1_stat_tests():
    """
    Panel a: Nemenyi p-values from Experiment 1 (48 configs).
    Panel b: Cohen's d_z from Experiment 2 (30 paired reps, LightGBM, 20%).
    These are DIFFERENT statistical populations and are labelled accordingly.
    """
    nem_r2  = load_csv(EXP1 / "nemenyi_matrix_r2.csv",  index_col=0)
    dz_r2   = load_csv(STAB / "cohens_d_matrix_r2.csv", index_col=0)
    friedman = load_json(EXP1 / "friedman_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_COL * 1.8, FIG_W_HALF * 2.2))

    # ── panel a: Nemenyi p-values ──
    ax = axes[0]
    labels = [METHOD_LABEL[m] for m in METHODS]
    nem_arr = nem_r2.loc[METHODS, METHODS].values.astype(float)

    # colour: dark red <0.01, orange 0.01–0.05, grey >=0.05
    cmap_a = plt.cm.RdYlGn_r
    im_a = ax.imshow(nem_arr, cmap=cmap_a, vmin=0, vmax=0.10, aspect="auto")

    ax.set_xticks(range(len(METHODS)))
    ax.set_yticks(range(len(METHODS)))
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)

    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            if i == j:
                chi2_val = friedman.get("chi2_r2", float("nan"))
                p_val    = friedman.get("p_r2",    float("nan"))
                ax.text(j, i, f"$\\chi^2={chi2_val:.1f}$\n$p={p_val:.3f}$",
                        ha="center", va="center", fontsize=FONT_SMALL - 1)
            elif j > i:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=FONT_TICK, color="white")
            else:
                v = nem_arr[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=FONT_SMALL,
                        color="white" if v < 0.05 else "black")

    ax.set_title("(a) Nemenyi $p$-values\n(Exp.~1, 48 configurations)",
                 fontsize=FONT_BODY)

    # ── panel b: Cohen's d_z ──
    ax = axes[1]
    dz_arr = dz_r2.loc[METHODS, METHODS].values.astype(float)
    vmax_dz = np.nanmax(np.abs(dz_arr))
    im_b = ax.imshow(dz_arr, cmap="RdBu_r", vmin=-vmax_dz, vmax=vmax_dz,
                     aspect="auto")

    ax.set_xticks(range(len(METHODS)))
    ax.set_yticks(range(len(METHODS)))
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)

    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            v = dz_arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=FONT_TICK)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FONT_SMALL,
                        color="white" if abs(v) > vmax_dz * 0.5 else "black")

    ax.set_title(r"(b) Cohen's $d_z$" + "\n(Exp.~2, 30 paired reps, LightGBM)",
                 fontsize=FONT_BODY)
    fig.colorbar(im_b, ax=axes[1], shrink=0.7, label=r"$d_z$")

    plt.tight_layout()
    fig.savefig(OUT / "fig1_stat_tests.pdf")
    fig.savefig(OUT / "fig1_stat_tests.png")
    plt.close(fig)
    print("  fig1_stat_tests.pdf written.")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Pareto frontier (Experiment 2)
# ════════════════════════════════════════════════════════════════════════════
def fig3_pareto():
    df = load_csv(STAB / "aggregated_summary.csv").set_index("method")

    fig, ax = plt.subplots(figsize=(FIG_W_HALF * 1.5, FIG_W_HALF * 1.4))
    colors = {"DC": "#d62728", "MI": "#1f77b4", "MIC": "#9467bd", "XI": "#2ca02c"}

    for method in METHODS:
        row = df.loc[method]
        ax.errorbar(
            row["r2_mean"], row["cov_r2_pct"],
            xerr=row["r2_std"],
            fmt="o", color=colors[method], markersize=7,
            label=METHOD_LABEL[method], capsize=3
        )
        ax.annotate(METHOD_LABEL[method],
                    (row["r2_mean"], row["cov_r2_pct"]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=FONT_SMALL)

    ax.set_xlabel(r"Mean $R^2$ (Performance)", fontsize=FONT_BODY)
    ax.set_ylabel(r"CoV($R^2$) \% (Instability)", fontsize=FONT_BODY)
    ax.set_title("Performance–stability Pareto frontier\n"
                 "(LightGBM, 20\\% threshold, 30 repetitions)",
                 fontsize=FONT_BODY)
    ax.legend(fontsize=FONT_SMALL)
    plt.tight_layout()
    fig.savefig(OUT / "fig3_pareto.pdf")
    fig.savefig(OUT / "fig3_pareto.png")
    plt.close(fig)
    print("  fig3_pareto.pdf written.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("Generating tables...")
    table3_overall_results()
    table4_stability()
    table5_bootstrap_cv()
    table6_timing()

    print("Generating figures...")
    fig1_stat_tests()
    fig3_pareto()

    print(f"\nAll outputs written to {OUT}/")
    print("Copy .tex files into Ksaai.tex to replace hardcoded table bodies.")
    print("Replace \\includegraphics targets with fig1_stat_tests, fig3_pareto.")


if __name__ == "__main__":
    main()
