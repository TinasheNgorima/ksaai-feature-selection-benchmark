"""
gen_feature_agreement.py
Regenerates feature_agreement_matrix.pdf and full_feature_agreement_matrix.pdf
from current feature score CSVs.

Outputs (copied to repo root for \includegraphics):
  feature_agreement_matrix.pdf      — features selected by >=2 methods (Fig 2)
  full_feature_agreement_matrix.pdf — all features in union (Fig 4 / appendix)

Run from repo root:
  python gen_feature_agreement.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── constants (match figures_tables.py) ──────────────────────────────────────
METHODS      = ["DC", "MI", "MIC", "XI"]
METHOD_LABEL = {"DC": "DC", "MI": "MI", "MIC": "MIC", "XI": r"$\xi_n$"}
THRESHOLD    = 20          # % retention threshold
FONT_BODY    = 10
FONT_TICK    = 9
FONT_SMALL   = 8
DPI          = 300
FIG_W_COL    = 11.0

plt.rcParams.update({
    "font.family":     "DejaVu Serif",
    "font.size":       FONT_BODY,
    "axes.titlesize":  FONT_BODY,
    "axes.labelsize":  FONT_BODY,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "figure.dpi":      DPI,
    "savefig.dpi":     DPI,
    "savefig.bbox":    "tight",
    "axes.facecolor":  "white",
    "figure.facecolor":"white",
})

SCORE_DIR = Path("results/experiment1/feature_scores")
OUT_DIR   = Path("results/figures_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── load scores and select top-N features at 20% threshold ───────────────────
def load_selected(method, n_total=81):
    csv = SCORE_DIR / f"{method.lower()}_scores.csv"
    df  = pd.read_csv(csv)
    # handle both 'score' and 'xi_score' / 'dc_score' etc column names
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    score_col = [c for c in df.columns if c != 'feature'][0]
    n_select  = max(1, int(np.ceil(n_total * THRESHOLD / 100)))
    selected  = set(df.nlargest(n_select, score_col)['feature'].tolist())
    return selected


selected = {m: load_selected(m) for m in METHODS}

# union of all selected features
union_all  = sorted(set.union(*selected.values()))

# features selected by >=2 methods
counts     = {f: sum(f in selected[m] for m in METHODS) for f in union_all}
multi      = sorted([f for f in union_all if counts[f] >= 2],
                    key=lambda f: -counts[f])
single     = sorted([f for f in union_all if counts[f] == 1])

print(f"Total union: {len(union_all)}")
print(f"Selected by >=2 methods: {len(multi)}")
print(f"Selected by exactly 1 method (xi_n-exclusive region): {len(single)}")
xi_only = selected['XI'] - selected['DC'] - selected['MI'] - selected['MIC']
print(f"XI-exclusive: {len(xi_only)} {sorted(xi_only)}")


def make_agreement_matrix(features, title, outname, show_divider=True):
    """Draw binary selection matrix: rows=features, cols=methods."""
    n_feat   = len(features)
    n_meth   = len(METHODS)
    labels   = [METHOD_LABEL[m] for m in METHODS]
    count_col = [counts[f] for f in features]

    # figure height scales with number of features
    fig_h = max(3, n_feat * 0.18 + 0.8)
    fig, ax = plt.subplots(figsize=(FIG_W_COL * 0.75, fig_h))

    # draw cells
    for row_i, feat in enumerate(features):
        for col_j, method in enumerate(METHODS):
            color = "#2166ac" if feat in selected[method] else "#f7f7f7"
            rect  = plt.Rectangle([col_j - 0.5, row_i - 0.5], 1, 1,
                                   facecolor=color, edgecolor="white", lw=0.5)
            ax.add_patch(rect)

    # divider between >=2 and ==1 method features
    if show_divider and multi and single:
        divider_y = len(multi) - 0.5
        ax.axhline(divider_y, color="#e08214", lw=1.5, ls="--")

    # axes
    ax.set_xlim(-0.5, n_meth - 0.5)
    ax.set_ylim(-0.5, n_feat - 0.5)
    ax.set_xticks(range(n_meth))
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(features, fontsize=FONT_SMALL)
    ax.xaxis.set_ticks_position("bottom")

    # agreement count on right axis
    ax2 = ax.twinx()
    ax2.set_ylim(-0.5, n_feat - 0.5)
    ax2.set_yticks(range(n_feat))
    ax2.set_yticklabels([str(count_col[i]) for i in range(n_feat)],
                         fontsize=FONT_SMALL)
    ax2.set_ylabel("Agreement\n(no. methods)", fontsize=FONT_SMALL)

    ax.set_xlabel("Method", fontsize=FONT_BODY)
    ax.set_ylabel("Feature", fontsize=FONT_BODY)
    ax.set_title(title, fontsize=FONT_BODY, pad=6)

    plt.tight_layout()
    out_pdf = OUT_DIR / f"{outname}.pdf"
    out_png = OUT_DIR / f"{outname}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    return out_pdf


# ── Fig 2: features selected by >=2 methods ──────────────────────────────────
make_agreement_matrix(
    multi + single,
    title=f"Feature selection agreement (20\\% threshold, $n=81$ descriptors)",
    outname="feature_agreement_matrix",
    show_divider=True,
)

# ── Fig 4 / appendix: all features in union ──────────────────────────────────
make_agreement_matrix(
    union_all,
    title=f"Complete feature selection agreement matrix "
          f"({len(union_all)} descriptors, 20\\% threshold)",
    outname="full_feature_agreement_matrix",
    show_divider=True,
)

# ── copy PDFs to repo root for \includegraphics ───────────────────────────────
import shutil
for name in ["feature_agreement_matrix", "full_feature_agreement_matrix"]:
    src = OUT_DIR / f"{name}.pdf"
    dst = Path(f"{name}.pdf")
    shutil.copy(src, dst)
    print(f"  copied {name}.pdf to repo root")

print("\nDone. Upload feature_agreement_matrix.pdf and full_feature_agreement_matrix.pdf to Overleaf.")
