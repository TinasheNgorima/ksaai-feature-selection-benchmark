"""
02_stability_30reps.py
======================
Experiment 2: Stability analysis — 30 independent repetitions of the optimal
configuration (LightGBM, 20% retention threshold) for all four feature-selection
methods.

Seed schedule (JMLR reproducibility criteria)
---------------------------------------------
  train-test split for repetition r : random_state = r  (r ∈ {0, …, 29})
  LightGBM / Optuna / mutual_info_regression : random_state = 42

Outputs (written to results/stability/)
---------------------------------------
  aggregated_summary.csv             — one row per method
  <METHOD>/repetitions.csv           — per-repetition metrics
  <METHOD>/jaccard_scores.csv        — all 435 pairwise Jaccard values
  <METHOD>/selected_features.csv     — feature set per repetition
  <METHOD>/summary.json
  latex_table.tex                    — Table 4 (stability) ready for inclusion
"""

# =============================================================================
# Imports
# =============================================================================
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import dcor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import mutual_info_regression as _mir
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================
SEED_MODEL     = 42          # LightGBM, Optuna, mutual_info_regression
THRESHOLD      = 0.20
N_FEATURES     = 81
N_TOP_FEATURES = int(N_FEATURES * THRESHOLD)   # 16
N_REPETITIONS  = 30
N_TRIALS       = 20
N_CV_FOLDS     = 5
TEST_SIZE      = 0.20
METHODS        = ["DC", "MI", "MIC", "XI"]

DATA_FILE   = Path("data/sc_mean.csv")           # merged CSV (features + target)
RESULTS_DIR = Path("results/stability")

# =============================================================================
# Logging
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / f"stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# Feature-selection methods
# =============================================================================

def xicor(x: np.ndarray, y: np.ndarray, ties: bool = True) -> float:
    n = len(x)
    y_sorted = y[np.argsort(x)]
    r = rankdata(y_sorted, method="average")
    if ties:
        l = rankdata(y_sorted, method="max")
        return float(1 - np.sum(np.abs(r[:-1] - r[1:])) / (2 * np.sum(l * (n - l) / n)))
    return float(1 - 3 * np.sum(np.abs(r[:-1] - r[1:])) / (n ** 2 - 1))


def score_xi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        try:
            scores[j] = xicor(X[:, j], y)
        except Exception:
            scores[j] = 0.0
    return scores


def score_mi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mutual_info_regression(X, y, random_state=SEED_MODEL)


def score_dc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X64 = X.astype(np.float64)
    y64 = y.astype(np.float64)
    scores = np.zeros(X64.shape[1])
    for j in range(X64.shape[1]):
        try:
            scores[j] = dcor.distance_correlation(X64[:, j], y64)
        except Exception:
            scores[j] = 0.0
    return scores
def score_mic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    def compute_mic(x, y): return _mir(x.reshape(-1,1), y.reshape(-1), random_state=42)[0]
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        try:
            scores[j] = compute_mic(X[:, j], y)
        except Exception:
            scores[j] = 0.0
    return scores


SCORERS = {"DC": score_dc, "MI": score_mi, "MIC": score_mic, "XI": score_xi}


def select_top_features(X: np.ndarray, y: np.ndarray, method: str, k: int) -> np.ndarray:
    scores = SCORERS[method](X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(scores)[-k:][::-1]


# =============================================================================
# LightGBM pipeline with Optuna tuning
# =============================================================================

def build_and_tune(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Tune LightGBM via Optuna (20 trials, 5-fold CV).
    Scaling is performed inside each fold; parameters are estimated on training
    folds only, preventing fold-wise preprocessing leakage.
    """
    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED_MODEL)

    def objective(trial: optuna.Trial) -> float:
        model = LGBMRegressor(
            n_estimators     = trial.suggest_int("n_estimators", 50, 300),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 30),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=SEED_MODEL, n_jobs=1, verbose=-1, force_col_wise=True,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        mae_scores = []
        for tr_idx, val_idx in cv.split(X_train):
            pipe.fit(X_train[tr_idx], y_train[tr_idx])
            mae_scores.append(mean_absolute_error(y_train[val_idx], pipe.predict(X_train[val_idx])))
        return float(np.mean(mae_scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED_MODEL),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    p = study.best_params
    best_model = LGBMRegressor(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], min_child_samples=p["min_child_samples"],
        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
        random_state=SEED_MODEL, n_jobs=-1, verbose=-1, force_col_wise=True,
    )
    best_pipe = Pipeline([("scaler", StandardScaler()), ("model", best_model)])
    best_pipe.fit(X_train, y_train)
    return best_pipe


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    df = pd.read_csv(DATA_FILE)
    y = df["critical_temp"].astype(np.float32).values
    X_df = df.drop(columns=["critical_temp", "material"], errors="ignore")
    X = X_df.astype(np.float32).values
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]  # positional indices
    assert X.shape == (15_542, 81), f"Unexpected shape: {X.shape}"
    log.info(f"Dataset loaded: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, feature_names


# =============================================================================
# Stability experiment — one method
# =============================================================================

def run_stability(X: np.ndarray, y: np.ndarray, feature_names: list, method: str) -> dict:
    """Run 30-repetition stability experiment for one feature-selection method."""
    log.info(f"\n{'='*70}\nMETHOD: {method}\n{'='*70}")

    rep_records    = []
    metric_tracker = {k: [] for k in ["r2_train", "r2_test", "mae_test", "rmse_test"]}
    all_feat_sets  = []
    all_timings    = []

    for rep in tqdm(range(N_REPETITIONS), desc=method):
        t0 = time.perf_counter()

        # Repetition r uses random_state = r  (distinct split per repetition)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=rep
        )

        # Feature selection on training data only
        top_idx = select_top_features(X_train, y_train, method, N_TOP_FEATURES)
        feat_set = [feature_names[i] for i in top_idx]
        all_feat_sets.append(feat_set)

        X_tr_sel = X_train[:, top_idx]
        X_te_sel = X_test[:, top_idx]

        # Hyperparameter optimisation (training CV only)
        best_pipe = build_and_tune(X_tr_sel, y_train)

        # Evaluation
        y_pred_train = best_pipe.predict(X_tr_sel)
        y_pred_test  = best_pipe.predict(X_te_sel)

        r2_tr  = float(r2_score(y_train, y_pred_train))
        r2_te  = float(r2_score(y_test,  y_pred_test))
        mae_te = float(mean_absolute_error(y_test, y_pred_test))
        rmse_te= float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        elapsed = time.perf_counter() - t0

        metric_tracker["r2_train"].append(r2_tr)
        metric_tracker["r2_test"].append(r2_te)
        metric_tracker["mae_test"].append(mae_te)
        metric_tracker["rmse_test"].append(rmse_te)
        all_timings.append(elapsed)

        rep_records.append({
            "rep": rep, "random_state_split": rep,
            "r2_train": r2_tr, "r2_test": r2_te,
            "mae_test": mae_te, "rmse_test": rmse_te,
            "delta_r2": r2_tr - r2_te,
            "elapsed_s": elapsed,
        })

    # Summary statistics
    def stats(values):
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std" : float(arr.std(ddof=1)),
            "cv"  : float(arr.std(ddof=1) / arr.mean() * 100),
            "min" : float(arr.min()),
            "max" : float(arr.max()),
        }

    summary = {k: stats(v) for k, v in metric_tracker.items()}

    # Jaccard similarity across all C(30,2) = 435 pairs
    jaccard_scores = []
    for i in range(N_REPETITIONS):
        for j in range(i + 1, N_REPETITIONS):
            s1, s2 = set(all_feat_sets[i]), set(all_feat_sets[j])
            jaccard_scores.append(len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0)

    jac_arr = np.array(jaccard_scores)
    jaccard_summary = {
        "mean": float(jac_arr.mean()),
        "std" : float(jac_arr.std()),
    }

    log.info(
        f"[{method}]  R² = {summary['r2_test']['mean']:.3f} ± {summary['r2_test']['std']:.3f}"
        f"  CV = {summary['r2_test']['cv']:.1f}%"
        f"  Jaccard = {jaccard_summary['mean']:.3f} ± {jaccard_summary['std']:.3f}"
    )

    return {
        "method"         : method,
        "summary_metrics": summary,
        "jaccard"        : jaccard_summary,
        "jaccard_all"    : jaccard_scores,
        "rep_records"    : rep_records,
        "feat_sets"      : all_feat_sets,
        "timings"        : all_timings,
    }


# =============================================================================
# Save results
# =============================================================================

def save_method_results(result: dict):
    method = result["method"]
    method_dir = RESULTS_DIR / method
    method_dir.mkdir(exist_ok=True)

    # Per-repetition CSV
    pd.DataFrame(result["rep_records"]).to_csv(method_dir / "repetitions.csv", index=False)

    # Jaccard scores CSV
    pd.DataFrame({
        "pair_id": range(len(result["jaccard_all"])),
        "jaccard" : result["jaccard_all"],
    }).to_csv(method_dir / "jaccard_scores.csv", index=False)

    # Selected feature sets
    pd.DataFrame({
        "rep"     : range(N_REPETITIONS),
        "features": [str(fs) for fs in result["feat_sets"]],
    }).to_csv(method_dir / "selected_features.csv", index=False)

    # Summary JSON
    summary_json = {
        "method"         : method,
        "n_repetitions"  : N_REPETITIONS,
        "threshold"      : THRESHOLD,
        "n_top_features" : N_TOP_FEATURES,
        "seed_split"     : "sequential (rep index)",
        "seed_model"     : SEED_MODEL,
        "summary_metrics": result["summary_metrics"],
        "jaccard"        : result["jaccard"],
    }
    with open(method_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)


# =============================================================================
# LaTeX table generator
# =============================================================================

def generate_latex_table(all_results: dict, out_path: Path):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Bootstrap stability metrics across feature-selection methods "
        r"(LightGBM, 20\% threshold, 30 repetitions).}",
        r"\label{tab:stability_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & $R^2$ (mean $\pm$ std) & CV$(R^2)$ (\%) & Jaccard (mean $\pm$ std) & "
        r"Train--Test Gap $\Delta R^2$ \\",
        r"\midrule",
    ]
    for method, res in all_results.items():
        r2  = res["summary_metrics"]["r2_test"]
        dg  = res["summary_metrics"]["r2_train"]  # approximation of gap via train mean
        jac = res["jaccard"]
        lines.append(
            f"{method} & {r2['mean']:.3f} $\\pm$ {r2['std']:.3f} & "
            f"{r2['cv']:.2f} & "
            f"{jac['mean']:.3f} $\\pm$ {jac['std']:.3f} & "
            f"{dg['mean'] - r2['mean']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)
    out_path.write_text(latex)
    log.info(f"LaTeX table written to {out_path}")
    print("\n" + latex)


# =============================================================================
# Main
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("Experiment 2: 30-repetition stability analysis — LightGBM, top 20%")
    log.info(f"Seed schedule — split: sequential (rep index), model/Optuna/MI: {SEED_MODEL}")
    log.info("=" * 70)

    X, y, feature_names = load_data()

    all_results = {}
    for method in METHODS:
        result = run_stability(X, y, feature_names, method)
        all_results[method] = result
        save_method_results(result)

    # Aggregated summary CSV
    rows = []
    for method, res in all_results.items():
        r2  = res["summary_metrics"]["r2_test"]
        mae = res["summary_metrics"]["mae_test"]
        rm  = res["summary_metrics"]["rmse_test"]
        jac = res["jaccard"]
        rows.append({
            "Method"            : method,
            "Mean_R2"           : r2["mean"],
            "Std_R2"            : r2["std"],
            "CV_R2_pct"         : r2["cv"],
            "Mean_RMSE"         : rm["mean"],
            "Std_RMSE"          : rm["std"],
            "Mean_MAE"          : mae["mean"],
            "Std_MAE"           : mae["std"],
            "Jaccard_mean"      : jac["mean"],
            "Jaccard_std"       : jac["std"],
        })

    agg_df = pd.DataFrame(rows)
    agg_df.to_csv(RESULTS_DIR / "aggregated_summary.csv", index=False)
    log.info("\n" + agg_df.to_string(index=False))

    generate_latex_table(all_results, RESULTS_DIR / "latex_table.tex")
    log.info(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
