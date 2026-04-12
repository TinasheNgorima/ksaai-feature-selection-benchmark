"""
01_experiment1_48configs.py
===========================
Experiment 1: Comprehensive comparison of four feature-selection methods across
48 configurations (4 methods × 3 thresholds × 4 models) on the superconductivity
dataset (n = 15,542, p = 81).

Seed schedule (JMLR reproducibility criteria)
---------------------------------------------
  train-test split      : random_state = 53
  cross-validation folds: random_state = 42
  model initialisation  : random_state = 42
  mutual_info_regression: random_state = 42
  Optuna                : random_state = 42

Outputs (written to results/experiment1/)
-----------------------------------------
  experiment_summary.csv          — one row per configuration
  feature_scores/                 — correlation scores for each method
  reduced_features/               — feature-reduced training/test CSV files
  experiment_<timestamp>.log
"""

# =============================================================================
# Imports
# =============================================================================
import logging
import itertools
import json
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import dcor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import mutual_info_regression as _mir
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================
SEED_SPLIT = 53          # train-test partition seed
SEED_MODEL = 42          # CV folds, model init, MI, Optuna
TEST_SIZE  = 0.20
VAL_SIZE   = 0.20        # fraction of train_val

THRESHOLDS     = [10, 15, 20]          # top-k percent
METHODS        = ["dc", "mi", "mic", "xi"]
MODEL_NAMES    = ["lasso", "elastic_net", "lightgbm", "rf"]
N_TRIALS_OPTUNA = 50
N_CV_FOLDS      = 5

DATA_FILE = Path("data/sc_mean.csv")
# TARGET_FILE removed — using sc_mean.csv
RESULTS_DIR   = Path("results/experiment1")

# =============================================================================
# Logging
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / f"experiment_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# Feature-selection methods
# =============================================================================

def xicor(x: np.ndarray, y: np.ndarray, ties: bool = True) -> float:
    """Chatterjee's ξₙ correlation coefficient."""
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
    """Mutual Information (sklearn). random_state=42 for reproducibility."""
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


SCORERS = {"xi": score_xi, "mi": score_mi, "dc": score_dc, "mic": score_mic}


def select_top_features(X: np.ndarray, y: np.ndarray, method: str, k: int) -> np.ndarray:
    """Return indices of top-k features ranked by method, computed on training data only."""
    scores = SCORERS[method](X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(scores)[-k:][::-1]


# =============================================================================
# Model building (Optuna Bayesian optimisation)
# =============================================================================

def build_and_tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    n_trials: int = N_TRIALS_OPTUNA,
) -> Pipeline:
    """
    Build a sklearn Pipeline with StandardScaler and tune hyperparameters
    via Optuna on 5-fold CV on the training split.
    Scaling parameters are estimated from training folds only (leak-proof).
    """
    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED_MODEL)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "lasso":
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            model = Lasso(alpha=alpha, max_iter=10_000, random_state=SEED_MODEL)
        elif model_name == "elastic_net":
            alpha  = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            l1r    = trial.suggest_float("l1_ratio", 0.1, 0.9)
            model  = ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=10_000, random_state=SEED_MODEL)
        elif model_name == "lightgbm":
            model = LGBMRegressor(
                n_estimators     = trial.suggest_int("n_estimators", 50, 400),
                max_depth        = trial.suggest_int("max_depth", 3, 10),
                learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                num_leaves       = trial.suggest_int("num_leaves", 20, 150),
                min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
                subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                random_state=SEED_MODEL,
                n_jobs=1,
                verbose=-1,
                force_col_wise=True,
            )
        elif model_name == "rf":
            model = RandomForestRegressor(
                n_estimators = trial.suggest_int("n_estimators", 50, 300),
                max_depth    = trial.suggest_int("max_depth", 3, 20),
                min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
                random_state=SEED_MODEL,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        mae_scores = []
        for tr_idx, val_idx in cv.split(X_train):
            pipe.fit(X_train[tr_idx], y_train[tr_idx])
            preds = pipe.predict(X_train[val_idx])
            mae_scores.append(mean_absolute_error(y_train[val_idx], preds))
        return float(np.mean(mae_scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED_MODEL))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Refit best pipeline on full training data
    best_params = study.best_params
    if model_name == "lasso":
        best_model = Lasso(alpha=best_params["alpha"], max_iter=10_000, random_state=SEED_MODEL)
    elif model_name == "elastic_net":
        best_model = ElasticNet(alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"],
                                max_iter=10_000, random_state=SEED_MODEL)
    elif model_name == "lightgbm":
        best_model = LGBMRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            min_child_samples=best_params["min_child_samples"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            random_state=SEED_MODEL, n_jobs=-1, verbose=-1, force_col_wise=True,
        )
    elif model_name == "rf":
        best_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            random_state=SEED_MODEL, n_jobs=-1,
        )

    best_pipe = Pipeline([("scaler", StandardScaler()), ("model", best_model)])
    best_pipe.fit(X_train, y_train)
    return best_pipe


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """Load superconductivity dataset; return (X, y, feature_names)."""
    df = pd.read_csv(DATA_FILE)
    X_df = df.drop(columns=["critical_temp", "material"], errors="ignore").astype(np.float32)
    y_df = df["critical_temp"].astype(np.float32)
    assert X_df.shape == (15_542, 81), f"Unexpected shape: {X_df.shape}"
    log.info(f"Dataset loaded: {X_df.shape[0]} samples × {X_df.shape[1]} features")
    return X_df.values, y_df.values, X_df.columns.tolist()


# =============================================================================
# Single-configuration runner
# =============================================================================

def run_config(
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    method: str,
    threshold: int,
    model_name: str,
) -> dict:
    """Run one (method, threshold, model) configuration."""
    n_features = int(round(81 * threshold / 100))
    # Feature selection on training data only
    top_idx = select_top_features(X_train, y_train, method, n_features)
    selected_names = [feature_names[i] for i in top_idx]

    X_tr_sel  = X_train[:, top_idx]
    X_val_sel = X_val[:, top_idx]
    X_te_sel  = X_test[:, top_idx]

    # Hyperparameter optimisation (uses X_train CV only)
    best_pipe = build_and_tune(X_tr_sel, y_train, model_name)

    # Evaluate
    y_pred_train = best_pipe.predict(X_tr_sel)
    y_pred_val   = best_pipe.predict(X_val_sel)
    y_pred_test  = best_pipe.predict(X_te_sel)

    def metrics(y_true, y_pred):
        return {
            "r2"  : float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae" : float(mean_absolute_error(y_true, y_pred)),
        }

    train_m = metrics(y_train, y_pred_train)
    val_m   = metrics(y_val,   y_pred_val)
    test_m  = metrics(y_test,  y_pred_test)

    return {
        "method"           : method,
        "threshold_pct"    : threshold,
        "n_features"       : n_features,
        "model"            : model_name,
        "selected_features": selected_names,
        "train_r2"         : train_m["r2"],
        "train_rmse"       : train_m["rmse"],
        "train_mae"        : train_m["mae"],
        "val_r2"           : val_m["r2"],
        "val_rmse"         : val_m["rmse"],
        "val_mae"          : val_m["mae"],
        "test_r2"          : test_m["r2"],
        "test_rmse"        : test_m["rmse"],
        "test_mae"         : test_m["mae"],
        "delta_r2"         : float(train_m["r2"] - test_m["r2"]),
    }


# =============================================================================
# Save feature scores
# =============================================================================

def compute_and_save_feature_scores(X_train: np.ndarray, y_train: np.ndarray, feature_names: list):
    """Compute and save correlation scores for all four methods on the training split."""
    scores_dir = RESULTS_DIR / "feature_scores"
    scores_dir.mkdir(exist_ok=True)
    score_cols = {"dc": "DC Score", "mi": "MI Score", "mic": "MIC Score", "xi": "Xi Score"}

    for method, scorer in SCORERS.items():
        log.info(f"Computing {method.upper()} scores …")
        scores = scorer(X_train, y_train)
        df = pd.DataFrame({"Feature": feature_names, score_cols[method]: scores})
        df = df.sort_values(score_cols[method], ascending=False).reset_index(drop=True)
        df.to_csv(scores_dir / f"{method}_scores.csv", index=False)
        log.info(f"  Saved {method}_scores.csv")

    return scores_dir


# =============================================================================
# Main
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("Experiment 1: 48-configuration comprehensive comparison")
    log.info(f"Seed schedule — split: {SEED_SPLIT}, model/CV/MI/Optuna: {SEED_MODEL}")
    log.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, feature_names = load_data()

    # ── Train / validation / test split ───────────────────────────────────────
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED_SPLIT
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE, random_state=SEED_SPLIT
    )
    log.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ── Feature scoring (on training split only) ──────────────────────────────
    compute_and_save_feature_scores(X_train, y_train, feature_names)

    # ── 48 configurations ─────────────────────────────────────────────────────
    configs = list(itertools.product(METHODS, THRESHOLDS, MODEL_NAMES))
    log.info(f"Running {len(configs)} configurations …")

    rows = []
    for method, threshold, model_name in tqdm(configs, desc="Configs"):
        try:
            row = run_config(
                X_train_val, y_train_val, X_test, y_test,
                X_val, y_val, X_train, y_train,
                feature_names, method, threshold, model_name,
            )
            rows.append(row)
            log.info(
                f"  {method.upper():3s} {threshold:2d}% {model_name:12s} "
                f"test R²={row['test_r2']:.4f}  RMSE={row['test_rmse']:.3f}  MAE={row['test_mae']:.3f}"
            )
        except Exception as exc:
            log.error(f"  FAILED {method} {threshold}% {model_name}: {exc}")

    # ── Save summary ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "experiment_summary.csv"
    results_df.to_csv(out_path, index=False)
    log.info(f"Results saved to {out_path}")
    log.info("\n" + results_df[["method", "threshold_pct", "model",
                                "test_r2", "test_rmse", "test_mae"]].to_string(index=False))

    return results_df


if __name__ == "__main__":
    main()
