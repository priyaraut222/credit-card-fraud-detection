"""
Shared definitions used by BOTH train_models.py and app.py.

This file exists so the feature order and preprocessing pipeline used at
training time can never silently drift from what's used at inference time
in the dashboard. Only one source of truth.

NOTE: the PyTorch Autoencoder class lives in autoencoder_utils.py, not here,
deliberately. This file is imported at the TOP of app.py on every single
page load/rerun. If `import torch` lived here, every click in the dashboard
would pay the cost of loading torch into memory -- even on pages that never
touch the Autoencoder. Keeping torch out of this file's import chain is
what makes the lazy-loading in app.py's load_ae() actually effective.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Column names in the original Kaggle "Credit Card Fraud Detection" CSV
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]  # V1 ... V28

# Raw columns as they appear in creditcard.csv (before any scaling)
RAW_FEATURE_ORDER = PCA_COLUMNS + ["Amount", "Time"]

MODELS_DIR = "models"


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Single source of truth for preprocessing. V1-V28 pass through
    unchanged (they're already PCA-transformed/scaled in the original
    dataset); Amount and Time each get their own StandardScaler.

    Wrapping this in a ColumnTransformer -- rather than fitting two
    separate scalers by hand and gluing the results back together --
    means every model's full pipeline (preprocessing + classifier) can
    be saved and loaded as ONE object. That removes an entire class of
    bugs: it's no longer possible to accidentally apply the wrong
    scaler, the wrong column order, or a stale scaler to new data,
    because there's nothing to mismatch -- one `.pkl` IS the complete
    transformation from raw transaction to prediction.
    """
    return ColumnTransformer(
        transformers=[
            ("pca_passthrough", "passthrough", PCA_COLUMNS),
            ("amount_scaler", StandardScaler(), ["Amount"]),
            ("time_scaler", StandardScaler(), ["Time"]),
        ],
        verbose_feature_names_out=False,
    )


def build_model_pipeline(classifier) -> Pipeline:
    """Wrap any sklearn-compatible classifier with the shared preprocessing."""
    return Pipeline(steps=[
        ("preprocess", build_preprocessing_pipeline()),
        ("classifier", classifier),
    ])


def cost_optimal_threshold(y_true, y_proba, cost_fn=500, cost_fp=5, return_curve=False):
    """
    Grid-search candidate thresholds to minimize assumed business cost:
        cost = (missed fraud x cost_fn) + (false alarms x cost_fp)

    If return_curve=True, also returns the full (thresholds, costs) arrays
    so the dashboard can plot cost vs. threshold, not just report the winner.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_curve

    _, _, thresholds = precision_recall_curve(y_true, y_proba)
    candidates = np.unique(np.concatenate([thresholds, [0.5]]))
    costs = []
    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)
    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    best_t = float(candidates[best_idx])
    if return_curve:
        return best_t, candidates, costs
    return best_t


def bootstrap_pr_auc_ci(y_true, y_proba, n_boot=300, ci=0.95, random_state=42):
    """
    Bootstrap confidence interval for PR-AUC: resample the test set with
    replacement n_boot times, recompute PR-AUC each time, and take the
    percentile interval. Turns "PR-AUC = 0.86" into "PR-AUC = 0.86,
    95% CI [0.83, 0.89]" -- lets you say whether a difference between two
    models' PR-AUC is actually distinguishable from noise.
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        # Skip degenerate resamples with only one class present
        if len(np.unique(y_true[idx])) < 2:
            scores[i] = np.nan
            continue
        scores[i] = average_precision_score(y_true[idx], y_proba[idx])
    scores = scores[~np.isnan(scores)]
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 - (1 - ci) / 2) * 100
    return float(np.percentile(scores, lo_pct)), float(np.percentile(scores, hi_pct))


def minmax(arr):
    arr = np.asarray(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)