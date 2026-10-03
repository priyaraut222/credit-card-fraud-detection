"""
train_models.py
================
Trains every model, tunes the deployment model's hyperparameters, computes
SHAP explanations, bootstrap confidence intervals, and an ensemble anomaly
score -- then saves everything app.py needs into models/.

Run once (or again whenever you retrain):

    python train_models.py

Expects data/creditcard.csv (the standard Kaggle 284,807-row / 492-fraud
"Credit Card Fraud Detection" dataset). Takes roughly 15-25 minutes --
the hyperparameter search and SMOTE+RF are the slowest steps.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.autoencoder_utils import Autoencoder
from src.model_utils import (
    MODELS_DIR,
    PCA_COLUMNS,
    RAW_FEATURE_ORDER,
    bootstrap_pr_auc_ci,
    build_model_pipeline,
    build_preprocessing_pipeline,
    cost_optimal_threshold,
    minmax,
)

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "creditcard.csv")
os.makedirs(MODELS_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"'{DATA_PATH}' not found. Download creditcard.csv from the Kaggle "
        "'Credit Card Fraud Detection' dataset and place it in data/."
    )
log("Loading creditcard.csv ...")
df_raw = pd.read_csv(DATA_PATH)
log(f"Loaded {len(df_raw):,} rows, {int(df_raw['Class'].sum()):,} fraud "
    f"({df_raw['Class'].mean()*100:.4f}%)")

dataset_stats = {
    "total_transactions": int(len(df_raw)),
    "legit_transactions": int((df_raw["Class"] == 0).sum()),
    "fraud_transactions": int((df_raw["Class"] == 1).sum()),
    "fraud_rate_pct": float(df_raw["Class"].mean() * 100),
}

correlations = (
    df_raw[PCA_COLUMNS + ["Amount", "Time", "Class"]]
    .corr()["Class"].drop("Class")
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .to_dict()
)

normal_rows = df_raw[df_raw["Class"] == 0]
sample_size = min(8000, len(normal_rows))
sample_df = pd.concat(
    [df_raw[df_raw["Class"] == 1], normal_rows.sample(n=sample_size, random_state=RANDOM_STATE)]
).sample(frac=1, random_state=RANDOM_STATE)
sample_df.to_csv(os.path.join(MODELS_DIR, "sample_data.csv"), index=False)

# ---------------------------------------------------------------------------
# 2. TIME-BASED split (not random stratified)
#
# `Time` is seconds elapsed since the first transaction -- a real temporal
# signal. A random split lets the model train on transactions that happen
# chronologically AFTER some of its test transactions, which never happens
# in production (you can't train on the future). Sorting by Time and taking
# the last 20% as the test set is a more honest estimate of how this model
# would perform deployed forward in time.
# ---------------------------------------------------------------------------
log("Splitting by Time (chronological), not randomly ...")
df_sorted = df_raw.sort_values("Time").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)

X = df_sorted[RAW_FEATURE_ORDER]
y = df_sorted["Class"]

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
log(f"Train: {len(X_train):,} rows ({y_train.mean()*100:.4f}% fraud, "
    f"Time <= {df_sorted['Time'].iloc[split_idx-1]:.0f}s)")
log(f"Test:  {len(X_test):,} rows ({y_test.mean()*100:.4f}% fraud, "
    f"Time > {df_sorted['Time'].iloc[split_idx-1]:.0f}s)")

test_export = X_test.copy()
test_export["Class"] = y_test.values
test_export.to_csv(os.path.join(MODELS_DIR, "test_holdout_sample.csv"), index=False)

metrics = {}


def evaluate(name, y_true, y_proba, threshold=0.5, compute_ci=True):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    entry = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm,
        "threshold_used": float(threshold),
    }
    if compute_ci:
        lo, hi = bootstrap_pr_auc_ci(y_true, y_proba, n_boot=300, random_state=RANDOM_STATE)
        entry["pr_auc_ci_low"] = lo
        entry["pr_auc_ci_high"] = hi
    metrics[name] = entry
    log(f"{name:42s} PR-AUC={entry['pr_auc']:.4f}  Recall={entry['recall']:.3f}  "
        f"Precision={entry['precision']:.3f}")


# ---------------------------------------------------------------------------
# 3. Logistic Regression: baseline / weighted / SMOTE -- each a full Pipeline
# ---------------------------------------------------------------------------
log("Training Logistic Regression (baseline) ...")
lr_base = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
lr_base.fit(X_train, y_train)
evaluate("Logistic Regression (baseline)", y_test, lr_base.predict_proba(X_test)[:, 1])
joblib.dump(lr_base, os.path.join(MODELS_DIR, "lr_baseline.pkl"))

log("Training Logistic Regression (class-weighted) ...")
lr_weighted = build_model_pipeline(
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
)
lr_weighted.fit(X_train, y_train)
evaluate("Logistic Regression (weighted)", y_test, lr_weighted.predict_proba(X_test)[:, 1])
joblib.dump(lr_weighted, os.path.join(MODELS_DIR, "lr_weighted.pkl"))

log("Applying SMOTE to training data only (post-preprocessing) ...")
# SMOTE needs numeric, already-preprocessed input, so fit the shared
# preprocessing once, resample in that space, then attach a fresh classifier.
preproc_for_smote = build_preprocessing_pipeline()
X_train_pre = preproc_for_smote.fit_transform(X_train)
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pre, y_train)

log("Training Logistic Regression (SMOTE) ...")
lr_smote_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_smote_clf.fit(X_train_smote, y_train_smote)
lr_smote = SkPipeline([("preprocess", preproc_for_smote), ("classifier", lr_smote_clf)])
evaluate("Logistic Regression (SMOTE)", y_test, lr_smote.predict_proba(X_test)[:, 1])
joblib.dump(lr_smote, os.path.join(MODELS_DIR, "lr_smote.pkl"))

# ---------------------------------------------------------------------------
# 4. Random Forest: baseline / weighted / SMOTE
# ---------------------------------------------------------------------------
log("Training Random Forest (baseline) ...")
rf_base = build_model_pipeline(
    RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
)
rf_base.fit(X_train, y_train)
evaluate("Random Forest (baseline)", y_test, rf_base.predict_proba(X_test)[:, 1])
joblib.dump(rf_base, os.path.join(MODELS_DIR, "rf_baseline.pkl"))

log("Training Random Forest (class-weighted) ...")
rf_weighted = build_model_pipeline(
    RandomForestClassifier(n_estimators=100, class_weight="balanced",
                            random_state=RANDOM_STATE, n_jobs=-1)
)
rf_weighted.fit(X_train, y_train)
evaluate("Random Forest (weighted)", y_test, rf_weighted.predict_proba(X_test)[:, 1])
joblib.dump(rf_weighted, os.path.join(MODELS_DIR, "rf_weighted.pkl"))

log("Training Random Forest (SMOTE) -- slow step ...")
rf_smote_clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_smote_clf.fit(X_train_smote, y_train_smote)
rf_smote = SkPipeline([("preprocess", preproc_for_smote), ("classifier", rf_smote_clf)])
evaluate("Random Forest (SMOTE)", y_test, rf_smote.predict_proba(X_test)[:, 1])
joblib.dump(rf_smote, os.path.join(MODELS_DIR, "rf_smote.pkl"))

# ---------------------------------------------------------------------------
# 5. XGBoost baseline, then hyperparameter-tuned weighted XGBoost
#    (this is the deployment model, so it's the one worth tuning properly)
# ---------------------------------------------------------------------------
log("Training XGBoost (baseline) ...")
xgb_base = build_model_pipeline(
    XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
)
xgb_base.fit(X_train, y_train)
evaluate("XGBoost (baseline)", y_test, xgb_base.predict_proba(X_test)[:, 1])
joblib.dump(xgb_base, os.path.join(MODELS_DIR, "xgb_baseline.pkl"))

log("Tuning XGBoost (weighted) with RandomizedSearchCV -- this is the slow one ...")
weight_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
X_train_pre_xgb = preproc_for_smote.transform(X_train)  # reuse already-fitted preprocessing

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}
search = RandomizedSearchCV(
    estimator=XGBClassifier(
        scale_pos_weight=weight_ratio, random_state=RANDOM_STATE,
        eval_metric="logloss", n_jobs=-1,
    ),
    param_distributions=param_dist,
    n_iter=20,                     # kept modest -- full grid would take hours on 227k rows
    scoring="average_precision",   # optimize PR-AUC directly, not accuracy
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train_pre_xgb, y_train)
log(f"Best params: {search.best_params_}")
log(f"Best CV PR-AUC: {search.best_score_:.4f}")

xgb_weighted_clf = search.best_estimator_
xgb_weighted = SkPipeline([("preprocess", preproc_for_smote), ("classifier", xgb_weighted_clf)])
evaluate("XGBoost (weighted, tuned)", y_test, xgb_weighted.predict_proba(X_test)[:, 1])
joblib.dump(xgb_weighted, os.path.join(MODELS_DIR, "xgb_weighted.pkl"))

with open(os.path.join(MODELS_DIR, "best_xgb_params.json"), "w") as f:
    json.dump({"best_params": search.best_params_, "best_cv_pr_auc": float(search.best_score_)}, f, indent=2)

feature_importance = dict(
    sorted(zip(PCA_COLUMNS + ["Amount", "Time"],
               xgb_weighted_clf.feature_importances_.astype(float)),
           key=lambda kv: kv[1], reverse=True)
)

# ---------------------------------------------------------------------------
# 6. SHAP values for the deployment model
#
# Feature importance (above) says which features matter ON AVERAGE across
# the whole dataset. SHAP says, for ONE specific transaction, how much each
# feature pushed the prediction up or down -- the thing an analyst actually
# needs when deciding whether to trust a given flagged transaction.
# A TreeExplainer on XGBoost is fast (no retraining, closed-form for trees),
# so it's computed here once and the explainer object is saved for the
# dashboard to reuse on any transaction a user enters live.
# ---------------------------------------------------------------------------
log("Computing SHAP explainer + background values ...")
explainer = shap.TreeExplainer(xgb_weighted_clf)
shap_sample_idx = np.random.RandomState(RANDOM_STATE).choice(
    len(X_test), size=min(500, len(X_test)), replace=False
)
X_test_pre_sample = preproc_for_smote.transform(X_test.iloc[shap_sample_idx])
shap_values_sample = explainer.shap_values(X_test_pre_sample)

joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.pkl"))
np.savez(
    os.path.join(MODELS_DIR, "shap_sample.npz"),
    shap_values=shap_values_sample,
    feature_values=X_test_pre_sample,
    feature_names=np.array(PCA_COLUMNS + ["Amount", "Time"]),
)

# ---------------------------------------------------------------------------
# 7. Isolation Forest (unsupervised)
# ---------------------------------------------------------------------------
log("Training Isolation Forest ...")
preproc_iso = build_preprocessing_pipeline()
X_train_pre_iso = preproc_iso.fit_transform(X_train)
X_test_pre_iso = preproc_iso.transform(X_test)

fraud_rate = y_train.mean()
iso_forest_clf = IsolationForest(
    n_estimators=100, contamination=fraud_rate, random_state=RANDOM_STATE, n_jobs=-1
)
iso_forest_clf.fit(X_train_pre_iso)
iso_scores_test = -iso_forest_clf.score_samples(X_test_pre_iso)
evaluate("Isolation Forest", y_test, iso_scores_test)

iso_forest = SkPipeline([("preprocess", preproc_iso), ("classifier", iso_forest_clf)])
joblib.dump(iso_forest, os.path.join(MODELS_DIR, "iso_forest.pkl"))
np.savez(os.path.join(MODELS_DIR, "iso_scores.npz"), scores=iso_scores_test, labels=y_test.values)

# ---------------------------------------------------------------------------
# 8. Autoencoder (PyTorch, trained only on normal transactions)
# ---------------------------------------------------------------------------
log("Training Autoencoder (normal transactions only) ...")
X_train_normal_pre = X_train_pre_iso[y_train.values == 0]
X_train_tensor = torch.tensor(X_train_normal_pre, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_pre_iso, dtype=torch.float32)
input_dim = X_train_tensor.shape[1]

ae = Autoencoder(input_dim)
train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=256, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

n_epochs = 20
for epoch in range(n_epochs):
    ae.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(ae(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    epoch_loss /= len(X_train_tensor)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        log(f"  Autoencoder epoch {epoch+1}/{n_epochs} - loss {epoch_loss:.6f}")

torch.save(ae.state_dict(), os.path.join(MODELS_DIR, "autoencoder.pth"))

ae.eval()
with torch.no_grad():
    recon = ae(X_test_tensor)
    recon_error = torch.mean((X_test_tensor - recon) ** 2, dim=1).numpy()

evaluate("Autoencoder", y_test, recon_error)
np.savez(os.path.join(MODELS_DIR, "reconstruction_errors.npz"), errors=recon_error, labels=y_test.values)

# ---------------------------------------------------------------------------
# 9. Ensemble anomaly score: average of min-max normalized IF + AE scores
#
# Each unsupervised model catches somewhat different anomalies (Isolation
# Forest reasons about feature-space isolation, the Autoencoder reasons
# about reconstruction fidelity). Averaging their normalized scores tests
# whether combining them beats either alone.
# ---------------------------------------------------------------------------
log("Building ensemble anomaly score (Isolation Forest + Autoencoder) ...")
ensemble_score = (minmax(iso_scores_test) + minmax(recon_error)) / 2
evaluate("Ensemble (Isolation Forest + Autoencoder)", y_test, ensemble_score)
np.savez(os.path.join(MODELS_DIR, "ensemble_scores.npz"), scores=ensemble_score, labels=y_test.values)

# ---------------------------------------------------------------------------
# 10. Cost-based threshold tuning + cost curve for the deployment model
# ---------------------------------------------------------------------------
log("Computing cost-optimal threshold + cost curve for XGBoost (weighted, tuned) ...")
xgb_test_proba = xgb_weighted.predict_proba(X_test)[:, 1]
best_threshold, curve_thresholds, curve_costs = cost_optimal_threshold(
    y_test, xgb_test_proba, cost_fn=500, cost_fp=5, return_curve=True
)
evaluate(
    "XGBoost (weighted, tuned threshold)", y_test, xgb_test_proba,
    threshold=best_threshold, compute_ci=False,
)
np.savez(
    os.path.join(MODELS_DIR, "cost_curve.npz"),
    thresholds=curve_thresholds, costs=curve_costs,
    best_threshold=best_threshold, cost_fn=500, cost_fp=5,
)

# ---------------------------------------------------------------------------
# 11. Save everything the dashboard needs
# ---------------------------------------------------------------------------
log("Saving metrics.json, dataset_stats.json, correlations.json, config.json ...")

with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(MODELS_DIR, "dataset_stats.json"), "w") as f:
    json.dump(dataset_stats, f, indent=2)
with open(os.path.join(MODELS_DIR, "correlations.json"), "w") as f:
    json.dump(correlations, f, indent=2)
with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
    json.dump(feature_importance, f, indent=2)

config = {
    "input_dim": input_dim,
    "best_model_by_pr_auc": max(
        (k for k in metrics if "tuned threshold" not in k),
        key=lambda k: metrics[k]["pr_auc"],
    ),
    "deployment_model": "XGBoost (weighted, tuned)",
    "deployment_model_file": "xgb_weighted.pkl",
    "chosen_threshold": best_threshold,
    "random_state": RANDOM_STATE,
    "split_strategy": "time-based (last 20% of transactions by Time held out)",
    "cost_fn_assumed": 500,
    "cost_fp_assumed": 5,
}
with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

results_df = pd.DataFrame(
    [{"Model": name, **{k: v for k, v in m.items() if k != "confusion_matrix"}}
     for name, m in metrics.items()]
).sort_values("pr_auc", ascending=False)
results_df.to_csv("model_comparison.csv", index=False)

log("Done! All artifacts saved to models/. Now run: streamlit run app.py")