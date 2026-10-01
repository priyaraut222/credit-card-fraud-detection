"""
train_models.py
================
This is the ONE script that actually trains models. It mirrors the exact
steps from fraud_detection.ipynb, but adds the piece the notebook was
missing: it SAVES every trained model, scaler, and metric to models/,
so app.py can load them instantly instead of retraining on every run.

Run this once (or again whenever you retrain):

    python train_models.py

Expects data/creditcard.csv to exist (the standard Kaggle 284,807-row /
492-fraud "Credit Card Fraud Detection" dataset). Takes a few minutes --
SMOTE + Random Forest is the slowest step.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.model_utils import Autoencoder, MODEL_FEATURE_ORDER, MODELS_DIR, PCA_COLUMNS

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
        "'Credit Card Fraud Detection' dataset and place it in the data/ folder."
    )
log("Loading creditcard.csv ...")
df_raw = pd.read_csv(DATA_PATH)
log(f"Loaded {len(df_raw):,} rows, {int(df_raw['Class'].sum()):,} fraud "
    f"({df_raw['Class'].mean()*100:.4f}%)")

# ---------------------------------------------------------------------------
# 2. Dataset stats + EDA artifacts the dashboard needs (Overview / Insights pages)
# ---------------------------------------------------------------------------
dataset_stats = {
    "total_transactions": int(len(df_raw)),
    "legit_transactions": int((df_raw["Class"] == 0).sum()),
    "fraud_transactions": int((df_raw["Class"] == 1).sum()),
    "fraud_rate_pct": float(df_raw["Class"].mean() * 100),
}

correlations = (
    df_raw[PCA_COLUMNS + ["Amount", "Time", "Class"]]
    .corr()["Class"]
    .drop("Class")
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .to_dict()
)

# Small sample kept for charts in the dashboard (keeps the app light --
# it never needs to load the full 284,807-row CSV)
normal_rows = df_raw[df_raw["Class"] == 0]
sample_size = min(8000, len(normal_rows))  # don't crash on smaller datasets
sample_df = pd.concat(
    [
        df_raw[df_raw["Class"] == 1],  # keep every fraud row, there are few
        normal_rows.sample(n=sample_size, random_state=RANDOM_STATE),
    ]
).sample(frac=1, random_state=RANDOM_STATE)
sample_df.to_csv(os.path.join(MODELS_DIR, "sample_data.csv"), index=False)

# ---------------------------------------------------------------------------
# 3. Preprocess -- mirrors the notebook exactly:
#    scale Amount and Time with THEIR OWN StandardScaler each, then drop
#    the raw columns. Two separate scalers are saved so inference on new
#    transactions uses identical transforms to what the models were trained on.
# ---------------------------------------------------------------------------
log("Scaling Amount and Time (separate scalers, matching the notebook) ...")
df = df_raw.copy()

amount_scaler = StandardScaler()
df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount"]])

time_scaler = StandardScaler()
df["Time_scaled"] = time_scaler.fit_transform(df[["Time"]])

df = df.drop(["Time", "Amount"], axis=1)

joblib.dump(amount_scaler, os.path.join(MODELS_DIR, "amount_scaler.pkl"))
joblib.dump(time_scaler, os.path.join(MODELS_DIR, "time_scaler.pkl"))

X = df[MODEL_FEATURE_ORDER]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
log(f"Train: {len(X_train):,} rows ({y_train.mean()*100:.4f}% fraud) | "
    f"Test: {len(X_test):,} rows ({y_test.mean()*100:.4f}% fraud)")

metrics = {}      # model_name -> dict of scores (what the dashboard reads)


def evaluate(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics[name] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm,  # [[tn, fp], [fn, tp]]
        "threshold_used": float(threshold),
    }
    log(f"{name:40s} PR-AUC={metrics[name]['pr_auc']:.4f}  "
        f"Recall={metrics[name]['recall']:.3f}  Precision={metrics[name]['precision']:.3f}")


def cost_optimal_threshold(y_true, y_proba, cost_fn=500, cost_fp=5):
    _, _, thresholds = precision_recall_curve(y_true, y_proba)
    candidates = np.unique(np.concatenate([thresholds, [0.5]]))
    best_t, best_cost = 0.5, float("inf")
    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t)


# ---------------------------------------------------------------------------
# 4. Logistic Regression: baseline / weighted / SMOTE
# ---------------------------------------------------------------------------
log("Training Logistic Regression (baseline) ...")
lr_base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_base.fit(X_train, y_train)
evaluate("Logistic Regression (baseline)", y_test, lr_base.predict_proba(X_test)[:, 1])
joblib.dump(lr_base, os.path.join(MODELS_DIR, "lr_baseline.pkl"))

log("Training Logistic Regression (class-weighted) ...")
lr_weighted = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
lr_weighted.fit(X_train, y_train)
evaluate("Logistic Regression (weighted)", y_test, lr_weighted.predict_proba(X_test)[:, 1])
joblib.dump(lr_weighted, os.path.join(MODELS_DIR, "lr_weighted.pkl"))

log("Applying SMOTE to training data only ...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

log("Training Logistic Regression (SMOTE) ...")
lr_smote = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_smote.fit(X_train_smote, y_train_smote)
evaluate("Logistic Regression (SMOTE)", y_test, lr_smote.predict_proba(X_test)[:, 1])
joblib.dump(lr_smote, os.path.join(MODELS_DIR, "lr_smote.pkl"))

# ---------------------------------------------------------------------------
# 5. Random Forest: baseline / weighted / SMOTE
# ---------------------------------------------------------------------------
log("Training Random Forest (baseline) ...")
rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_base.fit(X_train, y_train)
evaluate("Random Forest (baseline)", y_test, rf_base.predict_proba(X_test)[:, 1])
joblib.dump(rf_base, os.path.join(MODELS_DIR, "rf_baseline.pkl"))

log("Training Random Forest (class-weighted) ...")
rf_weighted = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_weighted.fit(X_train, y_train)
evaluate("Random Forest (weighted)", y_test, rf_weighted.predict_proba(X_test)[:, 1])
joblib.dump(rf_weighted, os.path.join(MODELS_DIR, "rf_weighted.pkl"))

log("Training Random Forest (SMOTE) -- this is the slow one, please wait ...")
rf_smote = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)
evaluate("Random Forest (SMOTE)", y_test, rf_smote.predict_proba(X_test)[:, 1])
joblib.dump(rf_smote, os.path.join(MODELS_DIR, "rf_smote.pkl"))

# ---------------------------------------------------------------------------
# 6. XGBoost: baseline / weighted
# ---------------------------------------------------------------------------
log("Training XGBoost (baseline) ...")
xgb_base = XGBClassifier(
    n_estimators=100, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1
)
xgb_base.fit(X_train, y_train)
evaluate("XGBoost (baseline)", y_test, xgb_base.predict_proba(X_test)[:, 1])
joblib.dump(xgb_base, os.path.join(MODELS_DIR, "xgb_baseline.pkl"))

log("Training XGBoost (weighted) ...")
weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_weighted = XGBClassifier(
    n_estimators=100, scale_pos_weight=weight_ratio, random_state=RANDOM_STATE,
    eval_metric="logloss", n_jobs=-1,
)
xgb_weighted.fit(X_train, y_train)
evaluate("XGBoost (weighted)", y_test, xgb_weighted.predict_proba(X_test)[:, 1])
joblib.dump(xgb_weighted, os.path.join(MODELS_DIR, "xgb_weighted.pkl"))

feature_importance = dict(
    sorted(
        zip(MODEL_FEATURE_ORDER, xgb_weighted.feature_importances_.astype(float)),
        key=lambda kv: kv[1],
        reverse=True,
    )
)

# ---------------------------------------------------------------------------
# 7. Isolation Forest (unsupervised)
# ---------------------------------------------------------------------------
log("Training Isolation Forest ...")
fraud_rate = y_train.mean()
iso_forest = IsolationForest(
    n_estimators=100, contamination=fraud_rate, random_state=RANDOM_STATE, n_jobs=-1
)
iso_forest.fit(X_train)
iso_scores_test = -iso_forest.score_samples(X_test)  # higher = more anomalous
evaluate("Isolation Forest", y_test, iso_scores_test)
joblib.dump(iso_forest, os.path.join(MODELS_DIR, "iso_forest.pkl"))
np.savez(os.path.join(MODELS_DIR, "iso_scores.npz"), scores=iso_scores_test, labels=y_test.values)

# ---------------------------------------------------------------------------
# 8. Autoencoder (PyTorch, trained only on normal transactions)
# ---------------------------------------------------------------------------
log("Training Autoencoder (normal transactions only) ...")
X_train_normal = X_train[y_train == 0]
X_train_tensor = torch.tensor(X_train_normal.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
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
# 9. Cost-based threshold tuning for the deployed model (XGBoost weighted)
# ---------------------------------------------------------------------------
log("Computing cost-optimal threshold for XGBoost (weighted) ...")
best_threshold = cost_optimal_threshold(y_test, xgb_weighted.predict_proba(X_test)[:, 1])
evaluate(
    "XGBoost (weighted, tuned threshold)",
    y_test, xgb_weighted.predict_proba(X_test)[:, 1], threshold=best_threshold,
)

# ---------------------------------------------------------------------------
# 10. Save everything the dashboard needs
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
    "feature_cols": MODEL_FEATURE_ORDER,
    "input_dim": input_dim,
    "best_model_by_pr_auc": max(
        (k for k in metrics if k != "XGBoost (weighted, tuned threshold)"),
        key=lambda k: metrics[k]["pr_auc"],
    ),
    "deployment_model": "XGBoost (weighted)",
    "deployment_model_file": "xgb_weighted.pkl",
    "chosen_threshold": best_threshold,
    "random_state": RANDOM_STATE,
}
with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

results_df = pd.DataFrame(
    [{"Model": name, **{k: v for k, v in m.items() if k != "confusion_matrix"}}
     for name, m in metrics.items()]
).sort_values("pr_auc", ascending=False)
results_df.to_csv("model_comparison.csv", index=False)

log("Done! All artifacts saved to models/. Now run: streamlit run app.py")
