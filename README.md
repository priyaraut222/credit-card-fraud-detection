# Credit Card Fraud Detection

Fraud detection on the standard Kaggle "Credit Card Fraud Detection" dataset
(284,807 transactions, 492 fraud, ~0.173% fraud rate), comparing several
supervised and unsupervised approaches, with a Streamlit analytics dashboard.

## Why this structure

**The original problem:** the Streamlit app expected files like
`models/xgboost_model.pkl` that the notebook never actually created — the
notebook trained models in memory and plotted results, but never called
`joblib.dump(...)` or `torch.save(...)` to persist them to disk. This project
splits things into two clear pieces so that can't happen silently again:

- **`train_models.py`** — the only file that trains anything. Run it once.
  It saves every trained model, both scalers, all metrics, and everything
  else the dashboard needs into `models/`.
- **`app.py`** — the dashboard. It only *loads* artifacts from `models/`.
  It never trains or fits anything, so it starts instantly and can't drift
  out of sync with what was actually trained.
- **`src/model_utils.py`** — shared code (the PyTorch Autoencoder class,
  feature column order) used by both of the above, so they can never
  silently disagree with each other.

## Project layout

```
.
├── data/
│   └── creditcard.csv          ← you add this (see Setup)
├── models/                     ← created by train_models.py, gitignored
│   ├── lr_baseline.pkl / lr_weighted.pkl / lr_smote.pkl
│   ├── rf_baseline.pkl / rf_weighted.pkl / rf_smote.pkl
│   ├── xgb_baseline.pkl / xgb_weighted.pkl
│   ├── iso_forest.pkl
│   ├── autoencoder.pth
│   ├── amount_scaler.pkl / time_scaler.pkl
│   ├── metrics.json / config.json / dataset_stats.json
│   ├── correlations.json / feature_importance.json
│   └── sample_data.csv
├── src/
│   └── model_utils.py
├── notebooks/
│   └── fraud_detection.ipynb   ← your original exploratory notebook
├── train_models.py             ← run this first
├── app.py                      ← then run this
├── pyproject.toml
└── README.md
```

## Setup

1. **Get the dataset.** Download `creditcard.csv` from the Kaggle
   "Credit Card Fraud Detection" dataset (mlg-ulb) and place it at
   `data/creditcard.csv`.

2. **Install dependencies.**
   ```powershell
   uv pip install -e .
   ```
   (or `pip install -e .` if you're not using `uv`)

3. **Train and save every model.** This is the step that was missing
   before — it's what actually creates the `models/` folder:
   ```powershell
   python train_models.py
   ```
   Takes a few minutes (SMOTE + Random Forest is the slowest part).
   You only need to re-run this if you change the training code or dataset.

4. **Run the dashboard:**
   ```powershell
   streamlit run app.py
   ```

## Models trained

| Category | Models |
|---|---|
| Supervised | Logistic Regression (baseline / class-weighted / SMOTE), Random Forest (baseline / weighted / SMOTE), XGBoost (baseline / weighted) |
| Unsupervised / anomaly detection | Isolation Forest, Autoencoder (PyTorch, trained only on legitimate transactions) |

All models are evaluated on the **same held-out test set** using Precision,
Recall, F1-score, PR-AUC and ROC-AUC — not accuracy, since accuracy is
meaningless on a dataset that's 99.8% one class.

## Notes

- `train_models.py` fits `Amount` and `Time` with **two separate**
  `StandardScaler` instances (matching the original notebook's
  preprocessing exactly) and saves both, so the dashboard's live
  prediction page transforms new input identically to training.
- SMOTE is applied to the training split only, never to the test set.
- The dashboard's "Fraud Detection" page defaults to the class-weighted
  XGBoost model with a cost-based tuned threshold, but lets you switch
  between any trained supervised model.
