# Credit Card Fraud Detection & Analytics Dashboard

End-to-end fraud detection system on the Kaggle "Credit Card Fraud Detection" dataset (284,807 transactions, 0.173% fraud rate): nine supervised and unsupervised modeling approaches, hyperparameter-tuned, SHAP-explained, time-validated, and served through an interactive Streamlit dashboard.

**[Live Demo](#) &nbsp;•&nbsp; [Findings Write-Up](FINDINGS.md) &nbsp;•&nbsp; [Results](#results)**

---

## Overview

Fraud detection is a canonical extreme class-imbalance problem — fewer than 0.2% of transactions here are fraudulent, which means naive accuracy is meaningless. This project treats that as the central design constraint, not an afterthought:

- **Time-based validation**, not a random split — the model is never trained on transactions that happen after the ones it's tested on, which is what a random split silently allows
- Every model evaluated on **PR-AUC, Precision, Recall, F1**, with **bootstrap confidence intervals** so differences between models can be judged against the noise floor, not just point estimates
- Class imbalance addressed two competing ways — **class-weighting vs. SMOTE** — compared head-to-head rather than assumed (see [Findings](FINDINGS.md): the answer isn't what most tutorials assume)
- The deployment model's hyperparameters are **tuned with `RandomizedSearchCV`**, optimizing PR-AUC directly
- Decision thresholds are **cost-based** (assumed cost of a missed fraud vs. a false alarm), not left at the default 0.5
- **SHAP** explains individual predictions — not just which features matter on average, but why *this specific transaction* was flagged
- Supervised classifiers benchmarked against **unsupervised anomaly detection** (Isolation Forest, a PyTorch Autoencoder, and an ensemble of both)
- Every model is a single `sklearn.Pipeline` bundling preprocessing and classifier together — inference can't apply mismatched preprocessing because there's nothing to mismatch

Shipped as a Streamlit dashboard, not a notebook, so the work is explorable by a non-technical reviewer — including live single-transaction scoring, batch CSV upload, and per-prediction SHAP explanations.

---

## Results

Final metrics (time-based validation) are written to `models/metrics.json` and `model_comparison.csv` after running `train_models.py` — see [Reproducing Results](#reproducing-results). Headline numbers from initial development (random-split) runs:

| Model | PR-AUC |
|---|---|
| Random Forest (SMOTE) | 0.868 |
| Random Forest (baseline) | 0.861 |
| Random Forest (class-weighted) | 0.850 |
| XGBoost (baseline) | 0.781 |
| Logistic Regression (baseline) | 0.744 |
| Logistic Regression (SMOTE) | 0.725 |
| Logistic Regression (class-weighted) | 0.719 |

**Key finding:** SMOTE gave a small edge for Random Forest, but underperformed the plain baseline for both models when combined with class-weighting-style assumptions. Full reasoning in [FINDINGS.md](FINDINGS.md) — including why this result makes sense given the PCA-transformed feature space, and how the switch to time-based validation changes the picture.

---

## Dashboard

| Page | What it shows |
|---|---|
| **Overview** | KPIs, class imbalance visualization, model leaderboard |
| **Fraud Detection** | Live single-transaction scoring with SHAP explanation, plus batch CSV upload/scoring with downloadable results |
| **Model Comparison** | Metrics with bootstrap confidence intervals, selectable confusion matrices, hyperparameter search results, cost-vs-threshold curve |
| **Anomaly Detection** | Isolation Forest, Autoencoder, and their ensemble — score distributions and confusion matrices |
| **Dataset Insights** | EDA — correlations, feature distributions by class, SHAP global feature impact |
| **About** | Problem framing, full methodology, explicit limitations |

---

## Methodology

**1. EDA** — quantified the 0.173% fraud rate, compared amount distributions by class, ranked `V1`–`V28` by correlation with fraud.

**2. Time-based split** — sorted by `Time`, held out the last 20% chronologically, instead of a random stratified split. See [Findings](FINDINGS.md) for why this matters.

**3. Baseline** — Logistic Regression, the bar every other model has to clear.

**4. Imbalance handling, compared not assumed** — every classifier trained three ways: unmodified, class-weighted (`class_weight='balanced'` / `scale_pos_weight`), and SMOTE (fit on the training split only, after preprocessing, never touching the test set).

**5. Tree-based models** — Random Forest and XGBoost, benchmarked against the linear baseline.

**6. Hyperparameter tuning** — `RandomizedSearchCV` (20 iterations, 3-fold stratified CV, scored on PR-AUC directly) over XGBoost's `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, and `n_estimators`.

**7. SHAP explainability** — `TreeExplainer` on the tuned XGBoost model, computed once at training time and reused live in the dashboard for any transaction a user enters.

**8. Cost-based threshold tuning** — grid search over thresholds minimizing `(missed fraud × assumed cost) + (false alarms × assumed cost)`, with the full cost curve surfaced in the dashboard, not just the winning threshold.

**9. Unsupervised anomaly detection** — Isolation Forest (feature-space isolation) and a PyTorch Autoencoder (trained only on legitimate transactions, flagging high reconstruction error), plus a min-max-normalized ensemble of both.

**10. Statistical rigor** — bootstrap confidence intervals (300 resamples) on every model's PR-AUC.

**11. Final comparison** — all approaches evaluated on the same held-out, time-ordered test set.

---

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | scikit-learn, XGBoost, imbalanced-learn (SMOTE), PyTorch |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |
| Deployment | Docker |
| Tooling | uv |

---

## Project Structure

```
credit-card-fraud-detection/
├── train_models.py        # Trains all 9 models, tunes, explains, saves everything
├── app.py                 # Streamlit dashboard — loads artifacts, never retrains
├── src/
│   └── model_utils.py     # Shared Pipeline builder, Autoencoder, cost/CI helpers
├── tests/
│   └── test_model_utils.py   # pytest coverage for preprocessing + evaluation logic
├── notebooks/
│   └── fraud_detection.ipynb # Original exploratory analysis
├── data/
│   └── creditcard.csv     # Kaggle dataset (not included — see Setup)
├── models/                # Generated: trained pipelines, SHAP explainer, metrics
├── Dockerfile
├── FINDINGS.md             # Narrative write-up of key results
└── pyproject.toml
```

Training and serving are deliberately separate: `train_models.py` runs once
and saves every artifact; `app.py` only loads them. This mirrors how ML
systems are actually structured in production, and means the dashboard
starts in seconds.

---

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd credit-card-fraud-detection

# Download creditcard.csv from the Kaggle "Credit Card Fraud Detection"
# dataset and place it at data/creditcard.csv

uv venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
uv pip install -e .

python train_models.py        # ~15-25 min: trains, tunes, explains, saves everything
streamlit run app.py
```

<a id="reproducing-results"></a>
`train_models.py` trains and saves: Logistic Regression ×3, Random Forest ×3,
XGBoost baseline + hyperparameter-tuned, Isolation Forest, PyTorch
Autoencoder, their ensemble, a SHAP explainer, bootstrap confidence
intervals, and the cost-optimal threshold curve — everything the dashboard
reads.

### Running tests

```bash
uv pip install -e ".[dev]"
pytest tests/ -v
```

### Docker

```bash
docker build -t fraud-dashboard .
docker run -p 8501:8501 fraud-dashboard
```

Requires `models/` to already be populated (`python train_models.py` first)
— the image serves the dashboard, it doesn't retrain inside the container.

---

## What This Project Demonstrates

- **Imbalanced classification** done rigorously: correct metrics, tested (not assumed) resampling strategy, statistically grounded comparisons
- **Realistic validation**: time-based split instead of a random split that silently leaks future information
- **Decision-relevant evaluation**: cost-based thresholds that map model output to a business decision, not just an AUC number
- **Explainability**: SHAP integrated into live predictions, not just a static feature-importance chart
- **Engineering discipline**: Pipeline-wrapped models, pytest coverage, Docker packaging, clean train/serve separation
- **Honest reporting**: a findings write-up that includes a result that *didn't* go the expected way (SMOTE), and a dashboard About page that states explicit limitations

---

## Limitations

- `V1`–`V28` are anonymized PCA components; real-world meaning is unknown, limiting interpretability
- Dataset is historical (European cardholders, September 2013) — fraud patterns evolve
- Performance here doesn't guarantee performance on live, real-world transaction streams
- An anomaly flag means "statistically unusual," not "confirmed fraud"
- The \$500 / \$5 cost assumptions used for threshold tuning are illustrative, not sourced from real loss data
- **This is a portfolio / applied-ML project, not a production fraud detection system.**

---

## Author

Built as an applied machine learning project exploring imbalanced classification, realistic validation, cost-sensitive decision-making, explainability, and the trade-offs between supervised and unsupervised fraud detection.

*Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud), ULB Machine Learning Group.*
