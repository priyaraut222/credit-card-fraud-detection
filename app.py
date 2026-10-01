"""
app.py -- Credit Card Fraud Detection & Analytics Dashboard

Reads ONLY pre-saved artifacts produced by train_models.py. This file
never trains, fits, or refits anything -- that keeps startup fast and
guarantees the preprocessing used here always matches what the models
were actually trained on.

Run:
    streamlit run app.py

If you see a "no artifacts found" message, run this first:
    python train_models.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model_utils import MODELS_DIR, PCA_COLUMNS, load_autoencoder

# ---------------------------------------------------------------------------
# Page config + light, restrained styling (no dark theme)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #f7f8fa; border-right: 1px solid #e3e5e8; }
    h1, h2, h3 { color: #1a1a2e; font-weight: 600; }
    div[data-testid="stMetric"] {
        background-color: #fafbfc; border: 1px solid #e3e5e8;
        border-radius: 6px; padding: 14px 16px;
    }
    .result-box { border-radius: 6px; padding: 22px 26px; margin: 12px 0 18px 0; border: 1px solid; }
    .result-legit { background-color: #f0f8f1; border-color: #bfe0c4; color: #1e7d34; }
    .result-fraud { background-color: #fdf0ef; border-color: #f0bcb8; color: #b3261e; }
    .result-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 6px; }
    .finding-box {
        background-color: #f5f8fb; border-left: 4px solid #2f5d8a;
        padding: 14px 18px; border-radius: 4px; margin: 10px 0;
    }
    .limitation-box {
        background-color: #fdf7ec; border-left: 4px solid #b06a00;
        padding: 14px 18px; border-radius: 4px; margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLOR_LEGIT = "#1e7d34"
COLOR_FRAUD = "#b3261e"
COLOR_NEUTRAL = "#2f5d8a"


# ---------------------------------------------------------------------------
# Artifact loading -- cached, so nothing reloads on every click
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_csv(name):
    path = os.path.join(MODELS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_resource
def load_scaler(name):
    path = os.path.join(MODELS_DIR, name)
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_sklearn_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_ae(input_dim):
    path = os.path.join(MODELS_DIR, "autoencoder.pth")
    return load_autoencoder(path, input_dim) if os.path.exists(path) else None


@st.cache_data
def load_npz(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


config = load_json("config.json")
metrics = load_json("metrics.json")
dataset_stats = load_json("dataset_stats.json")
correlations = load_json("correlations.json")
feature_importance = load_json("feature_importance.json")

if config is None or metrics is None or dataset_stats is None:
    st.error(
        "**No trained model artifacts found in `models/`.**\n\n"
        "This dashboard only loads pre-trained models -- it never trains "
        "anything itself. Run the training script first, from the project root:\n\n"
        "```\npython train_models.py\n```\n\n"
        "That trains every model from the notebook, evaluates them "
        "consistently, and saves everything this dashboard needs into "
        "`models/`. It takes a few minutes and only needs to be run once "
        "(or again if you retrain)."
    )
    st.stop()

amount_scaler = load_scaler("amount_scaler.pkl")
time_scaler = load_scaler("time_scaler.pkl")
feature_cols = config["feature_cols"]
input_dim = config["input_dim"]

SUPERVISED_MODEL_FILES = {
    "Logistic Regression (baseline)": "lr_baseline.pkl",
    "Logistic Regression (weighted)": "lr_weighted.pkl",
    "Logistic Regression (SMOTE)": "lr_smote.pkl",
    "Random Forest (baseline)": "rf_baseline.pkl",
    "Random Forest (weighted)": "rf_weighted.pkl",
    "Random Forest (SMOTE)": "rf_smote.pkl",
    "XGBoost (baseline)": "xgb_baseline.pkl",
    "XGBoost (weighted)": "xgb_weighted.pkl",
}
SUPERVISED_MODEL_FILES = {
    name: fname for name, fname in SUPERVISED_MODEL_FILES.items()
    if name in metrics and os.path.exists(os.path.join(MODELS_DIR, fname))
}
ANOMALY_MODELS = [m for m in ["Isolation Forest", "Autoencoder"] if m in metrics]
DEPLOYMENT_MODEL = config.get("deployment_model", "XGBoost (weighted)")


def metrics_dataframe(exclude_tuned=True):
    rows = []
    for name, m in metrics.items():
        if exclude_tuned and "tuned threshold" in name:
            continue
        rows.append({
            "Model": name, "Precision": m["precision"], "Recall": m["recall"],
            "F1-score": m["f1"], "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
        })
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Fraud Detection")
st.sidebar.caption("Analytics Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Fraud Detection", "Model Comparison", "Anomaly Detection",
     "Dataset Insights", "About"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(f"Models evaluated: {len(SUPERVISED_MODEL_FILES) + len(ANOMALY_MODELS)}\n\n"
                    f"Best by PR-AUC: {config['best_model_by_pr_auc']}")

# =============================================================================
# OVERVIEW
# =============================================================================
if page == "Overview":
    st.title("Credit Card Fraud Detection")
    st.caption("Project overview -- dataset, models, and headline results")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{dataset_stats['total_transactions']:,}")
    c2.metric("Legitimate", f"{dataset_stats['legit_transactions']:,}")
    c3.metric("Fraudulent", f"{dataset_stats['fraud_transactions']:,}")
    c4.metric("Fraud Rate", f"{dataset_stats['fraud_rate_pct']:.3f}%")
    best_name = config["best_model_by_pr_auc"]
    c5.metric("Best Model (PR-AUC)", best_name.split(" (")[0], f"{metrics[best_name]['pr_auc']:.3f}")

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Class Distribution")
        dist_df = pd.DataFrame({
            "Class": ["Legitimate", "Fraudulent"],
            "Count": [dataset_stats["legit_transactions"], dataset_stats["fraud_transactions"]],
        })
        fig = px.bar(dist_df, x="Class", y="Count", color="Class", text="Count",
                     color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraudulent": COLOR_FRAUD})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_type="log", yaxis_title="Count (log scale)",
                           plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Fraud accounts for only {dataset_stats['fraud_rate_pct']:.3f}% of "
                   "transactions -- log scale is used so the fraud bar is actually visible.")

    with right:
        st.subheader("Model Performance Summary")
        df_metrics = metrics_dataframe()
        st.dataframe(
            df_metrics.style.format({c: "{:.3f}" for c in
                                      ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]})
            .background_gradient(subset=["PR-AUC"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("Key Finding")
    st.markdown(
        f"""<div class="finding-box">
        Because fraudulent transactions make up only {dataset_stats['fraud_rate_pct']:.3f}%
        of the dataset, <b>accuracy is not a meaningful metric</b> -- a model predicting
        "legitimate" for every transaction would still score above
        {100 - dataset_stats['fraud_rate_pct']:.1f}% accuracy while catching zero fraud.
        <b>PR-AUC and recall</b> are used as the primary evaluation metrics instead.
        Based on PR-AUC, <b>{best_name}</b> performed best on this dataset
        (PR-AUC = {metrics[best_name]['pr_auc']:.3f}).
        </div>""",
        unsafe_allow_html=True,
    )

# =============================================================================
# FRAUD DETECTION
# =============================================================================
elif page == "Fraud Detection":
    st.title("Transaction Fraud Check")
    st.caption("Enter transaction feature values to get a fraud assessment")

    if not SUPERVISED_MODEL_FILES:
        st.warning("No supervised model artifacts found. Run train_models.py first.")
        st.stop()

    default_model = DEPLOYMENT_MODEL if DEPLOYMENT_MODEL in SUPERVISED_MODEL_FILES else \
        list(SUPERVISED_MODEL_FILES.keys())[0]
    model_choice = st.selectbox(
        "Model used for prediction", list(SUPERVISED_MODEL_FILES.keys()),
        index=list(SUPERVISED_MODEL_FILES.keys()).index(default_model),
    )

    with st.form("transaction_form"):
        st.markdown("**Transaction Information**")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            time_val = st.number_input("Time (seconds since first transaction)",
                                         min_value=0.0, value=50000.0, step=1000.0)
        with tcol2:
            amount_val = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)

        st.markdown("**PCA Components**")
        st.caption("V1-V28 are anonymized, PCA-transformed features (their real-world "
                   "meaning was removed from the original dataset for privacy).")
        v_values = {}
        with st.expander("V1 - V14", expanded=True):
            cols = st.columns(7)
            for i in range(1, 15):
                v_values[f"V{i}"] = cols[(i - 1) % 7].number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")
        with st.expander("V15 - V28", expanded=False):
            cols = st.columns(7)
            for i in range(15, 29):
                v_values[f"V{i}"] = cols[(i - 15) % 7].number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

        submitted = st.form_submit_button("Check Transaction", use_container_width=True)

    if submitted:
        model = load_sklearn_model(SUPERVISED_MODEL_FILES[model_choice])
        if model is None or amount_scaler is None or time_scaler is None:
            st.error("Required model or scaler artifact could not be loaded.")
        else:
            row = pd.DataFrame([v_values])[PCA_COLUMNS]
            row["Amount_scaled"] = amount_scaler.transform([[amount_val]])[:, 0]
            row["Time_scaled"] = time_scaler.transform([[time_val]])[:, 0]
            row = row[feature_cols]

            proba = float(model.predict_proba(row)[0, 1])
            threshold = config["chosen_threshold"] if model_choice == "XGBoost (weighted)" else 0.5
            is_fraud = proba >= threshold

            st.divider()
            if is_fraud:
                st.markdown(
                    """<div class="result-box result-fraud">
                    <div class="result-title">FRAUDULENT / SUSPICIOUS TRANSACTION</div>
                    This transaction was flagged for review.</div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """<div class="result-box result-legit">
                    <div class="result-title">LEGITIMATE TRANSACTION</div>
                    No fraud indicators detected for this input.</div>""",
                    unsafe_allow_html=True,
                )

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Fraud Probability", f"{proba*100:.2f}%")
            mcol2.metric("Decision Threshold", f"{threshold:.3f}")
            mcol3.metric("Model Used", model_choice.split(" (")[0])
            st.progress(min(proba, 1.0), text=f"Fraud probability: {proba*100:.2f}%")
            st.caption(
                "Probability is this model's own calibrated output (`predict_proba`), "
                f"compared against a threshold of {threshold:.3f}. This is a prediction "
                "from a model trained on historical, anonymized data -- not a confirmed "
                "fraud determination."
            )

# =============================================================================
# MODEL COMPARISON
# =============================================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Evaluated consistently on the same held-out test set")

    df_metrics = metrics_dataframe()
    st.dataframe(
        df_metrics.style.format({c: "{:.3f}" for c in
                                  ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]})
        .background_gradient(subset=["PR-AUC"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )
    best = df_metrics.iloc[0]
    st.markdown(
        f"""<div class="finding-box"><b>{best['Model']}</b> ranks highest on PR-AUC
        ({best['PR-AUC']:.3f}), the most appropriate ranking metric for this heavily
        imbalanced problem (ROC-AUC can look deceptively high here even for weak
        models, since true negatives dominate the dataset).</div>""",
        unsafe_allow_html=True,
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Precision vs Recall")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Precision", x=df_metrics["Model"], y=df_metrics["Precision"],
                              marker_color=COLOR_NEUTRAL))
        fig.add_trace(go.Bar(name="Recall", x=df_metrics["Model"], y=df_metrics["Recall"],
                              marker_color="#b06a00"))
        fig.update_layout(barmode="group", plot_bgcolor="white", xaxis_tickangle=-35,
                           yaxis_title="Score", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("PR-AUC Ranking")
        fig = px.bar(df_metrics.sort_values("PR-AUC"), x="PR-AUC", y="Model", orientation="h",
                     color="PR-AUC", color_continuous_scale="Greens")
        fig.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select a model", list(metrics.keys()))
    cm = np.array(metrics[selected_model]["confusion_matrix"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                     labels=dict(x="Predicted", y="Actual", color="Count"),
                     x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"])
    fig.update_layout(coloraxis_showscale=False, height=420, width=420)
    st.plotly_chart(fig)
    tn, fp, fn, tp = cm.ravel()
    st.caption(f"True Negatives: {tn:,}  |  False Positives: {fp:,}  |  "
               f"False Negatives: {fn:,}  |  True Positives: {tp:,}  "
               f"(threshold = {metrics[selected_model]['threshold_used']:.3f})")

# =============================================================================
# ANOMALY DETECTION
# =============================================================================
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.caption("Unsupervised approaches: Isolation Forest and Autoencoder")

    st.markdown(
        """<div class="finding-box">
        Unlike the supervised models, these two approaches never see fraud labels
        during training. They learn what <i>normal</i> transactions look like, and
        flag anything that deviates -- transactions with unusually high anomaly
        scores are more different from the patterns learned from normal
        transactions.</div>""",
        unsafe_allow_html=True,
    )

    if not ANOMALY_MODELS:
        st.info("No anomaly-detection artifacts found. Run train_models.py first.")
        st.stop()

    tabs = st.tabs(ANOMALY_MODELS)
    for tab, model_name in zip(tabs, ANOMALY_MODELS):
        with tab:
            m = metrics[model_name]
            c1, c2, c3 = st.columns(3)
            c1.metric("PR-AUC", f"{m['pr_auc']:.3f}")
            c2.metric("Recall", f"{m['recall']:.3f}")
            c3.metric("Precision", f"{m['precision']:.3f}")

            if model_name == "Autoencoder":
                data = load_npz("reconstruction_errors.npz")
                if data is not None:
                    err_df = pd.DataFrame({
                        "Reconstruction Error": data["errors"],
                        "Class": np.where(data["labels"] == 1, "Fraud", "Legitimate"),
                    })
                    fig = px.histogram(err_df, x="Reconstruction Error", color="Class",
                                        color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
                                        barmode="overlay", nbins=60, log_y=True)
                    fig.update_layout(plot_bgcolor="white", title="Reconstruction Error Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("The autoencoder was trained only on legitimate transactions, so it "
                               "reconstructs them well (low error). Transactions it reconstructs "
                               "poorly (high error) are flagged as anomalous.")
            elif model_name == "Isolation Forest":
                data = load_npz("iso_scores.npz")
                if data is not None:
                    score_df = pd.DataFrame({
                        "Anomaly Score": data["scores"],
                        "Class": np.where(data["labels"] == 1, "Fraud", "Legitimate"),
                    })
                    fig = px.histogram(score_df, x="Anomaly Score", color="Class",
                                        color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
                                        barmode="overlay", nbins=60, log_y=True)
                    fig.update_layout(plot_bgcolor="white", title="Isolation Forest Anomaly Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Higher scores mean the transaction was easier for the model to "
                               "isolate from the rest of the data -- i.e. more unusual.")

            cm = np.array(m["confusion_matrix"])
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"])
            fig.update_layout(coloraxis_showscale=False, height=380, width=380,
                               title=f"{model_name} Confusion Matrix")
            st.plotly_chart(fig)

# =============================================================================
# DATASET INSIGHTS
# =============================================================================
elif page == "Dataset Insights":
    st.title("Dataset Insights")
    st.caption("Exploratory findings from the original analysis")

    sample_df = load_csv("sample_data.csv")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Class Distribution")
        dist_df = pd.DataFrame({
            "Class": ["Legitimate", "Fraudulent"],
            "Count": [dataset_stats["legit_transactions"], dataset_stats["fraud_transactions"]],
        })
        fig = px.pie(dist_df, names="Class", values="Count", color="Class",
                     color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraudulent": COLOR_FRAUD})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Transaction Amount (Fraud vs Normal)")
        if sample_df is not None:
            fig = px.box(sample_df, x="Class", y="Amount", color="Class",
                         color_discrete_map={0: COLOR_LEGIT, 1: COLOR_FRAUD}, log_y=True,
                         labels={"Class": "0 = Legitimate, 1 = Fraud"})
            fig.update_layout(showlegend=False, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Log scale used -- transaction amounts are heavily right-skewed.")

    st.divider()
    st.subheader("Strongest Correlations with Fraud")
    if correlations:
        corr_df = pd.DataFrame([{"Feature": k, "Correlation": v} for k, v in correlations.items()])
        corr_df["AbsCorrelation"] = corr_df["Correlation"].abs()
        corr_df = corr_df.sort_values("AbsCorrelation", ascending=True).tail(15)
        fig = px.bar(corr_df, x="Correlation", y="Feature", orientation="h",
                     color="Correlation", color_continuous_scale="RdBu", range_color=[-1, 1])
        fig.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Feature Distributions: Legitimate vs Fraud")
    if sample_df is not None:
        feature_choice = st.selectbox("Select a feature", PCA_COLUMNS, index=13)  # default V14
        fig = px.histogram(sample_df, x=feature_choice,
                            color=sample_df["Class"].map({0: "Legitimate", 1: "Fraud"}),
                            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
                            barmode="overlay", nbins=60, histnorm="probability density")
        fig.update_layout(plot_bgcolor="white", legend_title_text="Class")
        st.plotly_chart(fig, use_container_width=True)

    if feature_importance:
        st.divider()
        st.subheader("Feature Importance (XGBoost)")
        fi_df = pd.DataFrame([{"Feature": k, "Importance": v} for k, v in feature_importance.items()]
                              ).head(15).sort_values("Importance")
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[COLOR_NEUTRAL])
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# ABOUT
# =============================================================================
elif page == "About":
    st.title("About This Project")

    st.subheader("Problem")
    st.write("Credit card fraud detection is an extremely imbalanced binary classification "
             "problem: fraudulent transactions are rare, costly, and must be distinguished "
             "from an overwhelming majority of legitimate transactions with similar-looking features.")

    st.subheader("Dataset")
    st.write(f"The dataset contains {dataset_stats['total_transactions']:,} transactions made "
             f"by European cardholders, of which {dataset_stats['fraud_transactions']:,} "
             f"({dataset_stats['fraud_rate_pct']:.3f}%) are fraudulent. Features V1-V28 are the "
             "result of a PCA transformation applied for confidentiality; only `Time` and "
             "`Amount` are in their original form.")

    st.subheader("Methods")
    st.markdown("""
- **Baseline:** Logistic Regression on scaled features
- **Imbalance handling:** class-weighting (`class_weight='balanced'` / `scale_pos_weight`)
  and SMOTE oversampling, compared directly
- **Supervised classifiers:** Logistic Regression, Random Forest, XGBoost
- **Threshold tuning:** cost-based threshold selection (assumed cost of a missed
  fraud vs. a false alarm), rather than the default 0.5 cutoff
- **Unsupervised anomaly detection:** Isolation Forest and a PyTorch Autoencoder
  (trained only on legitimate transactions, flagging high reconstruction error)
    """)

    st.subheader("Evaluation")
    st.write("Precision, Recall, F1-score, and PR-AUC are used instead of accuracy. "
             f"With fraud at roughly {dataset_stats['fraud_rate_pct']:.3f}% of transactions, "
             "a model predicting 'legitimate' every time would score very high accuracy "
             "while catching zero fraud. PR-AUC is a stricter, more honest metric here than "
             "ROC-AUC, because ROC-AUC is inflated by the large number of easy true negatives.")

    st.subheader("Limitations")
    st.markdown("""<div class="limitation-box">

- Features V1-V28 are anonymized PCA components -- their real-world meaning is unknown
- The dataset is historical (European cardholders, September 2013) and fraud patterns evolve over time
- Model performance on this dataset may not generalize to real-world, live transaction streams
- An anomaly flag from Isolation Forest or the Autoencoder does not mean confirmed fraud --
  it means the transaction looked statistically unusual
- **This application is a portfolio / analysis project, not a production fraud detection system.**

</div>""", unsafe_allow_html=True)
