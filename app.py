"""
app.py -- Credit Card Fraud Detection & Analytics Dashboard

Reads ONLY pre-saved artifacts produced by train_models.py. Never trains
or refits anything, so it starts fast and can't drift from what was
actually trained.

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
# NOTE: torch and shap are intentionally NOT imported here. They're loaded
# lazily, only inside the functions that actually need them (load_ae,
# load_shap_explainer), so pages that never touch the Autoencoder or SHAP
# explanations don't pay the cost of importing those heavy libraries on
# every sidebar click -- this matters a lot on Streamlit Community Cloud's
# limited free-tier CPU/RAM.

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model_utils import MODELS_DIR, PCA_COLUMNS, RAW_FEATURE_ORDER
# load_autoencoder is imported lazily inside load_ae() below, NOT here --
# see src/autoencoder_utils.py for why.

st.set_page_config(page_title="Fraud Detection Analytics", layout="wide", initial_sidebar_state="expanded")

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
COLOR_WARN = "#b06a00"


# ---------------------------------------------------------------------------
# Artifact loading -- cached
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(name):
    path = os.path.join(MODELS_DIR, name)
    return json.load(open(path)) if os.path.exists(path) else None


@st.cache_data
def load_csv(name):
    path = os.path.join(MODELS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_resource
def load_pipeline(filename):
    """Loads a complete sklearn Pipeline (preprocessing + classifier) as one object."""
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_ae(input_dim):
    # Both the torch-backed module AND its load_autoencoder function are
    # imported here, lazily -- only when Anomaly Detection actually needs
    # the Autoencoder, not on every page render.
    from src.autoencoder_utils import load_autoencoder
    path = os.path.join(MODELS_DIR, "autoencoder.pth")
    return load_autoencoder(path, input_dim) if os.path.exists(path) else None


@st.cache_resource
def load_shap_explainer():
    import shap  # lazy import -- only pulled in when a SHAP explanation is actually requested
    path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
    return joblib.load(path) if os.path.exists(path) else None


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
best_xgb_params = load_json("best_xgb_params.json")

if config is None or metrics is None or dataset_stats is None:
    st.error(
        "**No trained model artifacts found in `models/`.**\n\n"
        "This dashboard only loads pre-trained models -- run the training "
        "script first:\n\n```\npython train_models.py\n```\n\n"
        "Takes 15-25 minutes (hyperparameter tuning + SMOTE are the slow "
        "parts) and only needs to run once, or again if you retrain."
    )
    st.stop()

SUPERVISED_MODEL_FILES = {
    "Logistic Regression (baseline)": "lr_baseline.pkl",
    "Logistic Regression (weighted)": "lr_weighted.pkl",
    "Logistic Regression (SMOTE)": "lr_smote.pkl",
    "Random Forest (baseline)": "rf_baseline.pkl",
    "Random Forest (weighted)": "rf_weighted.pkl",
    "Random Forest (SMOTE)": "rf_smote.pkl",
    "XGBoost (baseline)": "xgb_baseline.pkl",
    "XGBoost (weighted, tuned)": "xgb_weighted.pkl",
}
SUPERVISED_MODEL_FILES = {
    name: fname for name, fname in SUPERVISED_MODEL_FILES.items()
    if name in metrics and os.path.exists(os.path.join(MODELS_DIR, fname))
}
ANOMALY_SCORE_SOURCES = {
    "Isolation Forest": "iso_scores.npz",
    "Autoencoder": "reconstruction_errors.npz",
    "Ensemble (Isolation Forest + Autoencoder)": "ensemble_scores.npz",
}
ANOMALY_SCORE_SOURCES = {k: v for k, v in ANOMALY_SCORE_SOURCES.items() if k in metrics}
DEPLOYMENT_MODEL = config.get("deployment_model", "XGBoost (weighted, tuned)")
DEPLOYMENT_MODEL_FILE = config.get("deployment_model_file", "xgb_weighted.pkl")


def metrics_dataframe(exclude_tuned=True):
    rows = []
    for name, m in metrics.items():
        if exclude_tuned and "tuned threshold" in name:
            continue
        rows.append({
            "Model": name, "Precision": m["precision"], "Recall": m["recall"],
            "F1-score": m["f1"], "PR-AUC": m["pr_auc"],
            "PR-AUC CI low": m.get("pr_auc_ci_low"), "PR-AUC CI high": m.get("pr_auc_ci_high"),
            "ROC-AUC": m["roc_auc"],
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
st.sidebar.caption(f"Models evaluated: {len(SUPERVISED_MODEL_FILES) + len(ANOMALY_SCORE_SOURCES)}\n\n"
                    f"Best by PR-AUC: {config['best_model_by_pr_auc']}\n\n"
                    f"Validation: {config.get('split_strategy', 'random split')}")

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
        fig.update_layout(showlegend=False, yaxis_type="log", yaxis_title="Count (log scale)", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Fraud accounts for only {dataset_stats['fraud_rate_pct']:.3f}% of transactions.")

    with right:
        st.subheader("Model Performance Summary")
        df_metrics = metrics_dataframe()
        display_cols = ["Model", "Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]
        st.dataframe(
            df_metrics[display_cols].style.format({c: "{:.3f}" for c in display_cols[1:]})
            .background_gradient(subset=["PR-AUC"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("Key Finding")
    st.markdown(
        f"""<div class="finding-box">
        Because fraudulent transactions make up only {dataset_stats['fraud_rate_pct']:.3f}%
        of the dataset, <b>accuracy is not a meaningful metric</b>. <b>PR-AUC and recall</b>
        are used instead. Based on PR-AUC, <b>{best_name}</b> performed best
        (PR-AUC = {metrics[best_name]['pr_auc']:.3f}). Models were validated using a
        <b>{config.get('split_strategy', 'random split')}</b> -- a stricter, more realistic
        test than a random split, since it never trains on transactions that happen
        after the ones it's evaluated on.
        </div>""",
        unsafe_allow_html=True,
    )

# =============================================================================
# FRAUD DETECTION
# =============================================================================
elif page == "Fraud Detection":
    st.title("Transaction Fraud Check")
    st.caption("Score a single transaction, or upload a batch as CSV")

    if not SUPERVISED_MODEL_FILES:
        st.warning("No supervised model artifacts found. Run train_models.py first.")
        st.stop()

    tab_single, tab_batch = st.tabs(["Single Transaction", "Batch Upload"])

    default_model = DEPLOYMENT_MODEL if DEPLOYMENT_MODEL in SUPERVISED_MODEL_FILES else \
        list(SUPERVISED_MODEL_FILES.keys())[0]

    # --- Single transaction ---------------------------------------------------
    with tab_single:
        model_choice = st.selectbox(
            "Model used for prediction", list(SUPERVISED_MODEL_FILES.keys()),
            index=list(SUPERVISED_MODEL_FILES.keys()).index(default_model),
            key="single_model_choice",
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
            st.caption("V1-V28 are anonymized, PCA-transformed features.")
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
            pipeline = load_pipeline(SUPERVISED_MODEL_FILES[model_choice])
            if pipeline is None:
                st.error("Selected model artifact could not be loaded.")
            else:
                row = pd.DataFrame([{**v_values, "Amount": amount_val, "Time": time_val}])[RAW_FEATURE_ORDER]

                proba = float(pipeline.predict_proba(row)[0, 1])
                threshold = config["chosen_threshold"] if model_choice == DEPLOYMENT_MODEL else 0.5
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

                # --- SHAP explanation (only available for the tuned XGBoost deployment model) ---
                if model_choice == DEPLOYMENT_MODEL:
                    explainer = load_shap_explainer()
                    if explainer is not None:
                        st.divider()
                        st.subheader("Why This Prediction?")
                        st.caption(
                            "Each bar shows how much a feature pushed this specific "
                            "prediction toward fraud (right, red) or toward legitimate "
                            "(left, green), relative to the model's average prediction."
                        )
                        preprocessed_row = pipeline.named_steps["preprocess"].transform(row)
                        shap_vals = explainer.shap_values(preprocessed_row)
                        feature_names = PCA_COLUMNS + ["Amount", "Time"]

                        shap_df = pd.DataFrame({
                            "Feature": feature_names,
                            "SHAP value": shap_vals[0],
                        }).sort_values("SHAP value", key=abs, ascending=True).tail(12)

                        fig = px.bar(
                            shap_df, x="SHAP value", y="Feature", orientation="h",
                            color=shap_df["SHAP value"] > 0,
                            color_discrete_map={True: COLOR_FRAUD, False: COLOR_LEGIT},
                        )
                        fig.update_layout(plot_bgcolor="white", showlegend=False,
                                           xaxis_title="Impact on fraud probability (SHAP value)")
                        st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Probability is this model's own calibrated output, compared against "
                    f"a threshold of {threshold:.3f}. This is a prediction from a model "
                    "trained on historical, anonymized data -- not a confirmed fraud determination."
                )

    # --- Batch upload -----------------------------------------------------------
    with tab_batch:
        st.caption(
            "Upload a CSV containing columns: Time, V1-V28, Amount (same schema as "
            "the training data, without the Class column)."
        )
        batch_model_choice = st.selectbox(
            "Model used for scoring", list(SUPERVISED_MODEL_FILES.keys()),
            index=list(SUPERVISED_MODEL_FILES.keys()).index(default_model),
            key="batch_model_choice",
        )
        uploaded = st.file_uploader("Upload transactions CSV", type="csv")

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                missing_cols = [c for c in RAW_FEATURE_ORDER if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
                else:
                    pipeline = load_pipeline(SUPERVISED_MODEL_FILES[batch_model_choice])
                    threshold = config["chosen_threshold"] if batch_model_choice == DEPLOYMENT_MODEL else 0.5

                    scored = batch_df.copy()
                    probs = pipeline.predict_proba(batch_df[RAW_FEATURE_ORDER])[:, 1]
                    scored["fraud_probability"] = probs
                    scored["flagged"] = probs >= threshold
                    scored = scored.sort_values("fraud_probability", ascending=False)

                    n_flagged = int(scored["flagged"].sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Transactions Scored", f"{len(scored):,}")
                    c2.metric("Flagged as Fraud", f"{n_flagged:,}")
                    c3.metric("Flag Rate", f"{n_flagged/len(scored)*100:.2f}%")

                    fig = px.histogram(scored, x="fraud_probability", nbins=50,
                                        title="Distribution of Fraud Probabilities")
                    fig.add_vline(x=threshold, line_dash="dash", line_color=COLOR_WARN,
                                  annotation_text=f"threshold={threshold:.3f}")
                    fig.update_layout(plot_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(scored, use_container_width=True, height=400)
                    st.download_button(
                        "Download Scored Transactions (CSV)",
                        scored.to_csv(index=False).encode("utf-8"),
                        file_name="scored_transactions.csv", mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Could not process the uploaded file: {e}")

# =============================================================================
# MODEL COMPARISON
# =============================================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption(f"Evaluated on the same held-out test set ({config.get('split_strategy', 'random split')})")

    df_metrics = metrics_dataframe()
    display_cols = ["Model", "Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]
    st.dataframe(
        df_metrics[display_cols].style.format({c: "{:.3f}" for c in display_cols[1:]})
        .background_gradient(subset=["PR-AUC"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )
    best = df_metrics.iloc[0]
    st.markdown(
        f"""<div class="finding-box"><b>{best['Model']}</b> ranks highest on PR-AUC
        ({best['PR-AUC']:.3f}), the most appropriate ranking metric for this heavily
        imbalanced problem.</div>""", unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("PR-AUC with 95% Confidence Intervals")
    st.caption(
        "Error bars from bootstrap resampling (300 resamples of the test set). "
        "Overlapping intervals mean the difference between two models' PR-AUC "
        "may not be statistically distinguishable from noise."
    )
    ci_df = df_metrics.dropna(subset=["PR-AUC CI low", "PR-AUC CI high"])
    if not ci_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ci_df["PR-AUC"], y=ci_df["Model"], orientation="h",
            error_x=dict(
                type="data", symmetric=False,
                array=ci_df["PR-AUC CI high"] - ci_df["PR-AUC"],
                arrayminus=ci_df["PR-AUC"] - ci_df["PR-AUC CI low"],
            ),
            marker_color=COLOR_NEUTRAL,
        ))
        fig.update_layout(plot_bgcolor="white", xaxis_title="PR-AUC")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Precision vs Recall")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Precision", x=df_metrics["Model"], y=df_metrics["Precision"], marker_color=COLOR_NEUTRAL))
        fig.add_trace(go.Bar(name="Recall", x=df_metrics["Model"], y=df_metrics["Recall"], marker_color=COLOR_WARN))
        fig.update_layout(barmode="group", plot_bgcolor="white", xaxis_tickangle=-35,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
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
    st.caption(f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,} "
               f"(threshold = {metrics[selected_model]['threshold_used']:.3f})")

    if best_xgb_params:
        st.divider()
        st.subheader("Hyperparameter Tuning (XGBoost Deployment Model)")
        st.caption(f"RandomizedSearchCV, optimizing PR-AUC directly, best CV PR-AUC = "
                   f"{best_xgb_params['best_cv_pr_auc']:.4f}")
        st.json(best_xgb_params["best_params"])

    st.divider()
    st.subheader("Cost-Based Threshold Selection")
    cost_curve = load_npz("cost_curve.npz")
    if cost_curve is not None:
        st.caption(
            f"Assumes a missed fraud costs \\${config.get('cost_fn_assumed', 500)} and a "
            f"false alarm costs \\${config.get('cost_fp_assumed', 5)}. The threshold "
            "minimizing total assumed cost is marked below, rather than using the default 0.5 cutoff."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cost_curve["thresholds"], y=cost_curve["costs"],
                                  mode="lines", line_color=COLOR_NEUTRAL))
        fig.add_vline(x=float(cost_curve["best_threshold"]), line_dash="dash", line_color=COLOR_FRAUD,
                      annotation_text=f"chosen threshold = {float(cost_curve['best_threshold']):.3f}")
        fig.update_layout(plot_bgcolor="white", xaxis_title="Threshold", yaxis_title="Total Assumed Cost ($)")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# ANOMALY DETECTION
# =============================================================================
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.caption("Unsupervised approaches: Isolation Forest, Autoencoder, and their ensemble")

    st.markdown(
        """<div class="finding-box">
        These approaches never see fraud labels during training -- they learn what
        <i>normal</i> transactions look like and flag deviations. The Ensemble
        combines both scores (min-max normalized and averaged) to test whether
        two different anomaly signals together beat either alone.</div>""",
        unsafe_allow_html=True,
    )

    if not ANOMALY_SCORE_SOURCES:
        st.info("No anomaly-detection artifacts found. Run train_models.py first.")
        st.stop()

    tabs = st.tabs(list(ANOMALY_SCORE_SOURCES.keys()))
    for tab, model_name in zip(tabs, ANOMALY_SCORE_SOURCES.keys()):
        with tab:
            m = metrics[model_name]
            c1, c2, c3 = st.columns(3)
            c1.metric("PR-AUC", f"{m['pr_auc']:.3f}")
            c2.metric("Recall", f"{m['recall']:.3f}")
            c3.metric("Precision", f"{m['precision']:.3f}")

            data = load_npz(ANOMALY_SCORE_SOURCES[model_name])
            if data is not None:
                score_key = "errors" if "errors" in data else "scores"
                label = "Reconstruction Error" if model_name == "Autoencoder" else "Anomaly Score"
                score_df = pd.DataFrame({
                    label: data[score_key],
                    "Class": np.where(data["labels"] == 1, "Fraud", "Legitimate"),
                })
                fig = px.histogram(score_df, x=label, color="Class",
                                    color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
                                    barmode="overlay", nbins=60, log_y=True)
                fig.update_layout(plot_bgcolor="white", title=f"{label} Distribution")
                st.plotly_chart(fig, use_container_width=True)

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
        feature_choice = st.selectbox("Select a feature", PCA_COLUMNS, index=13)
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
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[COLOR_NEUTRAL])
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    shap_data = load_npz("shap_sample.npz")
    if shap_data is not None:
        st.divider()
        st.subheader("Global Feature Impact (SHAP Summary)")
        st.caption("Average magnitude of each feature's effect on predictions across a sample of test transactions.")
        mean_abs_shap = np.abs(shap_data["shap_values"]).mean(axis=0)
        shap_summary = pd.DataFrame({
            "Feature": shap_data["feature_names"], "Mean |SHAP value|": mean_abs_shap,
        }).sort_values("Mean |SHAP value|", ascending=True).tail(15)
        fig = px.bar(shap_summary, x="Mean |SHAP value|", y="Feature", orientation="h",
                     color_discrete_sequence=[COLOR_FRAUD])
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
    st.write(f"{dataset_stats['total_transactions']:,} transactions from European cardholders, "
             f"{dataset_stats['fraud_transactions']:,} ({dataset_stats['fraud_rate_pct']:.3f}%) fraudulent. "
             "Features V1-V28 are PCA-transformed for confidentiality; only `Time` and `Amount` are original.")

    st.subheader("Methods")
    st.markdown(f"""
- **Validation:** {config.get('split_strategy', 'random split')} -- chosen over a random
  split because it never trains on transactions occurring after the ones it's tested on
- **Baseline:** Logistic Regression on preprocessed features
- **Imbalance handling:** class-weighting and SMOTE, compared directly rather than assumed
- **Supervised classifiers:** Logistic Regression, Random Forest, XGBoost
- **Hyperparameter tuning:** RandomizedSearchCV on the deployment model, optimizing PR-AUC directly
- **Threshold tuning:** cost-based threshold selection (assumed \\${config.get('cost_fn_assumed', 500)}
  cost per missed fraud vs. \\${config.get('cost_fp_assumed', 5)} per false alarm)
- **Explainability:** SHAP values on the deployment model, for per-transaction explanations
- **Unsupervised anomaly detection:** Isolation Forest, a PyTorch Autoencoder (trained only
  on legitimate transactions), and an ensemble of the two
- **Statistical rigor:** bootstrap confidence intervals on PR-AUC, so differences between
  models can be judged against the noise floor, not just point estimates
- **Architecture:** every model is a single `sklearn.Pipeline` bundling preprocessing and
  the classifier together, so inference can never apply mismatched preprocessing
    """)

    st.subheader("Evaluation")
    st.write("Precision, Recall, F1-score, and PR-AUC are used instead of accuracy, since "
             "accuracy is trivially high on a 99.8%-majority-class dataset. PR-AUC confidence "
             "intervals are reported via bootstrap resampling so model rankings reflect real "
             "differences, not noise.")

    st.subheader("Limitations")
    st.markdown("""<div class="limitation-box">

- Features V1-V28 are anonymized PCA components -- their real-world meaning is unknown
- The dataset is historical (European cardholders, September 2013)
- Performance here is not a guarantee of performance on live transaction streams
- An anomaly flag means "statistically unusual," not "confirmed fraud"
- SHAP explanations describe this model's reasoning, not ground-truth causal fraud mechanisms
- **This is a portfolio / applied-ML project, not a production fraud detection system.**

</div>""", unsafe_allow_html=True)