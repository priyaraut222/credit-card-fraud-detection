import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        color: #172033;
    }

    .subtitle {
        color: #667085;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #E4E7EC;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    }

    .metric-title {
        color: #667085;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #101828;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .success-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #ECFDF3;
        border: 1px solid #ABEFC6;
        color: #067647;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .danger-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #FEF3F2;
        border: 1px solid #FECDCA;
        color: #B42318;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #F2F4F7;
        border: 1px solid #D0D5DD;
        color: #344054;
    }

    section[data-testid="stSidebar"] {
        background-color: #F8F9FB;
        border-right: 1px solid #EAECF0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


model = load_model()
preprocessor = load_preprocessor()


# ============================================================
# DATASET INFORMATION
# ============================================================

TOTAL_TRANSACTIONS = 284807
FRAUD_TRANSACTIONS = 492
NORMAL_TRANSACTIONS = TOTAL_TRANSACTIONS - FRAUD_TRANSACTIONS

FRAUD_RATE = (FRAUD_TRANSACTIONS / TOTAL_TRANSACTIONS) * 100


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Credit Card Fraud Detection")

st.sidebar.caption(
    "Machine Learning • Anomaly Detection • Imbalanced Classification"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Fraud Detection",
        "Model Comparison",
        "Anomaly Detection",
        "Dataset Insights",
        "About"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("Credit Card Fraud Detection")
    st.markdown(
        '<div class="subtitle">'
        "Machine learning system for identifying fraudulent credit card transactions."
        "</div>",
        unsafe_allow_html=True
    )

    # KPI ROW
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            f"{TOTAL_TRANSACTIONS:,}"
        )

    with col2:
        st.metric(
            "Legitimate",
            f"{NORMAL_TRANSACTIONS:,}"
        )

    with col3:
        st.metric(
            "Fraudulent",
            f"{FRAUD_TRANSACTIONS:,}"
        )

    with col4:
        st.metric(
            "Fraud Rate",
            f"{FRAUD_RATE:.3f}%"
        )

    st.divider()

    # CLASS DISTRIBUTION
    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Transaction Class Distribution")

        class_df = pd.DataFrame({
            "Class": ["Legitimate", "Fraudulent"],
            "Count": [
                NORMAL_TRANSACTIONS,
                FRAUD_TRANSACTIONS
            ]
        })

        fig = px.bar(
            class_df,
            x="Class",
            y="Count",
            text="Count",
            title="Legitimate vs Fraudulent Transactions"
        )

        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Why Accuracy Is Misleading")

        st.markdown("""
        <div class="info-box">

        Fraudulent transactions represent only a very small fraction
        of the dataset.

        A model that predicted every transaction as legitimate could
        still achieve very high accuracy while detecting <b>zero</b>
        fraudulent transactions.

        Therefore, this project emphasizes:

        <b>Precision • Recall • F1-score • PR-AUC</b>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # MODEL SUMMARY
    st.subheader("Model Performance")

    if COMPARISON_PATH.exists():

        comparison = pd.read_csv(COMPARISON_PATH)

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "model_comparison.csv was not found. "
            "Add the model comparison results from the notebook."
        )


# ============================================================
# FRAUD DETECTION
# ============================================================

elif page == "Fraud Detection":

    st.title("Fraud Detection")
    st.markdown(
        '<div class="subtitle">'
        "Enter transaction features to generate a fraud prediction."
        "</div>",
        unsafe_allow_html=True
    )

    st.info(
        "Primary prediction model: Weighted XGBoost "
        "(best PR-AUC from the completed ML experiments)."
    )

    st.subheader("Transaction Information")

    col1, col2 = st.columns(2)

    with col1:
        time_value = st.number_input(
            "Time",
            min_value=0.0,
            value=0.0
        )

    with col2:
        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

    st.subheader("PCA Features")

    st.caption(
        "V1–V28 are anonymized PCA-transformed transaction features."
    )

    feature_values = {}

    # First 14 features
    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**V1 – V14**")

        for i in range(1, 15):

            feature_values[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"v{i}"
            )

    with col2:

        st.markdown("**V15 – V28**")

        for i in range(15, 29):

            feature_values[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"v{i}"
            )

    st.divider()

    predict_button = st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        input_data = {
            "Time": time_value,
            **feature_values,
            "Amount": amount_value
        }

        input_df = pd.DataFrame([input_data])

        try:

            # Apply EXACT training preprocessing
            processed_input = preprocessor.transform(input_df)

            probability = model.predict_proba(
                processed_input
            )[0][1]

            prediction = int(
                probability >= 0.5
            )

            st.divider()

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:

                    st.markdown(
                        """
                        <div class="danger-box">
                        🚨 Suspicious Transaction
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="success-box">
                        ✓ Likely Legitimate Transaction
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col2:

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2%}"
                )

            st.progress(
                float(probability)
            )

            if prediction == 1:

                st.warning(
                    "The model identified this transaction as suspicious. "
                    "This is a machine learning prediction and does not "
                    "constitute confirmation of fraud."
                )

            else:

                st.success(
                    "The model considers this transaction likely legitimate."
                )

        except Exception as e:

            st.error(
                "Prediction failed. Make sure the preprocessing pipeline "
                "and model were saved using the same feature order and "
                "transformations used during training."
            )

            st.exception(e)


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    st.markdown(
        '<div class="subtitle">'
        "Comparison of the machine learning approaches evaluated in the project."
        "</div>",
        unsafe_allow_html=True
    )

    if COMPARISON_PATH.exists():

        comparison = pd.read_csv(COMPARISON_PATH)

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Find likely metric columns
        numeric_columns = comparison.select_dtypes(
            include=np.number
        ).columns.tolist()

        if "PR-AUC" in comparison.columns:

            metric_column = "PR-AUC"

        elif "PR_AUC" in comparison.columns:

            metric_column = "PR_AUC"

        else:

            metric_column = None

        if metric_column:

            st.subheader("PR-AUC Comparison")

            sorted_df = comparison.sort_values(
                metric_column,
                ascending=True
            )

            fig = px.bar(
                sorted_df,
                x=metric_column,
                y=comparison.columns[0],
                orientation="h",
                text=metric_column,
                title="Precision-Recall AUC by Model"
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "model_comparison.csv was not found."
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.title("Anomaly Detection")

    st.markdown(
        '<div class="subtitle">'
        "Unsupervised and one-class approaches for identifying unusual transactions."
        "</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Isolation Forest")

        st.markdown("""
        Isolation Forest identifies observations that are easier
        to isolate from the rest of the dataset.

        Transactions that behave very differently from typical
        observations can therefore receive stronger anomaly scores.
        """)

    with col2:

        st.subheader("Autoencoder")

        st.markdown("""
        The autoencoder learns to reconstruct normal transaction
        patterns.

        Transactions with unusually large reconstruction errors
        can be flagged as potential anomalies.
        """)

    st.divider()

    st.subheader("Anomaly Detection Results")

    st.info(
        "Add the saved anomaly scores/results from the notebook here "
        "if you want the dashboard to display the actual score distributions."
    )


# ============================================================
# DATASET INSIGHTS
# ============================================================

elif page == "Dataset Insights":

    st.title("Dataset Insights")

    st.markdown(
        '<div class="subtitle">'
        "Exploratory analysis of transaction patterns and fraud distribution."
        "</div>",
        unsafe_allow_html=True
    )

    # CLASS DISTRIBUTION
    class_df = pd.DataFrame({
        "Class": ["Legitimate", "Fraudulent"],
        "Count": [
            NORMAL_TRANSACTIONS,
            FRAUD_TRANSACTIONS
        ]
    })

    fig = px.pie(
        class_df,
        names="Class",
        values="Count",
        title="Transaction Class Distribution"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Important Features")

    st.markdown("""
    The V1–V28 variables are PCA-transformed features from the
    anonymized transaction dataset.

    During exploratory analysis, several components showed
    noticeably different distributions between legitimate and
    fraudulent transactions.
    """)

    important_features = [
        "V17",
        "V14",
        "V12",
        "V10",
        "V16"
    ]

    feature = st.selectbox(
        "Select a feature",
        important_features
    )

    st.info(
        f"Selected feature: {feature}. "
        "The detailed distribution can be connected to the "
        "EDA results from the notebook."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.title("About the Project")

    st.markdown("""
    ## Credit Card Fraud Detection

    This project uses machine learning to identify potentially
    fraudulent credit card transactions under extreme class imbalance.

    ### Dataset

    The dataset contains **284,807 transactions**, including
    **492 fraudulent transactions**.

    Fraud therefore represents only approximately **0.173%**
    of all transactions.

    ### Machine Learning

    The project evaluates multiple approaches including supervised
    classification and anomaly detection.

    ### Evaluation

    Because of the severe class imbalance, the project focuses on:

    - Precision
    - Recall
    - F1-score
    - Precision-Recall AUC

    rather than relying only on accuracy.

    ### Important Limitation

    A model prediction should not be interpreted as confirmation
    that a transaction is fraudulent.

    The dataset also contains anonymized PCA-transformed features,
    meaning the individual V1–V28 components do not have directly
    interpretable real-world meanings.
    """)

    st.divider()

    st.caption(
        "Credit Card Fraud Detection • Machine Learning Project"
    )