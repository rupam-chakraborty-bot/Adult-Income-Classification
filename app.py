import streamlit as st
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Adult Income Classification",
    layout="wide"
)

# ==========================================================
# TITLE
# ==========================================================

st.title("💰 Adult Income Classification")

st.markdown("""
### Machine Learning Assignment 2

This application predicts whether an individual's annual income exceeds **$50K**
using multiple Machine Learning classification models trained on the Adult Income Dataset.

Upload **test_data.csv** generated from the notebook and evaluate model performance.
""")

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📂 Upload test_data.csv",
    type=["csv"]
)

# ==========================================================
# MODEL DROPDOWN
# ==========================================================

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

st.info(
    f"Selected Model: {model_name}"
)

# ==========================================================
# PROCESS FILE
# ==========================================================

if uploaded_file is not None:

    try:

        # --------------------------------------------------
        # Read Uploaded CSV
        # --------------------------------------------------

        df = pd.read_csv(uploaded_file)

        # --------------------------------------------------
        # Dataset Information
        # --------------------------------------------------

        st.subheader("📊 Dataset Information")

        col1, col2 = st.columns(2)

        col1.metric(
            "Rows",
            df.shape[0]
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        # --------------------------------------------------
        # Show Sample Data
        # --------------------------------------------------

        with st.expander(
            "🔍 View Sample of Uploaded Dataset"
        ):
            st.dataframe(df.head())

        # --------------------------------------------------
        # Check Target Column
        # --------------------------------------------------

        if "income" not in df.columns:

            st.error(
          
