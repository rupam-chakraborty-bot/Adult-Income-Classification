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
# PAGE HEADER
# ==========================================================

st.title("💰 Adult Income Classification")

st.markdown("""
### Machine Learning Assignment 2

Predict whether an individual's annual income exceeds **$50K**
using Machine Learning classification models trained on the
Adult Income Dataset.

Upload the generated **test_data.csv** file and evaluate model performance.
""")

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📂 Upload test_data.csv",
    type=["csv"]
)

# ==========================================================
# MODEL SELECTION
# ==========================================================

model_name = st.selectbox(
    "🤖 Select Model",
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
# MAIN PROCESSING
# ==========================================================

if uploaded_file is not None:

    try:

        # ==================================================
        # READ CSV
        # ==================================================

        df = pd.read_csv(uploaded_file)

        # ==================================================
        # DATASET INFORMATION
        # ==================================================

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

        # ==================================================
        # SHOW SAMPLE DATA
        # ==================================================

        with st.expander(
            "🔍 View Uploaded Dataset Sample"
        ):
            st.dataframe(
                df.head()
            )

        # ==================================================
        # VALIDATE TARGET
        # ==================================================

        if "income" not in df.columns:

            st.error(
                "Column 'income' not found in uploaded file."
            )

            st.stop()

        # ==================================================
        # FEATURES AND TARGET
        # ==================================================

        y_true = df["income"]

        X = df.drop(
            columns=["income"]
        )

        # ==================================================
        # LOAD SCALER
        # ==================================================

        scaler = joblib.load(
            "model/scaler.pkl"
        )

        # ==================================================
        # MODEL LOADING
        # ==================================================

        if model_name == "Logistic Regression":

            model = joblib.load(
                "model/logistic_model.pkl"
            )

            X_model = scaler.transform(X)

            y_pred = model.predict(X_model)

            y_score = model.predict_proba(
                X_model
            )[:, 1]

        elif model_name == "Decision Tree":

            model = joblib.load(
                "model/decision_tree.pkl"
            )

            X_model = X

            y_pred = model.predict(X_model)

            y_score = model.predict_proba(
                X_model
            )[:, 1]

        elif model_name == "KNN":

            model = joblib.load(
                "model/knn_model.pkl"
            )

            X_model = scaler.transform(X)

            y_pred = model.predict(X_model)

            y_score = model.predict_proba(
                X_model
            )[:, 1]

        elif model_name == "Naive Bayes":

            model = joblib.load(
                "model/naive_bayes.pkl"
            )

            X_model = scaler.transform(X)

            y_pred = model.predict(X_model)

            y_score = model.predict_proba(
                X_model
            )[:, 1]

        else:

            model = joblib.load(
                "model/svm_model.pkl"
            )

            X_model = scaler.transform(X)

            y_pred = model.predict(X_model)

            y_score = model.decision_function(
                X_model
            )

            st.success(
                "🏆 Best Performing Model Based on Notebook Evaluation"
            )

        # ==================================================
        # CALCULATE METRICS
        # ==================================================

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred
        )

        recall = recall_score(
            y_true,
            y_pred
        )

        f1 = f1_score(
            y_true,
            y_pred
        )

        auc = roc_auc_score(
            y_true,
            y_score
        )

        mcc = matthews_corrcoef(
            y_true,
            y_pred
        )

        # ==================================================
        # METRICS DASHBOARD
        # ==================================================

        st.markdown("---")

        st.subheader(
            "📈 Evaluation Metrics"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )

        c2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        c3.metric(
            "Precision",
            f"{precision*100:.2f}%"
        )

        c1.metric(
            "Recall",
            f"{recall*100:.2f}%"
        )

        c2.metric(
            "F1 Score",
            f"{f1*100:.2f}%"
        )

        c3.metric(
            "MCC",
            f"{mcc:.4f}"
        )

        st.markdown("---")

        # ==================================================
        # CONFUSION MATRIX
        # ==================================================

        st.subheader(
            "📋 Confusion Matrix"
        )

        st.caption(
            "Summary of correct and incorrect predictions made by the selected model."
        )

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(4, 3)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=1,
            linecolor="white",
            annot_kws={
                "size": 12
            },
            ax=ax
        )

        ax.set_xlabel(
            "Predicted",
            fontsize=10
        )

        ax.set_ylabel(
            "Actual",
            fontsize=10
        )

        ax.set_xticklabels(
            ["<=50K", ">50K"],
            rotation=0
        )

        ax.set_yticklabels(
            ["<=50K", ">50K"],
            rotation=0
        )

        ax.set_title(
            "Confusion Matrix",
            fontsize=12
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        # ==================================================
        # CLASSIFICATION REPORT
        # ==================================================

        st.markdown("---")

        st.subheader(
            "📄 Classification Report"
        )

        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(
            report_dict
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
