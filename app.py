import streamlit as st
import pandas as pd
import joblib

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

st.set_page_config(
    page_title="Adult Income Classification",
    layout="wide"
)

st.title("Adult Income Classification")
st.write("BITS WILP Machine Learning Assignment 2")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

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

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        if "income" not in df.columns:
            st.error(
                "Column 'income' not found in uploaded file."
            )
            st.stop()

        y_true = df["income"]

        X = df.drop(
            columns=["income"]
        )

        scaler = joblib.load(
            "model/scaler.pkl"
        )

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

            try:
                y_score = model.decision_function(
                    X_model
                )
            except:
                y_score = y_pred

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

        try:
            auc = roc_auc_score(
                y_true,
                y_score
            )
        except:
            auc = 0

        mcc = matthews_corrcoef(
            y_true,
            y_pred
        )

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            round(accuracy, 4)
        )

        col2.metric(
            "AUC",
            round(auc, 4)
        )

        col3.metric(
            "Precision",
            round(precision, 4)
        )

        col1.metric(
            "Recall",
            round(recall, 4)
        )

        col2.metric(
            "F1 Score",
            round(f1, 4)
        )

        col3.metric(
            "MCC",
            round(mcc, 4)
        )

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"]
            )
        )

        st.subheader("Classification Report")

        st.text(
            classification_report(
                y_true,
                y_pred
            )
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )