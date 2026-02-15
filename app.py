# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction")
st.write("Upload a dataset to test a pre-trained ML model and view its performance.")

# ===============================
# HELPER FUNCTIONS
# ===============================
def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_model(model_name):
    model_filename = os.path.join(get_base_dir(), "model", f"{model_name.replace(' ', '_').lower()}_model.pkl")
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    else:
        st.error(f"Model file {model_filename} not found!")
        return None

def preprocess_data(df):
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(df.median(), inplace=True)
    return df

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    st.subheader(f"🔍 Confusion Matrix: {model_name}")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'], ax=ax)
    st.pyplot(fig)

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    if 'target' not in df.columns:
        st.error("Dataset must contain 'target' column to calculate evaluation metrics.")
    else:
        df = preprocess_data(df)
        X = df.drop('target', axis=1)
        y = df['target']

        # Scale features for models that need it
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # List of models
        model_names = [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]

        # Dropdown to select a model
        selected_model = st.selectbox("Select Model", model_names)

        if st.button("Selected Model"):
            model = load_model(selected_model)
            if model is not None:
                # Use scaled data for some models
                use_scaled = selected_model in ["Logistic Regression", "KNN", "Naive Bayes"]
                X_test_input = X_scaled if use_scaled else X
                y_pred = model.predict(X_test_input)
                y_prob = model.predict_proba(X_test_input)[:, 1]

                # Show metrics
                metrics = calculate_metrics(y, y_pred, y_prob)
                st.subheader("📊 Model Performance Metrics")
                cols = st.columns(6)
                for col, (name, val) in zip(cols, metrics.items()):
                    col.metric(name, f"{val:.4f}")

                # Confusion matrix
                plot_confusion_matrix(y, y_pred, selected_model)
