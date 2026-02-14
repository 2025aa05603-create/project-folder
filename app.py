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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Prediction App")
st.write("Upload a dataset to train and evaluate different ML models.")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    if 'target' not in df.columns:
        st.error("Dataset must contain 'target' column.")
    else:
        X = df.drop('target', axis=1)
        y = df['target']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling (used only for some models)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ===============================
        # MODEL SELECTION DROPDOWN
        # ===============================
        model_name = st.selectbox(
            "Select Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            ]
        )

        if st.button("Train Selected Model"):

            # ===============================
            # MODEL INITIALIZATION
            # ===============================
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]

            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            elif model_name == "KNN":
                model = KNeighborsClassifier()
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]

            elif model_name == "Naive Bayes":
                model = GaussianNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            elif model_name == "Random Forest":
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            elif model_name == "XGBoost":
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            # ===============================
            # METRICS CALCULATION
            # ===============================
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # ===============================
            # DISPLAY METRICS
            # ===============================
            st.subheader("📊 Model Performance Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            # ===============================
            # CONFUSION MATRIX
            # ===============================
            st.subheader("🔍 Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                ax=ax
            )

            st.pyplot(fig)

            # ===============================
            # SAVE MODEL
            # ===============================
            os.makedirs("models", exist_ok=True)
            model_filename = f"models/{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_filename)

            st.success(f"{model_name} model saved successfully!")
