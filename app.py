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
st.write("Upload a dataset to train and evaluate the model.")

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if st.button("Train Logistic Regression Model"):

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.success(f"Accuracy: {acc:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Pred 0', 'Pred 1'],
                        yticklabels=['Actual 0', 'Actual 1'],
                        ax=ax)

            st.pyplot(fig)

            # Save model
            joblib.dump(model, "logistic_model.pkl")
            st.success("Model saved successfully!")
