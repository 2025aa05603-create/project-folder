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
import kagglehub
import zipfile

from sklearn.model_selection import train_test_split
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

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction")
st.write("Training data is fetched directly from Kaggle. Upload test_data.csv for evaluation.")

# ===============================
# DOWNLOAD TRAINING DATA FROM KAGGLE
# ===============================
dataset_name = "atousaomidvar/raw-merged-heart-dataset"

@st.cache_data
def load_kaggle_data():
    dataset_path = kagglehub.dataset_download(dataset_name)

    csv_file = None

    if os.path.isdir(dataset_path):
        for file in os.listdir(dataset_path):
            if file.endswith(".csv"):
                csv_file = os.path.join(dataset_path, file)
                break
    elif dataset_path.endswith(".zip"):
        extract_folder = "data"
        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
            for file in zip_ref.namelist():
                if file.endswith(".csv"):
                    csv_file = os.path.join(extract_folder, file)
                    break

    if csv_file is None:
        return None

    return pd.read_csv(csv_file)


df = load_kaggle_data()

if df is None:
    st.error("Failed to load dataset from Kaggle.")
    st.stop()

st.success("Training dataset loaded from Kaggle successfully!")

# ===============================
# PREPROCESSING FUNCTION
# ===============================
def preprocess(df):
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


df = preprocess(df)

if 'target' not in df.columns:
    st.error("Training dataset must contain 'target' column.")
    st.stop()

X_train = df.drop('target', axis=1)
y_train = df['target']

# ===============================
# SCALER
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ===============================
# MODEL SELECTION
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

# ===============================
# TRAIN MODEL
# ===============================
if st.button("Train Model"):

    if model_name == "Logistic Regression":
        model = LogisticRegression(C=0.005, penalty="l2", solver="lbfgs", max_iter=1000)
        model.fit(X_train_scaled, y_train)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=13, min_samples_split=200,
                                       min_samples_leaf=4, random_state=42)
        model.fit(X_train, y_train)

    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)

    elif model_name == "Naive Bayes":
        model = GaussianNB()
        model.fit(X_train, y_train)

    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=11,
                                       min_samples_split=10, random_state=42)
        model.fit(X_train, y_train)

    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(n_estimators=100, max_depth=7,
                                  learning_rate=0.1, random_state=42,
                                  eval_metric='logloss')
        model.fit(X_train, y_train)

    st.success(f"{model_name} trained successfully!")

    # ===============================
    # UPLOAD TEST DATA
    # ===============================
    st.subheader("📤 Upload test_data.csv for Evaluation")
    uploaded_test = st.file_uploader("Upload test_data.csv", type=["csv"])

    if uploaded_test is not None:

        test_df = pd.read_csv(uploaded_test)
        test_df = preprocess(test_df)

        if 'target' not in test_df.columns:
            st.error("Test data must contain 'target' column.")
            st.stop()

        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # Scale only if required
        if model_name in ["Logistic Regression", "KNN"]:
            X_test = scaler.transform(X_test)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ===============================
        # METRICS
        # ===============================
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("📊 Test Data Performance")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # ===============================
        # CONFUSION MATRIX
        # ===============================
        st.subheader("🔍 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        st.pyplot(fig)

        # ===============================
        # SAVE MODEL
        # ===============================
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{model_name.replace(' ', '_').lower()}.pkl")
        st.success("Model saved successfully!")
