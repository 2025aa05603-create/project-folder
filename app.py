# ==========================================
# HEART DISEASE PREDICTION - STREAMLIT APP
# ==========================================

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
st.write("Training dataset is downloaded from Kaggle. Upload a test CSV file for evaluation.")

# ===============================
# DOWNLOAD TRAINING DATA
# ===============================
@st.cache_data
def load_kaggle_dataset(dataset_name):
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


# ===============================
# PREPROCESS FUNCTION
# ===============================
def preprocess_data(df):
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg',
                        'exang', 'slope', 'ca', 'thal']

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df.replace("?", np.nan, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(df.median(), inplace=True)

    return df


# ===============================
# LOAD DATA FROM KAGGLE
# ===============================
dataset_name = st.text_input(
    "Enter Kaggle Dataset Name:",
    value="atousaomidvar/raw-merged-heart-dataset"
)

if st.button("Download Training Dataset"):

    with st.spinner("Downloading dataset from Kaggle..."):
        df = load_kaggle_dataset(dataset_name)

    if df is None:
        st.error("Failed to download dataset.")
        st.stop()

    st.success("Training dataset loaded successfully!")
    st.write("### Training Data Preview")
    st.dataframe(df.head())

    if 'target' not in df.columns:
        st.error("Dataset must contain 'target' column.")
        st.stop()

    # ===============================
    # PREPROCESS TRAINING DATA
    # ===============================
    df = preprocess_data(df)

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

    if st.button("Train Selected Model"):

        # ===============================
        # MODEL INITIALIZATION
        # ===============================
        if model_name == "Logistic Regression":
            model = LogisticRegression(
                C=0.005,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000
            )
            model.fit(X_train_scaled, y_train)

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=13,
                min_samples_split=200,
                min_samples_leaf=4,
                random_state=42
            )
            model.fit(X_train, y_train)

        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train_scaled, y_train)

        elif model_name == "Naive Bayes":
            model = GaussianNB()
            model.fit(X_train, y_train)

        elif model_name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=11,
                min_samples_split=10,
                random_state=42
            )
            model.fit(X_train, y_train)

        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)

        st.success(f"{model_name} trained successfully!")

        # ===============================
        # SAVE MODEL
        # ===============================
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{model_name.replace(' ', '_').lower()}.pkl")

        # ===============================
        # UPLOAD TEST CSV
        # ===============================
        st.subheader("📤 Upload Test Dataset (CSV Only)")
        uploaded_test = st.file_uploader("Upload test_data.csv", type=["csv"])

        if uploaded_test is not None:

            test_df = pd.read_csv(uploaded_test)
            st.write("### Test Data Preview")
            st.dataframe(test_df.head())

            if 'target' not in test_df.columns:
                st.error("Test dataset must contain 'target' column.")
                st.stop()

            test_df = preprocess_data(test_df)

            X_test = test_df.drop('target', axis=1)
            y_test = test_df['target']

            # Apply scaling if required
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

            st.subheader("📊 Evaluation Metrics")

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
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['Actual 0', 'Actual 1']
            )

            st.pyplot(fig)
