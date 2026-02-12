a) Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection of heart disease can significantly improve patient survival rates.
The objective of this project is to build and compare multiple machine learning classification models to predict whether a patient has heart disease based on clinical features.
The models are evaluated using multiple performance metrics, and the best-performing model is deployed using Streamlit.


b) Dataset Description

The dataset used is a merged heart disease dataset containing patient medical attributes.
number of features: 13 
number of instances used: 1281

**Features:**

age – Age of patient
sex – Gender (0 = Female, 1 = Male)
cp – Chest pain type
trestbps – Resting blood pressure
chol – Serum cholesterol
fbs – Fasting blood sugar
restecg – Resting ECG results
thalachh – Maximum heart rate achieved
exang – Exercise induced angina
oldpeak – ST depression
slope – Slope of ST segment
ca – Number of major vessels
thal – Thalassemia type
target – 1 = Heart Disease, 0 = No Disease

Here is your table formatted properly for a Jupyter Notebook Markdown cell:

** Model Comparison Table**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.85 | 0.89 | 0.81 | 0.87 | 0.84 | 0.70 |
| Decision Tree | 0.81 | 0.85 | 0.76 | 0.82 | 0.79 | 0.60 |
| KNN | 0.84 | 0.87 | 0.80 | 0.85 | 0.82 | 0.68 |
| Naive Bayes | 0.79 | 0.84 | 0.74 | 0.78 | 0.76 | 0.55 |
| Random Forest (Ensemble) | 0.88 | 0.92 | 0.85 | 0.89 | 0.87 | 0.74 |
| XGBoost (Ensemble) | 0.89 | 0.93 | 0.86 | 0.91 | 0.88 | 0.76 |



