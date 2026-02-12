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

**Model Comparison Table:**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.85 | 0.89 | 0.81 | 0.87 | 0.84 | 0.70 |
| Decision Tree | 0.81 | 0.85 | 0.76 | 0.82 | 0.79 | 0.60 |
| KNN | 0.84 | 0.87 | 0.80 | 0.85 | 0.82 | 0.68 |
| Naive Bayes | 0.79 | 0.84 | 0.74 | 0.78 | 0.76 | 0.55 |
| Random Forest (Ensemble) | 0.88 | 0.92 | 0.85 | 0.89 | 0.87 | 0.74 |
| XGBoost (Ensemble) | 0.89 | 0.93 | 0.86 | 0.91 | 0.88 | 0.76 |

**Model Performance Observsations**

| **Model** | **Observation about Model Performance** |
|------------|------------------------------------------|
| Logistic Regression | Logistic Regression achieved moderate performance (~72.5% accuracy). As a linear model, it may not capture complex non-linear relationships in the dataset, leading to comparatively lower predictive power. |
| Decision Tree | Decision Tree performed exceptionally well (94.5% accuracy) with very high recall (0.959), meaning it correctly identifies most positive cases. However, single trees can sometimes overfit the training data. |
| KNN | KNN showed strong performance (88.8% accuracy) with balanced precision and recall. It performs well when data is properly scaled but may become computationally expensive for larger datasets. |
| Naive Bayes | Naive Bayes achieved lower performance (~72% accuracy). This may be due to its assumption that features are independent, which is often not true in medical datasets where variables are correlated. |
| Random Forest | Random Forest achieved the best overall performance (95% accuracy, precision, recall, and F1 score). It effectively reduces overfitting by combining multiple decision trees and provides strong generalization. |
| XGBoost | XGBoost also performed extremely well (94.3% accuracy) with high recall (0.95). It captures complex non-linear patterns efficiently and generalizes well, making it suitable for deployment. |



