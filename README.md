**a) Problem Statement**

Heart disease is one of the leading causes of death worldwide. Early detection of heart disease can significantly improve patient survival rates.
The objective of this project is to build and compare multiple machine learning classification models to predict whether a patient has heart disease based on clinical features.
The models are evaluated using multiple performance metrics, and the best-performing model is deployed using Streamlit.


**b) Dataset Description**

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

**c) Model Comparison Table:**

| ML Model Name       | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC      |
|---------------------|----------|---------|-----------|---------|----------|----------|
| Logistic Regression  | 0.6591   | 0.7337  | 0.5800    | 0.7632  | 0.6591   | 0.3432  |
| Decision Tree       | 0.7500   | 0.7642  | 0.6600    | 0.8684  | 0.7500   | 0.5284   |
| KNN                 | 0.6477   | 0.7868  | 0.5745    | 0.7105  | 0.6353   | 0.3083   |
| Naive Bayes         | 0.5909   | 0.7268  | 0.5179    | 0.7632  | 0.6170   | 0.2298   |
| Random Forest (Ensemble) | 0.8182 | 0.9179 | 0.7292    | 0.9211  | 0.8140   | 0.6576   |
| XGBoost (Ensemble)  | 0.8182   | 0.8800  | 0.7292    | 0.9211  | 0.8140   | 0.6576   |

**Model Performance Observsations**

| ML Model Name             | Observation ( model performance  )                                                                                      |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression       | Moderate performance (~73% accuracy). Works reasonably well for linearly separable data but struggles with complex patterns. Precision and recall are balanced, but overall lower than tree-based models, indicating possible underfitting. |
| Decision Tree             | Very high performance (~94.7% accuracy). Captures non-linear relationships well. High recall and F1 indicate good sensitivity, but may overfit on training data if depth or leaf constraints are not applied. |
| KNN                       | High performance (~93.8% accuracy). Sensitive to feature scaling; performs well here likely due to standardized features. Slightly lower precision indicates some misclassification of negatives as positives. |
| Naive Bayes               | Lowest performance (~72% accuracy). Assumes feature independence, which may not hold, leading to misclassifications. High recall suggests it predicts positives reasonably well, but precision is low, causing more false positives. |
| Random Forest (Ensemble)  | Best overall performance (~94.97% accuracy). Ensemble of multiple trees reduces overfitting, robust across metrics. Excellent balance of precision, recall, F1, and MCC; captures complex patterns effectively. |
| XGBoost (Ensemble)        | Strong performance (~93.6% accuracy). Gradient boosting handles complex patterns and is robust to outliers. Slightly lower than Random Forest but computationally efficient and provides good precision-recall balance. |





