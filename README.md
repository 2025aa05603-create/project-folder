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

**Model Comparison Table:**

| ML Model Name       | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC      |
|---------------------|----------|---------|-----------|---------|----------|----------|
| Logistic Regression  | 0.7277   | 0.7276  | 0.7244    | 0.7409  | 0.7326   | 0.4553  |
| Decision Tree       | 0.9474   | 0.9473  | 0.9378    | 0.9591  | 0.9483   | 0.8950   |
| KNN                 | 0.9382   | 0.9380  | 0.9142    | 0.9682  | 0.9404   | 0.8779   |
| Naive Bayes         | 0.7208   | 0.7203  | 0.6929    | 0.8000  | 0.7426   | 0.4465   |
| Random Forest (Ensemble) | 0.9497 | 0.9497 | 0.9500    | 0.9500  | 0.9500   | 0.8993   |
| XGBoost (Ensemble)  | 0.9359   | 0.9360  | 0.9444    | 0.9273  | 0.9358   | 0.8720   |

**Model Performance Observsations**

| ML Model Name             | Observation ( model performance  )                                                                                      |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression       | Moderate performance (~73% accuracy). Works reasonably well for linearly separable data but struggles with complex patterns. Precision and recall are balanced, but overall lower than tree-based models, indicating possible underfitting. |
| Decision Tree             | Very high performance (~94.7% accuracy). Captures non-linear relationships well. High recall and F1 indicate good sensitivity, but may overfit on training data if depth or leaf constraints are not applied. |
| KNN                       | High performance (~93.8% accuracy). Sensitive to feature scaling; performs well here likely due to standardized features. Slightly lower precision indicates some misclassification of negatives as positives. |
| Naive Bayes               | Lowest performance (~72% accuracy). Assumes feature independence, which may not hold, leading to misclassifications. High recall suggests it predicts positives reasonably well, but precision is low, causing more false positives. |
| Random Forest (Ensemble)  | Best overall performance (~94.97% accuracy). Ensemble of multiple trees reduces overfitting, robust across metrics. Excellent balance of precision, recall, F1, and MCC; captures complex patterns effectively. |
| XGBoost (Ensemble)        | Strong performance (~93.6% accuracy). Gradient boosting handles complex patterns and is robust to outliers. Slightly lower than Random Forest but computationally efficient and provides good precision-recall balance. |





