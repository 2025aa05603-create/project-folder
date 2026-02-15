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
| Logistic Regression  | 0.7563   | 0.8154  | 0.6810    | 0.8778  | 0.7670   | 0.5386  |
| Decision Tree       | 0.7563   | 0.8330  | 0.6667    | 0.9333  | 0.7778   | 0.5611   |
| KNN                 | 0.7766   | 0.8307  | 0.7300    | 0.8111  | 0.7684   | 0.5568   |
| Naive Bayes         | 0.7411   | 0.7934  | 0.6789    | 0.8222  | 0.7437   | 0.4961   |
| Random Forest (Ensemble) | 0.8528 | 0.9283 | 0.8280    | 0.8556  | 0.8415   | 0.7045   |
| XGBoost (Ensemble)  | 0.8731   | 0.9248  | 0.8495    | 0.8778  | 0.8634   | 0.7453   |

**Model Performance Observsations**

| ML Model Name             | Observation ( model performance  )                                                                                      |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression       | Shows balanced performance with good Recall (0.8778), meaning it identifies most heart disease cases correctly. However, moderate MCC (0.5386) indicates limited overall correlation compared to ensemble models. Suitable as a strong baseline linear model. |
| Decision Tree             | Achieves very high Recall (0.9333), making it good for minimizing false negatives. However, slightly lower Precision suggests more false positives. Performance is decent but prone to overfitting compared to ensemble methods. |
| KNN                       | Provides balanced Accuracy (0.7766) with stable Precision and Recall. Performance is consistent but does not outperform ensemble methods. Sensitive to scaling and data distribution. |
| Naive Bayes               |Lowest overall performance among all models. Lower AUC (0.7934) and MCC (0.4961) indicate weaker predictive power. Assumption of feature independence likely limits performance on this dataset. Serves mainly as a baseline probabilistic model. |
| Random Forest (Ensemble)  |Strong performance across all metrics with high AUC (0.9283) and MCC (0.7045). Demonstrates good balance between Precision and Recall. Robust and less prone to overfitting due to ensemble averaging. |
| XGBoost (Ensemble)        | Best performing model overall. Highest Accuracy (0.8731), F1 Score (0.8634), and MCC (0.7453). Excellent balance between Precision and Recall. Strong generalization capability due to gradient boosting optimization. |





