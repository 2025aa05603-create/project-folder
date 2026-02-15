**a) Problem Statement**

Heart disease is one of the leading causes of death worldwide. Early detection of heart disease can significantly improve patient survival rates.
The objective of this project is to build and compare multiple machine learning classification models to predict whether a patient has heart disease based on clinical features.
The models are evaluated using multiple performance metrics, and the best-performing model is deployed using Streamlit.


**b) Dataset Description**

The dataset used is a merged heart disease dataset containing patient medical attributes.
number of features: 13 
number of instances used: >500
Dataset contains both categorical and numerical features, which were label-encoded for machine learning models.
Missing values represented by "?" were replaced with the median for numerical features.
Data was split into training (70%) and testing (30%) sets.
Feature scaling (StandardScaler) was applied for models sensitive to feature magnitudes (Logistic Regression, KNN, Naive Bayes,Decision Tree,Random Forest (Ensemble),XGBoost (Ensemble)).
This dataset combines multiple sources of heart disease data, providing a comprehensive set of risk factors for model training.


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
| Logistic Regression  | 0.7359  | 0.8100  | 0.7337    | 0.7492 | 0.7414   | 0.4717  |
| Decision Tree       | 0.8534   | 0.9109  | 0.8446    | 0.8701  | 0.8571  | 0.7071  |
| KNN                 | 0.9802   | 0.9917  | 0.9732    | 0.9879  | 0.9805  | 0.9604  |
| Naive Bayes         | 0.7359  | 0.8020  | 0.7147    | 0.7946  | 0.7525   | 0.4741   |
| Random Forest (Ensemble) | 0.9756 | 0.9980 | 0.9787    | 0.9728  | 0.9758   | 0.9512   |
| XGBoost (Ensemble)  | 0.9802   | 0.9976  | 0.9760    | 0.9849  | 0.9805   | 0.9603   |

**Model Performance Observations**

| ML Model Name             | Observation ( model performance  )                                                                                      |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression       | Moderate performance; decent accuracy (0.7359) and F1 Score (0.7414), but lower MCC (0.4717) indicates limited correlation between predictions and true labels. Good baseline model but underperforms compared to ensemble and KNN models. |
| Decision Tree             | Strong performance with high accuracy (0.8534) and F1 Score (0.8571); MCC (0.7071) shows good predictive power. May overfit on complex data, but interpretable and fast. |
| KNN                       |Excellent performance with very high accuracy (0.9802), F1 Score (0.9805), and MCC (0.9604). Sensitive to feature scaling and data size; may be slower for large datasets.|
| Naive Bayes               |Similar to Logistic Regression in accuracy (0.7359); precision lower (0.7147) but recall higher (0.7946), indicating it predicts positives reasonably well but misclassifies some negatives. Good for baseline probabilistic modelling.|
| Random Forest (Ensemble)  |Very strong performance; high accuracy (0.9756), F1 Score (0.9758), and MCC (0.9512). Robust to overfitting and handles complex interactions well; slightly lower recall than XGBoost but overall reliable.|
| XGBoost (Ensemble)        | This model outperformed performance; highest F1 Score (0.9805) and strong MCC (0.9603). Excels in handling complex data, missing values, and feature interactions. Slightly higher recall (0.9849) than Random Forest, making it better at capturing positives. |


**Conclusion:**
ensemble models, particularly XGBoost, are the most reliable and effective choice for this prediction task, offering a balance of accuracy, robustness, and generalization. Simpler models can still be valuable for quick insights or interpretability but are less competitive in predictive performance.


