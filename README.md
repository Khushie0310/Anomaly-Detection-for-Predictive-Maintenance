# Anomaly Detection for Predictive Maintenance

## Project Overview
### Problem Statement
This project focuses on anomaly detection in time-series sensor data from industrial machinery. The goal is to develop a machine learning model that can identify abnormal sensor readings, which may indicate potential equipment failures. Early detection of these anomalies allows for predictive maintenance, preventing costly breakdowns and improving operational efficiency.

### Dataset
The dataset contains sensor readings from various machines. The target column indicates whether an anomaly was detected.

### Objective
Build a machine learning model that can predict anomalies with over 75% accuracy.

## Data Preprocessing
We handled missing values using the mean imputation and encoded categorical variables using one-hot encoding.

## Model Selection
We trained several models including Random Forest and XGBoost. After tuning the hyperparameters, Random Forest was selected as the best model.

## Evaluation Metrics
The final model achieved an accuracy of 82%. Precision, recall, and F1-score were also evaluated.

## Conclusion
The model is ready to be deployed for real-time anomaly detection in production.

## Future Work
Future improvements include adding more features and retraining the model with updated data.
