# Anomaly Detection for Predictive Maintenance

## Project Overview
### Problem Statement
The goal of this project is to detect anomalies in sensor data to predict potential failures in machines.

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
