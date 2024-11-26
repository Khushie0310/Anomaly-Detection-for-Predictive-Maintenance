# Anomaly Detection in Time-Series Data

## Project Overview
This project aims to detect anomalies in time-series sensor data from machinery. The anomaly detection is performed using machine learning models like **XGBoost** and **Logistic Regression**.

## Dataset
The dataset consists of time-series sensor readings with multiple features and a binary target variable indicating whether the data point is an anomaly (1) or not (0).

### Dataset Columns:
- `time`: Timestamp of the observation.
- `x1`, `x2`, ..., `x60`: Sensor readings at different time intervals.
- `y`: Target variable (1 for anomaly, 0 for normal).

## Preprocessing
The dataset was cleaned and prepared for modeling with the following steps:
1. **Handling Missing Values**.
2. **Datetime Conversion**.
3. **Feature Selection**.

## Methodology
1. **Exploratory Data Analysis (EDA)**.
2. **Modeling** with **Logistic Regression** and **XGBoost**.
3. **Model Evaluation** using metrics like accuracy, precision, recall, F1-score.
4. **Feature Importance** and **SHAP** Interpretability.

## Results
- **XGBoost** performed better than **Logistic Regression** in terms of **F1-score** and **Recall**.

## Conclusion
- **XGBoost** was the final model chosen for anomaly detection.
- **SHAP** provided insights into feature contributions.

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Anomaly-Detection-Project.git
