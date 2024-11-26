# Automated Anomaly Detection for Predictive Maintenance

This project focuses on detecting anomalies in sensor data collected from industrial machinery, aiming to predict maintenance needs before failures occur. The solution leverages machine learning algorithms to classify sensor data as either "normal" or "anomalous," helping to prevent unplanned downtime.

The primary goal of this project is to develop a robust, automated system that identifies anomalies early, thus enhancing predictive maintenance processes.

## Problem Statement

In industrial settings, equipment failure often leads to unexpected downtime, affecting production efficiency. Predicting and identifying anomalies in the machinery sensor data can help prevent such failures. The objective of this project is to build an anomaly detection model that classifies sensor data as either "normal" or "anomalous," to help maintain machinery before a failure occurs.

## Dataset Description

. The dataset includes the following columns:

- **sensor1, sensor2, sensor3**: Continuous numerical values representing measurements from the machinery.
- **y**: The target variable indicating whether the observation is "normal" (0) or "anomalous" (1).

### Data Source:

- The raw data file (`anoma_data.csv`) is located in the `data/` folder of this repository. You can download it from there.

### Data Preprocessing:
- The raw data had missing values, which were handled by removing rows with missing critical data or imputing with mean values.
- Non-numeric columns (e.g., dates) were either dropped or converted into numeric format.
- The final dataset used for modeling was cleaned and transformed.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP
- **Jupyter Notebook**: For code execution and visualization
- **Modeling Techniques**: Logistic Regression, Random Forest, XGBoost
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve

## Modeling Techniques

- **Logistic Regression**: A simple linear model used for classification.
- **Random Forest**: A powerful ensemble learning method that improves performance by aggregating predictions from multiple decision trees.
- **XGBoost**: A gradient boosting algorithm known for its high accuracy and speed in training.

The models were evaluated on their ability to predict anomalies based on the sensor data.

## Evaluation Metrics

The performance of the models was evaluated using the following metrics:
- **Accuracy**: The proportion of correctly predicted labels.
- **Precision**: The ratio of true positive predictions to all positive predictions.
- **Recall**: The ratio of true positive predictions to all actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: To visualize the performance of classification models.
- **ROC Curve**: To evaluate the true positive rate versus the false positive rate.

## Conclusion

The XGBoost model outperformed Logistic Regression and Random Forest models in detecting anomalies with a higher precision and recall. The model showed great promise in accurately identifying anomalies, which is crucial for predictive maintenance in industrial settings.

## Future Work

- **Real-time Data**: Integrate real-time sensor data for continuous anomaly detection.
- **Model Optimization**: Further tuning the model’s hyperparameters to improve performance.
- **Deployment**: Building an API using Flask or FastAPI to deploy the model for real-time predictions in production environments.
