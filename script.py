# pip install pandas numpy scikit-learn mlflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Load the data
data = pd.read_csv('accumulated_data.csv')

# Define features (X) and target (y)
X = data[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]
y = data['dt']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.09, random_state=42)

# Create and train a machine learning model (Linear Regression in this example)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics to MLflow
with mlflow.start_run():
    mlflow.log_param('model', 'Linear Regression')
    mlflow.log_param('data_path', 'csv_output/coordinate_2.csv')
    mlflow.log_metric('Mean Squared Error', mse)
    mlflow.log_metric('Mean Absolute Error', mae)
    mlflow.log_metric('R-squared', r2)

    # Log the model
    mlflow.sklearn.log_model(model, 'model')

print("Model trained and logged to MLflow.")
