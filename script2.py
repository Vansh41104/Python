import os
import pandas as pd
from statsmodels.tsa.api import VAR
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Step 1: Create a list of file paths for your CSV files
csv_folder = 'csv_output'
csv_files = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

# Step 2: Initialize an empty DataFrame for data accumulation
all_data = pd.DataFrame()

# Step 3: Loop through files and append data
for file in tqdm(csv_files, desc='Processing Files'):
    data = pd.read_csv(file)
    data['dt'] = pd.to_datetime(data['dt'])
    data.set_index('dt', inplace=True)
    
    # Append data to the accumulation DataFrame
    all_data = all_data._append(data)

# Step 4: Train the VAR model using the accumulated data
model = VAR(all_data)
model_fit = model.fit()

# Example: Make forecasts for the next 5 steps
forecasted_values = model_fit.forecast(all_data.values[-1:], steps=5)

# Example: Calculate evaluation metrics (MSE and MAE)
# You can adjust this based on your testing data
test_data = all_data[-5:]
mse = mean_squared_error(test_data, forecasted_values)
mae = mean_absolute_error(test_data, forecasted_values)

# Example: Log the trained model and metrics to MLflow
with mlflow.start_run():
    mlflow.log_param('model', 'VAR')
    mlflow.log_metric('Mean Squared Error', mse)
    mlflow.log_metric('Mean Absolute Error', mae)
    
    # Log the model using MLflow
    mlflow.sklearn.log_model(model_fit, "var_model")

print("Model trained and logged to MLflow.")
