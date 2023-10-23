import os
import pandas as pd
import mlflow
import mlflow.sklearn
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load your data for the selected variable
#data = pd.read_csv('csv_output\coordinate_1.csv')  # Replace 'your_selected_variable.csv' with the actual file path
csv_folder = 'csv_output'  # Replace with the folder containing your CSV files
csv_files = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

all_data = pd.DataFrame()

# Data preprocessing (e.g., converting 'dt' to datetime and setting it as an index)
for file in tqdm(csv_files, desc='Processing Files'):
    data = pd.read_csv(file)
    data['dt'] = pd.to_datetime(data['dt'])
    data.set_index('dt', inplace=True)
        
    #Append data to the accumulation DataFrame
    all_data = all_data._append(data)

    # Ensure the data is stationary (e.g., by differencing if needed)
    # Example:
    # data['pm2_5_diff'] = data['pm2_5'] - data['pm2_5'].shift(1)
    # data = data.dropna()

    # Train ARIMA model
    p, d, q = 0.1, 0.1, 0.1  # Example order (you need to determine this based on your data)
    model = ARIMA(data['pm2_5'], order=(p, d, q))
    model_fit = model.fit()

    # Example: Make forecasts for the next 5 steps
    forecasted_values = model_fit.predict(start=len(data), end=len(data) + 4, typ='levels')

    # Example: Calculate evaluation metrics (MSE and MAE)
    # You can adjust this based on your testing data
    test_data = data['pm2_5'][-5:]
    mse = mean_squared_error(test_data, forecasted_values)
    mae = mean_absolute_error(test_data, forecasted_values)
    r2 = r2_score(test_data, forecasted_values)



    # Plot the test data and forecasted data
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-5:], test_data, label='Test Data', marker='o')
    plt.plot(data.index[-5:], forecasted_values, label='Forecasted Data', marker='x')
    plt.title('Test Data vs. Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()



# Example: Log the trained model and metrics to MLflow
with mlflow.start_run():
    mlflow.log_param('model', 'ARIMA')
    mlflow.log_metric('Mean Squared Error', mse)
    mlflow.log_metric('Mean Absolute Error', mae)
    mlflow.log_metric('R2', r2)
    
    # Log the model using MLflow
    mlflow.sklearn.log_model(model_fit, "arima_model")

print("ARIMA Model trained and logged to MLflow.")
