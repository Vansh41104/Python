import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

input_csv_path = 'accumulated_data_final.csv'
df = pd.read_csv(input_csv_path)
df.drop(columns=['dt'], inplace=True)
df.drop(columns=['main/aqi'], inplace=True)
df.drop(columns=['Unnamed: 9'], inplace=True)


X = df.drop(columns=['components/pm2_5'])
y = df['components/pm2_5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

average_predicted_value = y_pred.mean()

print(f"Average Predicted Value: {average_predicted_value:.2f}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
accuracy_percentage = r2 * 100
print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")

# Visualize the predictions vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
