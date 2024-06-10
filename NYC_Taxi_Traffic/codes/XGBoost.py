# Import necessary libraries
import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\data\\dataset.csv")
df = df.drop(df.columns[0], axis=1)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

df.loc[df['value'] > 35000, 'value'] = 20000

# Extract day, hour, weekday, and month from the timestamp
df['day'] = df.index.day
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['month'] = df.index.month

# Convert month and weekday to names
df["month"] = df["month"].apply(lambda x: calendar.month_name[x])
df['weekday'] = df['weekday'].apply(lambda x: calendar.day_name[x])
print(df.head())

values = df['value'].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values.reshape(-1, 1))

# Split data into training and test sets
train_size = int(len(scaled_values) * 0.8)
train_data = scaled_values[:train_size]
test_data = scaled_values[train_size:]

# Function to create sequences of data for XGBoost model
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Create sequences
sequence_length = 30  
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape data for XGBoost
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Build and train the XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["rmse"], eval_set=eval_set, verbose=True)

# Predict using the test data
predicted_values = model.predict(X_test)

# Inverse transform the scaled data
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
predicted_values_original = scaler.inverse_transform(predicted_values.reshape(-1, 1)).ravel()

# Calculate metrics
mae = mean_absolute_error(y_test_original, predicted_values_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predicted_values_original))
r2 = r2_score(y_test_original, predicted_values_original)

print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)

# Plot accuracy and loss
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

# Plot RMSE
plt.figure(figsize=(14, 7))
plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
plt.plot(x_axis, results['validation_1']['rmse'], label='Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\XGBoost_RMSE.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()
