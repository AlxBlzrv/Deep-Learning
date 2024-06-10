# Import necessary libraries
import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\data\\dataset.csv")
print(df.head())

# Drop the first unnamed column
df = df.drop(df.columns[0], axis=1)
print(df.head())

# Display dataset info
print(df.info())

# Convert the 'timestamp' column to datetime and set it as the index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
print(df.head())

# Check for missing values
print(df.isna().sum())

# Plot the entire time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], label='Passenger Count')
plt.title('NYC Taxi Passenger Count')
plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.legend()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\TS.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Identify and plot specific anomalies

# New York City Marathon (2014-11-02)
plt.figure(figsize=(10, 4))
df.loc["2014-10-30":"2014-11-03"].plot()
plt.title("NYC Marathon Anomaly (2014-11-02)")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend(["Passenger Count"])
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\NYC_Marathon_Anomaly.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Thanksgiving (2014-11-27)
plt.figure(figsize=(10, 4))
df.loc["2014-11-25":"2014-11-30"].plot()
plt.title("Thanksgiving Anomaly (2014-11-27)")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend(["Passenger Count"])
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\Thanksgiving_Anomaly.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Christmas (2014-12-25)
plt.figure(figsize=(10, 4))
df.loc["2014-12-20":"2014-12-30"].plot()
plt.title("Christmas Anomaly (2014-12-25)")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend(["Passenger Count"])
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\Christmas_Anomaly.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# New Year (2014-12-31 to 2015-01-01)
plt.figure(figsize=(10, 4))
df.loc["2014-12-29":"2015-01-03"].plot()
plt.title("New Year Anomaly (2014-12-31 to 2015-01-01)")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend(["Passenger Count"])
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\New_Year_Anomaly.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Snow storm (Insignificant, but plotted)
plt.figure(figsize=(10, 4))
df.loc["2014-11-15":"2015-01-31"].plot()
plt.title("Snow Storm Period")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend(["Passenger Count"])
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\Snow_Storm_Period.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Replace anomalies
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

# Resample data by day and plot
df_day = df['value'].resample('D').mean()
plt.figure(figsize=(12, 6))
df_day.plot(linewidth=1, color='blue')
plt.xlabel("Date")
plt.ylabel("Taxi Rides")
plt.title("Rides by Day")
plt.grid(linestyle='--', alpha=0.2)
plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\resample_TS.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot day rides by month
plt.figure(figsize=(12, 6))
sns.lineplot(x="day", y="value", data=df, hue="month", palette='bright')
plt.xlabel("Day")
plt.ylabel("Number of Rides")
plt.title("Day Rides by Month")
plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\Day_Rides_by_Month.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot hour rides by weekday
plt.figure(figsize=(12, 6))
sns.lineplot(x="hour", y="value", data=df, hue="weekday", palette='bright')
plt.xlabel("Hour")
plt.ylabel("Number of Rides")
plt.title("Hour Rides by Weekday")
plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\Hour_Rides_by_Weekday.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Decompose the time series and plot its components
df_numeric = df.select_dtypes(include=[np.number])
df_resampled = df_numeric.resample('D').mean()
decomposed = seasonal_decompose(df_resampled["value"], model='additive')

# Access individual components
trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

# Plot decomposed components
plt.figure(figsize=(12, 10))

plt.subplot(411)
plt.plot(df_resampled["value"], label='Original')
plt.legend()

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend()

plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend()

plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend()

plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\decomposed_TS.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot the seasonally adjusted trend
plt.figure(figsize=(10, 5))
plt.plot(trend, color='red', label='Trend')
plt.plot(df_resampled.index, df_resampled['value'], color='gray')
plt.ylabel('Number of Passengers')
plt.xlabel('Date')
plt.title('Trend')
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\Trend.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot the differenced series to check for stationarity
plt.figure(figsize=(12, 6))
df_resampled['value'].diff(30).plot()
plt.title('Differenced Time Series (30 Lags)')
plt.xlabel('Date')
plt.ylabel('Differenced Value')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\Differenced_TS.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Perform Dickey-Fuller test to check stationarity
result = adfuller(df_resampled['value'].diff(30)[30:])
print('Test Statistic:', result[0])
print('p-value:', result[1])
# The series is stationary based on the Dickey-Fuller test result

# Plot Autocorrelation Function (ACF)
plt.figure(figsize=(12, 6))
plot_acf(df_resampled['value'], lags=30)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\ACF.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot Partial Autocorrelation Function (PACF)
plt.figure(figsize=(12, 6))
plot_pacf(df_resampled['value'], lags=30)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\PACF.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Scaling the data for LSTM model
values = df['value'].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values.reshape(-1, 1))

# Split data into training and test sets
train_size = int(len(scaled_values) * 0.8)
train_data = scaled_values[:train_size]
test_data = scaled_values[train_size:]

# Function to create sequences of data for LSTM model
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

# Build and compile the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print('Mean Squared Error:', mse)

# Predict using the test data
predicted_values = model.predict(X_test)

# Inverse transform the scaled data
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
predicted_values_original = scaler.inverse_transform(predicted_values).ravel()

# Calculate other metrics
mae = mean_absolute_error(y_test_original, predicted_values_original)
rmse = np.sqrt(mean_squared_error(y_test_original, predicted_values_original))

print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(14, 6))
plt.plot(y_test_original, label='Actual')
plt.plot(predicted_values_original, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.tight_layout()
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\NYC_Taxi_Traffic\\plots\\Actual_vs_Predicted.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Calculate R^2 score
r2 = r2_score(y_test_original, predicted_values_original)
print('R^2 Score:', r2)

# Print the first 10 actual and predicted values for inspection
for i in range(10):
    print(f'Predicted: {predicted_values_original[i]}, Actual: {y_test_original[i]}')
