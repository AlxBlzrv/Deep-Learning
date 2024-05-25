import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset, parse the dates, and sort by sale date
data = pd.read_csv('D:\\Dell\\repos\\Deep-Learning\\House_Sales\\data\\ma_lga_12345.csv', parse_dates=['saledate'], dayfirst=True)
data = data.sort_values('saledate')

print(data.head())

# Function to create sequences of data for time series prediction
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append((seq, label))
    return sequences

# Function to build and train the model
def build_and_train_model(X_train, y_train, seq_length, epochs=120, batch_size=64):
    model = Sequential()
    
    # Add a Conv1D layer followed by MaxPooling
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, 1)))
    model.add(MaxPooling1D(pool_size=2))
    
    # Add LSTM layers with Dropout for regularization
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    
    # Add a Dense layer to produce the final output
    model.add(Dense(1))
    
    # Compile the model with RMSprop optimizer and mean squared error loss
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Train the model and return the history
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    
    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)
    r2 = r2_score(y_test, predictions)
    return r2, predictions, y_test

# Set the sequence length
seq_length = 8
property_types = data['type'].unique()
bedroom_counts = data['bedrooms'].unique()

models = {}
results = {}

# Train and evaluate models for each property type and bedroom count combination
for p_type in property_types:
    for bedrooms in bedroom_counts:
        subset = data[(data['type'] == p_type) & (data['bedrooms'] == bedrooms)]
        
        if len(subset) > seq_length:
            print(f'Training model for {p_type} with {bedrooms} bedrooms...')
            
            # Scale the data
            scaler = MinMaxScaler()
            subset[['MA']] = scaler.fit_transform(subset[['MA']])
            
            # Create sequences from the data
            sequences = create_sequences(subset[['MA']].values, seq_length)
            X = np.array([seq[0] for seq in sequences])
            y = np.array([seq[1] for seq in sequences])
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, random_state=42)
            
            # Build and train the model
            model, history = build_and_train_model(X_train, y_train, seq_length)
            
            # Evaluate the model
            r2, predictions, y_test = evaluate_model(model, X_test, y_test, scaler)
            print(f'R^2 Score for {p_type} with {bedrooms} bedrooms: {r2}')
            
            models[(p_type, bedrooms)] = model
            results[(p_type, bedrooms)] = (history, predictions, y_test)

# Plot and save the training and validation loss for each model
for (p_type, bedrooms), (history, _, _) in results.items():
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss for {p_type} with {bedrooms} bedrooms')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(f'D:\\Dell\\repos\\Deep-Learning\\House_Sales\\plots\\loss\\{p_type}_{bedrooms}_loss.png')
    plt.show()

# Plot and save the actual vs predicted prices for each model
for (p_type, bedrooms), (_, predictions, y_test) in results.items():
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f'Actual vs Predicted Prices for {p_type} with {bedrooms} bedrooms')
    plt.ylabel('Price')
    plt.xlabel('Time Step')
    plt.legend(loc='upper right')
    plt.savefig(f'D:\\Dell\\repos\\Deep-Learning\\House_Sales\\plots\\actual_vs_predicted_prices\\{p_type}_{bedrooms}_predictions.png')
    plt.show()
