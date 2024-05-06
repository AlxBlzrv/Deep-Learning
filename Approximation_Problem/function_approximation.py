import numpy as np
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Function to generate data based on different mathematical functions
def generate_data(function_name, x_min, x_max, num_points):
    # Generate data based on the selected function
    if function_name.lower() == "linear function":
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = 2 * x_values + 3 
    elif function_name.lower() == "quadratic function":
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = x_values ** 2
    elif function_name.lower() == "polynomial function":
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = x_values ** 3 - 2 * x_values + 5 
    elif function_name.lower() == "exponential function":
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = np.exp(x_values)
    elif function_name.lower() == "sigmoid function":
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = 1 / (1 + np.exp(-x_values))
    elif function_name.lower() == "cosine function":
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = np.cos(x_values)
    else:
        raise ValueError("Function is not supported")
    return x_values, y_values

# Function to plot the generated data
def plot_data(function_name, x_train, y_train, x_test, y_test):
    plt.figure(figsize=(10, 6))
    plt.title(function_name)
    # Plot train data
    plt.scatter(x_train, y_train, color='blue', label='Train Data', s=4, marker='o')
    # Plot test data
    plt.scatter(x_test, y_test, color='red', label='Test Data', s=6, marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig('D:\\Dell\\repos\\Deep-Learning\\Approximation_Problem\\plots\\function.png', 
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

# Function to build and train the neural network model
def build_and_train_model(function_name, x_train, y_train, epochs=100, batch_size=32):
    # Selecting activation function based on the function type
    activation_func = 'linear' if function_name.lower() == "linear function" else 'relu'

    # Building model architecture based on the function type
    if function_name.lower() in ["polynomial function", "exponential function"]:
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(1,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1) 
        ])
    elif function_name.lower() == "sigmoid":
        model = models.Sequential([
            layers.Dense(64, activation='sigmoid', input_shape=(1,)),
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(1) 
        ])
    elif function_name.lower() == "cosine function":
        model = models.Sequential([
            layers.Dense(128, activation='tanh', input_shape=(1,)),
            layers.Dense(256, activation='tanh'),
            layers.Dense(128, activation='tanh'),
            layers.Dense(64, activation='tanh'),
            layers.Dense(1)
        ])
    else:
        model = models.Sequential([
            layers.Dense(64, activation=activation_func, input_shape=(1,)),
            layers.Dense(64, activation=activation_func),
            layers.Dense(1) 
        ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model, history

# Function to evaluate the trained model
def evaluate_model(model, x_test, y_test):
    # Predicting values using the trained model
    predicted_y_values = model.predict(x_test)
    # Calculating R-squared score
    r2 = r2_score(y_test, predicted_y_values)
    return r2

def main():
    # Getting input from the user
    function_name = input("Enter the name of the function: ")
    x_min = float(input("Enter the minimum value of x: "))
    x_max = float(input("Enter the maximum value of x: "))
    num_points = int(input("Enter the number of points: "))
    
    x_values, y_values = generate_data(function_name, x_min, x_max, num_points)

    print("x_values:", x_values[:5])
    print("y_values:", y_values[:5])

    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=42)

    plot_data(function_name, x_train, y_train, x_test, y_test)

    model, history = build_and_train_model(function_name, x_train, y_train)

    # Plotting training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('D:\\Dell\\repos\\Deep-Learning\\Approximation_Problem\\plots\\loss.png', 
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    r2 = evaluate_model(model, x_test, y_test)
    print("Coefficient of Determination (R²) on test set:", r2)

if __name__ == "__main__":
    main()
