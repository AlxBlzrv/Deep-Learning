import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Set directories for data and annotations
data_dir = "D:\\Dell\\repos\\Deep-Learning\\Road_Sings\\images"
annotations_file = "D:\\Dell\\repos\\Deep-Learning\\Road_Sings\\annotations.csv"

# Load annotations from CSV file
annotations = pd.read_csv(annotations_file)
print(annotations.head())

# Define image dimensions
img_height, img_width = 224, 224

# Function to load and preprocess an image
def load_and_preprocess_image(file_path):
    img = load_img(file_path, target_size=(img_height, img_width))  # Load image with target size
    img_array = img_to_array(img)  # Convert image to array
    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNetV2
    return img_array

# Initialize lists for images and labels
images = []
labels = annotations['category'].values

# Plot category distribution
plt.figure(figsize=(20,20))
annotations['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution')
plt.savefig("D:\\Dell\\repos\\Deep-Learning\\Road_Sings\\category_plot.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Load and preprocess images
for file_name in annotations['file_name']:
    img_path = os.path.join(data_dir, file_name)
    img_array = load_and_preprocess_image(img_path)
    images.append(img_array)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load MobileNetV2 model without the top layer and with ImageNet weights
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer
x = Dense(1024, activation='relu')(x)  # Add a dense layer with ReLU activation
predictions = Dense(len(np.unique(labels)), activation='softmax')(x)  # Add a dense layer for predictions

# Define the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

# Function to plot training and validation metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Plot the training and validation metrics
plot_metrics(history)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
save_model(model, "D:\\Dell\\repos\\Deep-Learning\\Road_Sings\\cnn.h5", include_optimizer=False)
