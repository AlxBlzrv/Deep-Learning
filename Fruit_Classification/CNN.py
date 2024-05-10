import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import layers, models, callbacks # type: ignore
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Function to load data from directories
def load_data(data_directory, is_train=True):
    images = []
    labels = []

    if is_train:
        classes = os.listdir(data_directory)
        for class_name in classes:
            class_directory = os.path.join(data_directory, class_name)
            for filename in os.listdir(class_directory):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(class_directory, filename)
                    with Image.open(image_path) as image:
                        images.append(np.array(image))
                    labels.append(class_name)
    else:
        test_image_filenames = os.listdir(data_directory)
        for filename in test_image_filenames:
            if filename.endswith('.jpg'):
                image_path = os.path.join(data_directory, filename)
                with Image.open(image_path) as image:
                    images.append(np.array(image))

    return images, labels

# Function to preprocess images
def preprocess_images(images):
    images_np = np.array(images)
    images_normalized = images_np / 255.0
    return images_normalized

# Function to perform one-hot encoding on labels
def one_hot_encode(labels):
    label_binarizer = LabelBinarizer()
    one_hot_encoded_labels = label_binarizer.fit_transform(labels)
    return one_hot_encoded_labels, label_binarizer

# Function to create a CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Function to compile and train the model
def compile_and_train_model(model, train_generator, validation_data):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Define callbacks for early stopping and reducing learning rate
    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6)

    # Train the model
    history = model.fit(train_generator, epochs=5, validation_data=validation_data,
                        callbacks=[early_stopping, reduce_lr])

    return history

# Function to inverse one-hot encode labels
def inverse_one_hot_encode(labels, label_binarizer):
    label_classes = label_binarizer.classes_
    inverse_labels = [label_classes[label.argmax()] for label in labels]
    return inverse_labels

# Directories for train and test data
train_data_directory = "D:\\Dell\\repos\\Deep-Learning\\Fruit_Classification\\data\\train\\train"
test_data_directory = "D:\\Dell\\repos\\Deep-Learning\\Fruit_Classification\\data\\test\\test"

# Load train and test data
train_images, train_labels = load_data(train_data_directory, is_train=True)
test_images, _ = load_data(test_data_directory, is_train=False)

# Print unique classes in train labels
print(set(train_labels))

# Visualize distribution of images per category
data = pd.DataFrame({'Categories': train_labels})
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Categories', palette='viridis')
plt.xlabel('Categories')
plt.ylabel('Number of images')
plt.title('Distribution of the number of images by category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('D:\\Dell\\repos\\Deep-Learning\\Fruit_Classification\\plots\\countplot.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Preprocess train and test images
train_images_normalized = preprocess_images(train_images)
train_one_hot_encoded_labels, label_binarizer  = one_hot_encode(train_labels)
test_images_normalized = preprocess_images(test_images)

# Display sample test images
for i in range(3):
    plt.imshow(train_images_normalized[i])
    plt.title(f"Image {i+1}")
    plt.axis('off')
    plt.show()

# Define input shape and number of classes
input_shape = train_images_normalized[0].shape
num_classes = len(np.unique(train_labels))

# Split train data into train and validation sets
train_images_split, val_images_split, train_labels_split, val_labels_split = train_test_split(
    train_images_normalized, train_one_hot_encoded_labels, test_size=0.2, random_state=42)

# Data augmentation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_generator = train_datagen.flow(train_images_split, train_labels_split, batch_size=64)

# Create CNN model
model = create_cnn_model(input_shape, num_classes)
# Compile and train the model
history = compile_and_train_model(model, train_generator, validation_data=(val_images_split, val_labels_split))

# Predictions on test images
predictions = model.predict(test_images_normalized)
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels[:5])

# Inverse one-hot encode predicted labels
predicted_labels_words = inverse_one_hot_encode(predictions, label_binarizer)
print(predicted_labels_words[:10])

# Display sample test images with predictions
for i in range(5):
    plt.imshow(test_images_normalized[i])
    plt.title(f"Image {i+1}")
    plt.axis('off')
    plt.show()

# Plot training history (accuracy and loss)
history_df = pd.DataFrame(history.history)

history_df.plot()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training History')
plt.grid(True)
plt.savefig('D:\\Dell\\repos\\Deep-Learning\\Fruit_Classification\\plots\\accuracy.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Confusion matrix on validation set
val_predictions_prob = model.predict(val_images_split)
val_predictions = np.argmax(val_predictions_prob, axis=1)

cm = confusion_matrix(np.argmax(val_labels_split, axis=1), val_predictions)
cm_df = pd.DataFrame(cm, index = range(num_classes), columns = range(num_classes))

plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Oranges')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Validation Set')
plt.savefig('D:\\Dell\\repos\\Deep-Learning\\Fruit_Classification\\plots\\confusion_matrix.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

save_model(model, "D:\\Dell\\repos\\Deep-Learning\\Fruit_Classification\\cnn.h5", include_optimizer=False)
