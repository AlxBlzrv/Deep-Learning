# Fruit Classification Project

This project aims to classify various fruits and vegetables using convolutional neural networks (CNNs). It includes Python scripts for data loading, preprocessing, model creation, training, evaluation, and visualization.

## Contents

1. [About the Dataset](#about-the-dataset)
2. [Code](#code)
3. [Plots](#plots)
4. [Acknowledgements](#acknowledgements)

## About the Dataset

The dataset comprises a total of 22,495 images of fruits and vegetables, with the following details:

- Training set size: 16,854 images (one fruit or vegetable per image).
- Test set size: 5,641 images (one fruit or vegetable per image).
- Number of classes: 33 (fruits and vegetables).
- Image size: 100x100 pixels.
- Training data filename format: [fruit/vegetable name]_[id].jpg (e.g., Apple Braeburn_100.jpg).
- Testing data filename format: [4 digit id].jpg (e.g., 0001.jpg)

## Code

The code includes Python scripts for data loading, preprocessing, model creation, training, and evaluation. Key functionalities include:

- Loading data from directories (`load_data` function)
- Preprocessing images (`preprocess_images` function)
- One-hot encoding labels (`one_hot_encode` function)
- Creating a CNN model (`create_cnn_model` function)
- Compiling and training the model (`compile_and_train_model` function)
- Inverse one-hot encoding of labels (`inverse_one_hot_encode` function)

## Plots

The project generates several plots for visualization purposes:

- Countplot: Distribution of the number of images by category (`countplot.png`)
- Accuracy Plot: Training history showing accuracy over epochs (`accuracy.png`)
- Confusion Matrix: Evaluation of the model on the validation set (`confusion_matrix.png`)

## Acknowledgements

This project utilizes the TJ NMLO Public Dataset for fruit and vegetable images. Special thanks to the dataset creators for providing valuable resources for research and development in the field of deep learning and computer vision.
