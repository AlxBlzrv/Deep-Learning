# Road Sings Image Classification Project

This project focuses on classifying road signs using deep learning techniques. It utilizes a convolutional neural network (CNN) to classify images of road signs into different categories. The dataset consists of images of road signs stored in the `images` directory and their corresponding annotations in the `annotations.csv` file.

## Project Structure

- **images**: This directory contains images of road signs used for training and testing the model.
- **annotations.csv**: This CSV file contains annotations for the images, specifying their filenames and corresponding categories.
- **pretrained_model**: This directory contains the pre-trained MobileNetV2 model used as the base for the CNN. The trained model (`cnn.h5`) is also saved here.
- **category_plot.png**: This image file shows the distribution of different categories of road signs in the dataset.

## Usage

1. **Data Preparation**:
   - Ensure that the images are stored in the `images` directory and their annotations are provided in the `annotations.csv` file.
   
2. **Training the Model**:
   - Run the provided Python script to train the CNN model using the images and annotations.
   
3. **Evaluation**:
   - Evaluate the trained model's performance using the provided evaluation metrics such as accuracy, loss, and the confusion matrix.
   
4. **Visualization**:
   - Visualize the distribution of different categories of road signs using the `category_plot.png` file.
   - Plot training and validation metrics to analyze the model's performance during training.

## Requirements

- Python 3.x
- TensorFlow
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Acknowledgments

This project utilizes the MobileNetV2 architecture and TensorFlow library for deep learning tasks. The dataset is originating from Chinese Traffic Sign Recognition Database (Kaggle). It has been explored by Riga Data Science Club members in order to do some training on convolution neural networks.

## License

This project is licensed under the [MIT License](LICENSE).
