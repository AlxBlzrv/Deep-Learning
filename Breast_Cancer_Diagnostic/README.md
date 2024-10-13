# Breast Cancer Diagnostic

This project uses machine learning techniques to diagnose breast cancer using the "Breast Cancer Wisconsin (Diagnostic)" dataset. The goal is to classify whether a breast tumor is benign or malignant based on digitized images of fine needle aspirates (FNA) of breast masses.

## Project Structure
- **`breast-cancer-diagnostic.ipynb`**: Jupyter notebook containing the machine learning models and analysis.
- **`cancer_data.csv`**: The dataset used for training and testing the models. It contains features extracted from FNA images of breast tumors.

## About the Dataset
The dataset consists of 569 samples, with 30 features computed from digitized images of FNA of breast masses. These features describe characteristics of the cell nuclei present in the images, and the goal is to predict whether a tumor is malignant (M) or benign (B).

### Dataset Attributes:
1. **ID number**: Unique identifier for each sample.
2. **Diagnosis**: The target variable (M = malignant, B = benign).
3. **Features (3-32)**: Ten real-valued features computed for each cell nucleus, listed below:
   - **Radius**: Mean distance from center to points on the perimeter.
   - **Texture**: Standard deviation of gray-scale values.
   - **Perimeter**
   - **Area**
   - **Smoothness**: Local variation in radius lengths.
   - **Compactness**: (Perimeter^2 / area - 1.0).
   - **Concavity**: Severity of concave portions of the contour.
   - **Concave Points**: Number of concave portions of the contour.
   - **Symmetry**
   - **Fractal Dimension**: Approximation of the cell boundary's "roughness."

For each feature, the **mean**, **standard error**, and **worst (largest)** values were computed, resulting in 30 features for each sample.

### Class Distribution:
- 357 benign cases.
- 212 malignant cases.

### No missing attribute values.

### Source:
This dataset was originally published in:
- [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

Available through the UCI Machine Learning Repository: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

## Models and Techniques Used
The project implements several machine learning algorithms and a neural network to classify breast cancer tumors, including:
- **Logistic Regression**
- **Gaussian Naive Bayes**
- **K-Nearest Neighbors (KNN)** with hyperparameter tuning
- **Decision Tree Classifier**
- **Support Vector Classifier (SVC)** with Grid Search for hyperparameter tuning
- **Neural Network** using TensorFlow and Keras

## Installation and Setup
To run the project locally:
1. Clone the repository or download the files.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook (`breast-cancer-diagnostic.ipynb`) to explore and run the models.

## How to Use
1. **Data Visualization**: Explore the distribution of the diagnosis in the dataset using Seaborn.
2. **Preprocessing**: Features are normalized, and labels are mapped to numerical values.
3. **Model Training**: Various machine learning models are trained on the dataset.
4. **Evaluation**: Each model is evaluated based on its accuracy in predicting malignant or benign tumors.

## Results
Each machine learning model is evaluated for its performance in classifying breast tumors. The best-performing models include:
- **K-Nearest Neighbors (KNN)**: Achieved an accuracy of ~94%.
- **Support Vector Classifier**: Achieved an accuracy of ~95%.
- **Neural Network**: Achieved an accuracy of ~97%.

## Conclusion
This project demonstrates the use of various machine learning techniques for diagnosing breast cancer, providing a robust solution to identify malignant and benign tumors based on image features.

