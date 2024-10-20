# Heart Disease Prediction

This project focuses on the analysis and prediction of heart disease using a dataset sourced from Kaggle. The objective is to build machine learning models to predict the presence of heart disease based on various health metrics and lifestyle factors.

## Dataset

The dataset used in this project is the [Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). It contains medical data for patients with and without heart disease and includes attributes such as:

- **Age**: Age of the patient
- **Sex**: Gender of the patient (1 = male, 0 = female)
- **ChestPainType**: Type of chest pain experienced (typical angina, atypical angina, non-anginal pain, asymptomatic)
- **RestingBP**: Resting blood pressure (in mm Hg)
- **Cholesterol**: Serum cholesterol level (in mg/dl)
- **FastingBS**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **RestingECG**: Resting electrocardiogram results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (1 = yes, 0 = no)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: The slope of the peak exercise ST segment
- **HeartDisease**: Target variable (1 = heart disease, 0 = no heart disease)

## Project Overview

The primary goal of this project is to develop a predictive model to accurately classify patients as either having heart disease or not, based on their medical and lifestyle attributes. The project involves several stages:

1. **Data Preprocessing**: Cleaning and preparing the dataset for machine learning models by handling missing values, encoding categorical variables, and normalizing the data where necessary, augmentation etc.
2. **Exploratory Data Analysis (EDA)**: Performing statistical analysis and visualizations to understand the data distribution and identify important features.
3. **Modeling**: Building, training, and testing various machine learning models, such as:
   - Decision Trees / Random Forest and Gradient Boosting Classifier (as actual ensemble, 98% acc),
   - Logistic Regression (resulting ensemble model),
   
   deep learning models, such as:
   - Neural Networks (97% acc)

   and other:
   - Support Vector Machines (SVM)
   - K-Nearest Neighbors (KNN)

*Note*: Only the highest accuracy models were retained in the project report.

4. **Evaluation**: Comparing models based on accuracy, precision, recall, F1 score, and ROC-AUC curve to determine the best-performing model.
5. **Deployment**: Optionally deploying the model using a web-based interface for real-time predictions (implement according to need).

## Prerequisites

To run this project locally, you will need the following software and packages installed:

- Python 3.x
- Jupyter Notebook or any Python IDE
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow (for neural networks)

You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Project Structure

- `uci-heart-disease.ipynb`: Jupyter notebook with the entire workflow, including data exploration, model building, and evaluation.
- `data/`: Directory where the dataset is stored.

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/AlxBlzrv/Deep-Learning.git
   ```

2. Navigate to the project directory and open the Jupyter notebook:

   ```bash
   cd heart-disease-prediction
   jupyter notebook uci-heart-disease.ipynb
   ```

3. Follow the steps in the notebook to preprocess the data, train the models, and evaluate their performance.

## Results

The project will produce several machine learning models, and their performance will be compared based on classification metrics. The best model will be saved and can be used for predicting heart disease in new patients.

## Future Work

- **Model Improvement**: Experimenting with hyperparameter tuning and more advanced machine learning algorithms like Gradient Boosting or XGBoost.
- **Feature Engineering**: Adding domain-specific features to improve model accuracy.
- **Deployment**: Implementing a web application using Flask or Django for easy access to the model's predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
