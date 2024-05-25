# House Sales Project

## Introduction
This project aims to analyze and predict house sales prices based on multivariate time series data. The dataset spans from 2007 to 2019 and includes sales prices for houses and units with varying numbers of bedrooms in a specific region. The data includes the following features:

- Date of sale
- Price
- Property type: unit or house
- Number of bedrooms: 1, 2, 3, 4, 5
- 4-digit postcode (reference only)

The project involves preprocessing the data, creating sequences for time series prediction, building and training LSTM models, and evaluating model performance.

## Project Structure
The project directory structure is as follows:

- **data**: Contains raw sales data and preprocessed data files.
  - `raw_sales.csv`: Raw house sales data from 2007 to 2019.
  - `ma_lga_12345.csv`: Transformed sales data resampled at quarterly intervals with median price aggregation and moving average smoothing.

- **plots**: Includes plots generated during model training and evaluation.
  - **actual_vs_predicted_prices**: Contains plots comparing actual and predicted prices for each model.
  - **loss**: Contains plots of training and validation loss for each model.

- **time_series.py**: Python code file containing data preprocessing, model building, training, and evaluation.

## Dependencies
The project requires the following Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow

## Usage
1. Ensure all dependencies are installed.
2. Clone or download the repository.
3. Place the `raw_sales.csv` and `ma_lga_12345.csv` files in the `data` directory.
4. Run the `time_series.py` script to preprocess the data, train LSTM models, and evaluate their performance.
5. Plots of training and validation loss will be saved in the `plots/loss` directory, and plots comparing actual and predicted prices will be saved in the `plots/actual_vs_predicted_prices` directory.

## Note
- The `time_series.py` script contains functions to preprocess the data, create sequences for time series prediction, build LSTM models, and evaluate model performance.
- The models are trained and evaluated for each property type and bedroom count combination.
- Model performance metrics, such as R^2 score, are printed during training and evaluation.

## Acknowledgements

All data is taken from kaggle: https://www.kaggle.com/datasets/htagholdings/property-sales
