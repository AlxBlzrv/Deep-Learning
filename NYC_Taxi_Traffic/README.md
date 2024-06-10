# NYC_Taxi_Traffic

NYC_Taxi_Traffic is a project aimed at analyzing and predicting the passenger count for New York City taxis using time series data. The project employs different machine learning models to achieve accurate predictions and provides various visualizations for better understanding the data.

## Project Structure

The project contains the following directories and files:

- `codes/`: Contains the Python scripts for data preprocessing and model training.
  - `Preprocessing_And_LSTM.py`: Script for data preprocessing and LSTM model training.
  - `XGBoost.py`: Script for data preprocessing and XGBoost model training.
  
- `data/`: Contains the dataset used in the project.
  - `dataset.csv`: The dataset file with the following structure:
    ```csv
    Unnamed: 0,timestamp,value
    0,2014-07-01 00:00:00,10844
    1,2014-07-01 00:30:00,8127
    2,2014-07-01 01:00:00,6210
    3,2014-07-01 01:30:00,4656
    4,2014-07-01 02:00:00,3820
    ```
  
- `plots/`: Contains various plots generated during data analysis and model evaluation.
  - Example plots include time series analysis, anomaly detection, decomposed components, model performance metrics, etc.

## Installation

To run this project, you need to have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- tensorflow
- keras
- xgboost

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tensorflow keras xgboost
```

## Usage

### Preprocessing and LSTM Model

The script `Preprocessing_And_LSTM.py` performs data preprocessing, time series analysis, anomaly detection, and LSTM model training. To run the script:

```bash
python codes/Preprocessing_And_LSTM.py
```

This script will:

1. Load and preprocess the dataset.
2. Perform time series analysis and plot various visualizations.
3. Train an LSTM model on the preprocessed data.
4. Evaluate the model and plot the results.

### XGBoost Model

The script `XGBoost.py` performs data preprocessing and trains an XGBoost model. To run the script:

```bash
python codes/XGBoost.py
```

This script will:

1. Load and preprocess the dataset.
2. Train an XGBoost model on the preprocessed data.
3. Evaluate the model and plot the results.

## Results

The results of the analysis and model evaluations are stored in the `plots/` directory. These include:

- Time series plots of the entire dataset and specific anomalies (e.g., NYC Marathon, Thanksgiving, Christmas, New Year, and Snow Storm).
- Decomposed components of the time series (trend, seasonal, and residual).
- Differenced time series and stationarity tests (e.g., Dickey-Fuller test).
- Autocorrelation and partial autocorrelation functions.
- Actual vs. predicted values for both LSTM and XGBoost models.
- Model performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

## Acknowledgements

The dataset used in this project was sourced from Kaggle. Special thanks to Kaggle and the original dataset provider. The dataset can be found [here](https://www.kaggle.com/datasets/julienjta/nyc-taxi-traffic).

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Contributions, whether they involve fixing bugs, improving documentation, or adding new features, are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
