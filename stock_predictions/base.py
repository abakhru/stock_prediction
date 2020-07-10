#!env python

"""Find the best moving average.ipynb
https://towardsdatascience.com/an-algorithm-to-find-the-best-moving-average-for-stock-trading-1b024672299c
https://colab.research.google.com/github/gianlucamalato/machinelearning/blob/master/Find_the_best_moving_average.ipynb

pip install yfinance pandas_datareader sklearn keras tensorflow pydot graphviz aplha_vantage

SMA: Simple Moving Average
LWMA: Linear Weighted Moving Average
LSTM: Long Short Term Memory cells are like mini neural networks designed to allow for memory
     in a larger neural network.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
import yfinance
from alpha_vantage.timeseries import TimeSeries
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler

from stock_predictions import ALPHA_VANTAGE_APIKEY
from stock_predictions.logger import LOGGER
from stock_predictions.utils import pretty_print_df


class StockPricePrediction:

    def __init__(self, stock_symbol='FB',
                 start_date="2010-01-01",
                 end_date=datetime.now().strftime("%Y-%m-%d")):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.data_normaliser = None
        self.json_model_path = Path('/tmp', f'{self.stock_symbol}_model.json')
        self.model_file_path = Path('/tmp', f'{self.stock_symbol}_model.h5')

    def save_dataset(self, csv_path):
        """
    1. open  2. high    3. low  4. close   5. volume
1   360.700   365.00  357.5700    364.84  34380628.0
2   365.000   368.79  358.5200    360.06  48155849.0
3   364.000   372.38  362.2701    366.53  53038869.0
4   351.340   359.46  351.1500    358.87  33861316.0
5   354.635   356.56  345.1500    349.72  66118952.0
        :return:
        """
        ts = TimeSeries(key=ALPHA_VANTAGE_APIKEY, output_format='pandas')
        data, meta_data = ts.get_daily(self.stock_symbol, outputsize='full')
        data.to_csv(csv_path)

    def plot_moving_avg(self, n_forward=40):
        """Plats best moving average of the given stock symbol

        - Takes 10years worth of data set for the stock price.
        - Split this dataset into training and test sets
        - Apply different moving averages on the training set and, for each one,
        calculate the average return value after N days when the close price is over the moving
        average
        - Choose the moving average length that maximizes such average return
        - Use this moving average to calculate the average return on the test set
        - Verify that the average return on the test set is statistically similar to the
        average return achieved on the training set

        :param n_forward: number of days forward PREDICTION (int)
        :return:
        """
        ticker = yfinance.Ticker(self.stock_symbol)
        data = ticker.history(interval="1d", start=self.start_date, end=self.end_date)
        data['Forward Close'] = data['Close'].shift(-n_forward)
        data['Forward Return'] = (data['Forward Close'] - data['Close']) / data['Close']
        result = []
        train_size = 0.6
        for sma_length in range(20, 500):
            data['SMA'] = data['Close'].rolling(sma_length).mean()
            data['input'] = [int(x) for x in data['Close'] > data['SMA']]
            df = data.dropna()
            training = df.head(int(train_size * df.shape[0]))
            test = df.tail(int((1 - train_size) * df.shape[0]))
            tr_returns = training[training['input'] == 1]['Forward Return']
            test_returns = test[test['input'] == 1]['Forward Return']
            mean_forward_return_training = tr_returns.mean()
            mean_forward_return_test = test_returns.mean()
            pvalue = ttest_ind(tr_returns, test_returns, equal_var=False)[1]
            result.append({'sma_length': sma_length,
                           'training_forward_return': mean_forward_return_training,
                           'test_forward_return': mean_forward_return_test,
                           'p-value': pvalue})
        result.sort(key=lambda x: -x['training_forward_return'])
        LOGGER.info(f'Top 2 Results for "{n_forward}" days:\n'
                    f'{json.dumps(result[:2], indent=4, sort_keys=True)}')
        best_sma = result[0]['sma_length']
        data['SMA'] = data['Close'].rolling(best_sma).mean()
        plt.plot(data['Close'], label=self.stock_symbol)
        plt.plot(data['Close'].rolling(20).mean(), label="20-periods SMA")  # short-term
        plt.plot(data['Close'].rolling(50).mean(), label="50-periods SMA")  # medium-term
        plt.plot(data['Close'].rolling(200).mean(), label="200-periods SMA")  # long-term
        plt.plot(data['SMA'], label=f"{best_sma} periods SMA")
        plt.legend()
        plt.xlim((datetime.strptime(self.start_date, '%Y-%m-%d').date(),
                  datetime.strptime(self.end_date, '%Y-%m-%d').date()))
        plt.show()

    def csv_to_dataset(self, csv_path=None, history_points=50):
        if csv_path is None:
            csv_path = Path('/tmp', f'{self.stock_symbol}_daily.csv')
        if not csv_path.exists():
            self.save_dataset(csv_path)
        data = pd.read_csv(csv_path)
        data = data.drop('date', axis=1)
        data = data.drop(0, axis=0)
        data = data.values

        data_normaliser = MinMaxScaler()
        data_normalised = data_normaliser.fit_transform(data)

        # using the last {history_points} open high low close volume data points,
        # predict the next open value
        temp = list()
        for i in range(len(data_normalised) - history_points):
            temp.append(data_normalised[i: i + history_points].copy())
        ohlcv_histories_normalised = np.array(temp)

        temp = list()
        for i in range(len(data_normalised) - history_points):
            temp.append(data_normalised[:, 0][i + history_points].copy())
        next_day_open_values_normalised = np.array(temp)
        next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

        next_day_open_values = np.array(
                [data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
        next_day_open_values = np.expand_dims(next_day_open_values, -1)

        y_normaliser = MinMaxScaler()
        y_normaliser.fit(next_day_open_values)

        assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
        return (ohlcv_histories_normalised,
                next_day_open_values_normalised,
                next_day_open_values,
                y_normaliser)

    def test_prediction(self, end_date='2019-12-18'):
        LOGGER.info('==== Testing the prediction Model ====')
        # Get the quote
        stock_quote = pandas_datareader.DataReader(self.stock_symbol,
                                                   data_source='yahoo',
                                                   start=self.start_date,
                                                   end=end_date)
        # Create a new dataframe
        new_df = stock_quote.filter(['Close'])
        # Get thh last 60 day closing price
        last_60_days = new_df[-60:].values
        # Scale the data to be values between 0 and 1
        last_60_days_scaled = self.data_normaliser.transform(last_60_days)
        # Create an empty list
        x_test = list()
        # Append the past 60 days
        x_test.append(last_60_days_scaled)
        # Convert the x_test data set to a numpy array
        x_test = np.array(x_test)
        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # Get the predicted scaled price
        pred_price = self.model.predict(x_test)
        # undo the scaling
        pred_price = self.data_normaliser.inverse_transform(pred_price)
        pretty_print_df(pred_price)
        LOGGER.info(f"[{self.stock_symbol}:{end_date}] Predicted: Actual ==> "
                    f"{pred_price[0][0]}: {last_60_days['Close'][0]}")
