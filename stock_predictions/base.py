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

import matplotlib.pyplot as plt
import numpy as np
import yfinance
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler

from stock_predictions import ROOT, TODAY_DATE
from stock_predictions.data_utils import DataUtils
from stock_predictions.logger import LOGGER
from stock_predictions.utils import pretty_print_df


class StockPricePrediction(DataUtils):
    def __init__(
        self,
        stock_symbol='FB',
        start_date="2010-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
    ):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.data_normaliser = MinMaxScaler()
        self.model_dir = ROOT / 'models'
        self.data_dir = ROOT / 'data'
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.json_model_path = self.model_dir.joinpath(
            f'{self.stock_symbol.lower()}_' f'{TODAY_DATE}_model.json'
        )
        self.model_file_path = self.model_dir.joinpath(
            f'{self.stock_symbol.lower()}_' f'{TODAY_DATE}_model.h5'
        )

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
            result.append(
                {
                    'sma_length': sma_length,
                    'training_forward_return': mean_forward_return_training,
                    'test_forward_return': mean_forward_return_test,
                    'p-value': pvalue,
                }
            )
        result.sort(key=lambda x: -x['training_forward_return'])
        LOGGER.info(
            f'Top 2 Results for "{n_forward}" days:\n'
            f'{json.dumps(result[:2], indent=4, sort_keys=True)}'
        )
        best_sma = result[0]['sma_length']
        data['SMA'] = data['Close'].rolling(best_sma).mean()
        plt.plot(data['Close'], label=self.stock_symbol)
        plt.plot(data['Close'].rolling(20).mean(), label="20-periods SMA")  # short-term
        plt.plot(data['Close'].rolling(50).mean(), label="50-periods SMA")  # medium-term
        plt.plot(data['Close'].rolling(200).mean(), label="200-periods SMA")  # long-term
        plt.plot(data['SMA'], label=f"{best_sma} periods SMA")
        plt.legend()
        plt.xlim(
            (
                datetime.strptime(self.start_date, '%Y-%m-%d').date(),
                datetime.strptime(self.end_date, '%Y-%m-%d').date(),
            )
        )
        plt.show()

    def test_prediction(self, end_date='2019-12-18'):
        LOGGER.info('\n==== Testing the prediction Model ====')
        # Get the quote
        stock_quote = self.get_yahoo_stock_data(self.stock_symbol, self.start_date, self.end_date)
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
        LOGGER.info(
            f"[{self.stock_symbol}:{end_date}] Predicted: Actual ==> "
            f"{pred_price[0][0]}: {last_60_days[-1][0]}"
        )
