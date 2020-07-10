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
import logging

import click
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
import yfinance
from alpha_vantage.timeseries import TimeSeries
from keras.layers import Activation, Dense, Dropout, Input, LSTM
from keras.models import Sequential
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from tensorflow import optimizers
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.utils.vis_utils import plot_model

from stock_predictions.logger import LOGGER, pretty_print_df

logging.getLogger('matplotlib.font_manager').setLevel('ERROR')
logging.getLogger('urllib3.connectionpool').setLevel('ERROR')
ALPHA_VANTAGE_APIKEY = 'S9XC819851W24M08'
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")


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

    @staticmethod
    def visualize_price_history(data_frame):
        # Visualize the closing price history
        plt.figure(figsize=(16, 8))
        plt.title('Close Price History')
        plt.plot(data_frame['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.show()

    def predict_price_v1(self, epochs=50):
        """
        # Description: This program uses an artificial recurrent neural network called
        Long Short Term Memory (LSTM) to predict the closing stock price of a stock
        using the past 60 day stock price.

        :return:
        """
        df = pandas_datareader.DataReader(name=self.stock_symbol,
                                          data_source='yahoo',
                                          start=self.start_date, end=self.end_date)
        LOGGER.info(f"\n==== Stock price data for '{self.stock_symbol}' ===="
                    f"\n{tabulate(df[:10], headers='keys', tablefmt='sql')}")
        # self.visualize_price_history(df)
        # Create a new dataframe with only the 'Close' column
        data = df.filter(['Close'])
        # Converting the dataframe to a numpy array
        dataset = data.values
        # Get /Compute the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)
        # Scale the all of the data to be values between 0 and 1
        self.data_normaliser = MinMaxScaler(feature_range=(0, 1))
        data_normalised = self.data_normaliser.fit_transform(dataset)
        # Create the scaled training data set
        train_data = data_normalised[0:training_data_len, :]
        # Split the data into x_train and y_train data sets
        x_train, y_train = list(), list()
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
        # Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Reshape the data into the shape accepted by the LSTM
        # [number of samples, number of time steps, and number of features]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # Build the LSTM network model
        self.model = Sequential()
        if self.model_file_path.exists() and self.json_model_path.exists():
            self.model = model_from_json(self.json_model_path.read_text())
            self.model.load_weights(f'{self.model_file_path}')
            # self.model.compile(optimizer='adam', loss='mean_squared_error')
            self.model.compile(loss='binary_crossentropy',
                               optimizer='rmsprop',
                               metrics=['accuracy'])
        else:
            self.model.add(LSTM(units=50, return_sequences=True,
                                input_shape=(x_train.shape[1], 1)))
            self.model.add(LSTM(units=50, return_sequences=False))
            self.model.add(Dense(units=25))
            self.model.add(Dense(units=1))
            self.model.compile(loss='binary_crossentropy',
                               optimizer='rmsprop',
                               metrics=['accuracy'])
            LOGGER.info('Staring model training based on last 60 days price dataset ...')
            self.model.fit(x_train, y_train, batch_size=1, epochs=epochs, shuffle=True)
            self.json_model_path.write_text(self.model.to_json())
            self.model.save_weights(filepath=f'{self.model_file_path}')
        self.model.summary()
        scores = self.model.evaluate(x_train, y_train, verbose=0)
        LOGGER.info("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

        # if you need to visualize the model layers
        plot_model(self.model, to_file=f"{self.model_file_path.with_suffix('.jpg')}")

        # Test data set
        test_data = data_normalised[training_data_len - 60:, :]
        # Create the x_test and y_test data sets
        x_test = []
        # Get all of the rows from index 1603 to the
        # rest and all of the columns (in this case it's only column 'Close'),
        # so 2003 - 1603 = 400 rows of data
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])
        # print(tabulate(x_test[:10], headers="keys", tablefmt='sql'))
        # Convert x_test to a numpy array
        x_test = np.array(x_test)
        # Reshape the data into the shape accepted by the LSTM
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # Getting the self.models predicted price values
        predictions = self.model.predict(x_test)
        predictions = self.data_normaliser.inverse_transform(predictions)  # Undo scaling
        # Calculate/Get the value of RMSE
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        LOGGER.warning(f'RMSE: {rmse}')
        # Plot/Create the data for the graph
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        # plt.show()
        LOGGER.debug(f'Predicted Price:\n{valid[:5]}')

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

    def predict_price_v2(self, epochs=50, history_points=50):
        """
        # Description: This program uses an artificial recurrent neural network called
        Long Short Term Memory (LSTM) to predict the closing stock price of a stock
        using the past 60 day stock price.

        :return:
        """
        # df = pandas_datareader.DataReader(name=self.stock_symbol,
        #                                   data_source='yahoo',
        #                                   start=self.start_date, end=self.end_date)
        (ohlcv_histories, next_day_open_values,
         unscaled_y, y_normaliser) = self.csv_to_dataset(history_points=history_points)

        # LOGGER.info(f"\n==== Stock price data for '{self.stock_symbol}' ===="
        #             f"\n{tabulate(ohlcv_histories[:10], headers='keys', tablefmt='sql')}")

        test_split = 0.9  # the percent of data to be used for testing
        n = int(ohlcv_histories.shape[0] * test_split)
        # splitting the dataset up into train and test sets
        ohlcv_train = ohlcv_histories[:n]
        y_train = next_day_open_values[:n]
        ohlcv_test = ohlcv_histories[n:]
        y_test = next_day_open_values[n:]
        unscaled_y_test = unscaled_y[n:]
        # Build the LSTM network model
        # if self.model_file_path.exists() and self.json_model_path.exists():
        #     self.model = model_from_json(self.json_model_path.read_text())
        #     self.model.load_weights(f'{self.model_file_path}')
        #     # self.model.compile(optimizer='adam', loss='mean_squared_error')
        #     self.model.compile(loss='binary_crossentropy',
        #                        optimizer='rmsprop',
        #                        metrics=['accuracy'])
        # else:
        #     self.model.add(LSTM(units=50, return_sequences=True,
        #                         input_shape=(x_train.shape[1], 1)))
        #     self.model.add(LSTM(units=50, return_sequences=False))
        #     self.model.add(Dense(units=25))
        #     self.model.add(Dense(units=1))
        #     self.model.compile(loss='binary_crossentropy',
        #                        optimizer='rmsprop',
        #                        metrics=['accuracy'])
        #     LOGGER.info('Staring model training based on last 60 days price dataset ...')
        #     self.model.fit(x=x_train,
        #                   y=y_train,
        #                    batch_size=32,
        #                    epochs=10,
        #                    shuffle=True,
        #                    validation_split=0.1)
        #     self.json_model_path.write_text(self.model.to_json())
        #     self.model.save_weights(filepath=f'{self.model_file_path}')
        lstm_input = Input(shape=(history_points, 5), name='lstm_input')
        x = LSTM(units=50, name='lstm_0')(lstm_input)
        x = Dropout(0.2, name='lstm_dropout_0')(x)
        x = Dense(64, name='dense_0')(x)
        x = Activation('sigmoid', name='sigmoid_0')(x)
        x = Dense(1, name='dense_1')(x)
        output = Activation('linear', name='linear_output')(x)
        self.model = Model(inputs=lstm_input, outputs=output)
        self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
        self.model.summary()
        # if you need to visualize the model layers
        plot_model(self.model, to_file=f"{self.model_file_path.with_suffix('.jpg')}")

        self.model.fit(x=ohlcv_train,
                       y=y_train,
                       batch_size=32,
                       epochs=epochs,
                       shuffle=True,
                       validation_split=0.1)
        scores = self.model.evaluate(ohlcv_test, y_test)
        # LOGGER.debug(f'Scores: {scores}')

        y_test_predicted = self.model.predict(ohlcv_test)
        y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
        # also getting predictions for the entire dataset, just to see how it performs
        y_predicted = self.model.predict(ohlcv_histories)
        y_predicted = y_normaliser.inverse_transform(y_predicted)

        assert unscaled_y_test.shape == y_test_predicted.shape
        real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
        scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
        LOGGER.info(scaled_mse)
        plt.gcf().set_size_inches(22, 15, forward=True)
        plt.plot(unscaled_y_test[0:-1], label='real')
        plt.plot(y_test_predicted[0:-1], label='predicted')
        plt.legend(['Real', 'Predicted'])
        plt.show()

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


@click.command()
@click.option('-s', '--stock', default='FB', help='Stock name for prediction')
@click.option('-e', '--epochs', default=50, help='Number of times to train the model')
@click.option('--v1', is_flag=True, default=True, help='Build and use the v1 model')
@click.option('--v2', is_flag=True, default=False, help='Build and use the v2 model')
def main(stock, epochs, v1, v2):
    p = StockPricePrediction(stock_symbol=stock,
                             start_date='2012-01-01',
                             end_date='2019-12-17')
    # p.plot_moving_avg(stock_symbol=stock, n_forward=40)
    if v1:
        p.predict_price_v1(epochs=epochs)
    if v2:
        p.predict_price_v2(epochs=epochs)
    p.test_prediction(end_date=TODAY_DATE)


if __name__ == '__main__':
    main()
