#!/usr/bin/env python

"""
https://towardsdatascience.com/getting-rich-quick-with-machine-learning-and-stock-market-predictions-696802da94fe
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation, Dense, Dropout, Input, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow import optimizers
from tensorflow.python.keras.models import Model, model_from_json

from stock_predictions.base import StockPricePrediction
from stock_predictions.logger import LOGGER
from tensorflow.python.keras.utils.vis_utils import plot_model


class StockPredictionV2(StockPricePrediction):

    def __init__(self, stock_symbol='FB',
                 start_date="2010-01-01",
                 end_date=datetime.now().strftime("%Y-%m-%d")):
        super().__init__(stock_symbol, start_date, end_date)
        self.json_model_path = self.json_model_path.with_suffix('.v2.json')
        self.model_file_path = self.json_model_path.with_suffix('.v2.h5')

    def csv_to_dataset(self, csv_path=None, number_of_days=60):
        if csv_path is None:
            csv_path = self.data_dir / f'{self.stock_symbol}_daily.csv'
        if not csv_path.exists():
            self.alpha_vantage_get_dataset(self.stock_symbol, csv_path)
        data = pd.read_csv(csv_path)
        data = data.drop('date', axis=1)
        data = data.drop(0, axis=0)
        data = data.values
        data_normalised = self.data_normaliser.fit_transform(data)

        # using the last {number_of_days} open high low close volume data points,
        # predict the next open value
        temp = list()
        for i in range(len(data_normalised) - number_of_days):
            temp.append(data_normalised[i: i + number_of_days].copy())
        ohlcv_histories_normalised = np.array(temp)

        temp = list()
        for i in range(len(data_normalised) - number_of_days):
            temp.append(data_normalised[:, 0][i + number_of_days].copy())
        next_day_open_values_normalised = np.array(temp)
        next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

        next_day_open_values = np.array([data[:, 0][i + number_of_days].copy()
                                         for i in range(len(data) - number_of_days)])
        next_day_open_values = np.expand_dims(next_day_open_values, -1)

        y_normaliser = MinMaxScaler()
        y_normaliser.fit(next_day_open_values)

        assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
        return (ohlcv_histories_normalised,
                next_day_open_values_normalised,
                next_day_open_values,
                y_normaliser)

    def predict_price_v2(self, epochs=50, number_of_days=50):
        """
        # Description: This program uses an artificial recurrent neural network called
        Long Short Term Memory (LSTM) to predict the closing stock price of a stock
        using the past 60 day stock price.
        """
        (ohlcv_histories, next_day_open_values,
         unscaled_y, y_normaliser) = self.csv_to_dataset(number_of_days=number_of_days)

        test_split = 0.9  # the percent of data to be used for testing
        n = int(ohlcv_histories.shape[0] * test_split)
        # splitting the dataset up into train and test sets
        x_train = ohlcv_histories[:n]
        y_train = next_day_open_values[:n]

        x_test = ohlcv_histories[n:]
        y_test = next_day_open_values[n:]
        unscaled_y_test = unscaled_y[n:]
        # Build the LSTM network model
        if self.model_file_path.exists() and self.json_model_path.exists():
            self.model = model_from_json(self.json_model_path.read_text())
            self.model.load_weights(f'{self.model_file_path}')
            self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
        else:
            lstm_input = Input(shape=(number_of_days, 5), name='lstm_input')
            x = LSTM(units=50, name='lstm_0')(lstm_input)
            x = Dropout(0.2, name='lstm_dropout_0')(x)
            x = Dense(64, name='dense_0')(x)
            x = Activation('sigmoid', name='sigmoid_0')(x)
            x = Dense(1, name='dense_1')(x)
            output = Activation('linear', name='linear_output')(x)
            self.model = Model(inputs=lstm_input, outputs=output)
            self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
            LOGGER.info('Building V2 LSTM Stock Prediction Model')
            self.model.summary()
            # if you need to visualize the model layers
            plot_model(self.model, to_file=f"{self.model_file_path.with_suffix('.jpg')}")
            self.model.fit(x=x_train,
                           y=y_train,
                           batch_size=32,
                           epochs=epochs,
                           shuffle=True,
                           validation_split=0.1)
            self.json_model_path.write_text(self.model.to_json())
            self.model.save_weights(filepath=f'{self.model_file_path}')

        scores = self.model.evaluate(x_test, y_test)
        LOGGER.debug(f'Scores: {scores}')  # mean squared error of the normalised data

        y_test_predicted = self.model.predict(x_test)
        y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
        # also getting predictions for the entire dataset, just to see how it performs
        y_predicted = self.model.predict(ohlcv_histories)
        y_predicted = y_normaliser.inverse_transform(y_predicted)

        assert unscaled_y_test.shape == y_test_predicted.shape
        real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
        scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
        LOGGER.info(f'Scaled mean squared error: {scaled_mse}')

        plt.gcf().set_size_inches(22, 15, forward=True)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(unscaled_y_test[0:-1], label='real')
        plt.plot(y_test_predicted[0:-1], label='predicted')
        plt.legend(['Real', 'Predicted'])
        plt.show()
