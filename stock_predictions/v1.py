#!/usr/bin/env python
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader
from keras.layers import Dense, LSTM
from keras.models import Sequential
from tabulate import tabulate
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils.vis_utils import plot_model

from stock_predictions.base import StockPricePrediction
from stock_predictions.logger import LOGGER


class StockPredictionV1(StockPricePrediction):

    def __init__(self, stock_symbol='FB',
                 start_date="2010-01-01",
                 end_date=datetime.now().strftime("%Y-%m-%d")):
        super().__init__(stock_symbol, start_date, end_date)
        self.json_model_path = self.json_model_path.with_suffix('.v1.json')
        self.model_file_path = self.json_model_path.with_suffix('.v1.h5')

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
        # self.data_normaliser = MinMaxScaler(feature_range=(0, 1))
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
            # self.model.compile(optimizer='adam',
            #                    loss='mean_squared_error',
            #                    metrics=['accuracy'])
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
        LOGGER.info(f'\n==== Predicted Price ====\n{valid[:-5]}')
