#!/usr/bin/env python
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils.vis_utils import plot_model

from stock_predictions.base import StockPricePrediction
from stock_predictions.logger import LOGGER


class StockPredictionV1(StockPricePrediction):

    def __init__(self, stock_symbol='FB',
                 start_date="2010-01-01",
                 end_date=datetime.now().strftime("%Y-%m-%d"),
                 reset=False):
        super().__init__(stock_symbol, start_date, end_date)
        self.valid = None
        self.data = None
        self.rmse = None
        self.json_model_path = self.json_model_path.with_suffix('.v1.json')
        self.model_file_path = self.json_model_path.with_suffix('.v1.h5')
        if reset:
            LOGGER.debug('Deleting all model related files')
            self.model_file_path.unlink(missing_ok=True)
            self.json_model_path.unlink(missing_ok=True)

    def predict_price_v1(self, epochs=50, number_of_days=60):
        """
        # Description: This program uses an artificial recurrent neural network called
        Long Short Term Memory (LSTM) to predict the closing stock price of a stock
        using the past 60 day stock price.

        :epochs: epochs to use for trainings (int)
        :number_of_days: number of days data to use for training (int)
        """
        df = self.get_yahoo_stock_data(self.stock_symbol, self.start_date, self.end_date)
        # Create a new dataframe with only the 'Close' column
        self.data = df.filter(['Close'])
        # Converting the dataframe to a numpy array
        dataset = self.data.values
        # Get /Compute the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)
        # Scale the all of the data to be values between 0 and 1
        # self.data_normaliser = MinMaxScaler(feature_range=(0, 1))
        data_normalised = self.data_normaliser.fit_transform(dataset)
        # Create the scaled training data set
        train_data = data_normalised[0:training_data_len, :]
        # Split the data into x_train and y_train data sets
        x_train, y_train = list(), list()
        for i in range(number_of_days, len(train_data)):
            x_train.append(train_data[i - number_of_days:i, 0])
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
            # self.model.compile(loss='binary_crossentropy',
            #                    optimizer='rmsprop',
            #                    metrics=['accuracy'])
            self.model.compile(optimizer='adam',
                               loss='mean_squared_error',
                               metrics=['accuracy'])
            LOGGER.info(f'Staring model training based on last {number_of_days} days price '
                        f'dataset ...')
            self.model.fit(x_train, y_train,
                           batch_size=1,
                           epochs=epochs,
                           shuffle=True)
            self.json_model_path.write_text(self.model.to_json())
            self.model.save_weights(filepath=f'{self.model_file_path}')
        self.model.summary()
        scores = self.model.evaluate(x_train, y_train, verbose=0)
        LOGGER.info("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

        # if you need to visualize the model layers
        plot_model(self.model, to_file=f"{self.model_file_path.with_suffix('.jpg')}",
                   show_shapes=True)

        # Test data set
        test_data = data_normalised[training_data_len - number_of_days:, :]
        # Create the x_test and y_test data sets
        x_test = []
        # Get all of the rows from index 1603 to the
        # rest and all of the columns (in this case it's only column 'Close'),
        # so 2003 - 1603 = 400 rows of data
        y_test = dataset[training_data_len:, :]
        for i in range(number_of_days, len(test_data)):
            x_test.append(test_data[i - number_of_days:i, 0])
        # LOGGER.info(tabulate(x_test[:10], headers="keys", tablefmt='sql'))
        # Convert x_test to a numpy array
        x_test = np.array(x_test)
        # Reshape the data into the shape accepted by the LSTM
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # Getting the self.models predicted price values
        predictions = self.model.predict(x_test)
        predictions = self.data_normaliser.inverse_transform(predictions)  # Undo scaling
        # Calculate/Get the value of RMSE (Root Mean Squared Error)
        self.rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        LOGGER.warning(f'Root Mean Squared Error: {self.rmse}')
        # Plot/Create the data for the graph
        train = self.data[:training_data_len]
        self.valid = self.data[training_data_len:]
        self.valid['Predictions'] = predictions
        # Visualize the data
        plt.style.use('ggplot')
        plt.figure(figsize=(16, 8))
        plt.title(f'Model for {self.stock_symbol.upper()}')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(self.valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
        plt.show()
        LOGGER.info(f'\n==== Predicted Price ====\n{self.valid[:-5]}')

    def find_accuracy(self):
        """find the accuracy based on predicting day-to-day movements"""
        valid_movement = []
        pred_movement = []
        close_prices = self.valid.Close.tolist()
        pred_prices = self.valid.Predictions.tolist()
        n = 0
        for index, value in enumerate(close_prices[:-1]):
            if value > close_prices[index + 1]:
                valid_movement.append(1)
            else:
                valid_movement.append(0)
        for index, value in enumerate(pred_prices[:-1]):
            if value > pred_prices[index + 1]:
                pred_movement.append(1)
            else:
                pred_movement.append(0)
        for val, pred in zip(valid_movement, pred_movement):
            if val == pred:
                n = n + 1
            else:
                pass
        total = len(valid_movement)
        accuracy = n / total
        LOGGER.info(f'The accuracy of the LSTM Model predicting the movement of a stock each day '
                    f'is {100 * round(accuracy, 3)}%')
        dataframe = pd.DataFrame(list(zip(valid_movement, pred_movement)),
                                 columns=['Valid Movement', 'Predicted Movement'])
        LOGGER.info(dataframe)

        # get predicted price for next day
        last_60day = self.data[-60:].values
        last_60day_scaled = self.data_normaliser.transform(last_60day)
        xx_test = []
        xx_test.append(last_60day_scaled)
        xx_test = np.array(xx_test)
        xx_test = np.reshape(xx_test, (xx_test.shape[0], xx_test.shape[1], 1))
        pred = self.model.predict(xx_test)
        pred = self.data_normaliser.inverse_transform(pred)
        pred = pred[0]
        pred = pred[0]
        LOGGER.info("The predicted price for the next trading day is: {}".format(round(pred, 2)))

        # get stats
        # Root mean squared error
        LOGGER.info(f'The root mean squared error is {round(self.rmse, 2)}')
        error = mean_squared_error(self.valid['Close'].tolist(), self.valid['Predictions'].tolist())
        LOGGER.info('Testing Mean Squared Error: %.3f' % error)
