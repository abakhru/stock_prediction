#!/usr/bin/env python

"""
https://towardsdatascience.com/getting-rich-quick-with-machine-learning-and-stock-market-predictions-696802da94fe
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Dense, Dropout, Input, LSTM, concatenate
from tensorflow import optimizers
from tensorflow.python import set_random_seed
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.utils.vis_utils import plot_model

from stock_predictions.base import StockPricePrediction
from stock_predictions.logger import LOGGER

set_random_seed(4)
np.random.seed(4)


class StockPredictionV2(StockPricePrediction):

    def __init__(self, stock_symbol='FB',
                 start_date="2010-01-01",
                 end_date=datetime.now().strftime("%Y-%m-%d"),
                 reset=False):
        super().__init__(stock_symbol, start_date, end_date)
        self.json_model_path = self.json_model_path.with_suffix('.v2.json')
        self.model_file_path = self.json_model_path.with_suffix('.v2.h5')
        if reset:
            LOGGER.debug('Deleting all model related files')
            self.model_file_path.unlink(missing_ok=True)
            self.json_model_path.unlink(missing_ok=True)

    def basic_model(self, epochs=50, number_of_days=50):
        """
        # Description: This program uses an artificial recurrent neural network called
        Long Short Term Memory (LSTM) to predict the closing stock price of a stock
        using the past 60 day stock price.
        """
        (ohlcv_histories, next_day_open_values,
         unscaled_y, y_normaliser) = self.csv_to_dataset(
            csv_path=self.data_dir.joinpath(f'{self.stock_symbol}_daily.csv'),
            number_of_days=number_of_days)

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
            plot_model(self.model, to_file=f"{self.model_file_path.with_suffix('.jpg')}",
                       show_shapes=True)
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

    def model_with_sma_only(self, epochs=50, number_of_days=50):
        """
        includes SMA (Standard Moving Average technical indicator
        """
        (ohlcv_histories, next_day_open_values, unscaled_y,
         y_normaliser, technical_indicators) = self.csv_to_dataset(
            csv_path=self.data_dir.joinpath(f'{self.stock_symbol}_daily.csv'),
            number_of_days=number_of_days,
            with_tech_indicator=True)

        test_split = 0.9  # the percent of data to be used for testing
        n = int(ohlcv_histories.shape[0] * test_split)
        # splitting the dataset up into train and test sets
        x_train = ohlcv_histories[:n]
        tech_ind_train = technical_indicators[:n]
        y_train = next_day_open_values[:n]

        x_test = ohlcv_histories[n:]
        tech_ind_test = technical_indicators[n:]
        y_test = next_day_open_values[n:]

        unscaled_y_test = unscaled_y[n:]
        # Build the LSTM network model
        if self.model_file_path.exists() and self.json_model_path.exists():
            self.model = model_from_json(self.json_model_path.read_text())
            self.model.load_weights(f'{self.model_file_path}')
            self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
        else:
            lstm_input = Input(shape=(number_of_days, 5), name='lstm_input')
            dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')
            # the first branch operates on the first input
            x = LSTM(units=50, name='lstm_0')(lstm_input)
            x = Dropout(0.2, name='lstm_dropout_0')(x)
            lstm_branch = Model(inputs=lstm_input, outputs=x)
            # the second branch operates on the second input
            y = Dense(20, name='tech_dense_0')(dense_input)
            y = Activation("relu", name='tech_relu_0')(y)
            y = Dropout(0.2, name='tech_dropout_0')(y)
            technical_indicators_branch = Model(inputs=dense_input, outputs=y)
            # combine the output of the two branches
            combined = concatenate(inputs=[lstm_branch.output,
                                           technical_indicators_branch.output],
                                   name='concatenate')
            z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
            z = Dense(1, activation="linear", name='dense_out')(z)
            # our model will accept the inputs of the two branches and then output a single value
            self.model = Model(inputs=[lstm_branch.input,
                                       technical_indicators_branch.input],
                               outputs=z)
            self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
            LOGGER.info('Building V2 LSTM Stock Prediction Model')
            self.model.summary()
            # if you need to visualize the model layers
            plot_model(self.model, to_file=f"{self.model_file_path.with_suffix('.jpg')}",
                       show_shapes=True)
            self.model.fit(x=[x_train, tech_ind_train],
                           y=y_train,
                           batch_size=32,
                           epochs=epochs,
                           shuffle=True,
                           validation_split=0.1)
            self.json_model_path.write_text(self.model.to_json())
            self.model.save_weights(filepath=f'{self.model_file_path}')

        scores = self.model.evaluate(x=[x_test, tech_ind_test], y=y_test)
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

    def model_with_sma_mcad(self, epochs=50, number_of_days=50):
        """
        includes  technical indicators:
        - SMA (Standard Moving Average)
        - MCAD (Moving Average Convergence Divergence)

        The MCAD is calculated by subtracting the
        26-period Exponential Moving Average from the 12-period EMA[6]
        """
        (ohlcv_histories,
         next_day_open_values,
         unscaled_y,
         y_normaliser,
         technical_indicators) = self.csv_to_dataset(
                csv_path=self.data_dir.joinpath(f'{self.stock_symbol}_daily.csv'),
                number_of_days=number_of_days,
                with_tech_indicator=True)
        test_split = 0.9  # the percent of data to be used for testing
        n = int(ohlcv_histories.shape[0] * test_split)
        # splitting the dataset up into train and test sets
        x_train = ohlcv_histories[:n]
        tech_ind_train = technical_indicators[:n]
        y_train = next_day_open_values[:n]

        x_test = ohlcv_histories[n:]
        tech_ind_test = technical_indicators[n:]
        y_test = next_day_open_values[n:]

        unscaled_y_test = unscaled_y[n:]
        # Build the LSTM network model
        if self.model_file_path.exists() and self.json_model_path.exists():
            self.model = model_from_json(self.json_model_path.read_text())
            self.model.load_weights(f'{self.model_file_path}')
            self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
        else:
            lstm_input = Input(shape=(number_of_days, 5), name='lstm_input')
            dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')
            # the first branch operates on the first input
            x = LSTM(units=50, name='lstm_0')(lstm_input)
            x = Dropout(0.2, name='lstm_dropout_0')(x)
            lstm_branch = Model(inputs=lstm_input, outputs=x)
            # the second branch operates on the second input
            y = Dense(20, name='tech_dense_0')(dense_input)
            y = Activation("relu", name='tech_relu_0')(y)
            y = Dropout(0.2, name='tech_dropout_0')(y)
            technical_indicators_branch = Model(inputs=dense_input, outputs=y)
            # combine the output of the two branches
            combined = concatenate(inputs=[lstm_branch.output,
                                           technical_indicators_branch.output],
                                   name='concatenate')
            z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
            z = Dense(1, activation="linear", name='dense_out')(z)
            # our model will accept the inputs of the two branches and then output a single value
            self.model = Model(inputs=[lstm_branch.input,
                                       technical_indicators_branch.input],
                               outputs=z)
            self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
            LOGGER.info('Building V2 LSTM Stock Prediction Model')
            self.model.summary()
            # if you need to visualize the model layers
            plot_model(self.model, to_file=f"{self.model_file_path.with_suffix('.jpg')}",
                       show_shapes=True)
            self.model.fit(x=[x_train, tech_ind_train],
                           y=y_train,
                           batch_size=32,
                           epochs=epochs,
                           shuffle=True,
                           validation_split=0.1)
            self.json_model_path.write_text(self.model.to_json())
            self.model.save_weights(filepath=f'{self.model_file_path}')

        y_test_predicted = self.model.predict(x=[x_test, tech_ind_test])
        y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
        y_predicted = self.model.predict(x=[ohlcv_histories, technical_indicators])
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
