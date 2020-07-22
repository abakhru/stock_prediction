#!/usr/bin/env python
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_model import StockDataSet
from keras.layers import Activation, Dense, Dropout, Input, LSTM
# from model_rnn import LstmRNN
from tensorflow import optimizers
from tensorflow.python.keras.models import Model, model_from_json

from stock_predictions import ROOT
from stock_predictions.base import StockPricePrediction
from stock_predictions.logger import LOGGER

flags = tf.app.flags
flags.DEFINE_integer("stock_count", 100, "Stock count [100]")
flags.DEFINE_integer("input_size", 1, "Input size [1]")
flags.DEFINE_integer("num_steps", 30, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches. [50]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS


class StockPredictionRNN(StockPricePrediction):

    def __init__(self, stock_symbol='FB',
                 start_date="2010-01-01",
                 end_date=datetime.now().strftime("%Y-%m-%d")):
        super().__init__(stock_symbol, start_date, end_date)
        self.json_model_path = self.json_model_path.with_suffix('.v3.json')
        self.model_file_path = self.json_model_path.with_suffix('.v3.h5')
        self.log_dir = ROOT / 'logs'
        self.log_dir.mkdir(exist_ok=True)

    def predict_price_rnn(self, epochs=50, history_points=50):
        """
        # Description: This program uses an artificial recurrent neural network called
        Long Short Term Memory (LSTM) to predict the closing stock price of a stock
        using the past 60 day stock price.
        """
        (ohlcv_histories, next_day_open_values,
         unscaled_y, y_normaliser) = self.csv_to_dataset(history_points=history_points)

        test_split = 0.9  # the percent of data to be used for testing
        n = int(ohlcv_histories.shape[0] * test_split)
        # splitting the dataset up into train and test sets
        ohlcv_train = ohlcv_histories[:n]
        y_train = next_day_open_values[:n]
        ohlcv_test = ohlcv_histories[n:]
        y_test = next_day_open_values[n:]
        unscaled_y_test = unscaled_y[n:]
        # Build the LSTM network model
        if self.model_file_path.exists() and self.json_model_path.exists():
            self.model = model_from_json(self.json_model_path.read_text())
            self.model.load_weights(f'{self.model_file_path}')
            self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005))
        else:
            lstm_input = Input(shape=(history_points, 5), name='lstm_input')
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
            # plot_model(self.model, to_file=f"{self.model_file_path.with_suffix('.jpg')}")
            self.model.fit(x=ohlcv_train,
                           y=y_train,
                           batch_size=32,
                           epochs=epochs,
                           shuffle=True,
                           validation_split=0.1)
            self.json_model_path.write_text(self.model.to_json())
            self.model.save_weights(filepath=f'{self.model_file_path}')
        scores = self.model.evaluate(ohlcv_test, y_test)
        LOGGER.debug(f'Scores: {scores}')

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
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(unscaled_y_test[0:-1], label='real')
        plt.plot(y_test_predicted[0:-1], label='predicted')
        plt.legend(['Real', 'Predicted'])
        # plt.show()

    def show_all_variables(self):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def load_sp500(self, input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05):
        if target_symbol is not None:
            return [StockDataSet(target_symbol,
                                 input_size=input_size,
                                 num_steps=num_steps,
                                 test_ratio=test_ratio)]
        # Load metadata of s & p 500 stocks
        data = pd.read_csv(self.data_dir.joinpath('constituents-financials.csv'))
        data = data.rename(columns={col: col.lower().replace(' ', '_') for col in data.columns})
        data['file_exists'] = data['symbol'].map(lambda x: ROOT.joinpath(f"data/{x}.csv").exists())
        LOGGER.info(data['file_exists'].value_counts().to_dict())

        data = data[data['file_exists'] is True].reset_index(drop=True)
        data = data.sort('market_cap', ascending=False).reset_index(drop=True)
        if k is not None:
            data = data.head(k)
        LOGGER.info(f"Head of S&P 500 info:\n{data.head()}")
        # Generate embedding meta file
        data[['symbol', 'sector']].to_csv(self.log_dir.joinpath("metadata.tsv"),
                                          sep='\t', index=False)
        return [StockDataSet(row['symbol'],
                             input_size=input_size,
                             num_steps=num_steps,
                             test_ratio=0.05) for _, row in data.iterrows()]

    def main(self):
        pp.pprint(flags.FLAGS.__flags)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=run_config) as sess:
            rnn_model = LstmRNN(
                    sess,
                    FLAGS.stock_count,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    num_steps=FLAGS.num_steps,
                    input_size=FLAGS.input_size,
                    embed_size=FLAGS.embed_size,
                    )

            self.show_all_variables()

            stock_data_list = self.load_sp500(
                    FLAGS.input_size,
                    FLAGS.num_steps,
                    k=FLAGS.stock_count,
                    target_symbol=FLAGS.stock_symbol,
                    )

            if FLAGS.train:
                rnn_model.train(stock_data_list, FLAGS)
            else:
                if not rnn_model.load()[0]:
                    raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
