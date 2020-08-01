from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

from stock_predictions import ALPHA_VANTAGE_APIKEY, ROOT, TODAY_DATE
from stock_predictions.logger import LOGGER
from stock_predictions.utils import pretty_print_df, visualize_price_history


class DataUtils:

    def __init__(self):
        self.data_dir = ROOT.joinpath('data')
        self.data_normaliser = MinMaxScaler()

    @staticmethod
    def alpha_vantage_get_dataset(csv_path):
        """
        1. open  2. high    3. low  4. close   5. volume
        1   360.700   365.00  357.5700    364.84  34380628.0
        2   365.000   368.79  358.5200    360.06  48155849.0
        3   364.000   372.38  362.2701    366.53  53038869.0
        4   351.340   359.46  351.1500    358.87  33861316.0
        5   354.635   356.56  345.1500    349.72  66118952.0
        :return:
        """
        assert isinstance(csv_path, Path)
        stock_symbol = csv_path.name.split('_')[0].upper()
        ts = TimeSeries(key=ALPHA_VANTAGE_APIKEY, output_format='pandas')
        data, meta_data = ts.get_daily(stock_symbol, outputsize='full')
        data.to_csv(csv_path)

    def get_yahoo_stock_data(self, stock_symbol, start, end):
        """
                          High         Low        Open       Close    Volume   Adj Close
        Date
        2019-04-01  168.899994  167.279999  167.830002  168.699997  10381500  168.699997
        2019-04-02  174.899994  169.550003  170.139999  174.199997  23946500  174.199997
        2019-04-03  177.960007  172.949997  174.500000  173.539993  27590100  173.539993
        2019-04-04  178.000000  175.529999  176.020004  176.020004  17847700  176.020004
        """
        csv_path = self.data_dir / f'{stock_symbol.lower()}_quote_{TODAY_DATE}.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pandas_datareader.DataReader(name=stock_symbol,
                                              data_source='yahoo',
                                              start=start, end=end)
            df.to_csv(csv_path)
        LOGGER.info(f"\n==== Stock price data for '{stock_symbol}' ===="
                    f"\n{tabulate(df[:5], headers='keys', tablefmt='sql')}")
        # visualize_price_history(df)
        return df

    def xcsv_to_dataset(self, stock, number_of_days=60, with_tech_indicator=False):
        csv_path = self.data_dir / f'{stock.lower()}_daily.csv'
        if not csv_path.exists():
            self.alpha_vantage_get_dataset(stock, csv_path)
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

        ret_values = [ohlcv_histories_normalised,
                      next_day_open_values_normalised,
                      next_day_open_values,
                      y_normaliser]

        if with_tech_indicator:
            technical_indicators = []
            for his in ohlcv_histories_normalised:
                # since we are using his[3] we are taking the SMA of the closing price
                sma = np.mean(his[:, 3])
                macd = self.calc_ema(his, 12) - self.calc_ema(his, 26)
                technical_indicators.append(np.array([sma, macd]))
            technical_indicators = np.array(technical_indicators)
            technical_indicators_normalised = self.data_normaliser.fit_transform(
                    technical_indicators)
            assert ohlcv_histories_normalised.shape[0] == technical_indicators_normalised.shape[0]
            ret_values.append(technical_indicators_normalised)
            assert len(ret_values) == 5
        LOGGER.info(f'Returning values len: {len(ret_values)}')
        return tuple(ret_values)

    def csv_to_dataset(self, csv_path, number_of_days=60, with_tech_indicator=False):
        if not csv_path.exists():
            self.alpha_vantage_get_dataset(csv_path)
        data = pd.read_csv(csv_path)
        LOGGER.info(f'==== {csv_path.name} ====')
        pretty_print_df(data.tail())
        data = data.drop('date', axis=1)
        data = data.drop(0, axis=0)
        data = data.values
        data_normalised = self.data_normaliser.fit_transform(data)

        # using the last {number_of_days} open close high low volume data points,
        # predict the next open value
        ohlcv_histories_normalised = np.array([data_normalised[i:i + number_of_days].copy() for i in
                                               range(len(data_normalised) - number_of_days)])
        next_day_open_values_normalised = np.array(
                [data_normalised[:, 0][i + number_of_days].copy() for i in
                 range(len(data_normalised) - number_of_days)])
        next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

        next_day_open_values = np.array([data[:, 0][i + number_of_days].copy()
                                         for i in range(len(data) - number_of_days)])
        next_day_open_values = np.expand_dims(next_day_open_values, -1)

        y_normaliser = MinMaxScaler()
        y_normaliser.fit(next_day_open_values)

        def calc_ema(values, time_period):
            # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
            sma = np.mean(values[:, 3])
            ema_values = [sma]
            k = 2 / (1 + time_period)
            for i in range(len(his) - time_period, len(his)):
                close = his[i][3]
                ema_values.append(close * k + ema_values[-1] * (1 - k))
            return ema_values[-1]

        ret_values = [ohlcv_histories_normalised,
                      next_day_open_values_normalised,
                      next_day_open_values,
                      y_normaliser]

        if with_tech_indicator:
            technical_indicators = []
            for his in ohlcv_histories_normalised:
                # note since we are using his[3] we are taking the SMA of the closing price
                sma = np.mean(his[:, 3])
                macd = calc_ema(his, 12) - calc_ema(his, 26)
                # technical_indicators.append(np.array([sma]))
                technical_indicators.append(np.array([sma, macd]))

            technical_indicators = np.array(technical_indicators)

            tech_ind_scaler = MinMaxScaler()
            technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

            assert ohlcv_histories_normalised.shape[0] == technical_indicators_normalised.shape[0]
            ret_values.append(technical_indicators_normalised)
            assert len(ret_values) == 5
        return ret_values

    def multiple_csv_to_dataset(self, test_set_name, number_of_days=60):
        ohlcv_histories = 0
        technical_indicators = 0
        next_day_open_values = 0
        # for csv_file_path in list(filter(lambda x: x.endswith('daily.csv'), os.listdir('data'))):
        for csv_file_path in list(self.data_dir.glob('*_daily.csv')):
            # if not csv_file_path == test_set_name:
            if csv_file_path.name == test_set_name:
                LOGGER.debug(f'Processing ... {csv_file_path}')
                if type(ohlcv_histories) == int:
                    (ohlcv_histories, technical_indicators,
                     next_day_open_values, _, _) = self.csv_to_dataset(csv_file_path)
                else:
                    a, b, c, _, _ = self.csv_to_dataset(csv_file_path)
                    ohlcv_histories = np.concatenate((ohlcv_histories, a), 0)
                    technical_indicators = np.concatenate((technical_indicators, b), 0)
                    next_day_open_values = np.concatenate((next_day_open_values, c), 0)
        ohlcv_train = ohlcv_histories
        tech_ind_train = technical_indicators
        y_train = next_day_open_values
        (ohlcv_test, tech_ind_test, y_test,
         unscaled_y_test, y_normaliser) = self.csv_to_dataset(self.data_dir.joinpath(test_set_name),
                                                              number_of_days=number_of_days)
        return (ohlcv_train, tech_ind_train, y_train, ohlcv_test,
                tech_ind_test, y_test, unscaled_y_test, y_normaliser)


if __name__ == '__main__':
    p = DataUtils()
    # p.csv_to_dataset(stock='TSLA', number_of_days=60)
    # p.multiple_csv_to_dataset(test_set_name=['FB', 'AAPL', 'MSFT', 'AMZN', 'GOOGL'],

    p.multiple_csv_to_dataset(test_set_name='TSLA_daily.csv',
                              number_of_days=60)
