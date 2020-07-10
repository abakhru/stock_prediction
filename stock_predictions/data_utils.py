import pandas as pd
import pandas_datareader
from alpha_vantage.timeseries import TimeSeries
from tabulate import tabulate

from stock_predictions import ALPHA_VANTAGE_APIKEY, ROOT, TODAY_DATE
from stock_predictions.logger import LOGGER


class DataUtils:

    def __init__(self):
        self.data_dir = ROOT.joinpath('data')

    @staticmethod
    def alpha_vantage_get_dataset(stock_symbol, csv_path):
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
        # self.visualize_price_history(df)
        return df
