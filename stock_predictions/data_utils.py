from alpha_vantage.timeseries import TimeSeries

from stock_predictions import ALPHA_VANTAGE_APIKEY


class DataUtils:

    @staticmethod
    def save_dataset(stock_symbol, csv_path):
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
