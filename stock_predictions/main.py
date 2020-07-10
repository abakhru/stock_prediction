#!/usr/bin/env python

import click

from stock_predictions.v1 import StockPredictionV1
from stock_predictions.v2 import StockPredictionV2


@click.command()
@click.option('-s', '--stock', default='FB', help='Stock name for prediction')
@click.option('-e', '--epochs', default=50, help='Number of times to train the model')
@click.option('--v1', is_flag=True, default=False, help='Build and use the v1 model')
@click.option('--v2', is_flag=True, default=False, help='Build and use the v2 model')
def main(stock, epochs, v1, v2):
    # p.plot_moving_avg(stock_symbol=stock, n_forward=40)
    if v1:
        p = StockPredictionV1(stock_symbol=stock,
                              start_date='2012-01-01')
        p.predict_price_v1(epochs=epochs, number_of_days=600)
        p.find_accuracy()
        # p.test_prediction(end_date=TODAY_DATE)
    if v2:
        p = StockPredictionV2(stock_symbol=stock,
                              start_date='2012-01-01')
        p.predict_price_v2(epochs=epochs)


if __name__ == '__main__':
    main()
