#!/usr/bin/env python

import click

from stock_predictions.v1 import StockPredictionV1
from stock_predictions.v2 import StockPredictionV2


@click.command()
@click.option('-s', '--stock', default='FB', help='Stock name for prediction')
@click.option('-e', '--epochs', default=50, help='Number of times to train the model')
@click.option('--v1', is_flag=True, default=False, help='Build and use the v1 model')
@click.option('--v2', is_flag=True, default=False, help='Build and use the v2 model')
@click.option('-r', '--reset', is_flag=True, default=False, help='Reset models')
@click.option('-d', '--days', default=60, help='Total days data to use for training model')
def main(stock, epochs, v1, v2, reset, days):
    # p.plot_moving_avg(stock_symbol=stock, n_forward=40)
    if v1:
        p = StockPredictionV1(stock_symbol=stock, start_date='2012-01-01', reset=reset)
        p.predict_price_v1(epochs=epochs, number_of_days=days)
        p.find_accuracy()
        # p.test_prediction(end_date=TODAY_DATE)
    if v2:
        p = StockPredictionV2(stock_symbol=stock, start_date='2012-01-01', reset=reset)
        # p.basic_model(epochs=epochs, number_of_days=days)
        # p.model_with_sma_only(epochs=epochs, number_of_days=days)
        p.model_with_sma_mcad(epochs=epochs, number_of_days=days)


if __name__ == '__main__':
    main()
