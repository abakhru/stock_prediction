#!/usr/bin/env python

"""
- https://towardsdatascience.com/backtest-your-trading-strategy-with-only-3-lines-of-python-3859b4a4ab44
- TODO: https://www.backtrader.com/home/helloalgotrading/

"""
from datetime import datetime

import arrow
from fastquant import backtest, get_bt_news_sentiment, get_stock_data

from stock_predictions.base import StockPricePrediction


class FastQuantBackTestingV1(StockPricePrediction):

    def __init__(self, stock_symbol='FB',
                 start_date="2010-01-01",
                 end_date=datetime.now().strftime("%Y-%m-%d"),
                 reset=False):
        super().__init__(stock_symbol, start_date, end_date)
        today = arrow.utcnow()
        self.start_date = today.shift(months=-24).format('YYYY-MM-DD')
        self.end_date = today.format('YYYY-MM-DD')
        df = get_stock_data(symbol=stock_symbol,
                            start_date=self.start_date,
                            end_date=self.end_date,
                            format="ohlcv",
                            source='yahoo')
        # df = self.get_yahoo_stock_data(self.stock_symbol,
        #                                start=today.shift(months=-24).format('YYYY-MM-DD'),
        #                                end=today.format('YYYY-MM-DD'))
        # df = df['Close']
        print(df.head())
        print(df.tail())
        """# Backtest a simple moving average crossover (SMAC) strategy
        STRATEGY_MAPPING = {
            "rsi": RSIStrategy,
            "smac": SMACStrategy,
            "base": BaseStrategy,
            "macd": MACDStrategy,
            "emac": EMACStrategy,
            "bbands": BBandsStrategy,
            "buynhold": BuyAndHoldStrategy,
            "sentiment": SentimentStrategy,
            }
        """
        # backtest('smac', df, fast_period=15, slow_period=40)    # (-38285.86)
        # backtest(strategy='macd',
        #          data=df,
        #          fast_period=30,
        #          slow_period=50,
        #          verbose=True,
        #          plot=True)  # (12080.47)
        # res = backtest("smac", df, fast_period=range(15, 30, 3), slow_period=range(40, 55, 3),
        #                verbose=False, plot=False)
        # print(res[['fast_period', 'slow_period', 'final_value']].head())
        # backtest('macd', df, fast_period=12, slow_period=26, signal_period=9, sma_period=30,
        #          dir_period=10)
        # backtest('emac', df, fast_period=10, slow_period=30)
        # Bollinger Bands Strategy
        # backtest('bbands', df, period=20, devfactor=2.0)
        # backtest('rsi', df, rsi_period=14, rsi_upper=70, rsi_lower=30)
        sentiments = get_bt_news_sentiment(keyword='Tesla', page_nums=3)
        backtest(strategy="sentiment", data=df, sentiments=sentiments, senti=0.2)


if __name__ == '__main__':
    STOCK = 'TSLA'
    # STOCK = 'RELIANCE.NS'
    # STOCK = 'RCOM.NS'
    p = FastQuantBackTestingV1(stock_symbol=STOCK)
    # sentiments = get_bt_news_sentiment(keyword='Tesla', page_nums=3)
    # print(sentiments)
