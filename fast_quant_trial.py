#!/usr/bin/env python

"""
- https://github.com/enzoampil/fastquant
- https://towardsdatascience.com/backtest-your-trading-strategy-with-only-3-lines-of-python-3859b4a4ab44
"""

import arrow
from fastquant import backtest, get_stock_data

now = arrow.now().format('YYYY-MM-DD')

jfc = get_stock_data("JFC", "2018-01-01", "2019-01-01")
print(jfc.head())

# backtest('smac', jfc, fast_period=15, slow_period=40)
backtest('smac', jfc, fast_period=30, slow_period=50)
