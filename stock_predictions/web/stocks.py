import datetime
import math
import os
from pathlib import Path

import numpy as np
import requests
import yfinance as yf
from lxml import html
from ratelimit import limits

from stock_predictions.web.template import template

firstDate = "2019-04-01"
secondDate = "2019-12-02"
endDate = datetime.date.today()

# src https://drive.google.com/file/d/1skgUviLX-Zby_qyCSBLaEeMGxFBXvM8f/view
stock_symbols = ' '.join(Path(__file__).parent.joinpath('stock_symbols.txt').read_text().rsplit())
blueCategory = ' '.join(Path(__file__).parent.joinpath('blue_categories.txt').read_text().rsplit())


class StockResult:

    def __init__(self, stock_symbol, first_price, second_price,
                 current_price, bankruptcy_probability):
        self.stock_symbol = stock_symbol
        self.first_price = np.round(first_price, 2)
        self.second_price = np.round(second_price, 2)
        self.current_price = np.round(current_price, 2)
        self.first_percentage_movement = np.round(current_price / first_price * 100 - 100, 2)
        self.second_percentage_movement = np.round(current_price / second_price * 100 - 100, 2)
        self.bankruptcy_probability = bankruptcy_probability


def get_bankruptcy_probability(symbol):
    try:
        page = requests.get(f"https://www.macroaxis.com/invest/ratio/"
                            f"{symbol}--Probability-Of-Bankruptcy")
        tree = html.fromstring(page.content)
        bankruptcy_probability = tree.xpath("//div[contains(@class, "
                                            "\'importantValue\')]/text()")[0]
    except Exception as _:
        bankruptcy_probability = "unknown"
    return bankruptcy_probability


@limits(calls=1, period=1)  # slow down for rate limiting
def get_all_stocks_data():
    data = yf.download(stock_symbols, start=firstDate, end=endDate)
    return data


results_file_name = Path(__file__).parent.resolve().joinpath('docs/index.html')
if results_file_name.exists():
    results_file_name.unlink()

all_stocks_data = get_all_stocks_data()
stock_results = []

for stock_symbol in stock_symbols.split():
    first_price = all_stocks_data.Close[stock_symbol][firstDate]
    second_price = all_stocks_data.Close[stock_symbol][secondDate]
    current_price = all_stocks_data.Close[stock_symbol][-1]
    bankruptcy_probability = get_bankruptcy_probability(stock_symbol)
    result = StockResult(stock_symbol, first_price, second_price,
                         current_price, bankruptcy_probability)
    if not math.isnan(result.second_percentage_movement):
        stock_results.append(result)

stock_results.sort(key=lambda x: x.second_percentage_movement)

heading = ''
tableBody = ''

with open(results_file_name, "a") as results_file:
    heading += ('<h1>Diff generated at {} UTC</h1> \n'.format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

for result in stock_results:
    first_percentage_movementClass = 'positive-movement' if (
                result.first_percentage_movement > 0) else 'negative-movement'
    second_percentage_movementClass = 'positive-movement' if (
                result.second_percentage_movement > 0) else 'negative-movement'
    categoriesClasses = 'blue-category' if (result.stock_symbol in blueCategory) else ''

    tableBody += "<tr class='{}'> \n".format(categoriesClasses)
    tableBody += '<td><button onclick="renderChart(`{}`)">{}</button></td> \n'.format(
        result.stock_symbol, result.stock_symbol)
    tableBody += "<td>{}$</td> \n".format(result.first_price)
    tableBody += "<td>{}$</td> \n".format(result.second_price)
    tableBody += "<td>{}$</td> \n".format(result.current_price)
    tableBody += "<td class='{}'>{}%</td> \n".format(first_percentage_movementClass,
                                                     result.first_percentage_movement)
    tableBody += "<td class='{}'>{}%</td> \n".format(second_percentage_movementClass,
                                                     result.second_percentage_movement)
    tableBody += "<td>{}</td> \n".format(result.bankruptcy_probability)
    tableBody += "</tr> \n"

with open(results_file_name, "a") as results_file:
    results_file.write(template.format(heading, tableBody))
