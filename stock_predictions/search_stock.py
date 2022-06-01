#!/usr/bin/env python
import json
from datetime import datetime

import click
import requests
import yfinance
from pandas import DataFrame

from stock_predictions import ROOT
from stock_predictions.logger import LOGGER
from stock_predictions.utils import pretty_print_df

LOGGER.setLevel('INFO')
final_df = DataFrame(columns=['Symbol', 'Name', 'URL', 'High', 'Low', 'Open', 'Close', 'Volume'])


class SearchStockSymbol:

    def __init__(self, company_name,
                 exchange='NASDAQ',
                 start_date="2020-01-01",
                 end_date=datetime.now().strftime("%Y-%m-%d")):
        self.company_name = company_name
        self.stock_symbol = None
        self.exchange_name = exchange
        self.start_date = start_date
        self.end_date = end_date
        self.search_stock_symbol()

    def search_stock_symbol(self, retry=2):
        url = (f'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query='
               f'{self.company_name}&callback=YAHOO.Finance.SymbolSuggest.ssCallback&'
               f'lang=en')
        if self.company_name in final_df['Name'].values:
            LOGGER.warning(f'{self.company_name} already processed')
            return
        response = requests.get(url)
        assert response.status_code == 200
        data = json.loads(response.text.replace('YAHOO.Finance.SymbolSuggest.ssCallback(',
                                                '').replace(");", ""))
        if retry <= 0:
            LOGGER.critical(f'Stock symbol for "{self.company_name.capitalize()}" '
                            f'not found in {self.exchange_name} exchange')
            return
        try:
            nasdaq_result = [i for i in data['ResultSet']['Result']
                             if i['exchDisp'] == self.exchange_name][0]
            LOGGER.debug(f'{self.exchange_name} only:\n{json.dumps(nasdaq_result, indent=4)}')
            self.company_name = nasdaq_result['name']
            self.stock_symbol = nasdaq_result['symbol']
            LOGGER.info(f'[{self.stock_symbol}] ==> {self.company_name}')
        except IndexError as _:
            LOGGER.debug(f'All Exchanges results:\n{json.dumps(data, sort_keys=True, indent=True)}')
            self.company_name = self.company_name.split()[0]
            LOGGER.error(f'Retrying with only the first part of company: {self.company_name}')
            self.search_stock_symbol(retry=retry - 1)
            # sys.exit()

    def ticker_details(self):
        if self.stock_symbol is None:
            return
        ticker = yfinance.Ticker(self.stock_symbol)
        data = ticker.history(interval="1d", start=self.start_date, end=self.end_date)
        # data = DataReader(self.stock_symbol, data_source='yahoo', start=self.start_date,
        #                   end=self.end_date)
        LOGGER.debug(f'[{self.stock_symbol}] details:\n{data.tail()}')
        LOGGER.info(f'[{self.end_date}] Price: {data["Close"][-1]}')
        ab = data.drop(['Dividends', 'Stock Splits'], axis=1)
        cd = ab.tail(1)
        cd.insert(0, 'Symbol', self.stock_symbol)
        cd.insert(1, 'Name', self.company_name)
        cd.insert(2, 'URL', f'https://finance.yahoo.com/quote/{self.stock_symbol}')
        global final_df
        final_df = final_df.append(cd)
        return data


@click.command()
@click.option('-s', '--stock', default='FB', help='Stock name for prediction')
@click.option('-e', '--exchange', default='NASDAQ', help='Exchange name to search in')
def main(stock, exchange):
    p = SearchStockSymbol(stock, exchange)
    p.ticker_details()


if __name__ == '__main__':
    # m = ROOT.joinpath('data', 't.txt').read_text().splitlines()
    # s = [i.split(' ', 1)[1:] for i in m]
    # b = [i[0].strip() for i in s]
    # c = [i.split('EQ')[0].replace('-', '') for i in b]
    # for i in c:
    #     p = SearchStockSymbol(company_name=i, exchange='NSE')
    #     p.ticker_details()
    p = SearchStockSymbol(company_name='Reliance Communications', exchange='NSE')
    p.ticker_details()
    LOGGER.info(pretty_print_df(final_df.drop_duplicates()))
    # main()
