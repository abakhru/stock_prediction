from urllib.request import urlopen

import nltk
import requests
import time
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas import DataFrame

from stock_predictions.logger import LOGGER
from stock_predictions.utils import pretty_print_df


def get_internal_links(keyword, page_nums=3):
    nltk.download("vader_lexicon", quiet=True)  # download vader lexicon
    date_sentiments = dict()
    df = DataFrame(columns=['date', 'url', 'sentiment'])
    sia = SentimentIntensityAnalyzer()
    for i in range(1, page_nums + 1):
        url = f'https://www.businesstimes.com.sg/search/{keyword.replace(" ", "%20")}?page={i}'
        LOGGER.info(f'Processing url: {url}')
        data = requests.get(url=url, allow_redirects=True).content
        soup = BeautifulSoup(data, features='html.parser')
        posts = soup.findAll("div", {"class": "media-body"})
        LOGGER.debug(f'Found internal links: {posts}')
        for post in posts:
            time.sleep(1)
            url = post.a["href"]
            LOGGER.debug(f'Processing internal link : {url}')
            date = post.time.text
            try:
                link_page = urlopen(url).read()
            except ConnectionError as _:
                url = url[:-2]
                link_page = urlopen(url).read()
            link_soup = BeautifulSoup(link_page, features="lxml")
            sentences = link_soup.findAll("p")
            passage = ""
            for sentence in sentences:
                passage += sentence.text
            sentiment = sia.polarity_scores(passage)["compound"]
            date_sentiments.setdefault(date, []).append(sentiment)
            df.append([date, url, sentiment])
    LOGGER.info(f'Final Sentiments: {date_sentiments}')
    pretty_print_df(df)


stock = 'tesla'
get_internal_links(stock)
