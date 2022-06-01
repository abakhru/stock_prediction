"""Top-level package for stock_prediction."""

__author__ = """Amit Bakhru"""
__email__ = 'bakhru@me.com'
__version__ = '0.1.0'

import logging
import os
import warnings
from pathlib import Path

import arrow
import sys

if not sys.warnoptions:
    warnings.simplefilter('ignore')
os.environ.setdefault('TK_SILENCE_DEPRECATION', '1')

logging.getLogger('matplotlib.font_manager').setLevel('ERROR')
logging.getLogger('urllib3.connectionpool').setLevel('ERROR')
ALPHA_VANTAGE_APIKEY = 'S9XC819851W24M08'
now = arrow.utcnow()
TODAY_DATE = now.format('YYYY-MM-DD')
ROOT = Path(__file__).parent.parent.resolve()
