"""Top-level package for stock_prediction."""

__author__ = """Amit Bakhru"""
__email__ = 'bakhru@me.com'
__version__ = '0.1.0'

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

if not sys.warnoptions:
    warnings.simplefilter('ignore')
os.environ.setdefault('TK_SILENCE_DEPRECATION', '1')

logging.getLogger('matplotlib.font_manager').setLevel('ERROR')
logging.getLogger('urllib3.connectionpool').setLevel('ERROR')
ALPHA_VANTAGE_APIKEY = 'S9XC819851W24M08'
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
ROOT = Path(__file__).parent.parent.resolve()
