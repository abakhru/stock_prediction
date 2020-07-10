import logging

from colorlog import ColoredFormatter

LOGGER = logging.getLogger(__name__)

LOG_FORMAT = ('%(asctime)s '
              '%(log_color)s'
              '%(process)d %(name)s %(levelname)s | %(pathname)s:%(lineno)s | '
              '%(reset)s'
              '%(log_color)s%(message)s%(reset)s')
stream = logging.StreamHandler()
stream.setFormatter(ColoredFormatter(LOG_FORMAT))
logging.basicConfig(handlers=[stream])

LOGGER.setLevel('DEBUG')
