
from .candles import get_candles, store_candles, fake_candle, fake_range_candles, candles_from_close_prices, candlestick_chart
from .backtest import backtest
  
# def init() -> None:
    # import jesse.helpers as jh

    # from pydoc import locate
    # import os
    # import sys

    # fix directory issue
    # sys.path.insert(0, os.getcwd())

    # ls = os.listdir('.')
    # is_jesse_project = 'strategies' in ls and 'config.py' in ls and 'storage' in ls and 'routes.py' in ls

    # if not is_jesse_project:
        # print(
            # jh.color(
                # 'Invalid directory. To use Jesse inside notebooks, create notebooks inside the root of a Jesse project.',
                # 'red'
            # )
        # )

    # if is_jesse_project:
        # local_config = locate('config.config')
        # from jesse.config import set_config
        # set_config(local_config)


