
import numpy as np
from jesse import utils
from jesse import factories
from typing import Union
cimport cython
import random

def uuid4():
  s = '%032x' % random.getrandbits(128)
  return s[0:8]+'-'+s[8:12]+'-4'+s[13:16]+'-'+s[16:20]+'-'+s[20:32]
  
@cython.wraparound(True)
def get_candles(exchange: str, symbol: str, timeframe: str, start_date_: str, finish_date_: str) -> np.ndarray:
    """
    Returns candles from the database in numpy format

    :param exchange: str
    :param symbol: str
    :param timeframe: str
    :param start_date: str
    :param finish_date: str
    
    :return: np.ndarray
    """
    # symbol = symbol.upper()

    import arrow

    import jesse.helpers as jh
    from jesse.models import Candle
    from jesse.exceptions import CandleNotFoundInDatabase
    from jesse.services.candle import generate_candle_from_one_minutes
    # check if .env file exists
    if not jh.is_jesse_project():
        raise FileNotFoundError(
            'Invalid directory: ".env" file not found. To use Jesse inside notebooks, create notebooks inside the root of a Jesse project.'
        )

    symbol = symbol.upper()
    
    cdef long start_date, finish_date
    start_date = jh.arrow_to_timestamp(arrow.get(start_date_, 'YYYY-MM-DD'))
    finish_date = jh.arrow_to_timestamp(arrow.get(finish_date_, 'YYYY-MM-DD')) - 60000

    # validate
    if start_date == finish_date:
        raise ValueError('start_date and finish_date cannot be the same.')
    if start_date > finish_date:
        raise ValueError('start_date cannot be bigger than finish_date.')
    if finish_date > arrow.utcnow().int_timestamp * 1000:
        raise ValueError('Can\'t backtest the future!')

    # fetch from database
    candles_tuple = Candle.select(
        Candle.timestamp, Candle.open, Candle.close, Candle.high, Candle.low,
        Candle.volume
    ).where(
        Candle.timestamp.between(start_date, finish_date),
        Candle.exchange == exchange,
        Candle.symbol == symbol).order_by(Candle.timestamp.asc()).tuples()

    candles = np.array(tuple(candles_tuple))

    # validate that there are enough candles for selected period
    if len(candles) == 0 or candles[-1][0] != finish_date or candles[0][0] != start_date:
        raise CandleNotFoundInDatabase(f'Not enough candles for {symbol}. Try importing candles first.')

    if timeframe == '1m':
        return candles

    generated_candles = []
    for i in range(len(candles)):
        num = jh.timeframe_to_one_minutes(timeframe)

        if (i + 1) % num == 0:
            generated_candles.append(
                generate_candle_from_one_minutes(
                    timeframe,
                    candles[(i - (num - 1)):(i + 1)],
                    True
                )
            )

    return np.array(generated_candles)


def store_candles(candles: np.ndarray, exchange: str, symbol: str) -> None:
    """
    Stores candles in the database. The stored data can later be used for being fetched again via get_candles or even for running backtests on them.
    A common use case for this function is for importing candles from a CSV file so you can later use them for backtesting.
    """
    from jesse.services.db import store_candles as store_candles_from_list
    import jesse.helpers as jh

    # check if .env file exists
    if not jh.is_jesse_project():
        raise FileNotFoundError(
            'Invalid directory: ".env" file not found. To use Jesse inside notebooks, create notebooks inside the root of a Jesse project.'
        )

    # TODO: add validation for timeframe to make sure it's `1m`

    arr = [{
            'id': uuid4(),
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': c[0],
            'open': c[1],
            'close': c[2],
            'high': c[3],
            'low': c[4],
            'volume': c[5]
        } for c in candles]

    store_candles_from_list(arr)


def candlestick_chart(candles: np.ndarray):
    """
    Displays a candlestick chart from the numpy array
    """
    import mplfinance as mpf
    df = utils.numpy_candles_to_dataframe(candles)
    mpf.plot(df, type='candle')


def fake_candle(attributes: dict = None, reset: bool = False) -> np.ndarray:
    """
    Generates a fake candle.
    """
    return factories.fake_candle(attributes, reset)


def fake_range_candles(count: int) -> np.ndarray:
    """
    Generates a range of candles with random values.
    """
    return factories.range_candles(count)


def candles_from_close_prices(prices: Union[list, range]) -> np.ndarray:
    """
    Generates a range of candles from a list of close prices.
    The first candle has the timestamp of "2021-01-01T00:00:00+00:00"
    """
    return factories.candles_from_close_prices(prices)