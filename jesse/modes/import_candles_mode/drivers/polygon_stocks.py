from polygon.rest.client import RESTClient
from requests import HTTPError
from .stock_data_importing import MyRESTClient
import jesse.helpers as jh
from .interface import CandleExchange
import datetime

class Polygon_Stocks(CandleExchange):

    def __init__(self):
        super().__init__('Polygon', 5000, 0.01, stock_mode=True,backup_exchange_class=None)
        try:
            api_key = jh.get_config('env.exchanges.Polygon.api_key')
        except:
            raise ValueError("Polygon api_key missing in config.py")

        self.restclient = RESTClient(api_key)
        
    def init_backup_exchange(self):
        self.backup_exchange = None

    def get_starting_time(self, symbol):

        return None

    def fetch(self, symbol, start_timestamp):

        base = jh.base_asset(symbol)
        # Check if symbol exists. Raises HTTP 404 if it doesn't.
        try:
            details = self.restclient.reference_ticker_details(base)
        except HTTPError:
            raise ValueError("Symbol ({}) probably doesn't exist.".format(base))

        payload = {
            'unadjusted': 'false',
            'sort': 'asc',
            'limit': self.count,
        }

        # Polygon takes string dates not timestamps
        start = jh.timestamp_to_date(start_timestamp)
        end = jh.timestamp_to_date(start_timestamp + (self.count) * 60000)
        response = self.restclient.stocks_equities_aggregates(base, 1, 'minute', start, end, **payload)

        data = response.results

        candles = []

        for d in data:
            candles.append({
                'id': jh.generate_unique_id(),
                'symbol': symbol,
                'exchange': self.name,
                'timestamp': int(d['t']),
                'open': float(d['o']),
                'close': float(d['c']),
                'high': float(d['h']),
                'low': float(d['l']),
                'volume': int(d['v'])
            })

        return candles