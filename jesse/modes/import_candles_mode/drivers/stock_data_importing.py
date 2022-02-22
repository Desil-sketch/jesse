from polygon.rest.client import RESTClient
# from local_settings import polygon as settings
from datetime import date, datetime
from typing import Any, Optional
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

markets = ['crypto', 'stocks', 'fx']

#python stock_data_importing.py

class MyRESTClient(RESTClient):
    def __init__(self, auth_key: str=['api_key'], timeout:int=5):
        super().__init__(auth_key)
        retry_strategy = Retry(total=10,
                               backoff_factor=10,
                               status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount('https://', adapter)

    def get_tickers(self, market:str=None) -> pd.DataFrame:
        if not market in markets:
            raise Exception(f'Market must be one of {markets}.')

        resp = self.reference_tickers_v3(market=market)
        if hasattr(resp, 'results'):
            df = pd.DataFrame(resp.results)

            while hasattr(resp, 'next_url'):
                resp = self.reference_tickers_v3(next_url=resp.next_url)
                df = df.append(pd.DataFrame(resp.results))

            # if market == 'crypto':
                # Only use USD pairings.
            df = df[df['currency_symbol'] == 'USD']
            df['name'] = df['base_currency_name']
            df = df[['ticker', 'name', 'market', 'active']]

            df = df.drop_duplicates(subset='ticker')
            return df
        return None  

    def get_bars(self, market:str=None, ticker:str=None, multiplier:int=1,
                 timespan:str='minute', from_:date=None, to:date=None) -> pd.DataFrame:
                 
        payload = {
            'unadjusted': 'false',
            'sort': 'asc',
            'limit': 50000,
        }
        
        if not market in markets:
            raise Exception(f'Market must be one of {markets}.')

        if ticker is None:
            raise Exception('Ticker must not be None.')

        from_ = from_ if from_ else date(2000,1,1)
        to = to if to else date.today()

        if market == 'stocks':
            resp = self.stocks_equities_aggregates(ticker, multiplier, timespan,
                                          from_.strftime('%Y-%m-%d'), to.strftime('%Y-%m-%d'),
                                          **payload)
            df = pd.DataFrame(resp.results)
            last_minute = 0
            while resp.results[-1]['t'] > last_minute:
                last_minute = resp.results[-1]['t'] # Last minute in response
                last_minute_date = datetime.fromtimestamp(last_minute/1000).strftime('%Y-%m-%d')
                resp = self.stocks_equities_aggregates(ticker, multiplier, timespan,
                                          last_minute_date, to.strftime('%Y-%m-%d'),
                                          **payload)
                new_bars = pd.DataFrame(resp.results)
                df = df.append(new_bars[new_bars['t'] > last_minute])
                
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df['date'] = df['date'].astype('int64')//1e9
            df = df.rename(columns={'o':'open',
                                    'c':'close',
                                    'h':'high',
                                    'l':'low',
                                    'v':'volume',
                                    'vw':'vwap',
                                    'n':'transactions'})
            df = df[['date','open','close','high','low','volume']]

            return df
        # return None

start = datetime(2020,1,1)
client = MyRESTClient(auth_key = '_ivaT5WLTLgpsNksJFCM2g5JLTcJiebV')
ticker = 'AAPL'
df = client.get_bars(market='stocks',ticker=ticker,from_=start)
df = df.to_numpy()
# print(df[:10])
# print(df[:-10])
# df.to_csv(f'stock_data/{ticker}.csv', sep=',', index=False)
"""
# Imports
import pandas as pd
from sqlalchemy import create_engine

# This CSV doesn't have a header so pass
# column names as an argument
columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

# Load in the data
df = pd.read_csv(
    "polygon.csv",
    names=columns
)

# Instantiate sqlachemy.create_engine object
engine = create_engine('postgresql://jesse_user:password@localhost:5432/jesse_db')

# Save the data from dataframe to
# postgres table "candle"
df.to_sql(
    'candle', 
    engine,
    index=False # Not copying over the index
)
"""