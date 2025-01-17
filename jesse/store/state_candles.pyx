#cython: wraparound=True

import numpy as np 
cimport numpy as np 
from numpy cimport ndarray as ar 
cimport cython
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
import jesse.helpers as jh
from libc.math cimport fmax, fmin, abs
import jesse.store.state_positions 
import jesse.store.state_orders
from jesse.config import config
from jesse.enums import timeframes
from jesse.exceptions import RouteNotFound
# from jesse.libs import DynamicNumpyArray
from jesse.models import store_candle_into_db
from jesse.services.candle import generate_candle_from_one_minutes
from timeloop import Timeloop
from datetime import timedelta
from jesse.services import logger
from libc.time cimport time,time_t
from jesse.enums import order_statuses

cdef class CandlesState:
    cdef dict storage, initiated_pairs
    cdef bint are_all_initiated
    def __init__(self,dict storage = {}, dict initiated_pairs = {}, bint are_all_initiated=False) -> None:
        self.storage = storage
        self.are_all_initiated = are_all_initiated
        self.initiated_pairs = initiated_pairs

    def generate_new_candles_loop(self) -> None:
        """
        to prevent the issue of missing candles when no volume is traded on the live exchange
        """
        t = Timeloop()

        @t.job(interval=timedelta(seconds=1))
        def time_loop_per_second():
            # make sure all candles are already initiated
            if not self.are_all_initiated:
                return

            # only at first second on each minute
            if jh.now(True) % 60_000 != 1000:
                return

            for c in config['app']['considering_candles']:
                exchange, symbol = c[0], c[1]
                current_candle = self.get_current_candle(exchange, symbol, '1m')

                # fix for a bug
                if current_candle[0] <= 60_000:
                    continue

                if jh.now() >= current_candle[0] + 60_000:
                    new_candle = self._generate_empty_candle_from_previous_candle(current_candle)
                    self.add_one_candle(new_candle, exchange, symbol, '1m')

        t.start()

    @staticmethod
    @cython.wraparound(False)
    def _generate_empty_candle_from_previous_candle(previous_candle: np.ndarray) -> np.ndarray:
        new_candle = previous_candle.copy()
        new_candle[0] = previous_candle[0] + 60_000

        # new candle's open, close, high, and low all equal to previous candle's close
        new_candle[1] = previous_candle[2]
        new_candle[2] = previous_candle[2]
        new_candle[3] = previous_candle[2]
        new_candle[4] = previous_candle[2]
        # set volume to 0
        new_candle[5] = 0
        return new_candle

    @cython.wraparound(False)
    def mark_all_as_initiated(self) -> None:
        for k in self.initiated_pairs:
            self.initiated_pairs[k] = True
        self.are_all_initiated = True
        
    @cython.wraparound(False)
    def get_storage(self, exchange: str, symbol: str, timeframe: str) -> DynamicNumpyArray:
        key = f'{exchange}-{symbol}-{timeframe}' 

        try:
            return self.storage[key]
        except KeyError:
            raise RouteNotFound(
                f"Bellow route is required but missing in your routes:\n('{exchange}', '{symbol}', '{timeframe}')"
            )
            
    @cython.wraparound(False)
    def init_storage(self, bucket_size: int = 1000) -> None:
        cdef int total_bigger_timeframe
        for c in config['app']['considering_candles']:
            exchange, symbol = c[0], c[1]

            # initiate the '1m' timeframes
            key = f'{exchange}-{symbol}-{timeframes.MINUTE_1}'
            self.storage[key] = DynamicNumpyArray((bucket_size, 6))

            for timeframe in config['app']['considering_timeframes']:
                key = f'{exchange}-{symbol}-{timeframe}'
                # ex: 1440 / 60 + 1 (reserve one for forming candle)
                total_bigger_timeframe = int((bucket_size / jh.timeframe_to_one_minutes(timeframe)) + 1)
                self.storage[key] = DynamicNumpyArray((total_bigger_timeframe, 6))

    def add_candle(
            self,
            candle,
            exchange: str,
            symbol: str,
            timeframe: str,
            with_execution: bool = True,
            with_generation: bool = True,
            with_skip: bool = True
    ) -> None:

        # add only 1 candle
        if len(candle.shape) == 1:
            self.add_one_candle(
                candle,
                exchange,
                symbol,
                timeframe,
                with_execution,
                with_generation,
                with_skip)

        # add only multiple candles
        elif len(candle.shape) == 2:

            self.add_multiple_candles(
                candle,
                exchange,
                symbol,
                timeframe,
                with_execution,
                with_generation,
                )
                
    def add_one_candle(
            self,
            candle,
            exchange: str,
            symbol: str,
            timeframe: str,
            bint with_execution= True,
            bint with_generation = True,
            bint with_skip = True
    ) -> None:

        arr: DynamicNumpyArray = self.storage[f'{exchange}-{symbol}-{timeframe}']
        cdef long long old_index = arr.array[arr.index][0]
        cdef unsigned long long time_candle = candle[0] 
        

        # if it's not an initial candle, add it to the storage, if already exists, update it
        # if f'{exchange}-{symbol}' in self.initiated_pairs:
            # store_candle_into_db(exchange, symbol, candle, on_conflict='replace')
        #initial 
        if old_index == -1:
            arr.append(candle)
            
        # if it's new, add
        elif time_candle > old_index:
            arr.append(candle)
            
            # generate other timeframes
            if with_generation and timeframe == '1m':
                self.generate_bigger_timeframes(candle, exchange, symbol, with_execution)
               
            
        # if it's the last candle again, update
        elif time_candle == old_index:
            arr[-1] = candle

            # regenerate other timeframes
            if with_generation and timeframe == '1m':
                self.generate_bigger_timeframes(candle, exchange, symbol, with_execution)

        # past candles will be ignored (dropped)
        elif time_candle < old_index:
            return


    def add_multiple_candles(self,
                              candle: np.ndarray,
                              exchange: str,
                              symbol: str,
                              timeframe: str,
                              with_execution: bool = True,
                              with_generation: bool = True):

        arr: DynamicNumpyArray = self.storage[f'{exchange}-{symbol}-{timeframe}']
        # this is an array of candles
        if len(arr) == 0:
            arr.append_multiple(candle)

        # if it's new, add
        elif candle[-1,0] > arr.array[arr.index][0]:
            arr.append_multiple(candle)

            # generate other timeframes
            if with_generation and timeframe == '1m':
                self.generate_bigger_timeframes(candle, exchange, symbol, with_execution)
        else:
            print(f'new candle: {candle[-1,0]}, old candle: {arr.array[arr.index][0]}')
            raise ValueError('Try to insert list of candles into memory, but some already exist..')
            
            
    @cython.wraparound(False)
    def add_candle_from_trade(self, trade, exchange: str, symbol: str) -> None:
        """
        In few exchanges, there's no candle stream over the WS, for
        those we have to use cases the trades stream
        """
        if not jh.is_live():
            raise Exception('add_candle_from_trade() is for live modes only')

        # ignore if candle is still being initially imported
        if f'{exchange}-{symbol}' not in self.initiated_pairs:
            return

        # in some cases we might be missing the current forming candle like it is on FTX, hence
        # if that is the case, generate the current forming candle (it won't be super accurate)
        current_candle = self.get_current_candle(exchange, symbol, '1m')
        if jh.now() > current_candle[0] + 60_000:
            new_candle = self._generate_empty_candle_from_previous_candle(current_candle)
            self.add_one_candle(new_candle, exchange, symbol, '1m')

        # update position's current price
        self.update_position(exchange, symbol, trade['price'])

        current_candle = self.get_current_candle(exchange, symbol, '1m')
        new_candle = current_candle.copy()
        # close
        new_candle[2] = trade['price']
        # high
        new_candle[3] = max(new_candle[3], trade['price'])
        # low
        new_candle[4] = min(new_candle[4], trade['price'])
        # volume
        new_candle[5] += trade['volume']

        self.add_one_candle(new_candle, exchange, symbol, '1m')

    @staticmethod
    def update_position(exchange: str, symbol: str, price: float) -> None:
        # get position object
        cdef double p 
        key = f'{exchange}-{symbol}'
        p = jesse.store.state_positions.PositionsState().storage.get(key, None)

        # for extra_route candles, p == None, hence no further action is required
        if p is None:
            return

        p.current_price = price

    def generate_bigger_timeframes(self, candle: np.ndarray, exchange: str, symbol: str, with_execution: bool) -> None:
        cdef int number_of_candles, generate_from_count
        if not jh.is_live():
            return

        for timeframe in config['app']['considering_timeframes']:
            # skip '1m'
            if timeframe == '1m':
                continue

            last_candle = self.get_current_candle(exchange, symbol, timeframe)
            generate_from_count = int((candle[0] - last_candle[0]) / 60_000)
            number_of_candles = len(self.get_candles(exchange, symbol, '1m'))
            short_candles = self.get_candles(exchange, symbol, '1m')[-1 - generate_from_count:]

            if generate_from_count < 0:
                current_1m = self.get_current_candle(exchange, symbol, '1m')
                raise ValueError(
                    f'generate_from_count cannot be negative! '
                    f'generate_from_count:{generate_from_count}, candle[0]:{candle[0]}, '
                    f'last_candle[0]:{last_candle[0]}, current_1m:{current_1m[0]}, number_of_candles:{number_of_candles}')

            if len(short_candles) == 0:
                raise ValueError(
                    f'No candles were passed. More info:'
                    f'\nexchange:{exchange}, symbol:{symbol}, timeframe:{timeframe}, generate_from_count:{generate_from_count}'
                    f'\nlast_candle\'s timestamp: {last_candle[0]}'
                    f'\ncurrent timestamp: {jh.now()}'
                )

            # update latest candle
            generated_candle = generate_candle_from_one_minutes(
                timeframe,
                short_candles,
                accept_forming_candles=True
            )

            self.add_one_candle(generated_candle, exchange, symbol, timeframe, with_execution, with_generation=False)

    @cython.wraparound(False)
    def simulate_order_execution(self, exchange: str, symbol: str, timeframe: str, new_candle: np.ndarray) -> None:
        cdef dict orders
        previous_candle = self.get_current_candle(exchange, symbol, timeframe)
        orders = jesse.store.state_orders.OrdersState().get_orders(exchange, symbol) 

        if previous_candle[2] == new_candle[2]:
            return

        for o in orders:
            # skip inactive orders
            if not o.status == order_statuses.ACTIVE:
                continue

            if ((o.price >= previous_candle[2]) and (o.price <= new_candle[2])) or (
                    (o.price <= previous_candle[2]) and (o.price >= new_candle[2])):
                o.execute()
                
    @cython.wraparound(False)
    def batch_add_candle(self, candles: np.ndarray, exchange: str, symbol: str, timeframe: str,
                         with_generation: bool = True) -> None:
        for c in candles:
            self.add_one_candle(c, exchange, symbol, timeframe, with_execution=False, with_generation=with_generation, with_skip=False)

    @cython.wraparound(False)
    def forming_estimation(self, exchange: str, symbol: str, timeframe: str) -> tuple:
        cdef int current_1m_count, dif, required_1m_to_complete_count
        long_key = f'{exchange}-{symbol}-{timeframe}'
        short_key =  f'{exchange}-{symbol}-{"1m"}'
        required_1m_to_complete_count = jh.timeframe_to_one_minutes(timeframe)
        current_1m_count = len(self.storage[f'{exchange}-{symbol}-{"1m"}'])

        dif = current_1m_count % required_1m_to_complete_count
        return dif, long_key, short_key

    # # # # # # # # #
    # # # # # getters
    # # # # # # # # #
    def get_candles(self, exchange: str, symbol: str, timeframe: str) -> np.ndarray:
        cdef int long_count, short_count, dif 
        # no need to worry for forming candles when timeframe == 1m
        if timeframe == '1m':
            arr: c_DynamicNumpyArray = self.storage[f'{exchange}-{symbol}-{"1m"}' ]
            if len(arr) > -1:
                return arr.array[0:arr.index+1] 
            else:
                return np.zeros((0, 6))

        # other timeframes
        dif, long_key, short_key = self.forming_estimation(exchange, symbol, timeframe)
        long_count = len(self.storage[f'{exchange}-{symbol}-{timeframe}'])
        short_count = len(self.storage[f'{exchange}-{symbol}-{"1m"}'])

        if dif == 0 and long_count == 0:
            return np.zeros((0, 6))

        # complete candle
        if dif == 0 or self.storage[long_key].getslice(0,long_count)[-1][0] == self.storage[short_key].array[short_count-dif][0]:
            return self.storage[long_key].getslice(0,long_count)
        # generate forming
        else:
            return np.concatenate(
                (
                    self.storage[long_key].getslice(0,long_count),
                    np.array(
                        (
                            generate_candle_from_one_minutes(
                                timeframe,
                                self.storage[short_key].getslice(short_count-dif,short_count),
                                True
                            ),
                        )
                    )
                ), axis=0
            )

    def get_current_candle(self, exchange: str, symbol: str, timeframe: str) -> np.ndarray:
        cdef int long_count, short_count, dif
        # no need to worry for forming candles when timeframe == 1m
        if timeframe == '1m':
            arr: c_DynamicNumpyArray = self.storage[f'{exchange}-{symbol}-{"1m"}']
            if len(arr) > 0:
                return arr.array[arr.index]
            else:
                return np.zeros((0, 6))

        # other timeframes
        dif, long_key, short_key = self.forming_estimation(exchange, symbol, timeframe)
        long_count = len(self.storage[f'{exchange}-{symbol}-{timeframe}'])
        short_count = len(self.storage[f'{exchange}-{symbol}-{"1m"}'])

        # complete candle
        if dif != 0:
            return generate_candle_from_one_minutes(
                timeframe, self.storage[short_key].getslice(short_count-dif,short_count),
                True
            )
        if long_count != 0:
            return self.storage[long_key].array[self.storage[long_key].index]
        else:
            return np.zeros((0, 6))

class DynamicNumpyArray:
    def __init__(self, shape: tuple,int index = -1, attributes: dict = None):
        self.index = index
        self.array = np.zeros((shape),dtype=DTYPE)
        self.bucket_size = shape[0]
        self.shape = shape

    def __len__(self) -> int:
        return self.index + 1
     
    def getslice(self,int start = 0, int stop =0):
        stop = self.index+1 if stop == 0 else stop 
        return self.array[start:stop] 
        
    def __setitem__(self, int i, ar item):
        self.array[self.index] = item
        
    def append(self, ar item) -> None:
        self.index += 1
        cdef ar new_bucket
        cdef Py_ssize_t index = self.index 
        cdef int bucket_size = self.bucket_size
        if index != 0 and (index + 1) % bucket_size == 0:
            new_bucket = np.zeros((bucket_size,6),dtype=DTYPE)
            self.array = np.concatenate((self.array, new_bucket), axis=0, dtype=DTYPE)
        self.array[index] = item
        
    def append_multiple(self, double [:,::1] items) -> None:
        self.index += 1
        cdef Py_ssize_t new_index = self.index 
        cdef Py_ssize_t items_shape = items.shape[0]
        while items_shape > self.array.shape[0] - new_index:
            new_bucket = np.zeros(self.shape)
            self.array = np.concatenate((self.array, new_bucket), axis=0)

        self.array[new_index: new_index + items_shape] = items
        self.index += items_shape - 1
        
    def __getitem__(self, i):
        stop = self.index + 1
        return self.array[i.start:stop]
