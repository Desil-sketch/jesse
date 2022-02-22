#cython:wraparound=True
#cython:boundscheck=False

import time
from typing import Dict, Union, List

import arrow
import click
import numpy as np
cimport numpy as np 
import sys
from libc.math cimport fmin,fmax, NAN, abs, isnan, NAN
from numpy.math cimport INFINITY
import talib
cimport cython
np.import_array()
FTYPE = np.float64
ctypedef np.float64_t FTYPE_t
ctypedef double dtype_t
import pandas as pd
import random
def uuid4():
  s = '%032x' % random.getrandbits(128)
  return s[0:8]+'-'+s[8:12]+'-4'+s[13:16]+'-'+s[16:20]+'-'+s[20:32]
import jesse.helpers as jh
import jesse.services.metrics as stats
import jesse.services.required_candles as required_candles
# import jesse.services.selectors as selectors
from jesse import exceptions
from jesse.config import config
from jesse.enums import timeframes, order_types, order_roles, order_flags
from jesse.models import Candle, Order, Position
from jesse.modes.utils import save_daily_portfolio_balance
from jesse.routes import router
from jesse.services import charts
from jesse.services import quantstats
from jesse.services import report
from jesse.services.cache import cache
from jesse.services.candle import print_candle, candle_includes_price, split_candle
from jesse.services.candle import generate_candle_from_one_minutes
from jesse.services.numba_functions import monte_carlo_candles
from jesse.services.file import store_logs
from jesse.services.validators import validate_routes
from jesse.store import store
from jesse.services import logger
from jesse.services.failure import register_custom_exception_handler
from jesse.services.redis import sync_publish, process_status
from timeloop import Timeloop
from datetime import timedelta
from jesse.services.progressbar import Progressbar
from jesse.enums import order_statuses

#get_fixed jump candle is disabled 

def run(
        debug_mode,
        user_config: dict,
        routes: List[Dict[str, str]],
        extra_routes: List[Dict[str, str]],
        start_date: str,
        finish_date: str,
        candles: dict = None,
        chart: bool = False,
        tradingview: bool = False,
        full_reports: bool = False,
        csv: bool = False,
        json: bool = False
) -> None:
    if not jh.is_unit_testing():
        # at every second, we check to see if it's time to execute stuff
        status_checker = Timeloop()
        @status_checker.job(interval=timedelta(seconds=1))
        def handle_time():
            if process_status() != 'started':
                raise exceptions.Termination
        status_checker.start()

    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
           
    cdef list change,data
    cdef int routes_count, index
    # cdef float price_pct_change, bh_daily_returns_all_routes
    from jesse.config import config, set_config
    config['app']['trading_mode'] = 'backtest'

    # debug flag
    config['app']['debug_mode'] = debug_mode

    # inject config
    if not jh.is_unit_testing():
        set_config(user_config)

    # set routes
    router.initiate(routes, extra_routes)

    store.app.set_session_id()

    register_custom_exception_handler()

    # clear the screen
    if not jh.should_execute_silently():
        click.clear()

    # validate routes
    validate_routes(router)

    # initiate candle store
    store.candles.init_storage(500000)

    # load historical candles
    if candles is None:
        candles = load_candles(start_date, finish_date)
        click.clear()
        
    if not jh.should_execute_silently():
        sync_publish('general_info', {
            'session_id': jh.get_session_id(),
            'debug_mode': str(config['app']['debug_mode']),
        })

        # candles info
        key = f"{config['app']['considering_candles'][0][0]}-{config['app']['considering_candles'][0][1]}"
        sync_publish('candles_info', stats.candles_info(candles[key]['candles']))

        # routes info
        sync_publish('routes_info', stats.routes(router.routes))

    
    # run backtest simulation
    simulator(candles, run_silently=jh.should_execute_silently())

    # hyperparameters (if any)
    if not jh.should_execute_silently():
        sync_publish('hyperparameters', stats.hyperparameters(router.routes))
    
    if not jh.should_execute_silently():
        if store.completed_trades.count > 0:
            sync_publish('metrics', report.portfolio_metrics())

            routes_count = len(router.routes)
            more = f"-and-{routes_count - 1}-more" if routes_count > 1 else ""
            study_name = f"{router.routes[0].strategy_name}-{router.routes[0].exchange}-{router.routes[0].symbol}-{router.routes[0].timeframe}{more}-{start_date}-{finish_date}"
            store_logs(study_name, json, tradingview, csv)

            if chart:
                charts.portfolio_vs_asset_returns(study_name)

            sync_publish('equity_curve', charts.equity_curve())

            # QuantStats' report
            if full_reports:
                price_data = []
                # load close candles for Buy and hold and calculate pct_change
                for index, c in enumerate(config['app']['considering_candles']):
                    exchange, symbol = c[0], c[1]
                    if exchange in config['app']['trading_exchanges'] and symbol in config['app']['trading_symbols']:
                        # fetch from database
                        candles_tuple = Candle.select(
                            Candle.timestamp, Candle.close
                        ).where(
                            Candle.timestamp.between(jh.date_to_timestamp(start_date),
                                                     jh.date_to_timestamp(finish_date) - 60000),
                            Candle.exchange == exchange,
                            Candle.symbol == symbol
                        ).order_by(Candle.timestamp.asc()).tuples()

                        candles = np.array(candles_tuple)

                        timestamps = candles[:, 0]
                        price_data.append(candles[:, 1])

                price_data = np.transpose(price_data)
                price_df = pd.DataFrame(price_data, index=pd.to_datetime(timestamps, unit="ms"), dtype=float).resample(
                    'D').mean()
                price_pct_change = price_df.pct_change(1).fillna(0)
                bh_daily_returns_all_routes = price_pct_change.mean(1)
                quantstats.quantstats_tearsheet(bh_daily_returns_all_routes, study_name)
        else:
            sync_publish('equity_curve', None)
            sync_publish('metrics', None)

    # profiler.disable()
    # pr_stats = pstats.Stats(profiler).sort_stats('tottime')
    # pr_stats.print_stats(50)
    
    # close database connection
    from jesse.services.db import database
    database.close_connection()

@cython.wraparound(True)
def load_candles(start_date_str: str, finish_date_str: str) -> Dict[str, Dict[str, Union[str, np.ndarray]]]:
    cdef long start_date, finish_date, 
    cdef double required_candles_count
    cdef bint from_db
    cdef dict candles
    start_date = jh.date_to_timestamp(start_date_str)
    finish_date = jh.date_to_timestamp(finish_date_str) - 60000

    # validate
    if start_date == finish_date:
        raise ValueError('start_date and finish_date cannot be the same.')
    if start_date > finish_date:
        raise ValueError('start_date cannot be bigger than finish_date.')
    if finish_date > arrow.utcnow().int_timestamp * 1000:
        raise ValueError(
            "Can't load candle data from the future! The finish-date can be up to yesterday's date at most.")

    # load and add required warm-up candles for backtest
    if jh.is_backtesting():
        for c in config['app']['considering_candles']:
            required_candles.inject_required_candles_to_store(
                required_candles.load_required_candles(c[0], c[1], start_date_str, finish_date_str),
                c[0],
                c[1]
            )

    # download candles for the duration of the backtest
    candles = {}
    for c in config['app']['considering_candles']:
        exchange, symbol = c[0], c[1]

        from_db = False
        key =  f'{exchange}-{symbol}'

        cache_key = f"{start_date_str}-{finish_date_str}-{key}"
        if jh.get_config('env.caching.recycle'):
            print('Recycling enabled!')
            cached_value = cache.slice_pickles(cache_key, start_date_str, finish_date_str, key)
        else:
            cached_value = cache.get_value(cache_key)
            print('Recycling disabled, falling back to vanilla driver!')
        if cached_value:
            candles_tuple = cached_value
        # if cache exists use cache_value
        # not cached, get and cache for later calls in the next 5 minutes
        # fetch from database
        else:
            candles_tuple = cached_value or Candle.select(
                    Candle.timestamp, Candle.open, Candle.close, Candle.high, Candle.low,
                    Candle.volume
                ).where(
                    Candle.timestamp.between(start_date, finish_date),
                    Candle.exchange == exchange,
                    Candle.symbol == symbol
                ).order_by(Candle.timestamp.asc()).tuples()
            from_db = True
        # validate that there are enough candles for selected period
        required_candles_count = (finish_date - start_date) / 60_000
        if len(candles_tuple) == 0 or candles_tuple[-1][0] != finish_date or candles_tuple[0][0] != start_date:
            raise exceptions.CandleNotFoundInDatabase(
                f'Not enough candles for {symbol}. You need to import candles.'
            )
        elif len(candles_tuple) != required_candles_count + 1:
            raise exceptions.CandleNotFoundInDatabase(
                f'There are missing candles between {start_date_str} => {finish_date_str}')

        # cache it for near future calls
        if from_db:
            cache.set_value(cache_key, tuple(candles_tuple), expire_seconds=60 * 60 * 24 * 7)

        candles[key] = {
            'exchange': exchange,
            'symbol': symbol,
            'candles': np.array(candles_tuple)
        }

    return candles

def simulator(*args, **kwargs):	
    if jh.get_config('env.simulation.skip'):
        skip_simulator(*args, **kwargs)
    else:
        iterative_simulator(*args, **kwargs)	
    

def iterative_simulator(
        candles: dict, run_silently: bool, hyperparameters = None
) -> None:
    cdef Py_ssize_t length, count
    cdef double indicator1_f, indicator2_f
    cdef double [::1] indicator1_array, indicator2_array
    cdef bint precalc_bool
    cdef dict indicator1_storage, indicator2_storage
    begin_time_track = time.time()
    key = f"{config['app']['considering_candles'][0][0]}-{config['app']['considering_candles'][0][1]}"
    first_candles_set = candles[key]['candles']
    length = len(first_candles_set)
    # to preset the array size for performance
    try:
        store.app.starting_time = first_candles_set[0][0]
    except IndexError:
        raise IndexError('Check your "warm_up_candles" config value')
    store.app.time = first_candles_set[0][0]

    if jh.get_config('env.simulation.Montecarlo'):
        for j in candles:

            # candles[j]['candles'][:, 1], candles[j]['candles'][:, 2], candles[j]['candles'][:, 3], candles[j]['candles'][:, 4] = monte_carlo_candles(candles[j]['candles'][:])
            candles[j]['candles'][:, 1] = monte_carlo_candles(candles[j]['candles'][:, 1])
            candles[j]['candles'][:, 2] = monte_carlo_candles(candles[j]['candles'][:, 2])
            candles[j]['candles'][:, 3] = monte_carlo_candles(candles[j]['candles'][:, 3])
            candles[j]['candles'][:, 4] = monte_carlo_candles(candles[j]['candles'][:, 4])
            # candles[j]['candles'][:, 5] = monte_carlo_candles(candles[j]['candles'][:, 5])

    for r in router.routes:
        # if the r.strategy is str read it from file
        if isinstance(r.strategy_name, str):
            StrategyClass = jh.get_strategy_class(r.strategy_name)
        # else it is a class object so just use it
        else:
            StrategyClass = r.strategy_name

        try:
            r.strategy = StrategyClass()
        except TypeError:
            raise exceptions.InvalidStrategy(
                "Looks like the structure of your strategy directory is incorrect. Make sure to include the strategy INSIDE the __init__.py file."
                "\nIf you need working examples, check out: https://github.com/jesse-ai/example-strategies"
            )
        except:
            raise

        r.strategy.name = r.strategy_name
        r.strategy.exchange = r.exchange
        r.strategy.symbol = r.symbol
        r.strategy.timeframe = r.timeframe

        # read the dna from strategy's dna() and use it for injecting inject hyperparameters
        # first convert DNS string into hyperparameters
        if len(r.strategy.dna()) > 0 and hyperparameters is None:
            hyperparameters = jh.dna_to_hp(r.strategy.hyperparameters(), r.strategy.dna())

        # inject hyperparameters sent within the optimize mode
        if hyperparameters is not None:
            r.strategy.hp = hyperparameters

        # init few objects that couldn't be initiated in Strategy __init__
        # it also injects hyperparameters into self.hp in case the route does not uses any DNAs
        r.strategy._init_objects()
        key = f'{r.exchange}-{r.symbol}'
        store.positions.storage.get(key,None).strategy = r.strategy

    # add initial balance
    save_daily_portfolio_balance()
    cdef Py_ssize_t i
    dic = {
        timeframes.MINUTE_1: 1,
        timeframes.MINUTE_3: 3,
        timeframes.MINUTE_5: 5,
        timeframes.MINUTE_15: 15,
        timeframes.MINUTE_30: 30,
        timeframes.MINUTE_45: 45,
        timeframes.HOUR_1: 60,
        timeframes.HOUR_2: 60 * 2,
        timeframes.HOUR_3: 60 * 3,
        timeframes.HOUR_4: 60 * 4,
        timeframes.HOUR_6: 60 * 6,
        timeframes.HOUR_8: 60 * 8,
        timeframes.HOUR_12: 60 * 12,
        timeframes.DAY_1: 60 * 24,
    }
    
    progressbar = Progressbar(length, step=60)
    if jh.get_config('env.simulation.precalculation'):
        indicator1_storage,indicator2_storage = indicator_precalculation(candles,first_candles_set,store.positions.storage.get(key,None).strategy, False)
        precalc_bool = True
    else:
        precalc_bool = False
        indicator1 = None
        indicator2 = None
    for i in range(length):
        # update time
        store.app.time = first_candles_set[i][0] + 60_000
        # add candles
        for j in candles:
            short_candle = candles[j]['candles'][i]
            # if i != 0:
                # previous_short_candle = candles[j]['candles'][i - 1]
                # if previous_short_candle[2] < short_candle[1]:
                    # short_candle[1] = previous_short_candle[2]
                    # short_candle[4] = fmin(previous_short_candle[2], short_candle[4])
                # elif previous_short_candle[2] > short_candle[1]:
                    # short_candle[1] = previous_short_candle[2]
                    # short_candle[3] = fmax(previous_short_candle[2], short_candle[3])
                # short_candle = short_candle
            exchange = candles[j]['exchange']
            symbol = candles[j]['symbol']

            store.candles.add_one_candle(short_candle, exchange, symbol, '1m', with_execution=False,
                                     with_generation=False)

            # print short candle
            # if jh.is_debuggable('shorter_period_candles'):
                # print_candle(short_candle, True, symbol)

            _simulate_price_change_effect(short_candle, exchange, symbol)

            # generate and add candles for bigger timeframes
            for timeframe in config['app']['considering_timeframes']:
                # for 1m, no work is needed
                if timeframe == '1m':
                    continue

                count = dic[timeframe]
                # until = count - ((i + 1) % count)

                if (i + 1) % count == 0:
                    generated_candle = generate_candle_from_one_minutes(
                        timeframe,
                        candles[j]['candles'][(i - (count - 1)):(i + 1)])
                    store.candles.add_one_candle(generated_candle, exchange, symbol, timeframe, with_execution=False,
                                             with_generation=False)

        # update progressbar
        if not run_silently and i % 60 == 0:
            progressbar.update()
            sync_publish('progressbar', {
                'current': progressbar.current,
                'estimated_remaining_seconds': progressbar.estimated_remaining_seconds
            })

        # now that all new generated candles are ready, execute
        for r in router.routes:
            if precalc_bool:
                indicator_key = f'{r.exchange}-{r.symbol}-{r.timeframe}'
                indicator1_array = indicator1_storage[indicator_key]['array']
                indicator2_array = indicator2_storage[indicator_key]['array']
            count = dic[r.timeframe]
            # 1m timeframe
            if r.timeframe == timeframes.MINUTE_1:
                if precalc_bool:
                    indicator1_f = indicator1_array[i+1]
                    indicator2_f = indicator2_array[i+1]
                r.strategy._execute(indicator1_f,indicator2_f,precalc_bool)
            elif (i + 1) % count == 0:
                if precalc_bool:
                    indicator1_f = indicator1_array[(i/count)+1]
                    indicator2_f = indicator2_array[(i/count)+1]
                # print candle
                # if jh.is_debuggable('trading_candles'):
                    # print_candle(store.candles.get_current_candle(r.exchange, r.symbol, r.timeframe), False,
                                 # r.symbol)
                r.strategy._execute(indicator1_f,indicator2_f,precalc_bool)

        # now check to see if there's any MARKET orders waiting to be executed
        store.orders.execute_pending_market_orders()

        if i != 0 and i % 1440 == 0:
            save_daily_portfolio_balance()

    if not run_silently:
        # print executed time for the backtest session
        finish_time_track = time.time()
        sync_publish('alert', {
            'message': f'Successfully executed backtest simulation in: {round(finish_time_track - begin_time_track, 2)} seconds',
            'type': 'success'
        })

    for r in router.routes:
        r.strategy._terminate()
        store.orders.execute_pending_market_orders()

    # now that backtest is finished, add finishing balance
    save_daily_portfolio_balance()

# @cython.boundscheck(True)
cdef (double,double,double,double,double,double) c_sum(double [:,::1] array, Py_ssize_t rows) nogil:  
    cdef Py_ssize_t i 
    cdef double sum1
    cdef double min1 = INFINITY
    cdef double max1 = -INFINITY
    cdef double close1, open1, time1
    close1 = array[-1,2] if array[-1,2] == array[-1,2] else NAN
    open1 = array[0,1] if array[0,1] == array[0,1] else NAN
    time1 = array[0,0]
    # rows = len(array)
    if close1 is not NAN:
        for i in range(rows):
            sum1 = sum1 + array[i,5] 
            if array[i,4] < min1:
                min1 = array[i,4]
            if array[i,3] > max1:
                max1 = array[i,3] 
    else:
        sum1 = NAN
        min1 = NAN
        max1 = NAN
        
    return sum1, min1, max1, close1, open1, time1

def generate_candles_from_minutes(double [:,::1] first_candles_set, Py_ssize_t rows):
    sum1, min1, max1, close1, open1, time1 = c_sum(first_candles_set, rows)
    return np.array([
        time1,
        open1,
        close1,
        max1,
        min1,
        sum1,
    ])
    #52609
    #53112
    
def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]

    
def indicator_precalculation(dict candles,double [:,::1] first_candles_set,strategy, bint skip_1m):
    cdef Py_ssize_t  i, consider_timeframes, candle_prestorage_shape, index, offset, length,rows
    cdef np.ndarray candle_prestorage, partial_arrays
    cdef double [:,::1] new_candles, new_array
    cdef double [::1] indicator1,indicator2
    cdef dict indicator1_storage = {}
    cdef dict indicator2_storage = {}
    for j in candles:
        for timeframe in config['app']['considering_timeframes']:
            if timeframe == '1m' and skip_1m:
                continue
            exchange = candles[j]['exchange']
            symbol = candles[j]['symbol']
            new_candles = candles[j]['candles']
            key = f'{exchange}-{symbol}-{timeframe}'
            consider_timeframes = jh.timeframe_to_one_minutes(timeframe)
            candle_prestorage = store.candles.get_storage(exchange,symbol,"1m").array 
            candle_prestorage = trim_zeros(candle_prestorage) 
            candle_prestorage_shape = len(candle_prestorage)
            length = len(first_candles_set) + (candle_prestorage_shape)
            full_array = np.zeros((int(length/(consider_timeframes))+1,6))
            new_array = np.concatenate((candle_prestorage,new_candles),axis=0)
            partial_array = np.zeros((int(length/(consider_timeframes))+1,6))   
            index = 0
            for i in range(0,length):
                if ((i + 1) % consider_timeframes == 0):
                    partial_array[(index)] = generate_candles_from_minutes(new_array[(i - (consider_timeframes-1)):(i+1)],consider_timeframes)
                    index = index + 1 
            indicator1 = strategy.ema1(precalc_candles = partial_array)
            indicator2 = strategy.ema2(precalc_candles = partial_array)
            indicator1 = np.delete(indicator1,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
            indicator2 = np.delete(indicator2,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
            indicator1_storage[key] = {'array': indicator1 }
            indicator2_storage[key] = {'array': indicator2 } 
            
    return indicator1_storage, indicator2_storage

        
def skip_simulator(candles: dict, run_silently: bool, hyperparameters: dict = None) -> None:
    cdef Py_ssize_t i 
    cdef bint precalc_bool
    cdef dict indicator1_storage, indicator2_storage
    cdef int count, max_skip, length, min_timeframe_remainder
    cdef double [::1] indicator1_array, indicator2_array
    cdef double indicator1_f, indicator2_f
    begin_time_track = time.time()
    key = f"{config['app']['considering_candles'][0][0]}-{config['app']['considering_candles'][0][1]}"
    first_candles_set = candles[key]['candles']
    length = len(first_candles_set)
    # to preset the array size for performance
    store.app.starting_time = first_candles_set[0][0]
    store.app.time = first_candles_set[0][0]
    # initiate strategies
    min_timeframe, strategy = _initialized_strategies(hyperparameters)

    # add initial balance
    save_daily_portfolio_balance()
    
    i = min_timeframe_remainder = skip = min_timeframe
    cdef int update_dashboard = 240
    progressbar = Progressbar(length, step=min_timeframe * update_dashboard)
    # i is the i'th candle, which means that the first candle is i=1 etc..
    dic = {
        timeframes.MINUTE_1: 1,
        timeframes.MINUTE_3: 3,
        timeframes.MINUTE_5: 5,
        timeframes.MINUTE_15: 15,
        timeframes.MINUTE_30: 30,
        timeframes.MINUTE_45: 45,
        timeframes.HOUR_1: 60,
        timeframes.HOUR_2: 60 * 2,
        timeframes.HOUR_3: 60 * 3,
        timeframes.HOUR_4: 60 * 4,
        timeframes.HOUR_6: 60 * 6,
        timeframes.HOUR_8: 60 * 8,
        timeframes.HOUR_12: 60 * 12,
        timeframes.DAY_1: 60 * 24,
    }
    if jh.get_config('env.simulation.precalculation'):
        indicator1_storage,indicator2_storage = indicator_precalculation(candles,first_candles_set,strategy,True)
        precalc_bool = True
    else:
        precalc_bool = False
        indicator1 = None
        indicator2 = None
    while i <= length:
        # update time = open new candle, use i-1  because  0 < i <= length
        store.app.time = first_candles_set[i - 1][0] + 60_000

        # add candles
        for j in candles:

            short_candles = candles[j]['candles'][i - skip: i]
            # remove previous_short_candle fix
            exchange = candles[j]['exchange']
            symbol = candles[j]['symbol']

            store.candles.add_multiple_candles(short_candles, exchange, symbol, '1m', with_execution=False,
                                     with_generation=False)

            # print short candle
            # if jh.is_debuggable('shorter_period_candles'):
                # print_candle(short_candles[-1], True, symbol)

            # only to check for a limit orders in this interval, its not necessary that the short_candles is the size of
            # any timeframe candle
            current_temp_candle = generate_candle_from_one_minutes('',
                                                                   short_candles,
                                                                   accept_forming_candles=True)

            # if i - skip > 0:
                # current_temp_candle = _get_fixed_jumped_candle(candles[j]['candles'][i - skip - 1],
                                                               # current_temp_candle)
            # in this new prices update there might be an order that needs to be executed
            _simulate_price_change_effect(current_temp_candle, exchange, symbol)

            # generate and add candles for bigger timeframes
            for timeframe in config['app']['considering_timeframes']:
                # for 1m, no work is needed
                if timeframe == '1m':
                    continue

                # if timeframe is constructed by 1m candles without sync
                count = dic[timeframe]
                if count <= dic[timeframes.DAY_1]:
                    generate_new_candle = i % count == 0
                # elif timeframe == timeframes.MONTH_1:
                    # raise ValueError("1M timeframe not supported yet")
                else:
                    # if timeframe is timeframes.DAY_3 or timeframes.WEEK_1:

                    # anchor is just a random open-time candle of this timestamp to check if this candle is
                    # 3 days timestamp - 1637107200000.
                    # 1 week timestamp - 1636329600000.
                    anchor = {timeframes.DAY_3: 1637107200000,
                            timeframes.WEEK_1: 1636329600000}[timeframe]
                    generate_new_candle = abs(store.app.time - anchor) % (count * 60_000) == 0
                if generate_new_candle:
                    candles_1m = store.candles.get_storage(exchange, symbol, '1m')
                    generated_candle = generate_candle_from_one_minutes(
                        timeframe,
                        candles_1m[len(candles_1m) - count:])
                    store.candles.add_one_candle(generated_candle, exchange, symbol, timeframe, with_execution=False,
                                             with_generation=False)
                    for r in router.routes:
                        r.strategy.update_new_candle(generated_candle, exchange, symbol, timeframe)

        # update progressbar
        if not run_silently and i % (min_timeframe * update_dashboard) == 0:
            progressbar.update()
            sync_publish('progressbar', {
                'current': progressbar.current,
                'estimated_remaining_seconds': progressbar.estimated_remaining_seconds
            })

        # now that all new generated candles are ready, execute
        for r in router.routes:
            count = jh.timeframe_to_one_minutes(r.timeframe)
            if precalc_bool:
                indicator_key = f'{r.exchange}-{r.symbol}-{r.timeframe}'
                indicator1_array = indicator1_storage[indicator_key]['array']
                indicator2_array = indicator2_storage[indicator_key]['array']
            if i % count == 0:
                if precalc_bool:
                    indicator1_f = indicator1_array[i/count]
                    indicator2_f = indicator2_array[i/count]
                else:
                    indicator1_f = NAN
                    indicator2_f = NAN
                # print candle
                # if jh.is_debuggable('trading_candles'):
                    # print_candle(store.candles.get_current_candle(r.exchange, r.symbol, r.timeframe), False,
                                 # r.symbol)
                r.strategy._execute(indicator1_f,indicator2_f,precalc_bool)

        # now check to see if there's any MARKET orders waiting to be executed
        store.orders.execute_pending_market_orders()

        if i % 1440 == 0:
            save_daily_portfolio_balance()

        skip = _skip_n_candles(candles, min_timeframe_remainder, i)
        if skip < min_timeframe_remainder:
            min_timeframe_remainder -= skip
        elif skip == min_timeframe_remainder:
            min_timeframe_remainder = min_timeframe
        i += skip

    res = 0
    if not run_silently:
        # print executed time for the backtest session
        finish_time_track = time.time()
        sync_publish('alert', {
            'message': f'Successfully executed backtest simulation in: {round(finish_time_track - begin_time_track, 2)} seconds',
            'type': 'success'
        })

    for r in router.routes:
        r.strategy._terminate()
        store.orders.execute_pending_market_orders()

    # now that backtest is finished, add finishing balance
    save_daily_portfolio_balance()

def _initialized_strategies(hyperparameters: dict = None):
    for r in router.routes:
        StrategyClass = jh.get_strategy_class(r.strategy_name)

        try:
            r.strategy = StrategyClass()
        except TypeError:
            raise exceptions.InvalidStrategy(
                "Looks like the structure of your strategy directory is incorrect. "
                "Make sure to include the strategy INSIDE the __init__.py file.\n"
                "If you need working examples, check out: https://github.com/jesse-ai/example-strategies"
            )

        r.strategy.name = r.strategy_name
        r.strategy.exchange = r.exchange
        r.strategy.symbol = r.symbol
        r.strategy.timeframe = r.timeframe
        # inject hyper parameters (used for optimize_mode)
        # convert DNS string into hyperparameters
        if len(r.strategy.dna()) > 0 and hyperparameters is None:
            hyperparameters = jh.dna_to_hp(r.strategy.hyperparameters(), r.strategy.dna())

        # inject hyperparameters sent within the optimize mode
        if hyperparameters is not None:
            r.strategy.hp = hyperparameters

        # init few objects that couldn't be initiated in Strategy __init__
        # it also injects hyperparameters into self.hp in case the route does not uses any DNAs
        r.strategy._init_objects()
        key = f'{r.exchange}-{r.symbol}'
        store.positions.storage.get(key,None).strategy = r.strategy

    # search for minimum timeframe for skips
    consider_timeframes = [jh.timeframe_to_one_minutes(timeframe) for timeframe in
                           config['app']['considering_timeframes'] if timeframe != '1m']
    # smaller timeframe is dividing DAY_1 & I down want bigger timeframe to be the skipper
    # because it fast enough with 1 day + higher timeframes are better to check every day ( 1M / 1W / 3D )
    if timeframes.DAY_1 not in consider_timeframes:
        consider_timeframes.append(jh.timeframe_to_one_minutes(timeframes.DAY_1))

    # for cases where only 1m is used in this simulation
    if not consider_timeframes:
        return 1
    # take the greatest common divisor for that purpose
    return np.gcd.reduce(consider_timeframes),r.strategy


# def update_strategy_on_new_candle(candle, exchange, symbol, timeframe):
    # for r in router.routes:
        # r.strategy.update_new_candle(candle, exchange, symbol, timeframe)


cdef _execute_candles(i: int):
    for r in router.routes:
        count = jh.timeframe_to_one_minutes(r.timeframe)
        if i % count == 0:
            # print candle
            # if jh.is_debuggable('trading_candles'):
                # print_candle(store.candles.get_current_candle(r.exchange, r.symbol, r.timeframe), False,
                             # r.symbol)
            r.strategy._execute()

    # now check to see if there's any MARKET orders waiting to be executed
    store.orders.execute_pending_market_orders()


cdef _finish_simulation(begin_time_track: float, run_silently: bool):
    res = 0
    if not run_silently:
        # print executed time for the backtest session
        finish_time_track = time.time()
        sync_publish('alert', {
            'message': f'Successfully executed backtest simulation in: {round(finish_time_track - begin_time_track, 2)} seconds',
            'type': 'success'
        })

    for r in router.routes:
        r.strategy._terminate()
        store.orders.execute_pending_market_orders()

    # now that backtest is finished, add finishing balance
    save_daily_portfolio_balance()

cdef int _skip_n_candles(candles, max_skip: int, i: int):
    """
    calculate how many 1 minute candles can be skipped by checking if the next candles
    will execute limit and stop orders
    Use binary search to find an interval that only 1 or 0 orders execution is needed
    :param candles: np.ndarray - array of the whole 1 minute candles
    :max_skip: int - the interval that not matter if there is an order to be updated or not.
    :i: int - the current candle that should be executed
    :return: int - the size of the candles in minutes needs to skip
    """
    cdef int orders_counter
    while True:
        orders_counter = 0
        for r in router.routes:
            if store.orders.count_active_orders(r.exchange, r.symbol) < 2:
                continue

            orders = store.orders.get_orders(r.exchange, r.symbol)
            future_candles = candles[f'{r.exchange}-{r.symbol}']['candles']
            if i >= len(future_candles):
                # if there is a problem with i or with the candles it will raise somewhere else
                # for now it still satisfy the condition that no more than 2 orders will be execute in the next candle
                break

            current_temp_candle = generate_candle_from_one_minutes('',
                                                                   future_candles[i:i + max_skip],
                                                                   accept_forming_candles=True)

            for order in orders:
                if order.is_active() and candle_includes_price(current_temp_candle, order.price):
                    orders_counter += 1

        if orders_counter < 2 or max_skip == 1:
            # no more than 2 orders that can interfere each other in this candle.
            # or the candle is 1 minute candle, so I cant reduce it to smaller interval :/
            break

        max_skip //= 2

    return max_skip
    
def _get_fixed_jumped_candle(previous_candle: np.ndarray, candle: np.ndarray) -> np.ndarray:
    """
    A little workaround for the times that the price has jumped and the opening
    price of the current candle is not equal to the previous candle's close!

    :param previous_candle: np.ndarray
    :param candle: np.ndarray
    """
    if previous_candle[2] < candle[1]:
        candle[1] = previous_candle[2]
        candle[4] = fmin(previous_candle[2], candle[4])
    elif previous_candle[2] > candle[1]:
        candle[1] = previous_candle[2]
        candle[3] = fmax(previous_candle[2], candle[3])

    return candle


def _simulate_price_change_effect(real_candle: np.ndarray, exchange: str, symbol: str) -> None:
    cdef bint executed_order
    cdef Py_ssize_t index
    # cdef str key 
    cdef np.ndarray current_temp_candle
    cdef list orders = store.orders.storage.get(f'{exchange}-{symbol}',[])

    current_temp_candle = real_candle.copy()
    executed_order = False
    key = f'{exchange}-{symbol}'
    p = store.positions.storage.get(key, None)
    while True:
        if len(orders) == 0:
            executed_order = False
        else:
            for index, order in enumerate(orders):
                if index == len(orders) - 1 and not order.status == order_statuses.ACTIVE:
                    executed_order = False

                if not order.status == order_statuses.ACTIVE:
                    continue

                if (order.price >= current_temp_candle[4]) and (order.price <= current_temp_candle[3]): #candle_includes_price(current_temp_candle, order.price):
                    storable_temp_candle, current_temp_candle = split_candle(current_temp_candle, order.price)
                    store.candles.add_one_candle(
                        storable_temp_candle, exchange, symbol, '1m',
                        with_execution=False,
                        with_generation=False
                    )
                    # p = selectors.get_position(exchange, symbol)
                    p.current_price = storable_temp_candle[2]

                    executed_order = True

                    order.execute()

                    # break from the for loop, we'll try again inside the while
                    # loop with the new current_temp_candle
                    break
                else:
                    executed_order = False

        if not executed_order:
            # add/update the real_candle to the store so we can move on
            store.candles.add_one_candle(
                real_candle, exchange, symbol, '1m',
                with_execution=False,
                with_generation=False
            )
            # p = selectors.get_position(exchange, symbol)
            if p:
                p.current_price = real_candle[2]
            break
            
    p: Position = store.positions.storage.get(key, None)

    if not p:
        return

    # for now, we only support the isolated mode:
    if p.exchange.type == 'spot' or p.exchange.futures_leverage_mode == 'cross':
        return
        
    cdef double c_qty = p.qty
    cdef str c_type
    if c_qty == 0:
        c_liquidation_price = NAN
        c_type = 'close' 
    else:
        if c_qty > 0:   
            c_type = 'long'
            c_liquidation_price = p.entry_price * (1 - (1 / p.strategy.leverage) + 0.004)
        elif c_qty < 0:
            c_type = 'short'
            c_liquidation_price = p.entry_price * (1 + (1 / p.strategy.leverage) - 0.004)
        else:
            c_liquidation_price = NAN
            
    if (c_liquidation_price >= real_candle[4]) and (c_liquidation_price <= real_candle[3]):
        closing_order_side = jh.closing_side(c_type)

        # create the market order that is used as the liquidation order
        order = Order({
            'id':  uuid4(),
            'symbol': symbol,
            'exchange': exchange,
            'side': closing_order_side,
            'type': order_types.MARKET,
            'flag': order_flags.REDUCE_ONLY,
            'qty': jh.prepare_qty(p.qty, closing_order_side),
            'price': p.bankruptcy_price,
            'role': order_roles.CLOSE_POSITION
        })

        store.orders.add_order(order)

        store.app.total_liquidations += 1

        logger.info(f'{p.symbol} liquidated at {p.liquidation_price}')

        order.execute()
            

    # _check_for_liquidations(real_candle, exchange, symbol)


def _check_for_liquidations(candle: np.ndarray, exchange: str, symbol: str) -> None:
    key = f'{exchange}-{symbol}'
    p: Position = store.positions.storage.get(key, None)

    if not p:
        return

    # for now, we only support the isolated mode:
    if p.mode != 'isolated':
        return

    if candle_includes_price(candle, p.liquidation_price):
        closing_order_side = jh.closing_side(p.type)

        # create the market order that is used as the liquidation order
        order = Order({
            'id':  uuid4(),
            'symbol': symbol,
            'exchange': exchange,
            'side': closing_order_side,
            'type': order_types.MARKET,
            'flag': order_flags.REDUCE_ONLY,
            'qty': jh.prepare_qty(p.qty, closing_order_side),
            'price': p.bankruptcy_price,
            'role': order_roles.CLOSE_POSITION
        })

        store.orders.add_order(order)

        store.app.total_liquidations += 1

        logger.info(f'{p.symbol} liquidated at {p.liquidation_price}')

        order.execute()
