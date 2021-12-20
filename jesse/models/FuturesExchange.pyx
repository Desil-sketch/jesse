import numpy as np
cimport numpy as np 
DTYPE = np.float64
np.import_array()
cimport cython
from numpy cimport ndarray as ar 
# import jesse.helpers as jh
import jesse.services.logger as logger
from jesse.enums import sides, order_types
from jesse.exceptions import InsufficientMargin
# from jesse.libs import DynamicNumpyArray
from jesse.models import Order
from jesse.services import selectors
from .Exchange import Exchange
from libc.math cimport abs, fmax


class FuturesExchange(Exchange):
    # current holding assets
    assets = {}
    # current available assets (dynamically changes based on active orders)
    available_assets = {}

    buy_orders = {}
    sell_orders = {}

    def __init__(
            self,
            name: str,
            starting_assets: list,
            fee_rate: float,
            settlement_currency: str,
            futures_leverage_mode: str,
            futures_leverage: int
    ):
        super().__init__(name, starting_assets, fee_rate, 'futures')

        self.futures_leverage_mode = futures_leverage_mode
        self.futures_leverage = futures_leverage

        for item in starting_assets:
            self.buy_orders[item['asset']] = DynamicNumpyArray((10, 2))
            self.sell_orders[item['asset']] = DynamicNumpyArray((10, 2))

        # make sure trading routes exist in starting_assets
        from jesse.routes import router
        for r in router.routes:
            base = r.symbol.split('-')[0]
            if base not in self.assets:
                self.assets[base] = 0
                self.temp_reduced_amount[base] = 0
            if base not in self.buy_orders:
                self.buy_orders[base] = DynamicNumpyArray((10, 2))
            if base not in self.sell_orders:
                self.sell_orders[base] = DynamicNumpyArray((10, 2))

        self.starting_assets = self.assets.copy()
        self.available_assets = self.assets.copy()

        # start from 0 balance for self.available_assets which acts as a temp variable
        for k in self.available_assets:
            self.available_assets[k] = 0

        self.settlement_currency = settlement_currency.upper()

    def wallet_balance(self, symbol: str = '') -> float:
        return self.assets[self.settlement_currency]

    def available_margin(self, symbol: str = '') -> float:
        from jesse.store import store 
        # a temp which gets added to per each asset (remember that all future assets use the same currency for settlement)
        cdef double temp_credits = self.assets[self.settlement_currency]
        cdef double sum_buy_orders, sum_sell_orders
        cdef int c_leverage = self.futures_leverage
        
        # we need to consider buy and sell orders of ALL pairs
        # also, consider the value of all open positions
        for asset in self.assets:
            if asset == self.settlement_currency:
                continue
                
            key = f'{self.name}-{f"{asset}-{self.settlement_currency}"}'
            position = store.positions.storage.get(key, None)
            if position is None:
                continue

            if position.qty != 0:
                # add unrealized PNL
                temp_credits += position.pnl

            # only which of these has actual values, so we can count all of them!
            sum_buy_orders = (self.buy_orders[asset].array[0:self.buy_orders[asset].index+1][:,0] * self.buy_orders[asset].array[0:self.buy_orders[asset].index+1][:,1]).sum()
            sum_sell_orders = (self.sell_orders[asset].array[0:self.sell_orders[asset].index+1][:,0] * self.sell_orders[asset].array[0:self.sell_orders[asset].index+1][:,1]).sum()

            if position.qty != 0:
                temp_credits -= position.total_cost

            # Subtract the amount we paid for open orders. Notice that this does NOT include
            # reduce_only orders so either sum_buy_orders or sum_sell_orders is zero. We also
            # care about the cost we actually paid for it which takes into account the leverage
            temp_credits -= fmax(
                abs(sum_buy_orders) / c_leverage, abs(sum_sell_orders) / c_leverage
            )

        # count in the leverage
        return temp_credits * c_leverage

    def charge_fee(self, double amount) -> None:
        cdef double c_fee_rate = self.fee_rate
        cdef double fee_amount = abs(amount) * c_fee_rate
        cdef double new_balance = self.assets[self.settlement_currency] - fee_amount
        logger.info(
            f'Charged {round(fee_amount, 2)} as fee. Balance for {self.settlement_currency} on {self.name} changed from {round(self.assets[self.settlement_currency], 2)} to {round(new_balance, 2)}'
        )
        self.assets[self.settlement_currency] = new_balance

    def add_realized_pnl(self, double realized_pnl) -> None:
        cdef double new_balance = self.assets[self.settlement_currency] + realized_pnl
        logger.info(
            f'Added realized PNL of {round(realized_pnl, 2)}. Balance for {self.settlement_currency} on {self.name} changed from {round(self.assets[self.settlement_currency], 2)} to {round(new_balance, 2)}')
        self.assets[self.settlement_currency] = new_balance

    def on_order_submission(self, order: Order, bint skip_market_order = True) -> None:
        base_asset = order.symbol.split('-')[0]
        cdef double order_size, remaining_margin
        cdef double c_qty = order.qty 
        cdef double c_price = order.price
        # make sure we don't spend more than we're allowed considering current allowed leverage
        if (order.type != order_types.MARKET or skip_market_order) and not order.is_reduce_only:
            order_size = abs(order.qty * order.price)
            remaining_margin = self.available_margin()
            if order_size > remaining_margin:
                raise InsufficientMargin(
                    f'You cannot submit an order for ${round(order_size)} when your margin balance is ${round(remaining_margin)}')

        # skip market order at the time of submission because we don't have
        # the exact order.price. Instead, we call on_order_submission() one
        # more time at time of execution without "skip_market_order=False".
        if order.type == order_types.MARKET and skip_market_order:
            return

        self.available_assets[base_asset] += c_qty

        if not order.is_reduce_only:
            if order.side == sides.BUY:
                self.buy_orders[base_asset].append(np.array([c_qty, c_price]))
            else:
                self.sell_orders[base_asset].append(np.array([c_qty, c_price]))

    def on_order_execution(self, order: Order) -> None:
        base_asset = order.symbol.split('-')[0]
        cdef Py_ssize_t index
        cdef double c_qty = order.qty 
        cdef double c_price = order.price 
        if order.type == order_types.MARKET:
            self.on_order_submission(order, skip_market_order=False)

        if not order.is_reduce_only:
            if order.side == sides.BUY:
                # find and set order to [0, 0] (same as removing it)
                for index, item in enumerate(self.buy_orders[base_asset].array[0:self.buy_orders[base_asset].index+1]):
                    if item[0] == c_qty and item[1] == c_price:
                        self.buy_orders[base_asset][index] = np.array([0, 0])
                        break
            else:
                # find and set order to [0, 0] (same as removing it)
                for index, item in enumerate(self.sell_orders[base_asset].array[0:self.sell_orders[base_asset].index+1]):
                    if item[0] == c_qty and item[1] == c_price:
                        self.sell_orders[base_asset][index] = np.array([0, 0])
                        break

    def on_order_cancellation(self, order: Order) -> None:
        base_asset = order.symbol.split('-')[0]
        cdef Py_ssize_t index
        cdef double c_qty = order.qty 
        cdef double c_price = order.price 
        self.available_assets[base_asset] -= c_qty
        # self.available_assets[quote_asset] += order.qty * order.price
        if not order.is_reduce_only:
            if order.side == sides.BUY:
                # find and set order to [0, 0] (same as removing it)
                for index, item in enumerate(self.buy_orders[base_asset].array[0:self.buy_orders[base_asset].index+1]):
                    if item[0] == c_qty and item[1] == c_price:
                        self.buy_orders[base_asset][index] = np.array([0, 0])
                        break
            else:
                # find and set order to [0, 0] (same as removing it)
                for index, item in enumerate(self.sell_orders[base_asset].array[0:self.sell_orders[base_asset].index+1]):
                    if item[0] == c_qty and item[1] == c_price:
                        self.sell_orders[base_asset][index] = np.array([0, 0])
                        break


class DynamicNumpyArray:
    def __init__(self, shape: tuple,int index = -1, attributes: dict = None):
        self.index = index
        self.array = np.zeros((shape),dtype=DTYPE)
        self.bucket_size = shape[0]
        self.shape = shape

    def __len__(self) -> int:
        # cdef Py_ssize_t index = self.index 
        return self.index + 1
             
    def __setitem__(self, int i, ar item):
        cdef Py_ssize_t index = self.index
        if i < 0:
            i = (index + 1) - abs(i)
        self.array[i] = item
        
    def append(self, ar item) -> None:
        self.index += 1
        cdef ar new_bucket
        cdef Py_ssize_t index = self.index 
        cdef int bucket_size = self.bucket_size
        # expand if the arr is almost full
        if index != 0 and (index + 1) % bucket_size == 0:
            new_bucket = np.zeros((self.shape),dtype=DTYPE)
            self.array = np.concatenate((self.array, new_bucket), axis=0, dtype=DTYPE)
        self.array[index] = item
