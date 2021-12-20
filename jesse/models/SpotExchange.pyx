import jesse.helpers as jh
import jesse.services.logger as logger
from jesse.enums import sides, order_types
from jesse.exceptions import NegativeBalance, InvalidConfig
from jesse.models import Order
from .Exchange import Exchange
from libc.math cimport abs

class SpotExchange(Exchange):
    def add_realized_pnl(self, realized_pnl: float) -> None:
        pass

    def charge_fee(self, amount: float) -> None:
        pass

    # current holding assets
    assets = {}
    # current available assets (dynamically changes based on active orders)
    available_assets = {}

    def __init__(self, name: str, starting_assets: list, fee_rate: float):
        super().__init__(name, starting_assets, fee_rate, 'spot')

        from jesse.routes import router
        # check if base assets are configured
        for route in router.routes:
            base_asset = route.symbol.split('-')[0]
            if base_asset not in self.available_assets:
                raise InvalidConfig(
                    f"Jesse needs to know the balance of your base asset for spot mode. Please add {base_asset} to your exchanges assets config.")

    def wallet_balance(self, symbol: str = '') -> float:
        if symbol == '':
            raise ValueError
        quote_asset = symbol.split('-')[1]
        return self.assets[quote_asset]

    def available_margin(self, symbol: str = '') -> float:
        return self.wallet_balance(symbol)

    def on_order_submission(self, order: Order, skip_market_order=True):
        base_asset = order.symbol.split('-')[0]
        quote_asset = order.symbol.split('-')[1]
        cdef float c_qty = order.qty 
        cdef float c_price = order.price
        cdef double c_fee = self.fee_rate        
        # skip market order at the time of submission because we don't have
        # the exact order.price. Instead, we call on_order_submission() one
        # more time at time of execution without "skip_market_order=False".
        if order.type == order_types.MARKET and skip_market_order:
            return

        # used for logging balance change
        cdef float temp_old_quote_available_asset = self.available_assets[quote_asset]
        cdef float temp_old_base_available_asset = self.available_assets[base_asset]

        if order.side == sides.BUY:
            quote_balance = self.available_assets[quote_asset]
            self.available_assets[quote_asset] -= (abs(c_qty) * c_price) * (1 + c_fee)
            if self.available_assets[quote_asset] < 0:
                raise NegativeBalance(
                    f"Balance cannot go below zero in spot market. Available capital at {self.name} for {quote_asset} is {quote_balance} but you're trying to sell {abs(c_qty * c_price)}"
                )
        # sell order
        else:
            base_balance = self.available_assets[base_asset]
            new_base_balance = base_balance + c_qty
            if new_base_balance < 0:
                raise NegativeBalance(
                    f"Balance cannot go below zero in spot market. Available capital at {self.name} for {base_asset} is {base_balance} but you're trying to sell {abs(c_qty)}"
                )

            self.available_assets[base_asset] -= abs(c_qty)

        cdef float temp_new_quote_available_asset = self.available_assets[quote_asset]
        if jh.is_debuggable('balance_update') and temp_old_quote_available_asset != temp_new_quote_available_asset:
            logger.info(
                f'Available balance for {quote_asset} on {self.name} changed from {round(temp_old_quote_available_asset, 2)} to {round(temp_new_quote_available_asset, 2)}'
            )
        cdef float temp_new_base_available_asset = self.available_assets[base_asset]
        if jh.is_debuggable('balance_update') and temp_old_base_available_asset != temp_new_base_available_asset:
            logger.info(
                f'Available balance for {base_asset} on {self.name} changed from {round(temp_old_base_available_asset, 2)} to {round(temp_new_base_available_asset, 2)}'
            )

    def on_order_execution(self, order: Order) -> None:
        base_asset = order.symbol.split('-')[0]
        quote_asset = order.symbol.split('-')[1]
        cdef float c_qty = order.qty 
        cdef float c_price = order.price 
        cdef double c_fee = self.fee_rate
        
        if order.type == order_types.MARKET:
            self.on_order_submission(order, skip_market_order=False)

        # used for logging balance change
        cdef float temp_old_quote_asset = self.assets[quote_asset]
        cdef float temp_old_quote_available_asset = self.available_assets[quote_asset]
        cdef float temp_old_base_asset = self.assets[base_asset]
        cdef float temp_old_base_available_asset = self.available_assets[base_asset]

        # works for both buy and sell orders (sell order's qty < 0)
        self.assets[base_asset] += c_qty

        if order.side == sides.BUY:
            self.available_assets[base_asset] += c_qty
            self.assets[quote_asset] -= (abs(c_qty) * c_price) * (1 + c_fee)
        # sell order
        else:
            self.available_assets[quote_asset] += abs(c_qty) * c_price * (1 - c_fee)
            self.assets[quote_asset] += abs(c_qty) * c_price * (1 - c_fee)

        cdef float temp_new_quote_asset = self.assets[quote_asset]
        if jh.is_debuggable('balance_update') and temp_old_quote_asset != temp_new_quote_asset:
            logger.info(
                f'Balance for {quote_asset} on {self.name} changed from {round(temp_old_quote_asset, 2)} to {round(temp_new_quote_asset, 2)}'
            )
        cdef float temp_new_quote_available_asset = self.available_assets[quote_asset]
        if jh.is_debuggable('balance_update') and temp_old_quote_available_asset != temp_new_quote_available_asset:
            logger.info(
                f'Balance for {quote_asset} on {self.name} changed from {round(temp_old_quote_available_asset, 2)} to {round(temp_new_quote_available_asset, 2)}'
            )

        cdef float temp_new_base_asset = self.assets[base_asset]
        if jh.is_debuggable('balance_update') and temp_old_base_asset != temp_new_base_asset:
            logger.info(
                f'Balance for {base_asset} on {self.name} changed from {round(temp_old_base_asset, 2)} to {round(temp_new_base_asset, 2)}'
            )
        cdef float temp_new_base_available_asset = self.available_assets[base_asset]
        if jh.is_debuggable('balance_update') and temp_old_base_available_asset != temp_new_base_available_asset:
            logger.info(
                f'Balance for {base_asset} on {self.name} changed from {round(temp_old_base_available_asset, 2)} to {round(temp_new_base_available_asset, 2)}'
            )

    def on_order_cancellation(self, order: Order) -> None:
        base_asset = order.symbol.split('-')[0]
        quote_asset = order.symbol.split('-')[1]
        cdef float c_qty = order.qty 
        cdef float c_price = order.price 
        cdef double c_fee = self.fee_rate
        # used for logging balance change
        cdef float temp_old_quote_available_asset = self.available_assets[quote_asset]
        cdef float temp_old_base_available_asset = self.available_assets[base_asset]

        if order.side == sides.BUY:
            self.available_assets[quote_asset] += (abs(c_qty) * c_price) * (1 + c_fee)
        # sell order
        else:
            self.available_assets[base_asset] += abs(c_qty)

        cdef float temp_new_quote_available_asset = self.available_assets[quote_asset]
        if jh.is_debuggable('balance_update') and temp_old_quote_available_asset != temp_new_quote_available_asset:
            logger.info(
                f'Available balance for {quote_asset} on {self.name} changed from {round(temp_old_quote_available_asset, 2)} to {round(temp_new_quote_available_asset, 2)}'
            )
        cdef float temp_new_base_available_asset = self.available_assets[base_asset]
        if jh.is_debuggable('balance_update') and temp_old_base_available_asset != temp_new_base_available_asset:
            logger.info(
                f'Available balance for {base_asset} on {self.name} changed from {round(temp_old_base_available_asset, 2)} to {round(temp_new_base_available_asset, 2)}'
            )
