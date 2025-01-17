from typing import Union
cimport cython
from libc.math cimport abs

import jesse.helpers as jh
from jesse.enums import sides, order_flags
from jesse.exceptions import OrderNotAllowed, InvalidStrategy
from jesse.models import Order
from jesse.models import Position


class Broker:
    def __init__(self, position: Position, exchange: str, symbol: str, timeframe: str) -> None:
        self.position = position
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = exchange
        from jesse.services.api import api
        self.api = api

    @staticmethod
    def _validate_qty(qty: float) -> None:
        if qty == 0:
            raise InvalidStrategy('qty cannot be 0')

    def sell_at_market(self, qty: float, role: str = None) -> Union[Order, None]:
        self._validate_qty(qty)

        return self.api.market_order(
            self.exchange,
            self.symbol,
            abs(qty),
            self.position.current_price,
            sides.SELL,
            role, []
        )

    def sell_at(self, qty: float, price: float, role: str = None) -> Union[Order, None]:
        self._validate_qty(qty)

        if price < 0:
            raise ValueError('price cannot be negative.')

        return self.api.limit_order(
            self.exchange,
            self.symbol,
            abs(qty),
            price,
            sides.SELL,
            role,
            []
        )

    def buy_at_market(self, qty: float, role: str = None) -> Union[Order, None]:
        self._validate_qty(qty)

        return self.api.market_order(
            self.exchange,
            self.symbol,
            abs(qty),
            self.position.current_price,
            sides.BUY,
            role,
            []
        )

    def buy_at(self, qty: float, price: float, role: str = None) -> Union[Order, None]:
        self._validate_qty(qty)

        if price < 0:
            raise ValueError('price cannot be negative.')

        return self.api.limit_order(
            self.exchange,
            self.symbol,
            abs(qty),
            price,
            sides.BUY,
            role,
            []
        )

    def reduce_position_at(self, qty: float, price: float, role: str = None) -> Union[Order, None]:
        self._validate_qty(qty)

        qty = abs(qty)

        # validation
        if price < 0:
            raise ValueError('price cannot be negative.')

        # validation
        if self.position.is_close:
            raise OrderNotAllowed(
                'Cannot submit a reduce_position order when there is no open position'
            )

        side = jh.opposite_side(jh.type_to_side(self.position.type))

        if abs(price - self.position.current_price) < 0.0001:
            return self.api.market_order(
                self.exchange,
                self.symbol,
                qty,
                price,
                side,
                role,
                [order_flags.REDUCE_ONLY]
            )

        elif (side == 'sell' and self.position.type == 'long' and price > self.position.current_price) or (
                side == 'buy' and self.position.type == 'short' and price < self.position.current_price):
            return self.api.limit_order(
                self.exchange,
                self.symbol,
                qty,
                price,
                side,
                role,
                [order_flags.REDUCE_ONLY]
            )
        elif (side == 'sell' and self.position.type == 'long' and price < self.position.current_price) or (
                side == 'buy' and self.position.type == 'short' and price > self.position.current_price):
            return self.api.stop_order(
                self.exchange,
                self.symbol,
                abs(qty),
                price,
                side,
                role,
                [order_flags.REDUCE_ONLY]
            )
        else:
            raise OrderNotAllowed("This order doesn't seem to be for reducing the position.")

    def start_profit_at(self, side: str, qty: float, price: float, role: str = None) -> Union[Order, None]:
        self._validate_qty(qty)

        if price < 0:
            raise ValueError('price cannot be negative.')

        if side == 'buy' and price < self.position.current_price:
            raise OrderNotAllowed(
                f'A buy start_profit({price}) order must have a price higher than current_price({self.position.current_price}).'
            )
        if side == 'sell' and price > self.position.current_price:
            raise OrderNotAllowed(
                f'A sell start_profit({price}) order must have a price lower than current_price({self.position.current_price}).'
            )

        return self.api.stop_order(
            self.exchange,
            self.symbol,
            abs(qty),
            price,
            side,
            role,
            []
        )

    def cancel_all_orders(self) -> bool:
        return self.api.cancel_all_orders(self.exchange, self.symbol)

    def cancel_order(self, order_id: str) -> bool:
        return self.api.cancel_order(self.exchange, self.symbol, order_id)
