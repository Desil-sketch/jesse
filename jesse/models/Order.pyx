# from playhouse.postgres_ext import *

import jesse.helpers as jh
import jesse.services.logger as logger
import jesse.services.selectors as selectors
from jesse import sync_publish
from jesse.config import config
from jesse.services.notifier import notify
from jesse.enums import order_statuses, order_flags
from jesse.services.db import database
from libc.math cimport NAN
# from libc.time cimport time,time_t
import random
def uuid4():
  return '%032x' % random.getrandbits(128)
from jesse.services.db import database


if database.is_closed():
    database.open_connection()


class Order():
    # id generated by Jesse for database usage
    id = uuid4()
    trade_id = uuid4()
    session_id = uuid4()

    # id generated by market, used in live-trade mode
    exchange_id : str = None
    # some exchanges might require even further info
    vars : dict = None
    symbol: str 
    exchange : str 
    side : str 
    type : str 
    flag : str = None
    qty : float 
    price : float = NAN
    price : float = NAN
    status : str =order_statuses.ACTIVE
    created_at : int = None
    executed_at : int = None
    canceled_at : int = None
    role: str = None
    submitted_via: None

    class Meta:
        database = database.db
        indexes = ((('exchange', 'symbol'), False),)

    def __init__(self, attributes: dict = None, **kwargs) -> None:
        # Model.__init__(self, attributes=attributes, **kwargs)
        from jesse.store import store 
        
        if attributes is None:
            attributes = {}

        for a, value in attributes.items():
            setattr(self, a, value)

        if self.created_at is None:
            self.created_at = jh.now_to_timestamp()

        if jh.is_debuggable('order_submission'):
            txt = f'{"QUEUED" if self.is_queued else "SUBMITTED"} order: {self.symbol}, {self.type}, {self.side}, {self.qty}'
            if self.price:
                txt += f', ${round(self.price, 2)}'
            logger.info(txt)

        # handle exchange balance for ordered asset
        e = store.exchanges.storage.get(self.exchange, None)
        e.on_order_submission(self)

    def broadcast(self) -> None:
        sync_publish('order', self.to_dict)

    def notify_submission(self) -> None:
        self.broadcast()

        if config['env']['notifications']['events']['submitted_orders']:
            txt = f'{"QUEUED" if self.is_queued else "SUBMITTED"} order: {self.symbol}, {self.type}, {self.side}, {self.qty}'
            if self.price:
                txt += f', ${round(self.price, 2)}'
            notify(txt)

    @property
    def is_canceled(self) -> bool:
        return self.status == order_statuses.CANCELED

    @property
    def is_active(self) -> bool:
        return self.status == order_statuses.ACTIVE

    @property
    def is_queued(self) -> bool:
        """
        Used in live mode only: it means the strategy has considered the order as submitted,
        but the exchange does not accept it because of the distance between the current
        price and price of the order. Hence it's been queued for later submission.

        :return: bool
        """
        return self.status == order_statuses.QUEUED

    @property
    def is_new(self) -> bool:
        return self.is_active

    @property
    def is_executed(self) -> bool:
        return self.status == order_statuses.EXECUTED

    @property
    def is_filled(self) -> bool:
        return self.is_executed

    @property
    def is_reduce_only(self) -> bool:
        return self.flag == order_flags.REDUCE_ONLY

    @property
    def is_close(self) -> bool:
        return self.flag == order_flags.CLOSE
        
    @property
    def is_stop_loss(self):
        return self.submitted_via == 'stop-loss'

    @property
    def is_take_profit(self):
        return self.submitted_via == 'take-profit'
        
    @property
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'exchange_id': self.exchange_id,
            'symbol': self.symbol,
            'side': self.side,
            'type': self.type,
            'qty': self.qty,
            'price': self.price,
            'flag': self.flag,
            'status': self.status,
            'created_at': self.created_at,
            'canceled_at': self.canceled_at,
            'executed_at': self.executed_at,
        }

    def cancel(self, silent=False) -> None:
        from jesse.store import store 
        if self.is_canceled or self.is_executed:
            return

        self.canceled_at = jh.now_to_timestamp()
        self.status = order_statuses.CANCELED

        if not silent:
            txt = f'CANCELED order: {self.symbol}, {self.type}, {self.side}, {self.qty}'
            if self.price:
                txt += f', ${round(self.price, 2)}'
            if jh.is_debuggable('order_cancellation'):
                logger.info(txt)

        # handle exchange balance
        e = store.exchanges.storage.get(self.exchange, None)
        e.on_order_cancellation(self)

    def execute(self, silent=False) -> None:
        from jesse.store import store 
        if self.is_canceled or self.is_executed:
            return

        self.executed_at = jh.now_to_timestamp()
        self.status = order_statuses.EXECUTED

        if not silent:
            txt = f'EXECUTED order: {self.symbol}, {self.type}, {self.side}, {self.qty}'
            if self.price:
                txt += f', ${round(self.price, 2)}'
            # log
            if jh.is_debuggable('order_execution'):
                logger.info(txt)
        key = f'{self.exchange}-{self.symbol}' 
        p = store.positions.storage.get(key, None)

        if p:
            p._on_executed_order(self)

        # handle exchange balance for ordered asset
        e = store.exchanges.storage.get(self.exchange, None)
        e.on_order_execution(self)


# if database is open, create the table
# if database.is_open():
    # Order.create_table()
