# import numpy as np
# import peewee

from jesse.config import config
from jesse.services.db import database
import uuid
import jesse.helpers as jh
from jesse.config import config
from libc.math cimport abs, NAN
from jesse.services.db import database
        
if database.is_closed():
    database.open_connection()


class CompletedTrade():
    """A trade is made when a position is opened AND closed."""

    id: str = uuid.uuid4() # peewee.UUIDField(primary_key=True)
    strategy_name: str #peewee.CharField()
    symbol:str  # peewee.CharField()
    exchange: str #peewee.CharField()
    type: str #peewee.CharField()
    timeframe: str #peewee.CharField()
    entry_price: float = NAN#peewee.FloatField(default=NAN)
    exit_price:float =NAN  #peewee.FloatField(default=NAN)
    take_profit_at: float = NAN #peewee.FloatField(default=NAN)
    stop_loss_at: float = NAN # peewee.FloatField(default=NAN)
    qty: float = NAN #= peewee.FloatField(default=NAN)
    opened_at: int = None # = peewee.BigIntegerField()
    closed_at: int = None #peewee.BigIntegerField()
    entry_candle_timestamp: int = None #peewee.BigIntegerField()
    exit_candle_timestamp: int = None #peewee.BigIntegerField()
    leverage: int = None #peewee.IntegerField()

    orders = []

    class Meta:

        database = database.db
        indexes = ((('strategy_name', 'exchange', 'symbol'), False),)

    def __init__(self, attributes: dict = None, **kwargs) -> None:
        # peewee.Model.__init__(self, attributes=attributes, **kwargs)

        if attributes is None:
            attributes = {}

        for a, value in attributes.items():
            setattr(self, a, value)

    def toJSON(self) -> dict:
        orders = [o.__dict__ for o in self.orders]
        return {
            "id": self.id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "type": self.type,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "fee": self.fee,
            "size": self.size,
            "PNL": self.pnl,
            "PNL_percentage": self.pnl_percentage,
            "holding_period": self.holding_period,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "entry_candle_timestamp": self.entry_candle_timestamp,
            "exit_candle_timestamp": self.exit_candle_timestamp,
            "orders": orders,
        }

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'type': self.type,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'qty': self.qty,
            'opened_at': self.opened_at,
            'closed_at': self.closed_at,
            'entry_candle_timestamp': self.entry_candle_timestamp,
            'exit_candle_timestamp': self.exit_candle_timestamp,
            "fee": self.fee,
            "size": self.size,
            "PNL": self.pnl,
            "PNL_percentage": self.pnl_percentage,
            "holding_period": self.holding_period,
        }

    @property
    def fee(self) -> float:
        trading_fee = jh.get_config(f'env.exchanges.{self.exchange}.fee')
        return trading_fee * self.qty * (self.entry_price + self.exit_price)

    @property
    def size(self) -> float:
        return self.qty * self.entry_price

    @property
    def pnl(self) -> float:
        """PNL"""
        fee = config['env']['exchanges'][self.exchange]['fee']
        return jh.estimate_PNL(
            self.qty, self.entry_price, self.exit_price,
            self.type, fee
        )

    @property
    def pnl_percentage(self) -> float:
        """
        Alias for self.roi
        """
        return self.roi

    @property
    def roi(self) -> float:
        """
        Return on Investment in percentage
        More at: https://www.binance.com/en/support/faq/5b9ad93cb4854f5990b9fb97c03cfbeb
        """
        return self.pnl / self.total_cost * 100

    @property
    def total_cost(self) -> float:
        """
        How much we paid to open this position (currently does not include fees, should we?!)
        """
        return self.entry_price * abs(self.qty) / self.leverage

    @property
    def holding_period(self) -> int:
        """How many SECONDS has it taken for the trade to be done."""
        return (self.closed_at - self.opened_at) / 1000


# if database is open, create the table
# if database.is_open():
    # CompletedTrade.create_table()
