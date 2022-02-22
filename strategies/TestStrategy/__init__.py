from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
from datetime import datetime, timezone


class TestStrategy(Strategy):
    def hyperparameters(self):
        return [
            {'name': 'ema_period1', 'type': int, 'min': 6, 'max': 280, 'default': 90}, 
            {'name': 'ema_period2', 'type': int, 'min': 6, 'max': 280, 'default': 170}, 
        ]
         
    @property
    def ema_test(self):
        return ta.ema(self.candles, self.hp['ema_period1'],sequential=True)
        
    # @property
    def ema1(self,precalc_candles = None):
        return ta.ema(precalc_candles, period=self.hp['ema_period1'],sequential=True) 
   
        
    # @property
    def ema2(self,precalc_candles = None):
        return ta.ema(precalc_candles, self.hp['ema_period2'],sequential=True)
        
    @property 
    def longfilter(self) -> bool:
        return self._indicator1_value > self._indicator2_value 

    @property 
    def shortfilter(self) -> bool: 
        return self._indicator1_value < self._indicator2_value 
 
        
    def should_long(self) -> bool:
        return self.longfilter 

    def should_short(self) -> bool:
        return self.shortfilter 

    def go_long(self):
        entry = self.price
        qty = utils.size_to_qty(self.capital*0.01, entry, precision=3, fee_rate=self.fee_rate)
        self.buy = qty, entry

    def go_short(self):
        entry = self.price   
        qty = utils.size_to_qty(self.capital*0.01, entry, precision=3, fee_rate=self.fee_rate)
        self.sell = qty, entry

    def update_position(self):
        if self.is_long and (self.shortfilter): 
            self.liquidate()
        if self.is_short and (self.longfilter):  
            self.liquidate()
            
    def should_cancel(self) -> bool:
        pass

