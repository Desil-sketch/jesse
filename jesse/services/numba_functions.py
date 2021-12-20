from numba import njit 
import numpy as np

@njit
def generate_candle_from_one_minutes(timeframe: str,
                                     candles: np.ndarray ,
                                     accept_forming_candles:bool = False) :
    # if len(candles) == 0:
        # raise ValueError('No candles were passed')

    # if not accept_forming_candles and len(candles) != jh.timeframe_to_one_minutes(timeframe):
        # raise ValueError(
            # f'Sent only {len(candles)} candles but {jh.timeframe_to_one_minutes(timeframe)} is required to create a "{timeframe}" candle.'
        # )
    return np.array([
        candles[0][0],
        candles[0][1],
        candles[-1][2],
        (candles[:, 3]).max(),
        (candles[:, 4]).min(),
        (candles[:, 5]).sum(),
    ],dtype= np.float64)
    
@njit
def _get_fixed_jumped_candle_(previous_candle: np.ndarray, candle: np.ndarray) -> np.ndarray:
    """
    A little workaround for the times that the price has jumped and the opening
    price of the current candle is not equal to the previous candle's close!

    :param previous_candle: np.ndarray
    :param candle: np.ndarray
    """
    if previous_candle[2] < candle[1]:
        candle[1] = previous_candle[2]
        candle[4] = min(previous_candle[2], candle[4])
    elif previous_candle[2] > candle[1]:
        candle[1] = previous_candle[2]
        candle[3] = max(previous_candle[2], candle[3])
    return candle