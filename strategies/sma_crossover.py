"""
Sample SMA Crossover Strategy using VectorBT.
"""

import pandas as pd
import vectorbt as vbt


def run_sma_crossover_vbt(ohlcv_df: pd.DataFrame, fast_window: int = 10, slow_window: int = 30):
    """
    VectorBT Fast/Slow SMA Crossover Strategy Backtest.
    
    Parameters:
        ohlcv_df: DataFrame with 'close' column and DatetimeIndex.
        fast_window: Period for fast moving average.
        slow_window: Period for slow moving average.
        
    Returns:
        Portfolio object with backtest results.
    """
    price = ohlcv_df['close']
    
    # Calculate Fast and Slow Moving Averages
    fast_ma = vbt.MA.run(price, window=fast_window)
    slow_ma = vbt.MA.run(price, window=slow_window)
    
    # Generate Entry & Exit Signals
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    # Run Portfolio simulation
    portfolio = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        init_cash=100_000.0,
        fees=0.0005,  # 0.05% commission
        freq='1m'
    )
    
    return portfolio
