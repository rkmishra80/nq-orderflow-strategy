"""
Sample Backtrader Strategy Implementation.
"""

import backtrader as bt
import pandas as pd


class SmaCrossStrategy(bt.Strategy):
    """Simple Moving Average Crossover Strategy for Backtrader."""
    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )

    def __init__(self):
        # Moving averages
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period
        )
        # Crossover indicator (1: cross above, -1: cross below)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        # Agar market position nahi hai aur fast MA ne slow MA ko cross above kiya
        if not self.position:
            if self.crossover > 0:
                self.buy(size=1)
        # Agar already in position aur cross below hua
        elif self.crossover < 0:
            self.close()


def run_backtrader_simulation(ohlcv_df: pd.DataFrame, initial_cash: float = 100_000.0):
    """
    Runs Backtrader engine with pandas OHLCV DataFrame.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCrossStrategy)

    # Convert DatetimeIndex timezone-naive for standard backtrader feed
    df = ohlcv_df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    data_feed = bt.feeds.PandasData(
        dataname=df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=None,
    )

    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0005)

    start_value = cerebro.broker.getvalue()
    cerebro.run()
    end_value = cerebro.broker.getvalue()

    return {
        "Starting Portfolio Value": start_value,
        "Ending Portfolio Value": end_value,
        "Net Profit": end_value - start_value,
        "Return [%]": ((end_value - start_value) / start_value) * 100,
    }
