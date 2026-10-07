"""
Main Entry Point for Databento Quantitative Backtesting Project.
============================================================
Supports:
1. Advanced Order Flow + Volume Profile Strategy (NQ Futures)
2. Baseline SMA Crossover Strategy (VectorBT & Backtrader)
"""

import argparse
from pathlib import Path
import databento as db
import pandas as pd
from strategies.volume_profile_strategy import run_orderflow_strategy_pipeline
from strategies.sma_crossover import run_sma_crossover_vbt
from strategies.backtrader_strategy import run_backtrader_simulation


def get_data_files(data_dir: str = "data"):
    """data/ folder se saari .dbn aur .dbn.zst files dhoondhta hai."""
    path = Path(data_dir)
    files = sorted(list(path.glob("*.dbn")) + list(path.glob("*.dbn.zst")))
    return files


def load_dbn_trades(file_path: Path, contract_symbol: str = None) -> pd.DataFrame:
    """
    Databento .dbn / .dbn.zst file ko read karke DataFrame return karta hai.
    
    Agar contract_symbol pass nahi kiya toh sabse actively traded instrument (e.g., NQU5)
    automatically select hota hai taaki spread prices mix na hon.
    """
    print(f"[1/4] Loading DBN file: {file_path.name}...")
    store = db.DBNStore.from_file(file_path)
    df = store.to_df()

    # Timezone conversion: UTC to US/Eastern
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    # Filtering specific contract (front-month futures vs calendar spreads)
    if contract_symbol:
        df = df[df["symbol"] == contract_symbol]
    else:
        top_symbol = df["symbol"].value_counts().index[0]
        print(f"      Selected active symbol: '{top_symbol}' ({len(df[df['symbol'] == top_symbol]):,} trades)")
        df = df[df["symbol"] == top_symbol]

    return df


def aggregate_to_ohlcv(trades_df: pd.DataFrame, timeframe: str = "1min") -> pd.DataFrame:
    """
    High-frequency tick trades ko OHLCV bars mein convert karta hai.
    """
    print(f"[2/4] Resampling trades to {timeframe} OHLCV candles...")
    ohlcv = trades_df["price"].resample(timeframe).ohlc()
    ohlcv["volume"] = trades_df["size"].resample(timeframe).sum()
    ohlcv = ohlcv.dropna()
    print(f"      Generated {len(ohlcv)} candles.")
    return ohlcv


def run_orderflow_mode(trades_df: pd.DataFrame, timeframe: str = "5min"):
    """Order Flow + Volume Profile Strategy Mode"""
    print("\n" + "=" * 65)
    print("   [MODE] ADVANCED ORDER FLOW + VOLUME PROFILE STRATEGY")
    print(f"          Timeframe: {timeframe} | Target: NQ Futures | Sizing: 1 Contract")
    print("=" * 65)

    results = run_orderflow_strategy_pipeline(
        trades_df=trades_df,
        entry_timeframe=timeframe,
        tick_size=1.0,
        point_value=20.0,      # $20 per point on NQ E-mini
        enable_vbt=True
    )

    sim_res = results["sim_results"]
    summary = sim_res["summary"]
    trades_df_log = sim_res["trades_df"]

    print("\n" + "=" * 65)
    print("            PERFORMANCE SUMMARY (EVENT-DRIVEN CTC)")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:22}: {v}")
    print("-" * 65)

    if not trades_df_log.empty:
        print("\n[Trade Execution Log]:")
        display_cols = ["entry_time", "exit_time", "type", "direction", "entry_price", "exit_price", "pnl_pts", "pnl_usd", "exit_reason"]
        valid_cols = [c for c in display_cols if c in trades_df_log.columns]
        print(trades_df_log[valid_cols].to_string(index=False))

    # VectorBT Portfolio Stats
    vbt_pf = results["vbt_portfolio"]
    if vbt_pf is not None:
        print("\n" + "=" * 65)
        print("            VECTORBT BENCHMARK METRICS")
        print("=" * 65)
        try:
            stats = vbt_pf.stats()
            key_metrics = ["Start", "End", "Period", "Total Return [%]", "Benchmark Return [%]", "Max Drawdown [%]", "Total Trades", "Win Rate [%]"]
            avail = [m for m in key_metrics if m in stats.index]
            print(stats[avail].to_string())
        except Exception as e:
            print(f"VectorBT summary note: {e}")


def run_sma_mode(trades_df: pd.DataFrame, timeframe: str = "5min"):
    """Baseline SMA Crossover Strategy Mode (VectorBT + Backtrader)"""
    print("\n" + "=" * 65)
    print("   [MODE] BASELINE SMA CROSSOVER STRATEGY")
    print("=" * 65)

    ohlcv = aggregate_to_ohlcv(trades_df, timeframe=timeframe)

    # 1. VectorBT
    print("\n--- Running VectorBT Simulation ---")
    portfolio = run_sma_crossover_vbt(ohlcv, fast_window=10, slow_window=30)
    stats = portfolio.stats()
    key_metrics = ["Start", "End", "Period", "Total Return [%]", "Benchmark Return [%]", "Max Drawdown [%]", "Total Trades", "Win Rate [%]", "Profit Factor"]
    avail = [m for m in key_metrics if m in stats.index]
    print(stats[avail].to_string())

    # 2. Backtrader
    print("\n--- Running Backtrader Simulation ---")
    bt_res = run_backtrader_simulation(ohlcv)
    for k, v in bt_res.items():
        print(f"  {k:26}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Databento Backtesting Framework")
    parser.add_argument(
        "--strategy",
        choices=["orderflow", "sma"],
        default="orderflow",
        help="Strategy to run: 'orderflow' (default) or 'sma'"
    )
    parser.add_argument(
        "--timeframe",
        choices=["3min", "5min", "15min"],
        default="5min",
        help="Entry candle timeframe: '3min' or '5min' (default)"
    )
    parser.add_argument(
        "--file_index",
        type=int,
        default=0,
        help="Index of file in data/ folder to test (default: 0)"
    )
    parser.add_argument(
        "--all_files",
        action="store_true",
        help="Run memory-efficient backtest across all available files in data/"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("      Databento High-Frequency Backtesting Framework")
    print("=" * 65)

    if args.all_files:
        from backtest_multi_day import run_multi_day_backtest
        run_multi_day_backtest(data_dir="data", timeframe=args.timeframe)
        return

    files = get_data_files("data")
    if not files:
        print("\n[!] data/ folder mein koi .dbn ya .dbn.zst files nahi mili!")
        print("    Kripya apni .dbn files ko data/ folder mein copy karein.")
        return

    print(f"\n[OK] Found {len(files)} data file(s) in 'data/' folder.")
    target_file = files[min(args.file_index, len(files) - 1)]

    trades_df = load_dbn_trades(target_file)

    if args.strategy == "orderflow":
        run_orderflow_mode(trades_df, timeframe=args.timeframe)
    else:
        run_sma_mode(trades_df, timeframe=args.timeframe)

    print("\n[SUCCESS] Pipeline execution finished!")


if __name__ == "__main__":
    main()
