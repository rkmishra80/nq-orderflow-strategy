"""
Memory-Efficient Multi-Day Backtester for Databento Market Data.
================================================================
Processes high-frequency .dbn.zst files sequentially:
- Free memory after each day (gc.collect)
- Accumulates master trade journal & daily equity curve
- Calculates comprehensive performance metrics (Win Rate, Net Profit, Max Drawdown, Sharpe, etc.)
- Saves outputs to results/ directory
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional
import databento as db
import numpy as np
import pandas as pd

from strategies.volume_profile_strategy import (
    build_orderflow_bars,
    calculate_smart_vwap,
    calculate_volume_profile,
    detect_orderflow_features,
    generate_prior_day_orderflow_signals,
    simulate_orderflow_execution,
)


def extract_session_profile(file_path: Path) -> Optional[Dict]:
    """
    Ek single .dbn.zst file se RTH aur ETH volume profiles extract karta hai.
    """
    try:
        st = db.DBNStore.from_file(file_path)
        df = st.to_df()
        del st

        outright = df[~df["symbol"].str.contains("-")]
        if outright.empty:
            del df, outright
            gc.collect()
            return None

        top_symbol = outright["symbol"].value_counts().index[0]
        df_filtered = outright[outright["symbol"] == top_symbol].copy()
        del df, outright

        if df_filtered.index.tz is None:
            df_filtered.index = df_filtered.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df_filtered.index = df_filtered.index.tz_convert("America/New_York")

        times = df_filtered.index.time
        rth_start = pd.to_datetime("09:30").time()
        rth_end = pd.to_datetime("16:00").time()
        rth_trades = df_filtered[(times >= rth_start) & (times < rth_end)]
        eth_trades = df_filtered[times < rth_start]

        if len(rth_trades) < 200:
            del df_filtered, rth_trades, eth_trades
            gc.collect()
            return None

        session_date = df_filtered.index[0].strftime("%Y-%m-%d")
        rth_prof = calculate_volume_profile(rth_trades["price"], rth_trades["size"], tick_size=1.0)
        eth_prof = calculate_volume_profile(eth_trades["price"], eth_trades["size"], tick_size=1.0)

        del df_filtered, rth_trades, eth_trades
        gc.collect()

        return {
            "date": session_date,
            "file_path": file_path,
            "rth": rth_prof,
            "eth": eth_prof
        }
    except Exception as e:
        gc.collect()
        return None


def run_single_file_backtest(
    file_path: Path,
    prior_rth_profile: Optional[Dict],
    eth_profile: Optional[Dict],
    timeframe: str = "5min",
    tick_size: float = 1.0,
    point_value: float = 20.0
) -> Optional[pd.DataFrame]:
    """
    Prior Day Volume Profile + Strict Filters ke saath single day backtest run karta hai.
    """
    try:
        st = db.DBNStore.from_file(file_path)
        df = st.to_df()
        del st

        # Outright contract filter (spreads hatana)
        outright = df[~df["symbol"].str.contains("-")]
        if outright.empty:
            del df, outright
            gc.collect()
            return None

        # Front-month contract select karna
        top_symbol = outright["symbol"].value_counts().index[0]
        df_filtered = outright[outright["symbol"] == top_symbol].copy()
        del df, outright

        # Timezone conversion: UTC to US/Eastern
        if df_filtered.index.tz is None:
            df_filtered.index = df_filtered.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df_filtered.index = df_filtered.index.tz_convert("America/New_York")

        # RTH session check
        times = df_filtered.index.time
        rth_start = pd.to_datetime("09:30").time()
        rth_end = pd.to_datetime("16:00").time()
        rth_trades_count = ((times >= rth_start) & (times < rth_end)).sum()

        if rth_trades_count < 200:
            del df_filtered
            gc.collect()
            return None

        # Build orderflow bars & features
        bars = build_orderflow_bars(df_filtered, timeframe=timeframe)
        bars = calculate_smart_vwap(bars)
        bars = detect_orderflow_features(bars)

        del df_filtered
        gc.collect()

        # Prior Day RTH + Overnight ETH Strategy Signals
        bars = generate_prior_day_orderflow_signals(
            bars_df=bars,
            prior_rth_profile=prior_rth_profile,
            eth_profile=eth_profile,
            max_trades_per_day=2,
            buffer_pts=10.0,
            max_sl_pts=20.0,
            min_risk_pts=4.0,
            min_rr=1.8
        )

        sim_res = simulate_orderflow_execution(
            bars,
            point_value=point_value,
            slippage_pts=0.5,
            commission_per_contract=2.25,
            enable_trailing_ctc=True
        )

        day_trades = sim_res["trades_df"]
        del bars, sim_res
        gc.collect()

        return day_trades

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        gc.collect()
        return None


def calculate_portfolio_metrics(trades_df: pd.DataFrame, initial_capital: float = 100_000.0) -> Dict:
    """
    Cumulative equity curve aur comprehensive performance metrics calculate karta hai.
    """
    if trades_df.empty:
        return {"error": "No trades executed"}

    df = trades_df.copy().sort_values("entry_time").reset_index(drop=True)

    # Cumulative PnL aur Equity Curve
    df["cum_pnl"] = df["pnl_usd"].cumsum()
    df["equity"] = initial_capital + df["cum_pnl"]

    # Drawdown Calculation ($ and %)
    df["peak_equity"] = df["equity"].cummax()
    df["drawdown_usd"] = df["peak_equity"] - df["equity"]
    df["drawdown_pct"] = (df["drawdown_usd"] / df["peak_equity"]) * 100.0

    max_drawdown_usd = df["drawdown_usd"].max()
    max_drawdown_pct = df["drawdown_pct"].max()

    # Trade Statistics
    total_trades = len(df)
    wins = df[df["pnl_usd"] > 0]
    losses = df[df["pnl_usd"] <= 0]
    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0

    gross_profit = wins["pnl_usd"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl_usd"].sum()) if not losses.empty else 0.0
    net_profit = df["pnl_usd"].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

    avg_trade = df["pnl_usd"].mean()
    avg_win = wins["pnl_usd"].mean() if not wins.empty else 0.0
    avg_loss = abs(losses["pnl_usd"].mean()) if not losses.empty else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else np.nan

    # Consecutive Wins / Losses
    is_win = (df["pnl_usd"] > 0).astype(int)
    win_streaks = is_win.groupby((is_win != is_win.shift()).cumsum()).cumsum()
    max_cons_wins = int(win_streaks[is_win == 1].max()) if not win_streaks[is_win == 1].empty else 0

    is_loss = (df["pnl_usd"] <= 0).astype(int)
    loss_streaks = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumsum()
    max_cons_losses = int(loss_streaks[is_loss == 1].max()) if not loss_streaks[is_loss == 1].empty else 0

    # Daily Return Statistics & Sharpe / Sortino
    df["entry_date"] = pd.to_datetime(df["entry_time"]).dt.date
    daily_pnl = df.groupby("entry_date")["pnl_usd"].sum()

    daily_returns = daily_pnl / initial_capital
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if (std_daily and std_daily > 0) else np.nan

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino = (mean_daily / downside_std) * np.sqrt(252) if (downside_std and downside_std > 0) else np.nan

    # Strategy breakdown (Reversal vs Breakout)
    type_breakdown = {}
    if "type" in df.columns:
        for t, grp in df.groupby("type"):
            t_wins = grp[grp["pnl_usd"] > 0]
            type_breakdown[t] = {
                "trades": len(grp),
                "win_rate": round((len(t_wins) / len(grp)) * 100.0, 1),
                "net_profit": round(grp["pnl_usd"].sum(), 2)
            }

    return {
        "Total Trades": total_trades,
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Win Rate [%]": round(win_rate, 2),
        "Net Profit ($)": round(net_profit, 2),
        "Gross Profit ($)": round(gross_profit, 2),
        "Gross Loss ($)": round(gross_loss, 2),
        "Profit Factor": round(profit_factor, 2) if not np.isnan(profit_factor) else "N/A",
        "Max Drawdown ($)": round(max_drawdown_usd, 2),
        "Max Drawdown [%]": round(max_drawdown_pct, 2),
        "Avg Trade ($)": round(avg_trade, 2),
        "Avg Winner ($)": round(avg_win, 2),
        "Avg Loser ($)": round(avg_loss, 2),
        "Win/Loss Payoff Ratio": round(win_loss_ratio, 2) if not np.isnan(win_loss_ratio) else "N/A",
        "Max Consecutive Wins": max_cons_wins,
        "Max Consecutive Losses": max_cons_losses,
        "Sharpe Ratio (Annualized)": round(sharpe, 2) if not np.isnan(sharpe) else "N/A",
        "Sortino Ratio (Annualized)": round(sortino, 2) if not np.isnan(sortino) else "N/A",
        "Active Trading Days": len(daily_pnl),
        "Best Day ($)": round(daily_pnl.max(), 2) if not daily_pnl.empty else 0.0,
        "Worst Day ($)": round(daily_pnl.min(), 2) if not daily_pnl.empty else 0.0,
        "Strategy Breakdown": type_breakdown,
        "daily_pnl_series": daily_pnl,
        "trades_df_enriched": df
    }


def run_multi_day_backtest(
    data_dir: str = "data",
    timeframe: str = "5min",
    max_files: Optional[int] = None,
    point_value: float = 20.0,
    output_dir: str = "results",
    workers: int = 4
) -> Dict:
    """
    data/ directory ke saare .dbn.zst files ko parallel workers ke sath process karta hai.
    """
    path = Path(data_dir)
    all_files = sorted(list(path.glob("*.trades.dbn.zst")))

    if not all_files:
        print("[!] data/ folder mein koi .trades.dbn.zst file nahi mili!")
        return {}

    if max_files:
        all_files = all_files[:max_files]

    total_files = len(all_files)
    print("=" * 70)
    print(f"      STARTING MULTI-DAY CAUSAL ORDER FLOW BACKTEST ({total_files} FILES)")
    print(f"      Timeframe: {timeframe} | NQ Futures ($20/pt) | Fixed 1 Contract")
    print(f"      Execution: 0.5 pt Slippage/side + $4.50 RT Commission | Workers: {workers}")
    print("=" * 70)

    start_time = time.time()
    collected_trades: List[pd.DataFrame] = []

    print("\n[PHASE 1/2] Pre-calculating Daily Volume Profiles (Static Zero Look-Ahead)...")
    t_phase1 = time.time()
    valid_sessions = []

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fut_map = {executor.submit(extract_session_profile, fp): fp for fp in all_files}
            done_cnt = 0
            for fut in as_completed(fut_map):
                done_cnt += 1
                pct = (done_cnt / total_files) * 100.0
                print(f"  Extracting Profiles: [{done_cnt:3d}/{total_files:3d}] ({pct:5.1f}%)...", end="\r", flush=True)
                res = fut.result()
                if res is not None:
                    valid_sessions.append(res)
    else:
        for idx, fp in enumerate(all_files, 1):
            pct = (idx / total_files) * 100.0
            print(f"  Extracting Profiles: [{idx:3d}/{total_files:3d}] ({pct:5.1f}%)...", end="\r", flush=True)
            res = extract_session_profile(fp)
            if res is not None:
                valid_sessions.append(res)

    valid_sessions.sort(key=lambda x: x["date"])
    print(f"\n[DONE Phase 1] Extracted {len(valid_sessions)} valid trading sessions in {time.time() - t_phase1:.1f}s.")

    # Assign Prior Day RTH Volume Profile chronologically (zero look-ahead, 100% known at 09:30 ET)
    sim_tasks = []
    for idx, s in enumerate(valid_sessions):
        prior_rth = valid_sessions[idx - 1]["rth"] if idx > 0 else None
        sim_tasks.append({
            "file_path": s["file_path"],
            "date": s["date"],
            "prior_rth": prior_rth,
            "eth": s["eth"]
        })

    print("\n[PHASE 2/2] Simulating Prior-Day Order Flow Execution (Time Filters & Realistic Friction)...")
    t_phase2 = time.time()
    total_sim = len(sim_tasks)
    days_with_trades = 0
    days_skipped = 0

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fut_map = {
                executor.submit(
                    run_single_file_backtest,
                    task["file_path"],
                    task["prior_rth"],
                    task["eth"],
                    timeframe,
                    1.0,
                    point_value
                ): task for task in sim_tasks
            }
            done_cnt = 0
            for fut in as_completed(fut_map):
                done_cnt += 1
                pct = (done_cnt / total_sim) * 100.0
                print(f"  Simulating Days: [{done_cnt:3d}/{total_sim:3d}] ({pct:5.1f}%)...", end="\r", flush=True)
                try:
                    day_trades = fut.result()
                    if day_trades is not None and not day_trades.empty:
                        collected_trades.append(day_trades)
                        days_with_trades += 1
                    else:
                        days_skipped += 1
                except Exception as e:
                    task = fut_map[fut]
                    print(f"\nError simulating {task['date']}: {e}")
                    days_skipped += 1
    else:
        for idx, task in enumerate(sim_tasks, 1):
            pct = (idx / total_sim) * 100.0
            print(f"  Simulating Days: [{idx:3d}/{total_sim:3d}] ({pct:5.1f}%)...", end="\r", flush=True)
            day_trades = run_single_file_backtest(
                task["file_path"],
                task["prior_rth"],
                task["eth"],
                timeframe,
                1.0,
                point_value
            )
            if day_trades is not None and not day_trades.empty:
                collected_trades.append(day_trades)
                days_with_trades += 1
            else:
                days_skipped += 1

    total_duration = time.time() - start_time
    print(f"\n[DONE Phase 2] Completed strategy simulation in {time.time() - t_phase2:.1f}s.")
    print(f"[TOTAL TIME] Entire 1-Year dataset processed in {total_duration:.1f}s.")
    print(f"             Trading Days with trades: {days_with_trades} | Days skipped: {days_skipped}")

    if not collected_trades:
        print("[!] Kisi bhi din par trade signal generate nahi hua.")
        return {}

    # Combine all trades into master journal
    master_trades = pd.concat(collected_trades, ignore_index=True)
    master_trades = master_trades.sort_values("entry_time").reset_index(drop=True)

    # Calculate portfolio performance
    metrics = calculate_portfolio_metrics(master_trades)

    # Save results to disk
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    enriched_trades = metrics.pop("trades_df_enriched")
    daily_pnl = metrics.pop("daily_pnl_series")

    enriched_trades.to_csv(out_path / f"trades_log_{timeframe}.csv", index=False)
    daily_pnl.to_csv(out_path / f"daily_pnl_{timeframe}.csv", header=["daily_pnl"])

    print(f"\n[SAVED] Master Trade Log: {out_path / f'trades_log_{timeframe}.csv'}")
    print(f"[SAVED] Daily PnL Log: {out_path / f'daily_pnl_{timeframe}.csv'}")

    return {
        "metrics": metrics,
        "trades_df": enriched_trades,
        "daily_pnl": daily_pnl
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Day Databento Backtester")
    parser.add_argument("--timeframe", choices=["3min", "5min"], default="5min", help="Candle timeframe (default: 5min)")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files (default: all)")
    parser.add_argument("--point_value", type=float, default=20.0, help="Contract point value (default: 20 for NQ, 2 for MNQ)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    results = run_multi_day_backtest(
        data_dir="data",
        timeframe=args.timeframe,
        max_files=args.max_files,
        point_value=args.point_value,
        workers=args.workers
    )

    if not results:
        return

    metrics = results["metrics"]
    trades_df = results["trades_df"]

    print("\n" + "=" * 70)
    print("               MULTI-DAY BACKTEST PERFORMANCE SUMMARY")
    print("=" * 70)
    for k, v in metrics.items():
        if k == "Strategy Breakdown":
            print("\n  [Strategy Breakdown]:")
            for st_type, st_data in v.items():
                print(f"    - {st_type:16}: {st_data['trades']:3d} trades | Win Rate: {st_data['win_rate']}% | PnL: ${st_data['net_profit']:,.2f}")
        else:
            print(f"  {k:28}: {v}")
    print("=" * 70)

    print("\n[Recent 10 Trades]:")
    cols = ["entry_time", "exit_time", "type", "direction", "entry_price", "exit_price", "pnl_pts", "pnl_usd", "exit_reason"]
    avail_cols = [c for c in cols if c in trades_df.columns]
    print(trades_df[avail_cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
