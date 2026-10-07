"""
Complete Validation Suite for Order Flow & Volume Profile Strategy.
==================================================================
Performs:
1. 3-min vs 5-min baseline comparison
2. Slippage (0.5 pts/side) + Commission ($4.50 RT) re-run on both 5min and 3min
3. Walk-Forward: Train (Sep 2025 - Apr 2026) vs Test (May 2026 - Sep 2026)
4. Parameter Sensitivity Grid (Absorption x CVD Lookback x RR = 36 combos)
5. Top 20 Winning Trades Deep-Dive
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# VECTORIZED SIGNAL GENERATION & SIMULATION ENGINE ON CACHED BARS
# =============================================================================

def detect_signals_on_cached_bars(
    bars_df: pd.DataFrame,
    absorption_mult: float = 1.3,
    cvd_lookback: int = 6,
    risk_reward: float = 1.8,
    level_buffer_pts: float = 8.0
) -> pd.DataFrame:
    """
    Cached bars par fast vectorized + iterative signal generation.
    """
    df = bars_df.copy()

    # 1. Absorption Features
    high_vol = df["volume"] > (absorption_mult * df["vol_ma"])
    bullish_absorption = high_vol & (df["lower_wick"] >= 0.40 * df["candle_range"]) & (df["candle_range"] > 0)
    bearish_absorption = high_vol & (df["upper_wick"] >= 0.40 * df["candle_range"]) & (df["candle_range"] > 0)
    df["absorption_bull"] = bullish_absorption
    df["absorption_bear"] = bearish_absorption

    # 2. CVD Divergence
    price_low_rolling = df["low"].rolling(cvd_lookback, min_periods=3).min()
    price_high_rolling = df["high"].rolling(cvd_lookback, min_periods=3).max()
    cvd_low_rolling = df["cvd"].rolling(cvd_lookback, min_periods=3).min()
    cvd_high_rolling = df["cvd"].rolling(cvd_lookback, min_periods=3).max()

    price_at_low = df["low"] <= (price_low_rolling.shift(1) + 2.0)
    cvd_higher = df["cvd"] > cvd_low_rolling.shift(1)
    df["cvd_divergence_bull"] = price_at_low & cvd_higher

    price_at_high = df["high"] >= (price_high_rolling.shift(1) - 2.0)
    cvd_lower = df["cvd"] < cvd_high_rolling.shift(1)
    df["cvd_divergence_bear"] = price_at_high & cvd_lower

    # 3. Signals & Risk Targets
    times = df.index.time
    w1_start = pd.to_datetime("09:30").time()
    w1_end = pd.to_datetime("11:00").time()
    w2_start = pd.to_datetime("13:30").time()
    w2_end = pd.to_datetime("15:00").time()
    in_window = ((times >= w1_start) & (times <= w1_end)) | ((times >= w2_start) & (times <= w2_end))

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    deltas = df["delta"].values
    vwaps = df["vwap"].values
    vwap_slopes = df["vwap_slope"].values
    vwap_dist_sigmas = df["vwap_dist_sigma"].values
    volumes = df["volume"].values
    vol_mas = df["vol_ma"].values

    rth_pocs = df["rth_poc"].values
    rth_vahs = df["rth_vah"].values
    rth_vals = df["rth_val"].values

    abs_bulls = df["absorption_bull"].values
    abs_bears = df["absorption_bear"].values
    cvd_bulls = df["cvd_divergence_bull"].values
    cvd_bears = df["cvd_divergence_bear"].values

    signals = np.zeros(len(df), dtype=int)
    trade_types = [None] * len(df)
    sls = np.full(len(df), np.nan)
    tps = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        if not in_window[i]:
            continue

        c = closes[i]
        h = highs[i]
        l = lows[i]
        prev_c = closes[i - 1]
        delta = deltas[i]
        vwap = vwaps[i]
        slope = vwap_slopes[i]
        sigma = vwap_dist_sigmas[i]

        poc = rth_pocs[i]
        vah = rth_vahs[i]
        val = rth_vals[i]

        if np.isnan(val) or np.isnan(vah):
            continue

        dist_val = abs(l - val)
        dist_vah = abs(h - vah)

        # REVERSAL LONG
        near_val = dist_val <= level_buffer_pts
        bull_confirm = (abs_bulls[i] or cvd_bulls[i]) and (delta > 0)
        vwap_room_long = c <= (vwap + 5.0)

        if near_val and bull_confirm and vwap_room_long:
            sl = min(l, l - 8.0)
            risk = c - sl
            if risk > 4.0:
                tp_target = poc if (poc > c) else (vah if vah > c else c + (risk_reward * risk))
                tp = max(tp_target, c + (risk_reward * risk))
                signals[i] = 1
                trade_types[i] = "REVERSAL_LONG"
                sls[i] = sl
                tps[i] = tp
                continue

        # REVERSAL SHORT
        near_vah = dist_vah <= level_buffer_pts
        bear_confirm = (abs_bears[i] or cvd_bears[i]) and (delta < 0)
        vwap_room_short = c >= (vwap - 5.0)

        if near_vah and bear_confirm and vwap_room_short:
            sl = max(h, h + 8.0)
            risk = sl - c
            if risk > 4.0:
                tp_target = poc if (poc < c) else (val if val < c else c - (risk_reward * risk))
                tp = min(tp_target, c - (risk_reward * risk))
                signals[i] = -1
                trade_types[i] = "REVERSAL_SHORT"
                sls[i] = sl
                tps[i] = tp
                continue

        # BREAKOUT LONG
        crossed_vah = (prev_c <= vah + 2.0) and (c > vah + 2.0)
        bo_vol_long = (volumes[i] > 1.2 * vol_mas[i]) and (delta > 100)
        vwap_up = (c > vwap) and (slope > 0) and (sigma < 2.5)

        if crossed_vah and bo_vol_long and vwap_up:
            sl = vah - 6.0
            risk = c - sl
            if risk > 4.0:
                signals[i] = 1
                trade_types[i] = "BREAKOUT_LONG"
                sls[i] = sl
                tps[i] = c + (2.0 * risk)
                continue

        # BREAKOUT SHORT
        crossed_val = (prev_c >= val - 2.0) and (c < val - 2.0)
        bo_vol_short = (volumes[i] > 1.2 * vol_mas[i]) and (delta < -100)
        vwap_down = (c < vwap) and (slope < 0) and (sigma > -2.5)

        if crossed_val and bo_vol_short and vwap_down:
            sl = val + 6.0
            risk = sl - c
            if risk > 4.0:
                signals[i] = -1
                trade_types[i] = "BREAKOUT_SHORT"
                sls[i] = sl
                tps[i] = c - (2.0 * risk)
                continue

    df["signal"] = signals
    df["trade_type"] = trade_types
    df["stop_loss"] = sls
    df["take_profit"] = tps
    return df


def simulate_trades_fast(
    df: pd.DataFrame,
    point_value: float = 20.0,
    slippage_pts_per_side: float = 0.0,
    commission_round_turn: float = 4.10,
    enable_trailing_ctc: bool = True
) -> pd.DataFrame:
    """
    Simulation engine with realistic slippage and commission.
    """
    trades: List[Dict] = []
    current_position = 0
    entry_price = 0.0
    entry_time = None
    stop_loss = 0.0
    take_profit = 0.0
    trade_type = ""
    is_breakeven_set = False

    dates = df.index.date
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    signals = df["signal"].values
    sls = df["stop_loss"].values
    tps = df["take_profit"].values
    ttypes = df["trade_type"].values
    times = df.index

    for i in range(len(df)):
        h = highs[i]
        l = lows[i]
        c = closes[i]
        t = times[i]
        sig = signals[i]

        is_day_end = (i == len(df) - 1) or (dates[i] != dates[i + 1])

        if current_position == 1:
            if l <= stop_loss:
                raw_exit = stop_loss
                actual_entry = entry_price + slippage_pts_per_side
                actual_exit = raw_exit - slippage_pts_per_side
                pnl_pts = actual_exit - actual_entry
                pnl_usd = (pnl_pts * point_value) - commission_round_turn
                trades.append({
                    "entry_time": entry_time, "exit_time": t, "type": trade_type,
                    "direction": "LONG", "entry_price": actual_entry, "exit_price": actual_exit,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "STOP_LOSS"
                })
                current_position = 0
                continue

            if h >= take_profit:
                raw_exit = take_profit
                actual_entry = entry_price + slippage_pts_per_side
                actual_exit = raw_exit - slippage_pts_per_side
                pnl_pts = actual_exit - actual_entry
                pnl_usd = (pnl_pts * point_value) - commission_round_turn
                trades.append({
                    "entry_time": entry_time, "exit_time": t, "type": trade_type,
                    "direction": "LONG", "entry_price": actual_entry, "exit_price": actual_exit,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "TAKE_PROFIT"
                })
                current_position = 0
                continue

            if is_day_end:
                raw_exit = c
                actual_entry = entry_price + slippage_pts_per_side
                actual_exit = raw_exit - slippage_pts_per_side
                pnl_pts = actual_exit - actual_entry
                pnl_usd = (pnl_pts * point_value) - commission_round_turn
                trades.append({
                    "entry_time": entry_time, "exit_time": t, "type": trade_type,
                    "direction": "LONG", "entry_price": actual_entry, "exit_price": actual_exit,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "END_OF_SESSION"
                })
                current_position = 0
                continue

            if enable_trailing_ctc:
                risk_pts = entry_price - sls[i] if not np.isnan(sls[i]) else 10.0
                if not is_breakeven_set and (h - entry_price) >= abs(risk_pts):
                    stop_loss = entry_price + 1.0
                    is_breakeven_set = True
                elif is_breakeven_set:
                    new_sl = l - 3.0
                    if new_sl > stop_loss:
                        stop_loss = new_sl

        elif current_position == -1:
            if h >= stop_loss:
                raw_exit = stop_loss
                actual_entry = entry_price - slippage_pts_per_side
                actual_exit = raw_exit + slippage_pts_per_side
                pnl_pts = actual_entry - actual_exit
                pnl_usd = (pnl_pts * point_value) - commission_round_turn
                trades.append({
                    "entry_time": entry_time, "exit_time": t, "type": trade_type,
                    "direction": "SHORT", "entry_price": actual_entry, "exit_price": actual_exit,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "STOP_LOSS"
                })
                current_position = 0
                continue

            if l <= take_profit:
                raw_exit = take_profit
                actual_entry = entry_price - slippage_pts_per_side
                actual_exit = raw_exit + slippage_pts_per_side
                pnl_pts = actual_entry - actual_exit
                pnl_usd = (pnl_pts * point_value) - commission_round_turn
                trades.append({
                    "entry_time": entry_time, "exit_time": t, "type": trade_type,
                    "direction": "SHORT", "entry_price": actual_entry, "exit_price": actual_exit,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "TAKE_PROFIT"
                })
                current_position = 0
                continue

            if is_day_end:
                raw_exit = c
                actual_entry = entry_price - slippage_pts_per_side
                actual_exit = raw_exit + slippage_pts_per_side
                pnl_pts = actual_entry - actual_exit
                pnl_usd = (pnl_pts * point_value) - commission_round_turn
                trades.append({
                    "entry_time": entry_time, "exit_time": t, "type": trade_type,
                    "direction": "SHORT", "entry_price": actual_entry, "exit_price": actual_exit,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "END_OF_SESSION"
                })
                current_position = 0
                continue

            if enable_trailing_ctc:
                risk_pts = sls[i] - entry_price if not np.isnan(sls[i]) else 10.0
                if not is_breakeven_set and (entry_price - l) >= abs(risk_pts):
                    stop_loss = entry_price - 1.0
                    is_breakeven_set = True
                elif is_breakeven_set:
                    new_sl = h + 3.0
                    if new_sl < stop_loss:
                        stop_loss = new_sl

        # New Entry
        if current_position == 0 and sig != 0 and not is_day_end:
            current_position = sig
            entry_price = c
            entry_time = t
            stop_loss = sls[i]
            take_profit = tps[i]
            trade_type = ttypes[i]
            is_breakeven_set = False

    return pd.DataFrame(trades)


def compute_metrics(trades_df: pd.DataFrame, initial_capital: float = 100_000.0) -> Dict:
    if trades_df.empty:
        return {
            "Total Trades": 0, "Win Rate [%]": 0.0, "Net Profit ($)": 0.0,
            "Profit Factor": 0.0, "Max Drawdown ($)": 0.0, "Max Drawdown [%]": 0.0,
            "Avg Trade ($)": 0.0, "Avg Win ($)": 0.0, "Avg Loss ($)": 0.0,
            "Payoff Ratio": 0.0, "Sharpe": 0.0
        }

    df = trades_df.copy().sort_values("entry_time").reset_index(drop=True)
    df["cum_pnl"] = df["pnl_usd"].cumsum()
    df["equity"] = initial_capital + df["cum_pnl"]
    df["peak"] = df["equity"].cummax()
    df["dd_usd"] = df["peak"] - df["equity"]
    df["dd_pct"] = (df["dd_usd"] / df["peak"]) * 100.0

    wins = df[df["pnl_usd"] > 0]
    losses = df[df["pnl_usd"] <= 0]
    total_trades = len(df)
    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
    net_pnl = df["pnl_usd"].sum()
    gross_win = wins["pnl_usd"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl_usd"].sum()) if not losses.empty else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else np.nan

    avg_win = wins["pnl_usd"].mean() if not wins.empty else 0.0
    avg_loss = abs(losses["pnl_usd"].mean()) if not losses.empty else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else np.nan

    df["entry_date"] = pd.to_datetime(df["entry_time"], utc=True).dt.date
    daily_pnl = df.groupby("entry_date")["pnl_usd"].sum()
    daily_ret = daily_pnl / initial_capital
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if (daily_ret.std() and daily_ret.std() > 0) else np.nan

    return {
        "Total Trades": total_trades,
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Win Rate [%]": round(win_rate, 2),
        "Net Profit ($)": round(net_pnl, 2),
        "Gross Profit ($)": round(gross_win, 2),
        "Gross Loss ($)": round(gross_loss, 2),
        "Profit Factor": round(pf, 2) if not np.isnan(pf) else 0.0,
        "Max Drawdown ($)": round(df["dd_usd"].max(), 2),
        "Max Drawdown [%]": round(df["dd_pct"].max(), 2),
        "Avg Trade ($)": round(df["pnl_usd"].mean(), 2),
        "Avg Win ($)": round(avg_win, 2),
        "Avg Loss ($)": round(avg_loss, 2),
        "Payoff Ratio": round(payoff, 2) if not np.isnan(payoff) else 0.0,
        "Sharpe": round(sharpe, 2) if not np.isnan(sharpe) else 0.0
    }


# =============================================================================
# VALIDATION SUITE MAIN EXECUTION
# =============================================================================

def execute_validation():
    cache_dir = Path("data_cache")
    p5 = cache_dir / "bars_5min.parquet"
    p3 = cache_dir / "bars_3min.parquet"

    if not p5.exists() or not p3.exists():
        print("[!] Feature cache parquet files not ready yet.")
        return

    print("Loading cached bars (5min & 3min)...")
    bars_5m = pd.read_parquet(p5)
    bars_3m = pd.read_parquet(p3)
    print(f"Loaded: 5-min ({len(bars_5m):,} bars) | 3-min ({len(bars_3m):,} bars)")

    results_dir = Path("results/validation")
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1 & 2: 3-MIN VS 5-MIN & FRICTION (SLIPPAGE + COMMISSION)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [STEP 1 & 2] BASELINE VS REALISTIC FRICTION (5M VS 3M)")
    print("=" * 70)

    # 1. 5-min Baseline (Friction: $4.10 RT commission, 0 slippage)
    sig_5m = detect_signals_on_cached_bars(bars_5m, absorption_mult=1.3, cvd_lookback=6, risk_reward=1.8)
    tr_5m_base = simulate_trades_fast(sig_5m, slippage_pts_per_side=0.0, commission_round_turn=4.10)
    m_5m_base = compute_metrics(tr_5m_base)

    # 2. 5-min Realistic Friction (Slippage: 0.5 pts/side = $20 round-trip + Commission: $4.50 round-turn)
    tr_5m_fric = simulate_trades_fast(sig_5m, slippage_pts_per_side=0.5, commission_round_turn=4.50)
    m_5m_fric = compute_metrics(tr_5m_fric)

    # 3. 3-min Baseline
    sig_3m = detect_signals_on_cached_bars(bars_3m, absorption_mult=1.3, cvd_lookback=6, risk_reward=1.8)
    tr_3m_base = simulate_trades_fast(sig_3m, slippage_pts_per_side=0.0, commission_round_turn=4.10)
    m_3m_base = compute_metrics(tr_3m_base)

    # 4. 3-min Realistic Friction
    tr_3m_fric = simulate_trades_fast(sig_3m, slippage_pts_per_side=0.5, commission_round_turn=4.50)
    m_3m_fric = compute_metrics(tr_3m_fric)

    step1_2_table = pd.DataFrame({
        "5m Baseline": m_5m_base,
        "5m (0.5pt Slip + $4.50 Comm)": m_5m_fric,
        "3m Baseline": m_3m_base,
        "3m (0.5pt Slip + $4.50 Comm)": m_3m_fric
    })
    step1_2_table.to_csv(results_dir / "step1_step2_comparison.csv")
    print(step1_2_table.to_string())

    # -------------------------------------------------------------------------
    # STEP 3: WALK-FORWARD (TRAIN: SEP 2025 - APR 2026 vs TEST: MAY 2026 - SEP 2026)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [STEP 3] WALK-FORWARD: IN-SAMPLE (TRAIN) VS OUT-OF-SAMPLE (TEST)")
    print("=" * 70)

    # Using Realistic Friction on 5-min
    tr_wf = tr_5m_fric.copy()
    tr_wf["entry_date"] = pd.to_datetime(tr_wf["entry_time"], utc=True).dt.date
    split_date = pd.to_datetime("2026-05-01").date()

    train_trades = tr_wf[tr_wf["entry_date"] < split_date].copy()
    test_trades = tr_wf[tr_wf["entry_date"] >= split_date].copy()

    m_train = compute_metrics(train_trades)
    m_test = compute_metrics(test_trades)

    wf_table = pd.DataFrame({
        "Train (IS: Sep 2025 - Apr 2026)": m_train,
        "Test (OOS: May 2026 - Sep 2026)": m_test
    })
    wf_table.to_csv(results_dir / "step3_walk_forward.csv")
    print(wf_table.to_string())

    # -------------------------------------------------------------------------
    # STEP 4: PARAMETER SENSITIVITY GRID
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [STEP 4] PARAMETER SENSITIVITY GRID (36 COMBINATIONS)")
    print("=" * 70)

    absorptions = [1.1, 1.3, 1.5, 1.8]
    cvd_lookbacks = [3, 5, 8]
    risk_rewards = [1.5, 1.8, 2.0]

    sensitivity_rows = []
    t_start_grid = time.time()

    for abs_m in absorptions:
        for cvd_lb in cvd_lookbacks:
            for rr in risk_rewards:
                sig_grid = detect_signals_on_cached_bars(
                    bars_5m, absorption_mult=abs_m, cvd_lookback=cvd_lb, risk_reward=rr
                )
                tr_grid = simulate_trades_fast(
                    sig_grid, slippage_pts_per_side=0.5, commission_round_turn=4.50
                )
                m = compute_metrics(tr_grid)
                sensitivity_rows.append({
                    "Absorption": abs_m,
                    "CVD_Lookback": cvd_lb,
                    "RR_Target": rr,
                    "Trades": m["Total Trades"],
                    "Win_Rate_%": m["Win Rate [%]"],
                    "Net_Profit_$": m["Net Profit ($)"],
                    "Profit_Factor": m["Profit Factor"],
                    "Max_DD_$": m["Max Drawdown ($)"],
                    "Payoff_Ratio": m["Payoff Ratio"],
                    "Sharpe": m["Sharpe"]
                })

    sens_df = pd.DataFrame(sensitivity_rows).sort_values("Net_Profit_$", ascending=False)
    sens_df.to_csv(results_dir / "step4_parameter_sensitivity.csv", index=False)
    print(f"Grid finished in {time.time() - t_start_grid:.2f}s!")
    print("\nTop 10 Parameter Combinations:")
    print(sens_df.head(10).to_string(index=False))
    print("\nBottom 5 Parameter Combinations:")
    print(sens_df.tail(5).to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 5: TOP 20 BIGGEST WINNING TRADES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [STEP 5] TOP 20 BIGGEST WINNING TRADES DEEP-DIVE")
    print("=" * 70)

    top20 = tr_5m_fric.sort_values("pnl_usd", ascending=False).head(20).copy()
    top20["duration_min"] = (pd.to_datetime(top20["exit_time"]) - pd.to_datetime(top20["entry_time"])).dt.total_seconds() / 60.0
    cols_top20 = [
        "entry_time", "exit_time", "duration_min", "type", "direction",
        "entry_price", "exit_price", "pnl_pts", "pnl_usd", "exit_reason"
    ]
    top20_display = top20[cols_top20]
    top20_display.to_csv(results_dir / "step5_top20_winners.csv", index=False)
    print(top20_display.to_string(index=False))

    # Save summary json
    summary_data = {
        "step1_2": step1_2_table.to_dict(),
        "step3_wf": wf_table.to_dict(),
        "step4_top5": sens_df.head(5).to_dict(orient="records"),
        "step5_top20": top20_display.to_dict(orient="records")
    }
    with open(results_dir / "validation_summary.json", "w") as fp:
        json.dump(summary_data, fp, indent=2, default=str)
    print("\n[SUCCESS] Complete validation suite executed and saved to results/validation/!")


if __name__ == "__main__":
    execute_validation()
