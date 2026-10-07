"""
Advanced Order Flow + Volume Profile Strategy for NQ Futures.
============================================================
Features:
1. Volume Profile:
   - RTH Session Profile (09:30 - 16:00 ET)
   - ETH Session Profile (18:00 - 09:30 ET)
   - Levels: VAH, VAL, POC, LVNs calculated on 15-minute resolution.
2. Order Flow Confirmations:
   - Absorption (High volume bar with price rejection at key level)
   - Cumulative Volume Delta (CVD) Divergence (Bullish / Bearish)
   - Delta confirmation (Trigger bar delta sign)
3. Trade Types:
   - Reversal (Fade at VAH / VAL / LVN)
   - Breakout (Momentum expansion through VAH / VAL)
4. Execution Timeframe:
   - 3-minute or 5-minute candles
5. VWAP Confluence:
   - Session VWAP, Standard Deviation Bands, Slope and Distance filter
6. Time Filter:
   - 09:30 - 11:00 ET (Morning Window)
   - 13:30 - 15:00 ET (Afternoon Window)
7. Risk Management:
   - Fixed Position Size: 1 Contract
   - Dynamic Stop Loss (Beyond Level / Previous Swing)
   - Minimum 1:1.8 Risk-to-Reward Ratio (Target at next key level)
   - Trailing Stop Loss / CTC (Chasing The Close / Breakeven lock)
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import vectorbt as vbt


# =============================================================================
# 1. VOLUME PROFILE FUNCTIONS (RTH, ETH, POC, VAH, VAL, LVN)
# =============================================================================

def calculate_volume_profile(
    prices: pd.Series,
    volumes: pd.Series,
    tick_size: float = 1.0,
    va_pct: float = 0.70
) -> Dict:
    """
    Price aur volume distribution se Volume Profile calculate karta hai.
    
    Returns:
        poc: Point of Control (Sabse zyada volume wala price)
        vah: Value Area High (70% Value Area ki upper boundary)
        val: Value Area Low (70% Value Area ki lower boundary)
        lvns: Low Volume Nodes (Liquidity voids / rejection zones)
    """
    if len(prices) == 0 or len(volumes) == 0:
        return {"poc": None, "vah": None, "val": None, "lvns": [], "vp": pd.Series(dtype=float)}

    # Price ko tick size me bin karte hain (NQ ke liye 1.0 ya 0.25 point)
    binned_prices = np.round(prices / tick_size) * tick_size
    vp = pd.Series(volumes.values, index=binned_prices).groupby(level=0).sum().sort_index()

    if vp.empty:
        return {"poc": None, "vah": None, "val": None, "lvns": [], "vp": vp}

    # 1. Point of Control (POC)
    poc = float(vp.idxmax())
    total_volume = float(vp.sum())
    target_va_vol = total_volume * va_pct

    # 2. Value Area Calculation (70% Volume around POC)
    poc_idx = vp.index.get_loc(poc)
    accum_vol = float(vp.iloc[poc_idx])
    up_idx = poc_idx + 1
    down_idx = poc_idx - 1

    while accum_vol < target_va_vol and (up_idx < len(vp) or down_idx >= 0):
        vol_up = float(vp.iloc[up_idx]) if up_idx < len(vp) else 0.0
        vol_down = float(vp.iloc[down_idx]) if down_idx >= 0 else 0.0

        if vol_up >= vol_down and up_idx < len(vp):
            accum_vol += vol_up
            up_idx += 1
        elif down_idx >= 0:
            accum_vol += vol_down
            down_idx -= 1
        else:
            up_idx += 1

    val = float(vp.index[max(0, down_idx + 1)])
    vah = float(vp.index[min(len(vp) - 1, up_idx - 1)])

    # 3. Low Volume Nodes (LVNs) - local minima in profile
    smoothed_vp = vp.rolling(window=5, center=True).mean().dropna()
    lvns = []
    avg_vol = float(vp.mean())
    for i in range(1, len(smoothed_vp) - 1):
        is_valley = (smoothed_vp.iloc[i] < smoothed_vp.iloc[i - 1]) and (smoothed_vp.iloc[i] < smoothed_vp.iloc[i + 1])
        if is_valley and smoothed_vp.iloc[i] < (avg_vol * 0.65):
            lvns.append(float(smoothed_vp.index[i]))

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "lvns": lvns,
        "vp": vp,
        "total_volume": total_volume
    }


def calculate_levels_from_vp(vp_series: pd.Series, va_pct: float = 0.70) -> Tuple[float, float, float, List[float]]:
    """
    Given a sorted price-volume Series, computes POC, VAH, VAL, and LVNs.
    """
    if vp_series.empty or vp_series.sum() == 0:
        return np.nan, np.nan, np.nan, []

    poc = float(vp_series.idxmax())
    total_volume = float(vp_series.sum())
    target_va_vol = total_volume * va_pct

    poc_idx = vp_series.index.get_loc(poc)
    accum_vol = float(vp_series.iloc[poc_idx])
    up_idx = poc_idx + 1
    down_idx = poc_idx - 1
    n = len(vp_series)

    while accum_vol < target_va_vol and (up_idx < n or down_idx >= 0):
        vol_up = float(vp_series.iloc[up_idx]) if up_idx < n else 0.0
        vol_down = float(vp_series.iloc[down_idx]) if down_idx >= 0 else 0.0
        if vol_up >= vol_down and up_idx < n:
            accum_vol += vol_up
            up_idx += 1
        elif down_idx >= 0:
            accum_vol += vol_down
            down_idx -= 1
        else:
            up_idx += 1

    val = float(vp_series.index[max(0, down_idx + 1)])
    vah = float(vp_series.index[min(n - 1, up_idx - 1)])

    # LVNs: local valleys with smoothed VP
    smoothed = vp_series.rolling(window=5, center=True).mean().dropna()
    lvns = []
    avg_vol = float(vp_series.mean())
    for i in range(1, len(smoothed) - 1):
        if (smoothed.iloc[i] < smoothed.iloc[i - 1]) and (smoothed.iloc[i] < smoothed.iloc[i + 1]) and (smoothed.iloc[i] < (avg_vol * 0.65)):
            lvns.append(float(smoothed.index[i]))

    return poc, vah, val, lvns


def compute_session_profiles(
    trades_df: pd.DataFrame,
    bar_index: pd.DatetimeIndex = None,
    tick_size: float = 1.0,
    va_pct: float = 0.70
) -> Dict:
    """
    RTH (09:30 - 16:00 ET) aur ETH (Overnight) session ke Volume Profiles banata hai.

    Causal / Expanding Mode (Zero Look-Ahead Bias):
    Jab bar_index provide kiya jata hai, har bar t ke liye RTH profile strictly [09:30, bar_t]
    tak ke volume se banta hai. Jaise-jaise din aage badhta hai, profile expand hota hai.
    Overnight ETH profile 09:30 baje fully complete hoti hai aur pure RTH ke liye fixed rehti hai.
    """
    times = trades_df.index.time
    rth_start = pd.to_datetime("09:30").time()
    rth_end = pd.to_datetime("16:00").time()

    eth_trades = trades_df[times < rth_start]
    rth_trades = trades_df[(times >= rth_start) & (times < rth_end)]

    # 1. Overnight ETH Profile (Static throughout RTH session, zero lookahead)
    eth_profile = calculate_volume_profile(eth_trades["price"], eth_trades["size"], tick_size=tick_size, va_pct=va_pct)
    rth_profile = calculate_volume_profile(rth_trades["price"], rth_trades["size"], tick_size=tick_size, va_pct=va_pct)

    if bar_index is None:
        return {
            "rth": rth_profile,
            "eth": eth_profile
        }

    # 2. Causal Expanding RTH Profile bar-by-bar
    eth_poc = eth_profile.get("poc", np.nan)
    eth_vah = eth_profile.get("vah", np.nan)
    eth_val = eth_profile.get("val", np.nan)
    eth_lvns = eth_profile.get("lvns", [])

    rth_poc, rth_vah, rth_val, rth_lvns = np.nan, np.nan, np.nan, []
    expanding_records = []

    if rth_trades.empty or len(bar_index) == 0:
        for bt in bar_index:
            expanding_records.append({
                "bar_time": bt,
                "rth_poc": np.nan, "rth_vah": np.nan, "rth_val": np.nan, "rth_lvns": [],
                "eth_poc": eth_poc, "eth_vah": eth_vah, "eth_val": eth_val, "eth_lvns": eth_lvns
            })
        exp_df = pd.DataFrame(expanding_records).set_index("bar_time")
        return {
            "expanding": exp_df,
            "eth": eth_profile,
            "rth": rth_profile
        }

    # Pre-bin RTH trades
    binned_prices = (np.round(rth_trades["price"].values / tick_size) * tick_size)
    trade_sizes = rth_trades["size"].values
    trade_times = rth_trades.index.values

    # Map trades to bar indices
    bar_times_arr = bar_index.values
    bar_assigned = np.searchsorted(bar_times_arr, trade_times, side="right") - 1

    df_temp = pd.DataFrame({
        "bar_idx": bar_assigned,
        "price": binned_prices,
        "size": trade_sizes
    })
    grouped_bar_vp = df_temp.groupby(["bar_idx", "price"])["size"].sum()

    cum_vp = pd.Series(dtype=float)
    rth_bars_mask = (bar_index.time >= rth_start) & (bar_index.time < rth_end)

    for idx, bt in enumerate(bar_index):
        if not rth_bars_mask[idx]:
            expanding_records.append({
                "bar_time": bt,
                "rth_poc": np.nan, "rth_vah": np.nan, "rth_val": np.nan, "rth_lvns": [],
                "eth_poc": eth_poc, "eth_vah": eth_vah, "eth_val": eth_val, "eth_lvns": eth_lvns
            })
            continue

        if idx in grouped_bar_vp.index.levels[0]:
            bar_vp = grouped_bar_vp.loc[idx]
            cum_vp = cum_vp.add(bar_vp, fill_value=0).sort_index()

        poc, vah, val, lvns = calculate_levels_from_vp(cum_vp, va_pct=va_pct)
        rth_poc, rth_vah, rth_val, rth_lvns = poc, vah, val, lvns

        expanding_records.append({
            "bar_time": bt,
            "rth_poc": poc, "rth_vah": vah, "rth_val": val, "rth_lvns": lvns,
            "eth_poc": eth_poc, "eth_vah": eth_vah, "eth_val": eth_val, "eth_lvns": eth_lvns
        })

    exp_df = pd.DataFrame(expanding_records).set_index("bar_time")
    return {
        "expanding": exp_df,
        "eth": eth_profile,
        "rth": rth_profile
    }


# =============================================================================
# 2. ORDER FLOW & CANDLESTICK AGGREGATION
# =============================================================================

def build_orderflow_bars(trades_df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """
    Raw tick trades se OHLCV + Order Flow (Delta, Buy/Sell Volume, CVD) bars create karta hai.
    """
    # Databento: side == 'A' means aggressor bought at Ask; 'B' means aggressor sold at Bid
    trade_size = trades_df["size"].astype(np.int64)
    buy_vol = np.where(trades_df["side"] == "A", trade_size, 0)
    sell_vol = np.where(trades_df["side"] == "B", trade_size, 0)
    delta = buy_vol - sell_vol

    # Temporary dataframe for fast resampling
    tf_df = pd.DataFrame({
        "price": trades_df["price"],
        "size": trade_size,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "delta": delta,
    }, index=trades_df.index)

    resampled = pd.DataFrame()
    resampled["open"] = tf_df["price"].resample(timeframe).first()
    resampled["high"] = tf_df["price"].resample(timeframe).max()
    resampled["low"] = tf_df["price"].resample(timeframe).min()
    resampled["close"] = tf_df["price"].resample(timeframe).last()
    resampled["volume"] = tf_df["size"].resample(timeframe).sum()
    resampled["buy_vol"] = tf_df["buy_vol"].resample(timeframe).sum()
    resampled["sell_vol"] = tf_df["sell_vol"].resample(timeframe).sum()
    resampled["delta"] = tf_df["delta"].resample(timeframe).sum()

    resampled = resampled.dropna().copy()

    # Cumulative Volume Delta (CVD)
    resampled["cvd"] = resampled["delta"].cumsum()

    # Candle range and wicks for absorption analysis
    resampled["body_size"] = (resampled["close"] - resampled["open"]).abs()
    resampled["candle_range"] = resampled["high"] - resampled["low"]
    resampled["upper_wick"] = resampled["high"] - resampled[["open", "close"]].max(axis=1)
    resampled["lower_wick"] = resampled[["open", "close"]].min(axis=1) - resampled["low"]

    # Rolling average volume & ATR (14 bars)
    resampled["vol_ma"] = resampled["volume"].rolling(14, min_periods=3).mean()
    resampled["atr"] = resampled["candle_range"].rolling(14, min_periods=3).mean()

    return resampled


# =============================================================================
# 3. SMART VWAP CONFLUENCE (VWAP, BANDS & SLOPE)
# =============================================================================

def calculate_smart_vwap(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Session-anchored VWAP, standard deviation bands (1σ, 2σ) aur slope calculate karta hai.
    """
    df = bars_df.copy()
    times = df.index.time
    rth_start = pd.to_datetime("09:30").time()

    # Reset VWAP at 09:30 ET
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]

    # Group by trading date for session VWAP
    dates = df.index.date
    df["session_date"] = dates

    # Cumulative values per session
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()
    df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)

    # Standard deviation bands
    sq_diff = ((typical_price - df["vwap"]) ** 2) * df["volume"]
    cum_sq_diff = sq_diff.groupby(dates).cumsum()
    variance = cum_sq_diff / cum_vol.replace(0, np.nan)
    std = np.sqrt(variance.clip(lower=0))

    df["vwap_std"] = std
    df["vwap_upper_1"] = df["vwap"] + std
    df["vwap_lower_1"] = df["vwap"] - std
    df["vwap_upper_2"] = df["vwap"] + 2.0 * std
    df["vwap_lower_2"] = df["vwap"] - 2.0 * std

    # VWAP Slope (5-bar slope)
    df["vwap_slope"] = df["vwap"].diff(5)

    # Distance in StdDev units
    df["vwap_dist_sigma"] = (df["close"] - df["vwap"]) / df["vwap_std"].replace(0, np.nan)

    return df


# =============================================================================
# 4. ORDER FLOW CONFIRMATIONS (ABSORPTION & CVD DIVERGENCE)
# =============================================================================

def detect_orderflow_features(bars_df: pd.DataFrame, level_proximity_pts: float = 8.0) -> pd.DataFrame:
    """
    Order Flow confirmations identify karta hai:
    - Absorption: High volume lekin price aage nahi badh rahi (wick rejection / small body)
    - Cumulative Delta Divergence: Price HH/LL vs CVD LH/HL
    """
    df = bars_df.copy()

    # 1. Absorption Detection:
    # High volume (1.3x rolling avg) + prominent wick opposite to move
    high_vol = df["volume"] > (1.3 * df["vol_ma"])
    bullish_absorption = high_vol & (df["lower_wick"] >= 0.40 * df["candle_range"]) & (df["candle_range"] > 0)
    bearish_absorption = high_vol & (df["upper_wick"] >= 0.40 * df["candle_range"]) & (df["candle_range"] > 0)

    df["absorption_bull"] = bullish_absorption
    df["absorption_bear"] = bearish_absorption

    # 2. Cumulative Delta (CVD) Divergence (Lookback = 5 to 10 bars)
    lookback = 6
    price_low_rolling = df["low"].rolling(lookback, min_periods=3).min()
    price_high_rolling = df["high"].rolling(lookback, min_periods=3).max()
    cvd_low_rolling = df["cvd"].rolling(lookback, min_periods=3).min()
    cvd_high_rolling = df["cvd"].rolling(lookback, min_periods=3).max()

    # Bullish Divergence: Price near lowest low of lookback, but CVD higher than previous low
    price_at_low = df["low"] <= (price_low_rolling.shift(1) + 2.0)
    cvd_higher = df["cvd"] > cvd_low_rolling.shift(1)
    df["cvd_divergence_bull"] = price_at_low & cvd_higher

    # Bearish Divergence: Price near highest high of lookback, but CVD lower than previous high
    price_at_high = df["high"] >= (price_high_rolling.shift(1) - 2.0)
    cvd_lower = df["cvd"] < cvd_high_rolling.shift(1)
    df["cvd_divergence_bear"] = price_at_high & cvd_lower

    return df


# =============================================================================
# 5. SIGNAL GENERATION ENGINE (REVERSAL & BREAKOUT)
# =============================================================================

def generate_orderflow_signals(
    bars_df: pd.DataFrame,
    profile_levels: Optional[Dict] = None,
    level_buffer_pts: float = 8.0
) -> pd.DataFrame:
    """
    Causal Volume Profile Levels + Order Flow + VWAP + Time Filter ko combine karke
    Trading Signals generate karta hai.

    Zero Look-Ahead:
    Har bar t ke liye uss bar ke developing RTH VAH/VAL/POC levels ya completed overnight
    ETH levels ka use karta hai.
    """
    df = bars_df.copy()

    # If profile_levels has expanding df, merge into bars if not already present
    if profile_levels and "expanding" in profile_levels:
        exp_df = profile_levels["expanding"]
        for col in exp_df.columns:
            if col not in df.columns:
                df[col] = exp_df[col].values

    # Signal columns: 1 = Long, -1 = Short, 0 = No trade
    df["signal"] = 0
    df["trade_type"] = None   # 'REVERSAL_LONG', 'REVERSAL_SHORT', 'BREAKOUT_LONG', 'BREAKOUT_SHORT'
    df["stop_loss"] = np.nan
    df["take_profit"] = np.nan

    times = df.index.time
    # Time Filters (Eastern Time):
    # Window 1: 09:30 - 11:00 ET
    # Window 2: 13:30 - 15:00 ET
    w1_start = pd.to_datetime("09:30").time()
    w1_end = pd.to_datetime("11:00").time()
    w2_start = pd.to_datetime("13:30").time()
    w2_end = pd.to_datetime("15:00").time()

    in_window = ((times >= w1_start) & (times <= w1_end)) | ((times >= w2_start) & (times <= w2_end))

    for i in range(1, len(df)):
        if not in_window[i]:
            continue

        bar = df.iloc[i]
        prev_bar = df.iloc[i - 1]
        close = bar["close"]
        low = bar["low"]
        high = bar["high"]
        delta = bar["delta"]
        vwap = bar["vwap"]
        vwap_slope = bar["vwap_slope"]

        # Causal Levels for this specific bar i
        rth_poc = bar.get("rth_poc", np.nan)
        rth_vah = bar.get("rth_vah", np.nan)
        rth_val = bar.get("rth_val", np.nan)
        rth_lvns = bar.get("rth_lvns", [])
        if isinstance(rth_lvns, str):
            rth_lvns = [float(x) for x in rth_lvns.split(",") if x.strip()] if rth_lvns else []
        elif not isinstance(rth_lvns, list):
            rth_lvns = []

        eth_poc = bar.get("eth_poc", np.nan)
        eth_vah = bar.get("eth_vah", np.nan)
        eth_val = bar.get("eth_val", np.nan)
        eth_lvns = bar.get("eth_lvns", [])
        if isinstance(eth_lvns, str):
            eth_lvns = [float(x) for x in eth_lvns.split(",") if x.strip()] if eth_lvns else []
        elif not isinstance(eth_lvns, list):
            eth_lvns = []

        # Fallback to static if expanding not present
        if np.isnan(rth_poc) and np.isnan(rth_vah) and profile_levels is not None:
            rth_p = profile_levels.get("rth", {})
            eth_p = profile_levels.get("eth", {})
            rth_poc = float(rth_p.get("poc")) if rth_p.get("poc") is not None else np.nan
            rth_vah = float(rth_p.get("vah")) if rth_p.get("vah") is not None else np.nan
            rth_val = float(rth_p.get("val")) if rth_p.get("val") is not None else np.nan
            eth_poc = float(eth_p.get("poc")) if eth_p.get("poc") is not None else np.nan
            eth_vah = float(eth_p.get("vah")) if eth_p.get("vah") is not None else np.nan
            eth_val = float(eth_p.get("val")) if eth_p.get("val") is not None else np.nan

        # Active levels: developing RTH preferred, completed ETH secondary
        vah = rth_vah if not np.isnan(rth_vah) else eth_vah
        val = rth_val if not np.isnan(rth_val) else eth_val
        poc = rth_poc if not np.isnan(rth_poc) else eth_poc
        lvns = rth_lvns + eth_lvns

        if np.isnan(val) and np.isnan(vah):
            continue

        # Level distances
        dist_val = 999.0
        if not np.isnan(rth_val): dist_val = min(dist_val, abs(low - rth_val))
        if not np.isnan(eth_val): dist_val = min(dist_val, abs(low - eth_val))

        dist_vah = 999.0
        if not np.isnan(rth_vah): dist_vah = min(dist_vah, abs(high - rth_vah))
        if not np.isnan(eth_vah): dist_vah = min(dist_vah, abs(high - eth_vah))

        # Nearest LVN check
        dist_lvn = min([abs(close - lvn) for lvn in lvns], default=999.0) if lvns else 999.0

        # -------------------------------------------------------------
        # A. REVERSAL LONG (Support at VAL / lower LVN / lower VWAP)
        # -------------------------------------------------------------
        near_support = (dist_val <= level_buffer_pts) or (dist_lvn <= level_buffer_pts and (not np.isnan(poc) and close < poc))
        bull_confirmation = (bar["absorption_bull"] or bar["cvd_divergence_bull"]) and (delta > 0)
        vwap_support_confluence = (close <= vwap + 5.0)  # Room to run up to VWAP / POC

        if near_support and bull_confirmation and vwap_support_confluence:
            sl = min(low, bar["low"] - 8.0)
            risk = close - sl
            if risk > 4.0:
                # Target: POC if above, else VAH, minimum 1:1.8 RR
                tp_target = poc if (not np.isnan(poc) and poc > close) else (vah if not np.isnan(vah) and vah > close else close + (1.8 * risk))
                tp = max(tp_target, close + (1.8 * risk))
                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("trade_type")] = "REVERSAL_LONG"
                df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                df.iloc[i, df.columns.get_loc("take_profit")] = tp
                continue

        # -------------------------------------------------------------
        # B. REVERSAL SHORT (Resistance at VAH / upper LVN / upper VWAP)
        # -------------------------------------------------------------
        near_resistance = (dist_vah <= level_buffer_pts) or (dist_lvn <= level_buffer_pts and (not np.isnan(poc) and close > poc))
        bear_confirmation = (bar["absorption_bear"] or bar["cvd_divergence_bear"]) and (delta < 0)
        vwap_res_confluence = (close >= vwap - 5.0)  # Room to fall down to VWAP / POC

        if near_resistance and bear_confirmation and vwap_res_confluence:
            sl = max(high, bar["high"] + 8.0)
            risk = sl - close
            if risk > 4.0:
                tp_target = poc if (not np.isnan(poc) and poc < close) else (val if not np.isnan(val) and val < close else close - (1.8 * risk))
                tp = min(tp_target, close - (1.8 * risk))
                df.iloc[i, df.columns.get_loc("signal")] = -1
                df.iloc[i, df.columns.get_loc("trade_type")] = "REVERSAL_SHORT"
                df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                df.iloc[i, df.columns.get_loc("take_profit")] = tp
                continue

        # -------------------------------------------------------------
        # C. BREAKOUT LONG (Expansion above VAH with strong Volume + Delta)
        # -------------------------------------------------------------
        crossed_above_vah = (prev_bar["close"] <= vah + 2.0) and (close > vah + 2.0) if not np.isnan(vah) else False
        breakout_vol_long = bar["volume"] > (1.2 * bar["vol_ma"]) and (delta > 100)
        vwap_trending_up = (close > vwap) and (vwap_slope > 0) and (bar["vwap_dist_sigma"] < 2.5)

        if crossed_above_vah and breakout_vol_long and vwap_trending_up:
            sl = vah - 6.0  # SL just below breakout level
            risk = close - sl
            if risk > 4.0:
                tp = close + (2.0 * risk)  # 1:2.0 RR for momentum expansion
                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("trade_type")] = "BREAKOUT_LONG"
                df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                df.iloc[i, df.columns.get_loc("take_profit")] = tp
                continue

        # -------------------------------------------------------------
        # D. BREAKOUT SHORT (Expansion below VAL with strong Volume + Delta)
        # -------------------------------------------------------------
        crossed_below_val = (prev_bar["close"] >= val - 2.0) and (close < val - 2.0) if not np.isnan(val) else False
        breakout_vol_short = bar["volume"] > (1.2 * bar["vol_ma"]) and (delta < -100)
        vwap_trending_down = (close < vwap) and (vwap_slope < 0) and (bar["vwap_dist_sigma"] > -2.5)

        if crossed_below_val and breakout_vol_short and vwap_trending_down:
            sl = val + 6.0  # SL just above breakdown level
            risk = sl - close
            if risk > 4.0:
                tp = close - (2.0 * risk)
                df.iloc[i, df.columns.get_loc("signal")] = -1
                df.iloc[i, df.columns.get_loc("trade_type")] = "BREAKOUT_SHORT"
                df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                df.iloc[i, df.columns.get_loc("take_profit")] = tp
                continue

    return df


def generate_prior_day_orderflow_signals(
    bars_df: pd.DataFrame,
    prior_rth_profile: Optional[Dict] = None,
    eth_profile: Optional[Dict] = None,
    max_trades_per_day: int = 2,
    buffer_pts: float = 10.0,
    max_sl_pts: float = 20.0,
    min_risk_pts: float = 4.0,
    min_rr: float = 1.8
) -> pd.DataFrame:
    """
    Prior Day RTH Volume Profile Rebuilt Strategy (Zero Look-Ahead):
    1. Primary Levels:
       - Previous Day RTH Volume Profile (VAH, VAL, LVNs)
       - Fully known and static before 09:30 ET
    2. Secondary Levels:
       - Overnight ETH Profile (VAH, VAL)
    3. Strict Time Filters:
       - Initial Balance (09:30 - 10:00 ET) strictly skipped!
       - Trading Windows: 10:00 - 11:00 ET and 13:30 - 15:00 ET
    4. Strict Entry & Acceptance/Rejection Rules:
       - Proximity within buffer_pts (8-10 pts)
       - Support (VAL/LVN): Low probes <= lvl + 4, candle closes bullish (close > open) back above lvl - 2
       - Resistance (VAH/LVN): High probes >= lvl - 4, candle closes bearish (close < open) back below lvl + 2
       - Orderflow Confirmation: (Absorption OR CVD Divergence OR Delta sign)
       - VWAP Slope Filter:
         * Long: vwap_slope >= -0.1 (flat or positive)
         * Short: vwap_slope <= 0.1 (flat or negative)
    5. Risk Management:
       - Fixed 1 contract
       - SL beyond swing / level (min 4 pts, max 20 pts)
       - TP: Minimum 1:1.8 RR or Prior POC target
       - Max 2 trades per day limit
    """
    df = bars_df.copy()
    df["signal"] = 0
    df["trade_type"] = None
    df["stop_loss"] = np.nan
    df["take_profit"] = np.nan

    p_rth = prior_rth_profile if prior_rth_profile else eth_profile
    eth = eth_profile

    # Support levels: Prior VAL, Prior LVNs below POC, Overnight VAL
    support_levels = []
    if p_rth and p_rth.get("val"):
        support_levels.append(("PRIOR_VAL", p_rth["val"]))
    if eth and eth.get("val"):
        support_levels.append(("ETH_VAL", eth["val"]))
    if p_rth:
        for lvn in p_rth.get("lvns", []):
            if p_rth.get("poc") and lvn < p_rth["poc"]:
                support_levels.append(("PRIOR_LVN", lvn))

    # Resistance levels: Prior VAH, Prior LVNs above POC, Overnight VAH
    resistance_levels = []
    if p_rth and p_rth.get("vah"):
        resistance_levels.append(("PRIOR_VAH", p_rth["vah"]))
    if eth and eth.get("vah"):
        resistance_levels.append(("ETH_VAH", eth["vah"]))
    if p_rth:
        for lvn in p_rth.get("lvns", []):
            if p_rth.get("poc") and lvn > p_rth["poc"]:
                resistance_levels.append(("PRIOR_LVN", lvn))

    # Time filters (Eastern Time):
    # Window 1: 10:00 - 11:00 ET (Initial Balance 09:30-10:00 strictly skipped)
    # Window 2: 13:30 - 15:00 ET
    w1_start = pd.to_datetime("10:00").time()
    w1_end = pd.to_datetime("11:00").time()
    w2_start = pd.to_datetime("13:30").time()
    w2_end = pd.to_datetime("15:00").time()

    times = df.index.time
    in_win = ((times >= w1_start) & (times <= w1_end)) | ((times >= w2_start) & (times <= w2_end))

    signals_today = 0

    for i in range(1, len(df)):
        if not in_win[i]:
            continue
        if signals_today >= max_trades_per_day:
            break

        bar = df.iloc[i]
        c, o, h, l = bar["close"], bar["open"], bar["high"], bar["low"]
        delta = bar["delta"]
        v_slope = bar["vwap_slope"]

        # --- A. LONG REVERSAL AT SUPPORT (VAL / LVN) ---
        best_supp = None
        min_s_dist = 999.0
        for name, lvl in support_levels:
            d = abs(l - lvl)
            if d < min_s_dist:
                min_s_dist = d
                best_supp = (name, lvl)

        if best_supp and min_s_dist <= buffer_pts:
            s_name, s_lvl = best_supp
            probed = l <= s_lvl + 4.0
            rejected = (c > o) and (c >= s_lvl - 2.0)
            confirm = (bar["absorption_bull"] or bar["cvd_divergence_bull"] or delta > 0)
            vwap_ok = (v_slope >= -0.1)

            if probed and rejected and confirm and vwap_ok:
                raw_sl = min(l - 4.0, s_lvl - 6.0)
                sl = max(raw_sl, c - max_sl_pts)
                risk = c - sl
                if risk >= min_risk_pts:
                    poc_target = p_rth.get("poc") if p_rth else None
                    if poc_target and poc_target > c + (min_rr * risk):
                        tp = poc_target
                    else:
                        tp = c + (min_rr * risk)
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    df.iloc[i, df.columns.get_loc("trade_type")] = f"LONG_{s_name}"
                    df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                    df.iloc[i, df.columns.get_loc("take_profit")] = tp
                    signals_today += 1
                    continue

        # --- B. SHORT REVERSAL AT RESISTANCE (VAH / LVN) ---
        best_res = None
        min_r_dist = 999.0
        for name, lvl in resistance_levels:
            d = abs(h - lvl)
            if d < min_r_dist:
                min_r_dist = d
                best_res = (name, lvl)

        if best_res and min_r_dist <= buffer_pts:
            r_name, r_lvl = best_res
            probed = h >= r_lvl - 4.0
            rejected = (c < o) and (c <= r_lvl + 2.0)
            confirm = (bar["absorption_bear"] or bar["cvd_divergence_bear"] or delta < 0)
            vwap_ok = (v_slope <= 0.1)

            if probed and rejected and confirm and vwap_ok:
                raw_sl = max(h + 4.0, r_lvl + 6.0)
                sl = min(raw_sl, c + max_sl_pts)
                risk = sl - c
                if risk >= min_risk_pts:
                    poc_target = p_rth.get("poc") if p_rth else None
                    if poc_target and poc_target < c - (min_rr * risk):
                        tp = poc_target
                    else:
                        tp = c - (min_rr * risk)
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("trade_type")] = f"SHORT_{r_name}"
                    df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                    df.iloc[i, df.columns.get_loc("take_profit")] = tp
                    signals_today += 1
                    continue

    return df


# =============================================================================
# 6. RISK MANAGEMENT & EXECUTION ENGINE (1 CONTRACT, CTC TRAILING SL/TP)
# =============================================================================

def simulate_orderflow_execution(
    signals_df: pd.DataFrame,
    point_value: float = 20.0,            # $20 per point for NQ futures (Micro MNQ = $2)
    slippage_pts: float = 0.5,            # Realistic 0.5 pt slippage per execution
    commission_per_contract: float = 2.25, # $4.50 round-trip commission ($2.25 per side)
    enable_trailing_ctc: bool = True
) -> Dict:
    """
    Order Flow Strategy ka Event-Driven Execution Simulator:
    - Fixed 1 contract position
    - Realistic Friction: 0.5 pt slippage/side ($20/trade) + $4.50 RT commission ($24.50 total friction)
    - Dynamic Stop Loss & 1:1.8+ RR Take Profit
    - Trailing Stop Loss (CTC style - Close to Close / Breakeven lock)
    """
    df = signals_df.copy()
    trades: List[Dict] = []
    current_position = 0  # 1: Long, -1: Short, 0: Flat
    entry_price = 0.0
    actual_entry_price = 0.0
    entry_time = None
    stop_loss = 0.0
    take_profit = 0.0
    trade_type = ""
    is_breakeven_set = False
    initial_risk_pts = 10.0

    comm_rt = commission_per_contract * 2.0

    for i in range(len(df)):
        bar = df.iloc[i]
        time = df.index[i]
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]
        signal = bar["signal"]

        # -------------------------------------------------------------
        # In-Trade Management (Check SL, TP & Trailing CTC)
        # -------------------------------------------------------------
        if current_position == 1:
            # Check Stop Loss
            if low <= stop_loss:
                actual_exit_price = stop_loss - slippage_pts
                pnl_pts = actual_exit_price - actual_entry_price
                pnl_usd = (pnl_pts * point_value) - comm_rt
                trades.append({
                    "entry_time": entry_time, "exit_time": time, "type": trade_type,
                    "direction": "LONG", "entry_price": actual_entry_price, "exit_price": actual_exit_price,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd,
                    "exit_reason": "BREAKEVEN_STOP" if is_breakeven_set else "STOP_LOSS"
                })
                current_position = 0
                continue

            # Check Take Profit
            if high >= take_profit:
                actual_exit_price = take_profit - slippage_pts
                pnl_pts = actual_exit_price - actual_entry_price
                pnl_usd = (pnl_pts * point_value) - comm_rt
                trades.append({
                    "entry_time": entry_time, "exit_time": time, "type": trade_type,
                    "direction": "LONG", "entry_price": actual_entry_price, "exit_price": actual_exit_price,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "TAKE_PROFIT"
                })
                current_position = 0
                continue

            # CTC Trailing Management:
            if enable_trailing_ctc:
                if not is_breakeven_set and (high - entry_price) >= initial_risk_pts:
                    stop_loss = entry_price + 1.0  # Lock Breakeven + 1 pt
                    is_breakeven_set = True
                elif is_breakeven_set:
                    new_sl = bar["low"] - 3.0
                    if new_sl > stop_loss:
                        stop_loss = new_sl

        elif current_position == -1:
            # Check Stop Loss
            if high >= stop_loss:
                actual_exit_price = stop_loss + slippage_pts
                pnl_pts = actual_entry_price - actual_exit_price
                pnl_usd = (pnl_pts * point_value) - comm_rt
                trades.append({
                    "entry_time": entry_time, "exit_time": time, "type": trade_type,
                    "direction": "SHORT", "entry_price": actual_entry_price, "exit_price": actual_exit_price,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd,
                    "exit_reason": "BREAKEVEN_STOP" if is_breakeven_set else "STOP_LOSS"
                })
                current_position = 0
                continue

            # Check Take Profit
            if low <= take_profit:
                actual_exit_price = take_profit + slippage_pts
                pnl_pts = actual_entry_price - actual_exit_price
                pnl_usd = (pnl_pts * point_value) - comm_rt
                trades.append({
                    "entry_time": entry_time, "exit_time": time, "type": trade_type,
                    "direction": "SHORT", "entry_price": actual_entry_price, "exit_price": actual_exit_price,
                    "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "TAKE_PROFIT"
                })
                current_position = 0
                continue

            # CTC Trailing Management for Short:
            if enable_trailing_ctc:
                if not is_breakeven_set and (entry_price - low) >= initial_risk_pts:
                    stop_loss = entry_price - 1.0  # Lock Breakeven - 1 pt
                    is_breakeven_set = True
                elif is_breakeven_set:
                    new_sl = bar["high"] + 3.0
                    if new_sl < stop_loss:
                        stop_loss = new_sl

        # -------------------------------------------------------------
        # New Entry Trigger (Agar Flat hain aur valid signal mila)
        # -------------------------------------------------------------
        if current_position == 0 and signal != 0:
            current_position = signal
            entry_price = close
            actual_entry_price = close + slippage_pts if signal == 1 else close - slippage_pts
            entry_time = time
            stop_loss = bar["stop_loss"]
            take_profit = bar["take_profit"]
            trade_type = bar["trade_type"]
            initial_risk_pts = abs(entry_price - stop_loss) if (stop_loss and not np.isnan(stop_loss)) else 10.0
            is_breakeven_set = False

    # Close any open trade at end of data
    if current_position != 0:
        last_bar = df.iloc[-1]
        exit_price = last_bar["close"]
        if current_position == 1:
            actual_exit_price = exit_price - slippage_pts
            pnl_pts = actual_exit_price - actual_entry_price
        else:
            actual_exit_price = exit_price + slippage_pts
            pnl_pts = actual_entry_price - actual_exit_price
        pnl_usd = (pnl_pts * point_value) - comm_rt
        trades.append({
            "entry_time": entry_time, "exit_time": df.index[-1], "type": trade_type,
            "direction": "LONG" if current_position == 1 else "SHORT",
            "entry_price": actual_entry_price, "exit_price": actual_exit_price,
            "pnl_pts": pnl_pts, "pnl_usd": pnl_usd, "exit_reason": "END_OF_SESSION"
        })

    # Summary Statistics
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {
            "trades_df": trades_df,
            "summary": {
                "Total Trades": 0,
                "Net Profit ($)": 0.0,
                "Win Rate [%]": 0.0,
                "Profit Factor": 0.0,
                "Avg Trade ($)": 0.0
            }
        }

    wins = trades_df[trades_df["pnl_usd"] > 0]
    losses = trades_df[trades_df["pnl_usd"] <= 0]
    win_rate = (len(wins) / len(trades_df)) * 100.0
    total_profit = wins["pnl_usd"].sum() if not wins.empty else 0.0
    total_loss = abs(losses["pnl_usd"].sum()) if not losses.empty else 0.0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else np.nan

    summary = {
        "Total Trades": len(trades_df),
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Win Rate [%]": round(win_rate, 2),
        "Net Profit ($)": round(trades_df["pnl_usd"].sum(), 2),
        "Gross Profit ($)": round(total_profit, 2),
        "Gross Loss ($)": round(total_loss, 2),
        "Profit Factor": round(profit_factor, 2) if not np.isnan(profit_factor) else "N/A",
        "Avg Trade ($)": round(trades_df["pnl_usd"].mean(), 2),
        "Max Winner ($)": round(trades_df["pnl_usd"].max(), 2),
        "Max Loser ($)": round(trades_df["pnl_usd"].min(), 2),
    }

    return {
        "trades_df": trades_df,
        "summary": summary
    }


# =============================================================================
# 7. HIGH-LEVEL PIPELINE RUNNER (VECTORBT & DIRECT SIMULATOR)
# =============================================================================

def run_orderflow_strategy_pipeline(
    trades_df: pd.DataFrame,
    entry_timeframe: str = "5min",
    tick_size: float = 1.0,
    point_value: float = 20.0,
    enable_vbt: bool = True
) -> Dict:
    """
    Advanced Order Flow Strategy ka complete pipeline:
    1. Resample to entry timeframe (3min ya 5min) with tick delta & CVD
    2. Session-anchored VWAP & Bands
    3. Causal Expanding Volume Profiles (RTH & ETH) bar-by-bar
    4. Order Flow Confirmations (Absorption, CVD Divergence, Delta flip)
    5. Reversal & Breakout Signals
    6. Risk Management & CTC Trailing Simulation with realistic friction
    7. VectorBT Portfolio Backtest
    """
    print(f"\n--- [Step 1] Building {entry_timeframe} Order Flow Bars ---")
    bars = build_orderflow_bars(trades_df, timeframe=entry_timeframe)
    print(f"Generated {len(bars)} {entry_timeframe} candles with CVD & Delta.")

    print("\n--- [Step 2] Computing Session VWAP & Bands ---")
    bars = calculate_smart_vwap(bars)

    print("\n--- [Step 3] Calculating Causal Expanding Volume Profiles ---")
    profiles = compute_session_profiles(trades_df, bar_index=bars.index, tick_size=tick_size)
    if "expanding" in profiles:
        exp_df = profiles["expanding"]
        for col in exp_df.columns:
            bars[col] = exp_df[col].values

    rth_p = profiles["rth"]
    eth_p = profiles["eth"]
    print(f"Overnight ETH Profile -> POC: {eth_p['poc']}, VAH: {eth_p['vah']}, VAL: {eth_p['val']}")
    print(f"Final RTH Profile     -> POC: {rth_p['poc']}, VAH: {rth_p['vah']}, VAL: {rth_p['val']}")

    print("\n--- [Step 4] Detecting Order Flow Confirmations (Absorption & CVD Divergence) ---")
    bars = detect_orderflow_features(bars)

    print("\n--- [Step 5] Generating Trade Signals (Reversals & Breakouts with Time Filter) ---")
    bars = generate_orderflow_signals(bars, profiles)
    total_signals = (bars["signal"] != 0).sum()
    print(f"Generated {total_signals} actionable signal(s) in designated windows.")

    print("\n--- [Step 6] Running Risk Management Simulation (1 Contract, CTC Trailing, Friction) ---")
    sim_results = simulate_orderflow_execution(
        bars,
        point_value=point_value,
        slippage_pts=0.5,
        commission_per_contract=2.25
    )

    vbt_portfolio = None
    if enable_vbt and total_signals > 0:
        print("\n--- [Step 7] Running VectorBT Portfolio Engine ---")
        entries = bars["signal"] == 1
        exits = bars["signal"] == -1
        vbt_portfolio = vbt.Portfolio.from_signals(
            close=bars["close"],
            entries=entries,
            exits=exits,
            init_cash=100_000.0,
            fees=0.0002,
            freq=entry_timeframe
        )

    return {
        "bars": bars,
        "profiles": profiles,
        "sim_results": sim_results,
        "vbt_portfolio": vbt_portfolio
    }
