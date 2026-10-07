"""
Precomputes and Caches Resampled 5-min and 3-min Order Flow Bars
with Session Volume Profiles (RTH & ETH) and VWAP metrics.
"""

import gc
import time
from pathlib import Path
import databento as db
import numpy as np
import pandas as pd
from strategies.volume_profile_strategy import (
    compute_session_profiles,
    calculate_smart_vwap
)


def process_file_dual_timeframe(file_path: Path):
    try:
        st = db.DBNStore.from_file(file_path)
        df = st.to_df()
        del st

        outright = df[~df["symbol"].str.contains("-")]
        if outright.empty:
            del df, outright
            gc.collect()
            return None, None

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
        rth_trades = ((times >= rth_start) & (times < rth_end)).sum()
        if rth_trades < 200:
            del df_filtered
            gc.collect()
            return None, None

        # Signed Delta
        trade_size = df_filtered["size"].astype(np.int64)
        buy_vol = np.where(df_filtered["side"] == "A", trade_size, 0)
        sell_vol = np.where(df_filtered["side"] == "B", trade_size, 0)
        delta = buy_vol - sell_vol

        tf_df = pd.DataFrame({
            "price": df_filtered["price"],
            "size": trade_size,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "delta": delta,
        }, index=df_filtered.index)

        profiles = compute_session_profiles(df_filtered, tick_size=1.0)
        del df_filtered
        gc.collect()

        def build_bars(timeframe):
            b = pd.DataFrame()
            b["open"] = tf_df["price"].resample(timeframe).first()
            b["high"] = tf_df["price"].resample(timeframe).max()
            b["low"] = tf_df["price"].resample(timeframe).min()
            b["close"] = tf_df["price"].resample(timeframe).last()
            b["volume"] = tf_df["size"].resample(timeframe).sum()
            b["buy_vol"] = tf_df["buy_vol"].resample(timeframe).sum()
            b["sell_vol"] = tf_df["sell_vol"].resample(timeframe).sum()
            b["delta"] = tf_df["delta"].resample(timeframe).sum()
            b = b.dropna().copy()
            b["cvd"] = b["delta"].cumsum()
            b["body_size"] = (b["close"] - b["open"]).abs()
            b["candle_range"] = b["high"] - b["low"]
            b["upper_wick"] = b["high"] - b["close"].combine(b["open"], max)
            b["lower_wick"] = b["close"].combine(b["open"], min) - b["low"]
            b["vol_ma"] = b["volume"].rolling(14, min_periods=3).mean()
            b["atr"] = b["candle_range"].rolling(14, min_periods=3).mean()
            b = calculate_smart_vwap(b)

            rth = profiles["rth"]
            eth = profiles["eth"]
            b["symbol"] = top_symbol
            b["rth_poc"] = float(rth["poc"]) if rth["poc"] is not None else np.nan
            b["rth_vah"] = float(rth["vah"]) if rth["vah"] is not None else np.nan
            b["rth_val"] = float(rth["val"]) if rth["val"] is not None else np.nan
            b["rth_lvns"] = ",".join(map(str, rth.get("lvns", [])))
            b["eth_poc"] = float(eth["poc"]) if eth["poc"] is not None else np.nan
            b["eth_vah"] = float(eth["vah"]) if eth["vah"] is not None else np.nan
            b["eth_val"] = float(eth["val"]) if eth["val"] is not None else np.nan
            b["eth_lvns"] = ",".join(map(str, eth.get("lvns", [])))
            return b

        b5 = build_bars("5min")
        b3 = build_bars("3min")
        del tf_df
        gc.collect()
        return b5, b3

    except Exception as e:
        print(f"Error in {file_path.name}: {e}")
        gc.collect()
        return None, None


def main():
    data_dir = Path("data")
    out_dir = Path("data_cache")
    out_dir.mkdir(exist_ok=True, parents=True)

    files = sorted(list(data_dir.glob("*.trades.dbn.zst")))
    total = len(files)
    print(f"Starting feature cache generation for {total} files...")

    all_b5 = []
    all_b3 = []
    start_time = time.time()

    for idx, f in enumerate(files, 1):
        pct = (idx / total) * 100
        print(f"[{idx:3d}/{total:3d}] ({pct:5.1f}%) Caching: {f.name}...", end="\r", flush=True)
        b5, b3 = process_file_dual_timeframe(f)
        if b5 is not None and not b5.empty:
            all_b5.append(b5)
        if b3 is not None and not b3.empty:
            all_b3.append(b3)

    print(f"\nMerging and saving to parquet...")
    if all_b5:
        df5 = pd.concat(all_b5)
        df5.to_parquet(out_dir / "bars_5min.parquet")
        print(f"[OK] Saved {len(df5):,} 5-minute bars to {out_dir / 'bars_5min.parquet'}")
        del df5, all_b5
        gc.collect()

    if all_b3:
        df3 = pd.concat(all_b3)
        df3.to_parquet(out_dir / "bars_3min.parquet")
        print(f"[OK] Saved {len(df3):,} 3-minute bars to {out_dir / 'bars_3min.parquet'}")
        del df3, all_b3
        gc.collect()

    elapsed = time.time() - start_time
    print(f"Cache generation complete in {elapsed:.1f}s!")


if __name__ == "__main__":
    main()
