"""
MyFundedFutures (MFFU) $50k Evaluation & Funded Account Simulation.
===================================================================
Simulates MFFU $50,000 Evaluation Account Rules on 1-Year NQ Data:
- Starting Balance: $50,000
- Profit Target: $3,000 (Target Balance: $53,000)
- Max Drawdown: $2,000 EOD Trailing (Locks at $50,000 once peak >= $52,000)
- No Daily Loss Limit (Official MFFU Starter Plan)
- Realistic Friction: 0.5 pt/side Slippage + $4.50 Commission ($24.50/trade)
- Sizing: 1 Contract Fixed (Well within 5 contract scaling cap)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


def load_and_prep_trades():
    trades = pd.read_csv("results/trades_log_5min.csv")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    # Realistic friction adjustment (0.5 pt/side slippage = $20 + $0.40 extra comm = $20.40)
    trades["pnl_fric"] = trades["pnl_usd"] - 20.40
    return trades


def run_rolling_start_analysis(trades, use_friction=True):
    """
    Simulates starting an evaluation on every single available trading day (244 start dates).
    Answers: What is the probability of passing regardless of when you buy the account?
    """
    pnl_col = "pnl_fric" if use_friction else "pnl_usd"
    unique_dates = trades["entry_date"].unique()
    results = []

    for i in range(len(unique_dates)):
        start_date = unique_dates[i]
        subset = trades[trades["entry_date"] >= start_date]

        balance = 50000.0
        peak_eod = 50000.0
        threshold = 48000.0
        days = 0
        max_dd_from_peak = 0.0
        daily_pnls = []

        sub_dates = subset["entry_date"].unique()
        status = "IN_PROGRESS"

        for d in sub_dates:
            days += 1
            d_trades = subset[subset["entry_date"] == d]
            day_pnl = 0.0
            failed = False

            for _, t in d_trades.iterrows():
                pnl = t[pnl_col]
                balance += pnl
                day_pnl += pnl
                dd = peak_eod - balance
                if dd > max_dd_from_peak:
                    max_dd_from_peak = dd
                if balance < threshold:
                    failed = True
                    break

            if failed:
                status = "FAIL"
                break

            daily_pnls.append(day_pnl)
            if balance > peak_eod:
                peak_eod = balance
            # MFFU EOD trailing rule:
            threshold = min(50000.0, peak_eod - 2000.0)

            if (balance - 50000.0) >= 3000.0:
                status = "PASS"
                break

        results.append({
            "start_date": start_date,
            "status": status,
            "days_taken": days,
            "final_balance": balance,
            "net_profit": balance - 50000.0,
            "max_dd_from_peak": max_dd_from_peak,
            "best_day": max(daily_pnls) if daily_pnls else 0.0
        })

    return pd.DataFrame(results)


def run_sequential_evaluations(trades, use_friction=True):
    """
    Simulates buying accounts sequentially one after another as they Pass or Fail.
    """
    pnl_col = "pnl_fric" if use_friction else "pnl_usd"
    unique_dates = trades["entry_date"].unique()
    accounts = []
    acc_id = 1
    date_idx = 0

    while date_idx < len(unique_dates):
        acc_start = unique_dates[date_idx]
        balance = 50000.0
        peak_eod = 50000.0
        threshold = 48000.0
        days_traded = 0
        status = "IN_PROGRESS"
        max_dd = 0.0
        daily_pnls = []

        while date_idx < len(unique_dates):
            current_date = unique_dates[date_idx]
            days_traded += 1
            d_trades = trades[trades["entry_date"] == current_date]

            day_pnl = 0.0
            failed = False
            for _, t in d_trades.iterrows():
                pnl = t[pnl_col]
                balance += pnl
                day_pnl += pnl
                dd = peak_eod - balance
                if dd > max_dd:
                    max_dd = dd
                if balance < threshold:
                    failed = True
                    break

            if failed:
                status = "FAIL"
                date_idx += 1
                break

            daily_pnls.append(day_pnl)
            if balance > peak_eod:
                peak_eod = balance
            threshold = min(50000.0, peak_eod - 2000.0)

            if (balance - 50000.0) >= 3000.0:
                status = "PASS"
                date_idx += 1
                break

            date_idx += 1

        accounts.append({
            "account_id": acc_id,
            "start_date": acc_start,
            "end_date": current_date,
            "status": status,
            "days_traded": days_traded,
            "final_balance": balance,
            "net_profit": balance - 50000.0,
            "max_dd_from_peak": max_dd,
            "best_day": max(daily_pnls) if daily_pnls else 0.0
        })
        acc_id += 1

    return pd.DataFrame(accounts)


def run_funded_career_payout_simulation(trades):
    """
    Simulates a full trader career:
    Pass Evaluation -> Get Funded ($50k) -> Extract Payouts -> Re-evaluate if breached.
    """
    unique_dates = trades["entry_date"].unique()
    date_idx = 0
    phase = "EVAL"

    eval_attempts = 0
    eval_passes = 0
    eval_fails = 0
    funded_accounts = 0
    funded_blown = 0

    total_payouts_usd = 0.0
    payout_records = []

    balance = 50000.0
    peak_eod = 50000.0
    threshold = 48000.0
    eval_start_date = None
    funded_start_date = None
    funded_trading_days = 0

    while date_idx < len(unique_dates):
        current_date = unique_dates[date_idx]
        d_trades = trades[trades["entry_date"] == current_date]

        if phase == "EVAL":
            if eval_start_date is None:
                eval_start_date = current_date
                eval_attempts += 1
                balance = 50000.0
                peak_eod = 50000.0
                threshold = 48000.0

            failed = False
            for _, t in d_trades.iterrows():
                balance += t["pnl_fric"]
                if balance < threshold:
                    failed = True
                    break

            if failed:
                eval_fails += 1
                phase = "EVAL"
                eval_start_date = None
                date_idx += 1
                continue

            if balance > peak_eod:
                peak_eod = balance
            threshold = min(50000.0, peak_eod - 2000.0)

            # Target $3,000 Hit -> Transition to Funded
            if (balance - 50000.0) >= 3000.0:
                eval_passes += 1
                phase = "FUNDED"
                eval_start_date = None
                date_idx += 1
                continue

            date_idx += 1

        elif phase == "FUNDED":
            if funded_start_date is None:
                funded_start_date = current_date
                funded_accounts += 1
                balance = 50000.0
                peak_eod = 50000.0
                threshold = 48000.0
                funded_trading_days = 0

            funded_trading_days += 1
            failed = False
            for _, t in d_trades.iterrows():
                balance += t["pnl_fric"]
                if balance < threshold:
                    failed = True
                    break

            if failed:
                funded_blown += 1
                phase = "EVAL"
                funded_start_date = None
                date_idx += 1
                continue

            if balance > peak_eod:
                peak_eod = balance
            threshold = min(50100.0, peak_eod - 2000.0)

            # Payout Cycle: Every 10 trading days if balance > buffer ($52,100)
            if funded_trading_days >= 10 and balance > 52100.0:
                available_profit = balance - 50100.0
                payout_amount = min(2000.0, available_profit)
                trader_split = payout_amount * 0.90  # 90% payout split
                total_payouts_usd += trader_split
                balance -= payout_amount
                funded_trading_days = 0
                payout_records.append({
                    "date": str(current_date),
                    "funded_account_id": funded_accounts,
                    "gross_payout": payout_amount,
                    "net_payout_90pct": trader_split,
                    "remaining_balance": balance
                })

            date_idx += 1

    return {
        "eval_attempts": eval_attempts,
        "eval_passes": eval_passes,
        "eval_fails": eval_fails,
        "funded_accounts": funded_accounts,
        "funded_blown": funded_blown,
        "total_payouts_usd": total_payouts_usd,
        "total_payout_events": len(payout_records),
        "payout_records": payout_records
    }


def main():
    trades = load_and_prep_trades()
    out_dir = Path("results/mffu_simulation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Rolling Start Analysis (244 start dates)...")
    rolling_df = run_rolling_start_analysis(trades, use_friction=True)
    rolling_df.to_csv(out_dir / "rolling_evaluations.csv", index=False)

    print("Running Sequential Accounts Simulation...")
    seq_df = run_sequential_evaluations(trades, use_friction=True)
    seq_df.to_csv(out_dir / "sequential_accounts.csv", index=False)

    print("Running Funded Account Payout Simulation...")
    career = run_funded_career_payout_simulation(trades)
    with open(out_dir / "prop_career_payouts.json", "w") as fp:
        json.dump(career, fp, indent=2)

    print("Done! Summary files generated in results/mffu_simulation/")


if __name__ == "__main__":
    main()
