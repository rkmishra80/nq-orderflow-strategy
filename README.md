# Databento Quantitative Backtesting Framework

Python project setup for processing Databento Market Data (`.dbn` / `.dbn.zst`) and running algorithmic backtests using **VectorBT** and **Backtrader**.

---

## 📁 Project Structure

```text
GLBX-20260911-RYJCWBGU8U/
│
├── .venv/                      # Python virtual environment
├── data/                       # Databento data files (.dbn, .dbn.zst)
│   ├── metadata.json
│   └── glbx-mdp3-*.trades.dbn.zst
│
├── strategies/                 # Strategy modules
│   ├── __init__.py
│   ├── sma_crossover.py        # VectorBT SMA Strategy
│   └── backtrader_strategy.py  # Backtrader SMA Strategy
│
├── main.py                     # Project entry point
├── requirements.txt            # Project dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Virtual Environment Activate karein:
PowerShell mein:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(CMD mein: `.\.venv\Scripts\activate.bat`)*

### 2. Main Script Run karein:
```powershell
python main.py
```

---

## 🛠️ Features Included:
1. **Databento DBN Loader**: `.dbn` aur `.dbn.zst` files ko efficiently load karta hai.
2. **Spread Filtering**: Outright futures contracts (`NQU5`) ko calendar spreads se alag karta hai taaki price spikes na aayein.
3. **Tick to OHLCV Resampling**: High-frequency trade events ko 1-minute, 5-minute ya custom timeframe candles mein aggregate karta hai.
4. **VectorBT & Backtrader**: Dono frameworks ke sample strategies ready hain.
