"""
Dataset Generator for Stock Market Analytics Project
Generates 3 years of realistic OHLCV + Forex data for 6 major tickers
"""

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

TICKERS = {
    "AAPL":   {"start_price": 130.0, "volatility": 0.018, "trend": 0.0004,  "sector": "Technology"},
    "MSFT":   {"start_price": 280.0, "volatility": 0.016, "trend": 0.0005,  "sector": "Technology"},
    "GOOGL":  {"start_price": 100.0, "volatility": 0.020, "trend": 0.0003,  "sector": "Technology"},
    "JPM":    {"start_price": 140.0, "volatility": 0.017, "trend": 0.0003,  "sector": "Finance"},
    "XOM":    {"start_price":  85.0, "volatility": 0.019, "trend": 0.0002,  "sector": "Energy"},
    "EUR/USD":{"start_price":   1.08,"volatility": 0.006, "trend": 0.00005, "sector": "Forex"},
}

MARKET_EVENTS = [
    {"date": "2022-02-24", "shock": -0.04},
    {"date": "2022-05-10", "shock": -0.05},
    {"date": "2022-06-13", "shock": -0.06},
    {"date": "2022-10-13", "shock":  0.05},
    {"date": "2023-03-10", "shock": -0.04},
    {"date": "2023-05-25", "shock":  0.06},
    {"date": "2023-11-01", "shock":  0.04},
    {"date": "2024-01-15", "shock":  0.03},
    {"date": "2024-07-11", "shock": -0.03},
    {"date": "2024-11-06", "shock":  0.05},
]

def generate_price_series(config, dates):
    prices = [config["start_price"]]
    event_map = {e["date"]: e["shock"] for e in MARKET_EVENTS}
    for i in range(1, len(dates)):
        shock = event_map.get(dates[i].strftime("%Y-%m-%d"), 0)
        daily_return = config["trend"] + config["volatility"] * np.random.randn() + shock
        prices.append(prices[-1] * (1 + daily_return))
    return prices

def generate_ohlcv(close_prices, volatility, base_volume=5_000_000):
    rows = []
    for price in close_prices:
        spread = price * volatility * 0.5
        open_  = price + np.random.uniform(-spread, spread)
        high   = max(price, open_) + abs(np.random.normal(0, spread))
        low    = min(price, open_) - abs(np.random.normal(0, spread))
        volume = int(np.random.lognormal(np.log(base_volume), 0.5))
        rows.append({"Open": round(open_,4), "High": round(high,4),
                     "Low":  round(low,4),   "Close": round(price,4),
                     "Volume": volume})
    return rows

all_dates = pd.bdate_range(datetime(2022,1,3), datetime(2024,12,31))
records   = []

for ticker, config in TICKERS.items():
    close_prices = generate_price_series(config, list(all_dates))
    ohlcv        = generate_ohlcv(close_prices, config["volatility"])
    for i, date in enumerate(all_dates):
        row = {"Date": date.strftime("%Y-%m-%d"), "Ticker": ticker,
               "Sector": config["sector"], **ohlcv[i]}
        records.append(row)

df = pd.DataFrame(records)

# Inject realistic data quality issues for cleaning exercise
null_idx  = df.sample(frac=0.01, random_state=1).index
df.loc[null_idx, "Volume"] = np.nan
null_idx2 = df.sample(frac=0.005, random_state=2).index
df.loc[null_idx2, "Close"] = np.nan
dupes = df.sample(15, random_state=3)
df    = pd.concat([df, dupes], ignore_index=True)
for i in range(5):
    idx = np.random.randint(0, len(df))
    df.loc[idx, "High"] = df.loc[idx, "High"] * 10  # fat-finger outliers

df.to_csv("/home/claude/Stock_Market_Analytics_Project/datasets/stock_market_data.csv", index=False)
print(f"Dataset saved: {len(df):,} rows x {len(df.columns)} columns")
