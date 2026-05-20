"""
SCRIPT 1: data_cleaning.py  —  Stock Market Analytics Project
Purpose : Load raw data, fix all quality issues, export clean CSV + features
"""

import pandas as pd
import numpy as np
import os

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW        = os.path.join(BASE, "datasets", "stock_market_data.csv")
CLEAN      = os.path.join(BASE, "datasets", "stock_market_clean.csv")
REPORT_DIR = os.path.join(BASE, "outputs", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: Loading raw dataset")
print("=" * 60)
df = pd.read_csv(RAW)
print(f"  Raw shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Columns   : {list(df.columns)}")

print("\n" + "=" * 60)
print("STEP 2: Fixing data types")
print("=" * 60)
df["Date"] = pd.to_datetime(df["Date"])
for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("  Date converted to datetime64")
print("  OHLCV columns confirmed numeric")
print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

print("\n" + "=" * 60)
print("STEP 3: Removing duplicate rows")
print("=" * 60)
before = len(df)
df = df.drop_duplicates(subset=["Date", "Ticker"])
after  = len(df)
print(f"  Duplicates removed: {before - after}")
print(f"  Rows remaining    : {after:,}")

print("\n" + "=" * 60)
print("STEP 4: Handling missing values")
print("=" * 60)
print("\n  Null counts before:")
print(df.isnull().sum())
df = df.sort_values(["Ticker", "Date"])
df["Close"]  = df.groupby("Ticker")["Close"].ffill()
median_vol   = df.groupby("Ticker")["Volume"].transform("median")
df["Volume"] = df["Volume"].fillna(median_vol)
for col in ["Open", "High", "Low"]:
    df[col] = df[col].fillna(df["Close"])
print("\n  Null counts after:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("STEP 5: Detecting and removing outliers")
print("=" * 60)
outlier_mask  = df["High"] > df["Close"] * 3
outlier_count = outlier_mask.sum()
df.loc[outlier_mask, "High"] = df.loc[outlier_mask, "Close"] * 1.05
print(f"  Outlier rows fixed (High > 3x Close): {outlier_count}")

print("\n" + "=" * 60)
print("STEP 6: Feature engineering")
print("=" * 60)
df["Daily_Return"]   = df.groupby("Ticker")["Close"].pct_change() * 100
df["MA_20"]          = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
df["MA_50"]          = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50, min_periods=1).mean())
df["Volatility_20d"] = df.groupby("Ticker")["Daily_Return"].transform(lambda x: x.rolling(20, min_periods=1).std())
df["Price_Range"]    = df["High"] - df["Low"]
df["Year"]           = df["Date"].dt.year
df["Month"]          = df["Date"].dt.month
df["Month_Name"]     = df["Date"].dt.strftime("%b")
df["Quarter"]        = df["Date"].dt.quarter
df["DayOfWeek"]      = df["Date"].dt.day_name()
df["Cum_Return"]     = df.groupby("Ticker")["Daily_Return"].transform(
    lambda x: (1 + x / 100).cumprod() - 1) * 100

for col in ["Open","High","Low","Close","Daily_Return","MA_20","MA_50",
            "Volatility_20d","Price_Range","Cum_Return"]:
    df[col] = df[col].round(4)
df["Volume"] = df["Volume"].fillna(0).astype(int)

print(f"\n  Final shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
df.to_csv(CLEAN, index=False)
print(f"  Clean dataset saved: {CLEAN}")

summary = pd.DataFrame([{
    "raw_rows": before, "clean_rows": len(df),
    "duplicates_removed": before - after, "outliers_fixed": int(outlier_count),
    "tickers": ", ".join(sorted(df["Ticker"].unique())),
    "date_from": str(df["Date"].min().date()), "date_to": str(df["Date"].max().date())
}])
summary.to_csv(os.path.join(REPORT_DIR, "cleaning_summary.csv"), index=False)
print("  Cleaning summary saved: outputs/reports/cleaning_summary.csv")
print("\nDATA CLEANING COMPLETE")
