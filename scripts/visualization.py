"""
SCRIPT 3: visualization.py  —  All Charts
Purpose : Generate 8 professional charts and save to outputs/charts/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN      = os.path.join(BASE, "datasets", "stock_market_clean.csv")
CHART_DIR  = os.path.join(BASE, "outputs", "charts")
REPORT_DIR = os.path.join(BASE, "outputs", "reports")
os.makedirs(CHART_DIR, exist_ok=True)

# ── GLOBAL STYLE ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",       "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",      "grid.color": "#21262d",
    "grid.linewidth": 0.5,         "font.family": "monospace",
    "font.size": 10,
})
PALETTE = ["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657","#79c0ff"]
TICKERS = ["AAPL","MSFT","GOOGL","JPM","XOM","EUR/USD"]

df = pd.read_csv(CLEAN, parse_dates=["Date"])

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1: Price History — Line Chart
# WHY: Shows how each asset moved over 3 years. Investors look at this first
#      to spot bull/bear phases and compare relative performance.
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Stock Price History (2022–2024)", fontsize=16, color="#e6edf3", y=1.01)
axes = axes.flatten()
for i, (ticker, color) in enumerate(zip(TICKERS, PALETTE)):
    d = df[df["Ticker"] == ticker].sort_values("Date")
    axes[i].plot(d["Date"], d["Close"], color=color, linewidth=1.5, label="Close")
    axes[i].plot(d["Date"], d["MA_20"], color="white", linewidth=0.8, alpha=0.6, linestyle="--", label="MA20")
    axes[i].plot(d["Date"], d["MA_50"], color="#ffa657", linewidth=0.8, alpha=0.6, linestyle=":", label="MA50")
    axes[i].set_title(ticker, color=color, fontsize=12, fontweight="bold")
    axes[i].legend(fontsize=7, loc="upper left")
    axes[i].grid(True, alpha=0.4)
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "01_price_history.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Chart 1 saved: 01_price_history.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2: Cumulative Returns — Line Chart
# WHY: Normalised to 0% start, so you compare apples-to-apples regardless of
#      different starting prices ($1 vs $280). Recruiters love this chart.
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
for ticker, color in zip(TICKERS, PALETTE):
    d = df[df["Ticker"] == ticker].sort_values("Date")
    ax.plot(d["Date"], d["Cum_Return"], color=color, linewidth=2, label=ticker)
ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
ax.fill_between(df["Date"].unique(), 0, 0, alpha=0.1)
ax.set_title("Cumulative Returns by Asset (2022–2024)", fontsize=14, color="#e6edf3")
ax.set_ylabel("Cumulative Return (%)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "02_cumulative_returns.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Chart 2 saved: 02_cumulative_returns.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3: Daily Return Distribution — Histogram + KDE
# WHY: Fat tails (kurtosis) mean more extreme days than a normal distribution.
#      Risk managers use this to stress-test portfolios.
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Daily Return Distribution per Asset", fontsize=14, color="#e6edf3")
axes = axes.flatten()
for i, (ticker, color) in enumerate(zip(TICKERS, PALETTE)):
    d = df[df["Ticker"] == ticker]["Daily_Return"].dropna()
    axes[i].hist(d, bins=60, color=color, alpha=0.6, density=True, edgecolor="none")
    d.plot.kde(ax=axes[i], color="white", linewidth=1.5)
    axes[i].axvline(d.mean(), color="#ffa657", linewidth=1.5, linestyle="--", label=f"Mean {d.mean():.2f}%")
    axes[i].set_title(ticker, color=color, fontsize=11, fontweight="bold")
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.4)
    axes[i].set_xlabel("Daily Return (%)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "03_return_distributions.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Chart 3 saved: 03_return_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4: Correlation Heatmap
# WHY: Reveals diversification opportunities. Assets with low correlation
#      reduce portfolio risk — the foundation of Modern Portfolio Theory.
# ═══════════════════════════════════════════════════════════════════════════════
ret_pivot = df.pivot_table(index="Date", columns="Ticker", values="Daily_Return")
corr      = ret_pivot.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
            ax=ax, linewidths=0.5, linecolor="#21262d",
            annot_kws={"size": 11, "weight": "bold"},
            cbar_kws={"shrink": 0.8})
ax.set_title("Return Correlation Matrix — Diversification Analysis",
             fontsize=13, color="#e6edf3", pad=15)
ax.set_facecolor("#161b22")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "04_correlation_heatmap.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Chart 4 saved: 04_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5: Volatility Over Time — Rolling 20-Day
# WHY: Volatility clustering is a key market phenomenon (GARCH effects).
#      Spikes in volatility coincide with market events (wars, rate hikes).
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
for ticker, color in zip(TICKERS, PALETTE):
    d = df[df["Ticker"] == ticker].sort_values("Date")
    ax.plot(d["Date"], d["Volatility_20d"], color=color, linewidth=1.5, alpha=0.85, label=ticker)
market_events = [
    ("2022-02-24","Russia-Ukraine"),("2022-06-13","Inflation Peak"),
    ("2023-03-10","SVB Collapse"),("2023-05-25","AI Boom"),
    ("2024-11-06","US Election")
]
for evt_date, label in market_events:
    ax.axvline(pd.to_datetime(evt_date), color="#8b949e", linewidth=0.8, linestyle=":")
    ax.text(pd.to_datetime(evt_date), ax.get_ylim()[1]*0.85, label,
            rotation=90, fontsize=7, color="#8b949e", va="top")
ax.set_title("Rolling 20-Day Volatility — Market Event Impact", fontsize=14, color="#e6edf3")
ax.set_ylabel("Volatility (Std Dev of Daily Returns %)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "05_rolling_volatility.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Chart 5 saved: 05_rolling_volatility.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6: Volume Analysis — Bar Chart
# WHY: Volume confirms price moves. High volume + price rise = strong signal.
#      Low volume + price rise = weak, potentially reversing move.
# ═══════════════════════════════════════════════════════════════════════════════
stock_df = df[df["Ticker"] != "EUR/USD"].copy()
monthly_vol = stock_df.groupby(["Ticker","Year","Month"])["Volume"].mean().reset_index()
monthly_vol["Period"] = monthly_vol["Year"].astype(str) + "-" + monthly_vol["Month"].astype(str).str.zfill(2)

fig, ax = plt.subplots(figsize=(14, 7))
for ticker, color in zip(["AAPL","MSFT","GOOGL","JPM","XOM"], PALETTE):
    d = monthly_vol[monthly_vol["Ticker"] == ticker].sort_values("Period")
    ax.plot(range(len(d)), d["Volume"]/1e6, color=color, linewidth=1.5, label=ticker, alpha=0.85)
ax.set_title("Average Monthly Trading Volume (Millions) — Stocks Only", fontsize=14, color="#e6edf3")
ax.set_ylabel("Volume (Millions)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
ax.set_xticks([0, 12, 24, 35])
ax.set_xticklabels(["Jan 2022","Jan 2023","Jan 2024","Dec 2024"])
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "06_volume_analysis.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Chart 6 saved: 06_volume_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7: Risk vs Return Scatter (Sharpe-style)
# WHY: The classic portfolio optimisation chart. Interviewers LOVE this.
#      Shows which assets give the most return per unit of risk taken.
# ═══════════════════════════════════════════════════════════════════════════════
risk_ret = df.groupby("Ticker").agg(
    Ann_Return = ("Daily_Return", lambda x: x.mean() * 252),
    Ann_Vol    = ("Daily_Return", lambda x: x.std()  * np.sqrt(252))
).reset_index()

fig, ax = plt.subplots(figsize=(10, 7))
for i, row in risk_ret.iterrows():
    color = PALETTE[i % len(PALETTE)]
    ax.scatter(row["Ann_Vol"], row["Ann_Return"], s=250, color=color, zorder=5, edgecolors="white", linewidth=1)
    ax.annotate(row["Ticker"], (row["Ann_Vol"], row["Ann_Return"]),
                textcoords="offset points", xytext=(10, 5),
                fontsize=11, color=color, fontweight="bold")
ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
ax.set_title("Risk vs Return Analysis (Annualised)", fontsize=14, color="#e6edf3")
ax.set_xlabel("Annualised Volatility (Risk)", fontsize=11)
ax.set_ylabel("Annualised Return", fontsize=11)
ax.grid(True, alpha=0.4)
ax.text(0.02, 0.98, "TOP-LEFT = Best\n(High Return, Low Risk)",
        transform=ax.transAxes, fontsize=9, color="#3fb950", va="top")
ax.text(0.65, 0.02, "BOTTOM-RIGHT = Worst\n(Low Return, High Risk)",
        transform=ax.transAxes, fontsize=9, color="#f78166", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "07_risk_return_scatter.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Chart 7 saved: 07_risk_return_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8: AAPL Candlestick-style (OHLC bar chart)
# WHY: The standard chart in trading dashboards. Shows open, high, low, close
#      in one view — more information-dense than a simple line chart.
# ═══════════════════════════════════════════════════════════════════════════════
aapl = df[df["Ticker"] == "AAPL"].sort_values("Date").tail(90).reset_index(drop=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("AAPL — OHLC Price & Volume (Last 90 Days)", fontsize=14, color="#e6edf3")

for i, row in aapl.iterrows():
    color = "#3fb950" if row["Close"] >= row["Open"] else "#f78166"
    ax1.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=1)
    ax1.bar(i, abs(row["Close"] - row["Open"]), bottom=min(row["Open"], row["Close"]),
            color=color, width=0.6, alpha=0.9)

ax1.plot(range(len(aapl)), aapl["MA_20"], color="white",  linewidth=1, linestyle="--", label="MA20")
ax1.plot(range(len(aapl)), aapl["MA_50"], color="#ffa657", linewidth=1, linestyle=":", label="MA50")
ax1.set_ylabel("Price (USD)", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, len(aapl))

bar_colors = ["#3fb950" if r["Close"] >= r["Open"] else "#f78166" for _, r in aapl.iterrows()]
ax2.bar(range(len(aapl)), aapl["Volume"]/1e6, color=bar_colors, alpha=0.7, width=0.6)
ax2.set_ylabel("Vol (M)", fontsize=9)
ax2.grid(True, alpha=0.3)
tick_positions = [0, 15, 30, 45, 60, 75, 89]
ax2.set_xticks(tick_positions)
ax2.set_xticklabels([aapl["Date"].iloc[i].strftime("%b %d") for i in tick_positions], rotation=30)
ax1.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "08_aapl_ohlc_candlestick.png"), dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Chart 8 saved: 08_aapl_ohlc_candlestick.png")

print("\nALL 8 CHARTS SAVED to outputs/charts/")
