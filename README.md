# 📈 Stock Market & Forex Trading Analytics

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-brightgreen)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> End-to-end data analytics project: 3 years of multi-asset market data → cleaning → EDA → 11 visualisations → ML price-direction predictor

---

## 📌 Project Overview

This project analyses 3 years (2022–2024) of daily OHLCV data across **5 equities** (AAPL, MSFT, GOOGL, JPM, XOM) and **EUR/USD Forex** to answer three core business questions:

1. **Which assets delivered the best risk-adjusted returns?**
2. **How did major macro events (rate hikes, SVB collapse, AI boom) impact volatility?**
3. **Can a machine learning model predict next-day price direction better than random chance?**

---

## 🏢 Business Problem

A boutique asset management firm needs a data-driven dashboard to:
- Monitor multi-asset portfolio performance in one view
- Identify diversification opportunities using correlation analysis
- Build an early-warning system for volatility spikes
- Pilot ML-based directional signals for AAPL

---

## 🛠 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core programming language |
| Pandas | 2.0 | Data manipulation, time-series operations |
| NumPy | 1.24 | Vectorised numerical computations |
| Matplotlib | 3.7 | Base charting engine |
| Seaborn | 0.12 | Statistical visualisations (heatmap) |
| Scikit-learn | 1.3 | ML models, cross-validation, evaluation |

---

## 📂 Folder Structure

```
Stock_Market_Analytics_Project/
│
├── datasets/
│   ├── generate_dataset.py       # Synthetic OHLCV + Forex data generator
│   ├── stock_market_data.csv     # Raw dataset (intentional data quality issues)
│   └── stock_market_clean.csv   # Clean dataset with engineered features
│
├── scripts/
│   ├── data_cleaning.py          # Step-by-step data cleaning pipeline
│   ├── eda.py                    # Exploratory data analysis & reports
│   ├── visualization.py         # 8 professional charts
│   └── model_building.py        # Random Forest price direction model
│
├── outputs/
│   ├── charts/                   # 11 saved PNG charts
│   └── reports/                  # 8 CSV summary reports + insights text
│
├── presentation/
│   └── project_summary.md       # Non-technical executive summary
│
├── README.md                     # This file
├── requirements.txt              # All dependencies
├── insights.md                   # Deep business insights

```

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| Assets | AAPL, MSFT, GOOGL, JPM, XOM, EUR/USD |
| Period | 2022-01-03 to 2024-12-31 |
| Frequency | Daily (business days) |
| Records | ~4,700 rows after cleaning |
| Columns | Date, Ticker, Sector, Open, High, Low, Close, Volume + 11 engineered features |

**Intentional data quality issues** (for realistic cleaning practice):
- 1% missing Volume values
- 0.5% missing Close prices
- 15 duplicate rows
- 5 fat-finger outlier prices (High × 10)
- Date stored as string (requires type conversion)

---

## 🧹 Data Cleaning Steps

1. **Type conversion** — Date → datetime64, OHLCV → float64
2. **Duplicate removal** — 15 exact duplicates removed (same Date + Ticker)
3. **Null handling** — Close: forward-fill (last known price); Volume: median imputation
4. **Outlier detection** — IQR method: High > 3× Close flagged and capped
5. **OHLC validation** — High ≥ max(Open,Close) and Low ≤ min(Open,Close) enforced

---

## 🔍 EDA Highlights

| Metric | Finding |
|---|---|
| Best cumulative return | EUR/USD (+10.4%) — benefited from USD strength cycles |
| Worst cumulative return | MSFT (−56.5%) — hit hard by 2022 rate hike cycle |
| Highest Sharpe ratio | EUR/USD (0.32) — best return per unit of risk |
| Most volatile asset | GOOGL (34.8% annualised) |
| Least volatile asset | EUR/USD (12.7% annualised) |
| Highest equity correlation | AAPL ↔ MSFT (0.14) — tech cluster effect |
| Best diversifier | EUR/USD (avg equity correlation < 0.20) |

---

## 📈 Visualisations

| # | Chart | Business Purpose |
|---|---|---|
| 01 | Price History + MA20/MA50 | Spot trend direction and support/resistance |
| 02 | Cumulative Returns | Compare all assets on equal footing |
| 03 | Daily Return Distributions | Risk profiling, fat-tail assessment |
| 04 | Correlation Heatmap | Portfolio diversification analysis |
| 05 | Rolling 20d Volatility | Real-time risk monitoring |
| 06 | Monthly Volume Trends | Liquidity analysis |
| 07 | Risk vs Return Scatter | Portfolio optimisation decision support |
| 08 | AAPL OHLC Candlestick | Technical analysis dashboard |
| 09 | Feature Importance | ML interpretability |
| 10 | Confusion Matrix | Model error analysis |
| 11 | Model Comparison | Algorithm selection justification |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ritish-reddy/Stock_Market_Analytics.git
cd Stock_Market_Analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python datasets/generate_dataset.py

# 4. Run the full pipeline in order
python scripts/data_cleaning.py
python scripts/eda.py
python scripts/visualization.py
python scripts/model_building.py
```

All outputs (charts + reports) are saved to the `outputs/` folder.

---

## 💡 Key Business Insights

1. **EUR/USD outperformed all equity indices on a risk-adjusted basis** — validating FX as a portfolio diversifier
2. **Tech stocks moved in lockstep** (AAPL/MSFT/GOOGL correlation > 0.10–0.14) — concentrated tech bets amplify risk
3. **Volatility spiked 3–4× normal during macro shocks** — a volatility monitoring alert system would have flagged these before large drawdowns
4. **AAPL's Moving Average crossover (MA20 > MA50)** correctly identified 4 out of 5 major bull phases in the 2022–2024 period
5. **XOM outperformed tech in 2022** (energy super-cycle) demonstrating sector-rotation value

---

## 🔮 Future Improvements

- [ ] Add Bollinger Bands and MACD as features
- [ ] Build a Streamlit dashboard for live chart interaction
- [ ] Implement a backtesting engine to simulate trading strategy P&L
- [ ] Add LSTM/Transformer model for sequence modelling
- [ ] Pull live data using yfinance API
- [ ] Add portfolio optimisation (Markowitz Efficient Frontier)

---

## 👤 Author

**[Ritish Reddy]**
Data Analyst | Python | Financial Analytics  
📧 v.ritish85@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/ritishreddy04) | [GitHub](https://github.com/ritish-reddy)

---

*Built as a portfolio project demonstrating end-to-end data analytics skills for Data Analyst roles.*
