# Stock Market Analytics — Executive Summary

## The Business Problem
An asset management firm needs to monitor 5 equity stocks and EUR/USD forex across 3 years, identify diversification opportunities, monitor risk in real-time, and explore ML-based trading signals.

## What We Found
- **EUR/USD** was the best risk-adjusted performer (Sharpe 0.32, lowest volatility at 12.7%)
- **Tech stocks** (AAPL, MSFT, GOOGL) move together — buying all three is NOT diversification
- **Volatility spiked 3–4×** during 5 key macro events — predictable triggers for risk alerts
- **ML model** confirms price direction has limited predictability (~50%) — honest and consistent with market efficiency theory

## Recommended Actions
1. Add 15–20% EUR/USD allocation to reduce portfolio volatility
2. Set automated alerts when 20-day volatility exceeds 2× its 60-day average
3. Underweight growth tech during Fed tightening cycles
4. Use MA20/MA50 Golden Cross as a low-effort AAPL entry signal

## How It Was Built
Full Python pipeline: data cleaning → EDA → 11 visualisations → 3 ML models compared. All code reproducible and version-controlled on GitHub.
