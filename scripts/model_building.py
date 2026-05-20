"""
SCRIPT 4: model_building.py  —  ML Price Direction Predictor
Purpose : Predict next-day price direction (Up/Down) using Random Forest.
          This is a CLASSIFICATION problem — the most common ML task in finance.

WHY RANDOM FOREST (not Neural Networks)?
  - Tabular financial data: tree-based models consistently outperform NNs
  - Interpretable: you can explain feature importance to non-technical managers
  - Fast to train and evaluate
  - No need to normalise features
  - Handles non-linear relationships naturally
  - Resistant to overfitting (ensemble of 100+ trees)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN      = os.path.join(BASE, "datasets", "stock_market_clean.csv")
CHART_DIR  = os.path.join(BASE, "outputs", "charts")
REPORT_DIR = os.path.join(BASE, "outputs", "reports")

print("=" * 60)
print("ML MODEL: Next-Day Price Direction Predictor")
print("=" * 60)

df = pd.read_csv(CLEAN, parse_dates=["Date"])

# ── FOCUS ON AAPL (most liquid, clearest signals) ────────────────────────────
ticker = "AAPL"
d = df[df["Ticker"] == ticker].sort_values("Date").copy()

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FOR ML
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[1] Engineering features for {ticker}")

# Lag features: yesterday's and 2-days-ago returns
# WHY lags? Markets have short-term memory (momentum/mean-reversion)
d["Return_1d"]  = d["Daily_Return"].shift(1)
d["Return_2d"]  = d["Daily_Return"].shift(2)
d["Return_5d"]  = d["Daily_Return"].shift(5)
d["Return_10d"] = d["Daily_Return"].shift(10)

# Price above/below moving averages (binary signals used by technical analysts)
d["Above_MA20"] = (d["Close"] > d["MA_20"]).astype(int)
d["Above_MA50"] = (d["Close"] > d["MA_50"]).astype(int)
d["MA_Cross"]   = (d["MA_20"] > d["MA_50"]).astype(int)  # Golden/Death cross

# Volume signal: is today's volume above its 20-day average?
d["Vol_MA20"]   = d["Volume"].rolling(20, min_periods=1).mean()
d["High_Volume"]= (d["Volume"] > d["Vol_MA20"]).astype(int)

# Relative Strength Index proxy (simplified RSI)
delta     = d["Close"].diff()
gain      = delta.clip(lower=0).rolling(14).mean()
loss      = (-delta.clip(upper=0)).rolling(14).mean()
d["RSI"]  = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# Price momentum
d["Momentum_5d"]  = d["Close"] / d["Close"].shift(5)  - 1
d["Momentum_20d"] = d["Close"] / d["Close"].shift(20) - 1

# Volatility relative to recent average
d["Vol_Regime"] = d["Volatility_20d"] / d["Volatility_20d"].rolling(60).mean()

# ── TARGET VARIABLE ──────────────────────────────────────────────────────────
# Binary: 1 = price goes UP tomorrow, 0 = price goes DOWN/flat
# WHY this target? Predicting direction is more useful than predicting exact price.
# Even 55% accuracy is profitable if you size positions correctly.
d["Target"] = (d["Close"].shift(-1) > d["Close"]).astype(int)

FEATURES = ["Return_1d","Return_2d","Return_5d","Return_10d",
            "Above_MA20","Above_MA50","MA_Cross","High_Volume",
            "RSI","Momentum_5d","Momentum_20d","Vol_Regime",
            "Volatility_20d","Price_Range","Volume"]

d_clean = d[FEATURES + ["Target","Date"]].dropna()
X = d_clean[FEATURES]
y = d_clean["Target"]
dates = d_clean["Date"]

print(f"  Features   : {len(FEATURES)}")
print(f"  Samples    : {len(X):,}")
print(f"  Target dist: {y.value_counts().to_dict()} ({y.mean()*100:.1f}% Up days)")

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT — TIME SERIES AWARE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] Time-series train/test split")

# WHY TimeSeriesSplit instead of random split?
# CRITICAL CONCEPT FOR INTERVIEWS: Using random split on time-series = DATA LEAKAGE
# You would be training on "future" data and testing on "past" data.
# TimeSeriesSplit always trains on past, predicts future — realistic evaluation.

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"  Train set  : {len(X_train):,} samples ({dates.iloc[0].date()} to {dates.iloc[split_idx-1].date()})")
print(f"  Test set   : {len(X_test):,}  samples ({dates.iloc[split_idx].date()} to {dates.iloc[-1].date()})")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] Comparing 3 models")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=200, max_depth=6,
                                                   min_samples_leaf=10, random_state=42),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
}

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # Needed for Logistic Regression
X_test_sc  = scaler.transform(X_test)

results = {}
for name, model in models.items():
    Xtr = X_train_sc if "Logistic" in name else X_train
    Xte = X_test_sc  if "Logistic" in name else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    results[name] = {"accuracy": acc, "auc": auc, "model": model,
                     "y_pred": y_pred, "y_prob": y_prob}
    print(f"  {name:22s} | Accuracy: {acc:.3f} | AUC: {auc:.3f}")

# ── BEST MODEL: RANDOM FOREST ────────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["auc"])
best       = results[best_name]
rf_model   = results["Random Forest"]["model"]
print(f"\n  Best model: {best_name}")

print("\n[4] Detailed evaluation — Random Forest")
print(classification_report(y_test, results["Random Forest"]["y_pred"],
                             target_names=["Down","Up"]))

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("[5] Feature importance")
fi = pd.DataFrame({
    "Feature"   : FEATURES,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)
print(fi.to_string(index=False))
fi.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS: Feature Importance + Confusion Matrix + Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#0d1117","axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",  "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",      "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",     "grid.color": "#21262d",
    "font.family": "monospace",   "font.size": 10,
})

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 7))
colors  = ["#58a6ff" if i < 5 else "#8b949e" for i in range(len(fi))]
ax.barh(fi["Feature"], fi["Importance"], color=colors, alpha=0.85, edgecolor="none")
ax.set_title("Random Forest — Feature Importance\n(Top features drive prediction accuracy)",
             fontsize=13, color="#e6edf3")
ax.set_xlabel("Importance Score", fontsize=11)
ax.grid(True, axis="x", alpha=0.4)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "09_feature_importance.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()

# Confusion matrix
import seaborn as sns
cm   = confusion_matrix(y_test, results["Random Forest"]["y_pred"])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred Down","Pred Up"],
            yticklabels=["Actual Down","Actual Up"],
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title(f"Confusion Matrix — Random Forest\nAccuracy: {results['Random Forest']['accuracy']:.1%}",
             fontsize=13, color="#e6edf3")
ax.set_facecolor("#161b22")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "10_confusion_matrix.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()

# Model comparison bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
model_names = list(results.keys())
accs = [results[n]["accuracy"] for n in model_names]
aucs = [results[n]["auc"]      for n in model_names]
PALETTE = ["#58a6ff","#3fb950","#f78166"]
ax1.bar(model_names, accs, color=PALETTE, alpha=0.85, edgecolor="none")
ax1.set_title("Model Accuracy Comparison", color="#e6edf3")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.4, 0.7)
ax1.axhline(0.5, color="#ffa657", linewidth=1, linestyle="--", label="Random baseline")
ax1.legend(fontsize=9)
for i, v in enumerate(accs):
    ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11, color="white")
ax2.bar(model_names, aucs, color=PALETTE, alpha=0.85, edgecolor="none")
ax2.set_title("Model AUC-ROC Comparison", color="#e6edf3")
ax2.set_ylabel("AUC-ROC")
ax2.set_ylim(0.4, 0.7)
ax2.axhline(0.5, color="#ffa657", linewidth=1, linestyle="--", label="Random baseline")
ax2.legend(fontsize=9)
for i, v in enumerate(aucs):
    ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11, color="white")
for ax in [ax1, ax2]:
    ax.set_facecolor("#161b22")
    ax.grid(True, axis="y", alpha=0.4)
    ax.tick_params(axis="x", rotation=10)
plt.suptitle(f"ML Model Comparison — {ticker} Price Direction Prediction",
             fontsize=13, color="#e6edf3")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "11_model_comparison.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()

# Save model results
model_report = pd.DataFrame([
    {"Model": n, "Accuracy": results[n]["accuracy"], "AUC": results[n]["auc"]}
    for n in model_names
])
model_report.to_csv(os.path.join(REPORT_DIR, "model_results.csv"), index=False)

print("\nCharts 9-11 saved")
print("Model results saved: outputs/reports/model_results.csv")
print("\nML MODEL COMPLETE")
