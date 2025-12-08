import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Tickers to analyze
symbols = ["NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "HPE", "CRDO", "TSM"]

# Fetch unadjusted prices so we can explicitly use Adj Close
prices = yf.download(symbols, period="1y", auto_adjust=False)["Adj Close"]

# Daily returns and correlation
returns = prices.pct_change().dropna()
corr_matrix = returns.corr()

# Colorful correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1,
    vmax=1,
    linewidths=0.6,
    linecolor="white",
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Correlation"},
)
plt.title("Equity Correlation (1Y)")
plt.tight_layout()
plt.show()