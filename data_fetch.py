import yfinance as yf
import pandas as pd

print("FinScope - Data Fetch Module Starting...")

tickers = ["AAPL", "BTC-USD", "JNJ", "SPY", "XOM"]

# Download 3+ years of data
data = yf.download(
    tickers,
    start="2020-01-01",
    end="2025-01-01"
)["Close"]

data = data.dropna()

data.to_csv("clean_prices.csv")

print(" Data saved successfully!")
print(data.head())
