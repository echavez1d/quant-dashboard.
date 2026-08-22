import pandas as pd
import numpy as np
import yfinance as yf

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# Test real-time data fetching
df = yf.download("AAPL", period="5d", progress=False)
print("\nRecent Market Data:")
print(df.tail(2))
