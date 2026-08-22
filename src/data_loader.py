import yfinance as yf
import pandas as pd

def fetch_market_data(ticker: str, period: str = "1Y") -> pd.DataFrame:
    """
    Fetches raw OHLCV market data from Yahoo Finance and cleans basic anomalies.
    
    Parameters:
        ticker (str): Asset ticker symbol.
        period (str): Lookback horizon ('1M', '3M', '6M', '1Y', '3Y', '5Y', '10Y', 'MAX').
        
    Returns:
        pd.DataFrame: Cleaned daily pricing time series.
    """
    period_map = {
        "1M": "1mo", "3M": "3mo", "6M": "6mo",
        "1Y": "1y", "3Y": "3y", "5Y": "5y",
        "10Y": "10y", "MAX": "max"
    }
    yf_period = period_map.get(period.upper(), "1y")
    
    try:
        data = yf.download(ticker, period=yf_period, progress=False)
        
        if data.empty:
            return pd.DataFrame()
            
        # Flatten MultiIndex columns if present in yfinance response
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data = data.sort_index()
        
        # Handle non-trading day missing values: forward fill then backward fill
        data = data.ffill().bfill()
        
        return data
        
    except Exception as e:
        print(f"Data retrieval failed for {ticker}: {e}")
        return pd.DataFrame()