import pandas as pd
import numpy as np

def calculate_historical_var(returns_series: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculates Historical Value at Risk (VaR) at a given confidence level.
    """
    clean_returns = returns_series.dropna()
    if clean_returns.empty:
        return np.nan
    
    # For a 95% confidence level, we want the 5th percentile (1 - 0.95 = 0.05)
    percentile = (1 - confidence_level) * 100
    var = np.percentile(clean_returns, percentile)
    
    return float(var)

def calculate_historical_cvar(returns_series: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculates Historical Expected Shortfall (CVaR) at a given confidence level.
    """
    clean_returns = returns_series.dropna()
    if clean_returns.empty:
        return np.nan
    
    var = calculate_historical_var(clean_returns, confidence_level)
    
    # CVaR is the mean of all returns that are worse than or equal to the VaR threshold
    tail_losses = clean_returns[clean_returns <= var]
    cvar = tail_losses.mean() if len(tail_losses) > 0 else np.nan
    
    return float(cvar)