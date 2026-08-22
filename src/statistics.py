import pandas as pd
import numpy as np
from scipy import stats

def compute_summary_statistics(
    returns_series: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> dict:
    """
    Computes higher-order moments and risk-adjusted metrics for returns.
    """
    clean_returns = returns_series.dropna()

    if clean_returns.empty:
        return {}

    mean_daily = clean_returns.mean()
    median_daily = clean_returns.median()
    vol_daily = clean_returns.std()
    var_daily = clean_returns.var()

    skewness = float(stats.skew(clean_returns))
    kurtosis_excess = float(stats.kurtosis(clean_returns))  # Fisher kurtosis (Normal = 0)

    ann_return = mean_daily * trading_days
    ann_volatility = vol_daily * np.sqrt(trading_days)

    daily_rf = risk_free_rate / trading_days
    excess_returns = clean_returns - daily_rf

    sharpe = (excess_returns.mean() / vol_daily) * np.sqrt(trading_days) if vol_daily != 0 else np.nan

    downside_returns = clean_returns[clean_returns < daily_rf]
    downside_std = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0.0
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(trading_days) if downside_std != 0 else np.nan

    return {
        "Mean (Daily)": mean_daily,
        "Median (Daily)": median_daily,
        "Variance (Daily)": var_daily,
        "Volatility (Daily)": vol_daily,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_volatility,
        "Skewness": skewness,
        "Excess Kurtosis": kurtosis_excess,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
    }

def calculate_max_drawdown(price_series: pd.Series) -> tuple[float, pd.Series]:
    """
    Computes maximum historical drawdown percentage and continuous drawdown series.
    """
    prices = price_series.dropna()
    rolling_peak = prices.cummax()
    drawdown_series = (prices - rolling_peak) / rolling_peak
    max_drawdown = float(drawdown_series.min())

    return max_drawdown, drawdown_series

def run_normality_tests(returns_series: pd.Series) -> pd.DataFrame:
    """
    Runs formal statistical tests to evaluate if returns follow a normal distribution.
    Null Hypothesis (H0): The data is normally distributed.
    """
    clean_returns = returns_series.dropna()
    n = len(clean_returns)
    
    if n < 3:
        return pd.DataFrame()

    # 1. Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(clean_returns)

    # 2. Jarque-Bera Test (Tests skewness and kurtosis jointly)
    jb_stat, jb_p = stats.jarque_bera(clean_returns)

    # 3. Kolmogorov-Smirnov Test (Requires standardized data to test against Standard Normal)
    standardized_returns = (clean_returns - clean_returns.mean()) / clean_returns.std()
    ks_stat, ks_p = stats.kstest(standardized_returns, 'norm')

    # Compile Results
    results = {
        "Statistical Test": [
            "Jarque-Bera (Skew/Kurtosis)", 
            "Shapiro-Wilk (Overall Shape)", 
            "Kolmogorov-Smirnov (CDF Match)"
        ],
        "Test Statistic": [jb_stat, shapiro_stat, ks_stat],
        "p-value": [jb_p, shapiro_p, ks_p],
        "Conclusion (α=0.05)": [
            "Reject H0 (Not Normal)" if p < 0.05 else "Fail to Reject (Normal)" 
            for p in [jb_p, shapiro_p, ks_p]
        ]
    }
    
    return pd.DataFrame(results)