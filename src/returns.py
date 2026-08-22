import pandas as pd
import numpy as np


def calculate_returns(
    df: pd.DataFrame,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Computes daily simple, log, and cumulative returns.

    Requirements:
    - df must contain price_col
    - prices must be positive for log returns
    """
    if price_col not in df.columns:
        raise KeyError(f"Column {price_col!r} not found in DataFrame.")

    data = df.copy()

    # Ensure chronological order
    data = data.sort_index()

    prices = pd.to_numeric(data[price_col], errors="coerce")

    if (prices.dropna() <= 0).any():
        raise ValueError("Prices must be positive to calculate log returns.")

    data["Daily_Return"] = prices.pct_change()
    data["Log_Return"] = np.log(prices).diff()

    data["Cumulative_Return"] = (
        (1 + data["Daily_Return"]).cumprod() - 1
    )

    return data.dropna(
        subset=["Daily_Return", "Log_Return", "Cumulative_Return"]
    )


def resample_returns(
    df: pd.DataFrame,
    freq: str = "ME",
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Resamples prices and calculates point-to-point period returns.

    Examples of freq:
    - "W-FRI": weekly prices ending Friday
    - "ME": month-end, supported by newer pandas versions
    - "YE": year-end, supported by newer pandas versions
    """
    if price_col not in df.columns:
        raise KeyError(f"Column {price_col!r} not found in DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
    prices = prices.sort_index()

    resampled_prices = prices.resample(freq).last().dropna()
    period_returns = resampled_prices.pct_change().dropna()

    return period_returns.to_frame(name="Period_Return")
