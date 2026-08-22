import pandas as pd
import numpy as np
from src.returns import calculate_returns

def test_calculate_returns():
    """
    Tests the return engine using known synthetic data.
    Prices: 100 -> 110 (+10%) -> 121 (+10%)
    Expected Cumulative Return: 21%
    """
    # 1. Setup Dummy Data
    dates = pd.date_range(start="2023-01-01", periods=3)
    df = pd.DataFrame({"Close": [100.0, 110.0, 121.0]}, index=dates)
    
    # 2. Execute Function
    result = calculate_returns(df, price_col="Close")
    
    # 3. Assert (Verify) Results
    # Daily return for the second day should be exactly 0.10
    np.testing.assert_almost_equal(result["Daily_Return"].iloc[0], 0.10)
    
    # Daily return for the third day should be exactly 0.10
    np.testing.assert_almost_equal(result["Daily_Return"].iloc[1], 0.10)
    
    # Cumulative return at the end should be exactly 0.21 (21%)
    np.testing.assert_almost_equal(result["Cumulative_Return"].iloc[-1], 0.21)