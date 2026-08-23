import pandas as pd
import numpy as np

def calculate_price_zones(price_series: pd.Series, num_bins: int = 50) -> pd.DataFrame:
    """
    Groups historical prices into zones to identify quantitative support/resistance levels
    based on the time spent at those prices (Time-at-Price density).
    """
    prices = price_series.dropna()
    if prices.empty:
        return pd.DataFrame()

    # Create statistical bins (price zones)
    counts, bin_edges = np.histogram(prices, bins=num_bins)
    
    # Structure into a DataFrame
    zones_df = pd.DataFrame({
        "Zone_Bottom": bin_edges[:-1],
        "Zone_Top": bin_edges[1:],
        "Days_in_Zone": counts
    })
    
    # Calculate the midpoint of the zone for clean display
    zones_df["Zone_Center"] = (zones_df["Zone_Bottom"] + zones_df["Zone_Top"]) / 2
    
    # Calculate significance (percentage of total time spent in this zone)
    total_days = len(prices)
    zones_df["Significance"] = (zones_df["Days_in_Zone"] / total_days) * 100
    
    # Sort by the most historically significant zones (highest density)
    zones_df = zones_df.sort_values(by="Days_in_Zone", ascending=False).reset_index(drop=True)
    
    return zones_df
