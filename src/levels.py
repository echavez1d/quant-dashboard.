import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def calculate_price_zones(df: pd.DataFrame, left_bars: int = 5, right_bars: int = 5, cluster_tolerance: float = 0.005) -> pd.DataFrame:
    """
    Advanced Support/Resistance detection using Swing Points and Price Clustering.
    """
    # 1. SWING DETECTION
    # We need High and Low prices for true swing detection
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    
    # A swing point must be the highest/lowest point within this total window size
    window = left_bars + right_bars + 1
    if n < window:
        return pd.DataFrame()

    # Create sliding windows to find local extrema
    high_windows = sliding_window_view(highs, window)
    low_windows = sliding_window_view(lows, window)
    
    # The center bars we are evaluating
    centers_high = highs[left_bars : n - right_bars]
    centers_low = lows[left_bars : n - right_bars]
    
    # Find where the center is the absolute max/min of its window
    is_high = centers_high >= high_windows.max(axis=1)
    is_low = centers_low <= low_windows.min(axis=1)
    
    # Extract the prices where these swings occurred
    swing_prices = np.concatenate([centers_high[is_high], centers_low[is_low]])
    
    if len(swing_prices) == 0:
        return pd.DataFrame()

    # 2. PRICE CLUSTERING
    # Sort all turning points from lowest to highest price
    sorted_swings = np.sort(swing_prices)
    
    clusters = []
    current_cluster = [sorted_swings[0]]
    
    # Greedily group swings that are close to each other
    for price in sorted_swings[1:]:
        running_mean = np.mean(current_cluster)
        
        # If the price is within the % tolerance of the cluster's average, add it
        if abs(price - running_mean) / running_mean <= cluster_tolerance:
            current_cluster.append(price)
        else:
            # Start a new cluster
            clusters.append(current_cluster)
            current_cluster = [price]
    clusters.append(current_cluster)

    # 3. FORMAT FOR DASHBOARD
    zones_data = []
    total_touches = len(swing_prices)
    
    for cluster in clusters:
        zone_bottom = min(cluster)
        zone_top = max(cluster)
        
        # Give a tiny buffer for perfectly identical touches so the UI search works
        if zone_bottom == zone_top:
            zone_bottom *= 0.999
            zone_top *= 1.001
            
        zones_data.append({
            "Zone_Bottom": zone_bottom,
            "Zone_Top": zone_top,
            "Zone_Center": np.mean(cluster),
            "Days_in_Zone": len(cluster), # Now represents actual 'Touches' or 'Reversals'
            "Significance": (len(cluster) / total_touches) * 100
        })
        
    zones_df = pd.DataFrame(zones_data)
    # Sort by the zones with the most historical touches
    zones_df = zones_df.sort_values(by="Days_in_Zone", ascending=False).reset_index(drop=True)
    
    return zones_df
