import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def calculate_price_zones(df: pd.DataFrame, left_bars: int = 5, right_bars: int = 5, cluster_tolerance: float = 0.005) -> pd.DataFrame:
    """
    Advanced Support/Resistance detection using Pivot Swing Points and Price Clustering.
    """
    if "High" not in df.columns or "Low" not in df.columns or len(df) == 0:
        return pd.DataFrame()

    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    
    window = left_bars + right_bars + 1
    if n < window:
        return pd.DataFrame()

    # 1. SWING PIVOT DETECTION
    high_windows = sliding_window_view(highs, window)
    low_windows = sliding_window_view(lows, window)
    
    centers_high = highs[left_bars : n - right_bars]
    centers_low = lows[left_bars : n - right_bars]
    
    is_high = centers_high >= high_windows.max(axis=1)
    is_low = centers_low <= low_windows.min(axis=1)
    
    swing_prices = np.concatenate([centers_high[is_high], centers_low[is_low]])
    
    if len(swing_prices) == 0:
        return pd.DataFrame()

    # 2. GREEDY CLUSTERING
    sorted_swings = np.sort(swing_prices)
    clusters = []
    current_cluster = [sorted_swings[0]]
    
    for price in sorted_swings[1:]:
        running_mean = np.mean(current_cluster)
        if abs(price - running_mean) / running_mean <= cluster_tolerance:
            current_cluster.append(price)
        else:
            clusters.append(current_cluster)
            current_cluster = [price]
    clusters.append(current_cluster)

    # 3. METRICS & SIGNIFICANCE SCORING
    zones_data = []
    total_reversals = len(swing_prices)
    
    for cluster in clusters:
        zone_bottom = min(cluster)
        zone_top = max(cluster)
        
        if zone_bottom == zone_top:
            zone_bottom *= 0.999
            zone_top *= 1.001
            
        reversals_count = len(cluster)
        significance_pct = (reversals_count / total_reversals) * 100
            
        zones_data.append({
            "Zone_Bottom": zone_bottom,
            "Zone_Top": zone_top,
            "Zone_Center": np.mean(cluster),
            "Reversal_Touches": reversals_count,
            "Days_in_Zone": reversals_count, # Maintained for compatibility
            "Significance": significance_pct
        })
        
    zones_df = pd.DataFrame(zones_data)
    zones_df = zones_df.sort_values(by="Reversal_Touches", ascending=False).reset_index(drop=True)
    
    return zones_df