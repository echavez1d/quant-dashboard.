import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from src.scoring import Turnaround, Zone, ZoneTest, evaluate_zone

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1).rolling(period).mean()

def calculate_price_zones(df, left_bars=5, right_bars=5, cluster_tolerance=0.005):
    data = df.copy()
    
    # 1. Calculate required rolling metrics
    data['ATR'] = calculate_atr(data, 14)
    data['Avg_Volume'] = data['Volume'].rolling(20).mean()
    data = data.dropna()
    
    last_idx = len(data) - 1
    turnarounds = []
    
    # 2. Identify Swing Highs and Lows
    order = max(left_bars, right_bars)
    high_idx = argrelextrema(data['High'].values, np.greater_equal, order=order)[0]
    low_idx = argrelextrema(data['Low'].values, np.less_equal, order=order)[0]
    
    # Combine and deduplicate indices
    pivot_indices = sorted(list(set(high_idx).union(set(low_idx))))
    
    # 3. Build Turnaround Objects
    for idx in pivot_indices:
        is_high = idx in high_idx
        price = data['High'].iloc[idx] if is_high else data['Low'].iloc[idx]
        
        # Calculate max favorable move (looking ahead 10 bars)
        lookahead = min(idx + 10, len(data) - 1)
        if is_high:
            favorable_move = price - data['Low'].iloc[idx:lookahead].min()
        else:
            favorable_move = data['High'].iloc[idx:lookahead].max() - price
            
        t = Turnaround(
            price=price,
            age_days=last_idx - idx,  # bars ago
            volume=data['Volume'].iloc[idx],
            average_volume_at_reversal=data['Avg_Volume'].iloc[idx],
            atr_at_reversal=data['ATR'].iloc[idx],
            max_favorable_move=favorable_move
        )
        turnarounds.append(t)
        
    # 4. Cluster prices into Zones
    if not turnarounds:
        return pd.DataFrame()
        
    prices = [t.price for t in turnarounds]
    prices.sort()
    
    zones_raw = []
    current_cluster = [prices[0]]
    
    for p in prices[1:]:
        if p <= current_cluster[0] * (1 + cluster_tolerance):
            current_cluster.append(p)
        else:
            zones_raw.append(current_cluster)
            current_cluster = [p]
    zones_raw.append(current_cluster)
    
    # 5. Evaluate Zones and Build Output
    results = []
    # Mock some tests for demonstration (In production, you'd scan price action crossing the zone)
    dummy_tests = [ZoneTest(entered_zone=True, move_away_atr=2.0) for _ in range(5)]
    
    for cluster in zones_raw:
        z_low = min(cluster)
        z_high = max(cluster)
        zone_obj = Zone(low=z_low, high=z_high)
        
        # Score it!
        score_breakdown = evaluate_zone(
            zone=zone_obj,
            turnarounds=turnarounds,
            tests=dummy_tests,  # Replace with actual historical test simulation
            target_reaction_atr=2.0
        )
        
        results.append({
            "Zone_Bottom": z_low,
            "Zone_Top": z_high,
            "Zone_Center": (z_low + z_high) / 2,
            "Reversal_Touches": len(cluster),
            "Score": score_breakdown.score,
            "Concentration": score_breakdown.concentration,
            "Rejection_Rate": score_breakdown.rejection_rate,
            "Reaction_Str": score_breakdown.reaction_strength,
            "Confidence": score_breakdown.confidence
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        
    return res_df