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

def simulate_historical_tests(df, z_low, z_high, lookahead_bars=10):
    """
    Scans price data to track every time price touches the zone band
    and measures the subsequent reaction strength in ATR units.
    """
    tests = []
    in_test = False
    
    # Iterate through each price bar
    for idx in range(len(df) - lookahead_bars):
        bar_high = df['High'].iloc[idx]
        bar_low = df['Low'].iloc[idx]
        current_atr = df['ATR'].iloc[idx]
        
        # Detect zone intersection (price low <= zone_high and price high >= zone_low)
        intersects = (bar_low <= z_high) and (bar_high >= z_low)
        
        if intersects and not in_test:
            in_test = True
            
            # Look ahead N bars to measure reaction
            future_slice = df.iloc[idx + 1 : idx + 1 + lookahead_bars]
            
            # Determine if approach was from above (Support test) or below (Resistance test)
            prev_close = df['Close'].iloc[idx - 1] if idx > 0 else df['Close'].iloc[idx]
            zone_center = (z_low + z_high) / 2
            
            if prev_close >= zone_center:
                # Support test: Measure maximum upward bounce
                max_favorable = future_slice['High'].max() - z_high
                move_atr = max(0.0, max_favorable / current_atr) if current_atr > 0 else 0.0
            else:
                # Resistance test: Measure maximum downward bounce
                max_favorable = z_low - future_slice['Low'].min()
                move_atr = max(0.0, max_favorable / current_atr) if current_atr > 0 else 0.0
                
            tests.append(ZoneTest(entered_zone=True, move_away_atr=move_atr))
            
        elif not intersects:
            in_test = False  # Reset debounce state once price exits zone
            
    return tests

def calculate_price_zones(df, left_bars=5, right_bars=5, cluster_tolerance=0.005):
    data = df.copy()
    
    # 1. Calculate required rolling metrics
    data['ATR'] = calculate_atr(data, 14)
    data['Avg_Volume'] = data['Volume'].rolling(20).mean()
    data = data.dropna().reset_index(drop=True)
    
    if data.empty:
        return pd.DataFrame()
        
    last_idx = len(data) - 1
    turnarounds = []
    
    # 2. Identify Swing Highs and Lows
    order = max(left_bars, right_bars)
    high_idx = argrelextrema(data['High'].values, np.greater_equal, order=order)[0]
    low_idx = argrelextrema(data['Low'].values, np.less_equal, order=order)[0]
    
    pivot_indices = sorted(list(set(high_idx).union(set(low_idx))))
    
    # 3. Build Turnaround Objects
    for idx in pivot_indices:
        is_high = idx in high_idx
        price = data['High'].iloc[idx] if is_high else data['Low'].iloc[idx]
        
        lookahead = min(idx + 10, len(data) - 1)
        if is_high:
            favorable_move = price - data['Low'].iloc[idx:lookahead].min()
        else:
            favorable_move = data['High'].iloc[idx:lookahead].max() - price
            
        t = Turnaround(
            price=price,
            age_days=last_idx - idx,
            volume=data['Volume'].iloc[idx],
            average_volume_at_reversal=data['Avg_Volume'].iloc[idx],
            atr_at_reversal=data['ATR'].iloc[idx],
            max_favorable_move=favorable_move
        )
        turnarounds.append(t)
        
    if not turnarounds:
        return pd.DataFrame()
        
    # 4. Cluster prices into Zones
    prices = sorted([t.price for t in turnarounds])
    zones_raw = []
    current_cluster = [prices[0]]
    
    for p in prices[1:]:
        if p <= current_cluster[0] * (1 + cluster_tolerance):
            current_cluster.append(p)
        else:
            zones_raw.append(current_cluster)
            current_cluster = [p]
    zones_raw.append(current_cluster)
    
    # 5. Evaluate Zones with Real Simulation
    results = []
    
    for cluster in zones_raw:
        z_low = min(cluster)
        z_high = max(cluster)
        zone_obj = Zone(low=z_low, high=z_high)
        
        # Run real test simulation against price series
        real_tests = simulate_historical_tests(data, z_low, z_high, lookahead_bars=10)
        
        score_breakdown = evaluate_zone(
            zone=zone_obj,
            turnarounds=turnarounds,
            tests=real_tests,
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