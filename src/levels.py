import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from src.scoring import (
    Turnaround, 
    Zone, 
    ZoneTest, 
    merge_overlapping_zones, 
    evaluate_chart_zones
)

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1).rolling(period).mean()

def simulate_historical_tests(df, z_low, z_high, lookahead_bars=10):
    tests = []
    in_test = False
    
    for idx in range(len(df) - lookahead_bars):
        bar_high = df['High'].iloc[idx]
        bar_low = df['Low'].iloc[idx]
        current_atr = df['ATR'].iloc[idx]
        
        intersects = (bar_low <= z_high) and (bar_high >= z_low)
        
        if intersects and not in_test:
            in_test = True
            future_slice = df.iloc[idx + 1 : idx + 1 + lookahead_bars]
            prev_close = df['Close'].iloc[idx - 1] if idx > 0 else df['Close'].iloc[idx]
            zone_center = (z_low + z_high) / 2
            
            if prev_close >= zone_center:
                max_favorable = future_slice['High'].max() - z_high
                move_atr = max(0.0, max_favorable / current_atr) if current_atr > 0 else 0.0
            else:
                max_favorable = z_low - future_slice['Low'].min()
                move_atr = max(0.0, max_favorable / current_atr) if current_atr > 0 else 0.0
                
            tests.append(ZoneTest(entered_zone=True, move_away_atr=move_atr))
            
        elif not intersects:
            in_test = False
            
    return tests

def calculate_price_zones(df, left_bars=5, right_bars=5, cluster_tolerance=0.005):
    data = df.copy()
    
    data['ATR'] = calculate_atr(data, 14)
    data['Avg_Volume'] = data['Volume'].rolling(20).mean()
    data = data.dropna().reset_index(drop=True)
    
    if data.empty:
        return pd.DataFrame()
        
    last_idx = len(data) - 1
    turnarounds = []
    
    order = max(left_bars, right_bars)
    high_idx = argrelextrema(data['High'].values, np.greater_equal, order=order)[0]
    low_idx = argrelextrema(data['Low'].values, np.less_equal, order=order)[0]
    
    pivot_indices = sorted(list(set(high_idx).union(set(low_idx))))

    # Check for actual date metadata for accurate calendar math vs. bar math
    has_datetime_index = pd.api.types.is_datetime64_any_dtype(data.index)
    has_date_col = 'Date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Date'])
    
    for idx in pivot_indices:
        is_high = idx in high_idx
        price = data['High'].iloc[idx] if is_high else data['Low'].iloc[idx]
        
        lookahead = min(idx + 10, len(data) - 1)
        if is_high:
            favorable_move = price - data['Low'].iloc[idx:lookahead].min()
        else:
            favorable_move = data['High'].iloc[idx:lookahead].max() - price
            
        # ---------------------------------------------------------
        # ANNUALIZATION FIX: Convert bar spacing to calendar days
        # ---------------------------------------------------------
        if has_datetime_index:
            age_days = (data.index[-1] - data.index[idx]).total_seconds() / 86400.0
        elif has_date_col:
            age_days = (data['Date'].iloc[-1] - data['Date'].iloc[idx]).total_seconds() / 86400.0
        else:
            # Convert 252 trading bars/year to 365.25 calendar days/year
            age_days = (last_idx - idx) * (365.25 / 252.0)

        t = Turnaround(
            price=price,
            age_days=age_days,
            volume=data['Volume'].iloc[idx],
            average_volume_at_reversal=data['Avg_Volume'].iloc[idx],
            atr_at_reversal=data['ATR'].iloc[idx],
            max_favorable_move=favorable_move
        )
        turnarounds.append(t)
        
    if not turnarounds:
        return pd.DataFrame()
        
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
    
    avg_atr = data['ATR'].iloc[-1]
    
    # PHASE 1: Generate Candidate Zones
    candidate_zones = []
    for cluster in zones_raw:
        raw_low = min(cluster)
        raw_high = max(cluster)
        center = (raw_low + raw_high) / 2
        
        half_width = max((raw_high - raw_low) / 2, avg_atr * 0.25)
        z_low = center - half_width
        z_high = center + half_width
        
        candidate_zones.append(Zone(low=z_low, high=z_high))

    # PHASE 2: Merge Overlapping Zones
    merged_zones = merge_overlapping_zones(candidate_zones)

    # PHASE 3: Gather historical tests for merged zones
    zone_test_pairs = []
    for zone in merged_zones:
        real_tests = simulate_historical_tests(data, zone.low, zone.high, lookahead_bars=10)
        zone_test_pairs.append((zone, real_tests))

    # PHASE 4: Batch Evaluation with Relative Chart Normalization
    if not zone_test_pairs:
        return pd.DataFrame()

    evaluated_zones = evaluate_chart_zones(
        zone_test_pairs=zone_test_pairs,
        turnarounds=turnarounds,
        target_reaction_atr=2.0,
        final_scale_to_100=True
    )

    # PHASE 5: Build final DataFrame
    results = []
    for zone, breakdown in evaluated_zones:
        touches = sum(1 for t in turnarounds if zone.contains(t.price))
        
        results.append({
            "Zone_Bottom": zone.low,
            "Zone_Top": zone.high,
            "Zone_Center": (zone.low + zone.high) / 2,
            "Reversal_Touches": touches,
            "Score": breakdown.score,
            "Concentration": breakdown.concentration,
            "Rejection_Rate": breakdown.rejection_rate,
            "Reaction_Str": breakdown.reaction_strength,
            "Confidence": breakdown.confidence
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        
    return res_df