"""
Support / Resistance Zone Strength Score
=========================================

A composite, normalized score for how strong a price support/resistance
zone is. 

Includes ReWTS dynamic temporal chunking to prevent catastrophic forgetting, 
and Chart-Relative Normalization to prevent multiplicative score compression.

Formulas:
    Raw Concentration (C) = In-Zone ReWTS Weight / Total ReWTS Weight
    Relative C = Raw C / Max(Raw C on Chart)
    Zone Strength Score = 100 * (Relative C)**alpha * R**beta * S**gamma * D**delta * F
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import math


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Turnaround:
    """A single significant price reversal (swing high or swing low)."""
    price: float
    age_days: float
    volume: float
    average_volume_at_reversal: float
    atr_at_reversal: float
    max_favorable_move: float


@dataclass
class Zone:
    """A candidate support/resistance zone, defined by its price bounds."""
    low: float
    high: float

    @property
    def width(self) -> float:
        return max(self.high - self.low, 0.0)

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


@dataclass
class ZoneTest:
    """One time price approached/entered the zone."""
    entered_zone: bool
    move_away_atr: float
    rejection_threshold_atr: float = 1.0

    @property
    def is_valid(self) -> bool:
        return self.entered_zone

    @property
    def is_rejection(self) -> bool:
        return self.entered_zone and self.move_away_atr >= self.rejection_threshold_atr


@dataclass
class ZoneScoreBreakdown:
    concentration: float                 # C (Relative to chart max)
    raw_concentration: float             # Original un-normalized C
    rejection_rate: float                # R
    reaction_strength: float             # S (normalized)
    confidence: float                    # F
    density_advantage: Optional[float]   # D
    score: float                         # final Zone Strength Score, 0-100


# ---------------------------------------------------------------------------
# 0. Zone Pre-Processing (Fixes Overlapping Boxes)
# ---------------------------------------------------------------------------

def merge_overlapping_zones(zones: List[Zone]) -> List[Zone]:
    """
    Merges candidate zones that overlap in price bounds to prevent double-counting
    and overlapping visual boxes on the chart.
    """
    if not zones:
        return []

    # Sort zones by lower price bound
    sorted_zones = sorted(zones, key=lambda z: z.low)
    merged = [sorted_zones[0]]

    for current in sorted_zones[1:]:
        prev = merged[-1]
        # If the current zone overlaps or touches the previous zone's upper bound
        if current.low <= prev.high:
            # Merge them into a single wider zone
            merged[-1] = Zone(
                low=min(prev.low, current.low),
                high=max(prev.high, current.high)
            )
        else:
            merged.append(current)

    return merged


# ---------------------------------------------------------------------------
# 1. ReWTS-Weighted Turnaround Concentration  ->  C
# ---------------------------------------------------------------------------

def recency_weight(age_days: float, half_life: float = 90.0) -> float:
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return 2 ** (-age_days / half_life)


def reaction_strength(t: Turnaround, cap: float = 5.0) -> float:
    if t.atr_at_reversal <= 0:
        return 0.0
    return min(t.max_favorable_move / t.atr_at_reversal, cap)


def volume_factor(t: Turnaround, v_max: float = 3.0) -> float:
    if t.average_volume_at_reversal <= 0:
        return 1.0
    return min(t.volume / t.average_volume_at_reversal, v_max)


def compute_rewts_chunk_weights(
    turnarounds: List[Turnaround],
    chunk_size_days: float = 30.0,
    half_life: float = 180.0,
    gamma: float = 1.0,
) -> Dict[int, float]:
    if not turnarounds:
        return {}

    recent_turnarounds = [t for t in turnarounds if t.age_days <= chunk_size_days]
    if not recent_turnarounds:
        recent_turnarounds = sorted(turnarounds, key=lambda x: x.age_days)[:5]

    current_atr = sum(t.atr_at_reversal for t in recent_turnarounds) / len(recent_turnarounds)
    current_vol_factor = sum(volume_factor(t) for t in recent_turnarounds) / len(recent_turnarounds)

    chunks = defaultdict(list)
    for t in turnarounds:
        chunk_idx = int(t.age_days // chunk_size_days)
        chunks[chunk_idx].append(t)

    chunk_weights = {}
    for chunk_idx, chunk_turnarounds in chunks.items():
        chunk_avg_atr = sum(t.atr_at_reversal for t in chunk_turnarounds) / len(chunk_turnarounds)
        chunk_avg_vol = sum(volume_factor(t) for t in chunk_turnarounds) / len(chunk_turnarounds)

        atr_diff = (chunk_avg_atr - current_atr) / max(current_atr, 1e-5)
        vol_diff = chunk_avg_vol - current_vol_factor
        regime_distance_sq = (atr_diff ** 2) + (vol_diff ** 2)

        similarity_weight = math.exp(-gamma * regime_distance_sq)
        chunk_age_days = chunk_idx * chunk_size_days
        base_decay = 2 ** (-chunk_age_days / half_life)

        chunk_weights[chunk_idx] = similarity_weight * base_decay

    return chunk_weights


def event_weight_rewts(
    t: Turnaround,
    chunk_weights: Dict[int, float],
    chunk_size_days: float = 30.0,
    reaction_cap: float = 5.0,
    volume_cap: float = 3.0,
) -> float:
    chunk_idx = int(t.age_days // chunk_size_days)
    rewts_weight = chunk_weights.get(chunk_idx, 1.0)
    return rewts_weight * reaction_strength(t, reaction_cap) * volume_factor(t, volume_cap)


def turnaround_concentration(
    turnarounds: List[Turnaround],
    zone: Zone,
    half_life: float = 90.0,
    reaction_cap: float = 5.0,
    volume_cap: float = 3.0,
    use_rewts: bool = True,
    chunk_size_days: float = 30.0,
) -> float:
    if not turnarounds:
        return 0.0

    if use_rewts:
        chunk_weights = compute_rewts_chunk_weights(
            turnarounds, chunk_size_days=chunk_size_days, half_life=half_life * 2
        )
        weights = [
            event_weight_rewts(t, chunk_weights, chunk_size_days, reaction_cap, volume_cap)
            for t in turnarounds
        ]
    else:
        weights = [
            recency_weight(t.age_days, half_life) * reaction_strength(t, reaction_cap) * volume_factor(t, volume_cap)
            for t in turnarounds
        ]

    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0

    in_zone_weight = sum(w for w, t in zip(weights, turnarounds) if zone.contains(t.price))
    return in_zone_weight / total_weight


# ---------------------------------------------------------------------------
# 2. Zone Rejection Rate  ->  R
# ---------------------------------------------------------------------------

def zone_rejection_rate(tests: List[ZoneTest]) -> float:
    valid = [t for t in tests if t.is_valid]
    if not valid:
        return 0.0
    rejections = sum(1 for t in valid if t.is_rejection)
    return rejections / len(valid)


# ---------------------------------------------------------------------------
# 3. Reaction Strength (normalized)  ->  S
# ---------------------------------------------------------------------------

def average_reaction_in_zone(turnarounds: List[Turnaround], zone: Zone, cap: float = 5.0) -> float:
    in_zone = [t for t in turnarounds if zone.contains(t.price)]
    if not in_zone:
        return 0.0
    return sum(reaction_strength(t, cap) for t in in_zone) / len(in_zone)


def normalized_reaction_strength(average_reaction_atr: float, target_reaction_atr: float) -> float:
    if target_reaction_atr <= 0:
        return 0.0
    return min(average_reaction_atr / target_reaction_atr, 1.0)


# ---------------------------------------------------------------------------
# 4. Band-Width / Density Adjustment  ->  D
# ---------------------------------------------------------------------------

def reversal_density(weighted_turnaround_count: float, zone_width: float) -> float:
    if zone_width <= 0:
        return 0.0
    return weighted_turnaround_count / zone_width


def density_advantage(zone_density: float, expected_density: float) -> float:
    if expected_density <= 0:
        return 1.0
    return zone_density / expected_density


# ---------------------------------------------------------------------------
# 5. Sample-Size Confidence Factor  ->  F
# ---------------------------------------------------------------------------

def confidence_factor(n_valid_tests: int, k: float = 5.0) -> float:
    if n_valid_tests < 0:
        raise ValueError("n_valid_tests can't be negative")
    return n_valid_tests / (n_valid_tests + k)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def zone_strength_score(
    concentration: float,
    rejection_rate: float,
    reaction_strength_norm: float,
    confidence: float,
    density_advantage_value: Optional[float] = None,
    alpha: float = 0.5,  # Changed default to 0.5 to soften C's penalty
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> float:
    score = (
        100
        * (concentration ** alpha)
        * (rejection_rate ** beta)
        * (reaction_strength_norm ** gamma)
        * confidence
    )
    if density_advantage_value is not None:
        score *= density_advantage_value ** delta
    return score


# ---------------------------------------------------------------------------
# Chart-Level Evaluator (Fixes Score Compression)
# ---------------------------------------------------------------------------

def evaluate_chart_zones(
    zone_test_pairs: List[Tuple[Zone, List[ZoneTest]]],
    turnarounds: List[Turnaround],
    target_reaction_atr: float,
    *,
    half_life: float = 90.0,
    reaction_cap: float = 5.0,
    volume_cap: float = 3.0,
    confidence_k: float = 5.0,
    expected_density: Optional[float] = None,
    weights: Optional[dict] = None,
    use_rewts: bool = True,
    chunk_size_days: float = 30.0,
    final_scale_to_100: bool = True,
) -> List[Tuple[Zone, ZoneScoreBreakdown]]:
    """
    Evaluates all zones on a chart simultaneously to allow relative scoring.
    This prevents the "highest score is a 12" problem by ensuring the densest
    zone gets a Concentration (C) of 1.0.
    
    NOTE: Pass ALREADY MERGED zones into this function to avoid double-counting.
    """
    weights = weights or {}
    alpha = weights.get("alpha", 0.5) # Default to 0.5 square-root scaling
    
    raw_breakdowns = []
    
    # Pass 1: Compute Raw Metrics
    for zone, tests in zone_test_pairs:
        raw_c = turnaround_concentration(
            turnarounds, zone, half_life, reaction_cap, volume_cap, use_rewts, chunk_size_days
        )
        rej_rate = zone_rejection_rate(tests)
        avg_rxn = average_reaction_in_zone(turnarounds, zone, reaction_cap)
        rxn_norm = normalized_reaction_strength(avg_rxn, target_reaction_atr)
        conf = confidence_factor(sum(1 for t in tests if t.is_valid), confidence_k)
        
        density_adv = None
        if expected_density is not None:
            chunk_w = compute_rewts_chunk_weights(turnarounds, chunk_size_days) if use_rewts else {}
            weighted_count = sum(
                event_weight_rewts(t, chunk_w, chunk_size_days, reaction_cap, volume_cap) if use_rewts
                else (recency_weight(t.age_days, half_life) * reaction_strength(t, reaction_cap) * volume_factor(t, volume_cap))
                for t in turnarounds if zone.contains(t.price)
            )
            density_adv = density_advantage(reversal_density(weighted_count, zone.width), expected_density)

        raw_breakdowns.append({
            "zone": zone,
            "raw_c": raw_c,
            "rej_rate": rej_rate,
            "rxn_norm": rxn_norm,
            "conf": conf,
            "density_adv": density_adv
        })

    if not raw_breakdowns:
        return []

    # Pass 2: Relative Normalization
    max_raw_c = max((b["raw_c"] for b in raw_breakdowns), default=1.0)
    if max_raw_c == 0:
        max_raw_c = 1.0

    results = []
    for b in raw_breakdowns:
        relative_c = b["raw_c"] / max_raw_c
        
        raw_score = zone_strength_score(
            concentration=relative_c,
            rejection_rate=b["rej_rate"],
            reaction_strength_norm=b["rxn_norm"],
            confidence=b["conf"],
            density_advantage_value=b["density_adv"],
            alpha=alpha,
            beta=weights.get("beta", 1.0),
            gamma=weights.get("gamma", 1.0),
            delta=weights.get("delta", 1.0),
        )
        
        breakdown = ZoneScoreBreakdown(
            concentration=relative_c,
            raw_concentration=b["raw_c"],
            rejection_rate=b["rej_rate"],
            reaction_strength=b["rxn_norm"],
            confidence=b["conf"],
            density_advantage=b["density_adv"],
            score=raw_score
        )
        results.append((b["zone"], breakdown))

    # Pass 3 (Optional): Scale the absolute best zone to 100 for visual consistency
    if final_scale_to_100 and results:
        max_score = max((b.score for _, b in results), default=1.0)
        if max_score > 0:
            for _, b in results:
                b.score = (b.score / max_score) * 100.0

    return results