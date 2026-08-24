"""
Support / Resistance Zone Strength Score
=========================================

A composite, normalized score for how strong a price support/resistance
zone is, combining five ideas:

    1. ReWTS-Weighted Turnaround Concentration   -> C
    2. Zone Rejection Rate                        -> R
    3. Reaction Strength (normalized)              -> S
    4. Band-Width / Density Adjustment             -> D   (optional)
    5. Sample-Size Confidence Factor               -> F

Full formula:
    Zone Strength Score = 100 * C**alpha * R**beta * S**gamma * D**delta * F
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import defaultdict
import math


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Turnaround:
    """
    A single significant price reversal (swing high or swing low).
    """
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


# ---------------------------------------------------------------------------
# 1. ReWTS-Weighted Turnaround Concentration  ->  C
# ---------------------------------------------------------------------------

def recency_weight(age_days: float, half_life: float = 90.0) -> float:
    """Standard exponential decay fall-back."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return 2 ** (-age_days / half_life)


def reaction_strength(t: Turnaround, cap: float = 5.0) -> float:
    """Reaction_Strength_i = max_favorable_move / ATR_at_reversal."""
    if t.atr_at_reversal <= 0:
        return 0.0
    return min(t.max_favorable_move / t.atr_at_reversal, cap)


def volume_factor(t: Turnaround, v_max: float = 3.0) -> float:
    """Volume_Factor_i = min(volume_i / rolling_average_volume_i, v_max)."""
    if t.average_volume_at_reversal <= 0:
        return 1.0
    return min(t.volume / t.average_volume_at_reversal, v_max)


def compute_rewts_chunk_weights(
    turnarounds: List[Turnaround],
    chunk_size_days: float = 30.0,
    half_life: float = 180.0,
    gamma: float = 1.0,
) -> Dict[int, float]:
    """
    Groups turnarounds into temporal chunks and calculates dynamic weights 
    based on regime similarity between historical chunks and current look-back data.
    """
    if not turnarounds:
        return {}

    # 1. Identify current regime (from most recent turnarounds in look-back window)
    recent_turnarounds = [t for t in turnarounds if t.age_days <= chunk_size_days]
    if not recent_turnarounds:
        # Fall back to top 5 youngest turnarounds
        recent_turnarounds = sorted(turnarounds, key=lambda x: x.age_days)[:5]

    current_atr = sum(t.atr_at_reversal for t in recent_turnarounds) / len(recent_turnarounds)
    current_vol_factor = sum(volume_factor(t) for t in recent_turnarounds) / len(recent_turnarounds)

    # 2. Partition historical turnarounds into temporal chunks
    chunks = defaultdict(list)
    for t in turnarounds:
        chunk_idx = int(t.age_days // chunk_size_days)
        chunks[chunk_idx].append(t)

    chunk_weights = {}

    # 3. Calculate regime similarity for each chunk
    for chunk_idx, chunk_turnarounds in chunks.items():
        chunk_avg_atr = sum(t.atr_at_reversal for t in chunk_turnarounds) / len(chunk_turnarounds)
        chunk_avg_vol = sum(volume_factor(t) for t in chunk_turnarounds) / len(chunk_turnarounds)

        # Normalized feature distance between current look-back regime & historical chunk
        atr_diff = (chunk_avg_atr - current_atr) / max(current_atr, 1e-5)
        vol_diff = chunk_avg_vol - current_vol_factor
        regime_distance_sq = (atr_diff ** 2) + (vol_diff ** 2)

        # RBF Kernel for dynamic regime recall
        similarity_weight = math.exp(-gamma * regime_distance_sq)

        # Soft base decay (prevents distant identical regimes from over-dominating)
        chunk_age_days = chunk_idx * chunk_size_days
        base_decay = 2 ** (-chunk_age_days / half_life)

        # Combined ReWTS Chunk Weight
        chunk_weights[chunk_idx] = similarity_weight * base_decay

    return chunk_weights


def event_weight_rewts(
    t: Turnaround,
    chunk_weights: Dict[int, float],
    chunk_size_days: float = 30.0,
    reaction_cap: float = 5.0,
    volume_cap: float = 3.0,
) -> float:
    """
    Event_Weight_i = ReWTS_Chunk_Weight * Reaction_Strength_i * Volume_Factor_i
    """
    chunk_idx = int(t.age_days // chunk_size_days)
    rewts_weight = chunk_weights.get(chunk_idx, 1.0)

    return (
        rewts_weight
        * reaction_strength(t, reaction_cap)
        * volume_factor(t, volume_cap)
    )


def turnaround_concentration(
    turnarounds: List[Turnaround],
    zone: Zone,
    half_life: float = 90.0,
    reaction_cap: float = 5.0,
    volume_cap: float = 3.0,
    use_rewts: bool = True,
    chunk_size_days: float = 30.0,
) -> float:
    """
    Calculates C (0.0 - 1.0) as the ratio of in-zone event weights to total event weights.
    Uses ReWTS dynamic chunking by default to prevent catastrophic forgetting.
    """
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
            recency_weight(t.age_days, half_life)
            * reaction_strength(t, reaction_cap)
            * volume_factor(t, volume_cap)
            for t in turnarounds
        ]

    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0

    in_zone_weight = sum(
        w for w, t in zip(weights, turnarounds) if zone.contains(t.price)
    )
    return in_zone_weight / total_weight


# ---------------------------------------------------------------------------
# 2. Zone Rejection Rate  ->  R
# ---------------------------------------------------------------------------

def zone_rejection_rate(tests: List[ZoneTest]) -> float:
    """R = successful rejections / valid zone tests."""
    valid = [t for t in tests if t.is_valid]
    if not valid:
        return 0.0
    rejections = sum(1 for t in valid if t.is_rejection)
    return rejections / len(valid)


# ---------------------------------------------------------------------------
# 3. Reaction Strength (normalized)  ->  S
# ---------------------------------------------------------------------------

def average_reaction_in_zone(
    turnarounds: List[Turnaround], zone: Zone, cap: float = 5.0
) -> float:
    """Average (capped) reaction strength of turnarounds inside the zone."""
    in_zone = [t for t in turnarounds if zone.contains(t.price)]
    if not in_zone:
        return 0.0
    return sum(reaction_strength(t, cap) for t in in_zone) / len(in_zone)


def normalized_reaction_strength(
    average_reaction_atr: float, target_reaction_atr: float
) -> float:
    """S = min(average_reaction_in_ATRs / target_reaction_in_ATRs, 1)."""
    if target_reaction_atr <= 0:
        return 0.0
    return min(average_reaction_atr / target_reaction_atr, 1.0)


# ---------------------------------------------------------------------------
# 4. Band-Width / Density Adjustment  ->  D   (optional)
# ---------------------------------------------------------------------------

def atr_based_zone_width(atr: float, k: float = 1.0) -> float:
    return k * atr


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
    """F = n / (n + k)."""
    if n_valid_tests < 0:
        raise ValueError("n_valid_tests can't be negative")
    return n_valid_tests / (n_valid_tests + k)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def zone_strength_score(
    concentration: float,                # C
    rejection_rate: float,               # R
    reaction_strength_norm: float,       # S
    confidence: float,                   # F
    density_advantage_value: Optional[float] = None,  # D, optional
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> float:
    """
    Full:        100 * C**alpha * R**beta * S**gamma * D**delta * F
    Simplified:  100 * C * R * S * F              (when D is None)
    """
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
# End-to-end helper
# ---------------------------------------------------------------------------

@dataclass
class ZoneScoreBreakdown:
    concentration: float                 # C
    rejection_rate: float                # R
    reaction_strength: float             # S (normalized)
    confidence: float                    # F
    density_advantage: Optional[float]   # D
    score: float                         # final Zone Strength Score, ~0-100


def evaluate_zone(
    zone: Zone,
    turnarounds: List[Turnaround],
    tests: List[ZoneTest],
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
) -> ZoneScoreBreakdown:
    """
    Evaluates zone strength using ReWTS-weighted concentration by default.
    """
    weights = weights or {}

    concentration = turnaround_concentration(
        turnarounds,
        zone,
        half_life=half_life,
        reaction_cap=reaction_cap,
        volume_cap=volume_cap,
        use_rewts=use_rewts,
        chunk_size_days=chunk_size_days,
    )
    rejection_rate = zone_rejection_rate(tests)
    avg_reaction = average_reaction_in_zone(turnarounds, zone, reaction_cap)
    reaction_norm = normalized_reaction_strength(avg_reaction, target_reaction_atr)
    n_valid = sum(1 for t in tests if t.is_valid)
    confidence = confidence_factor(n_valid, confidence_k)

    density_adv = None
    if expected_density is not None:
        chunk_w = compute_rewts_chunk_weights(turnarounds, chunk_size_days) if use_rewts else {}
        weighted_count = sum(
            event_weight_rewts(t, chunk_w, chunk_size_days, reaction_cap, volume_cap) if use_rewts
            else (recency_weight(t.age_days, half_life) * reaction_strength(t, reaction_cap) * volume_factor(t, volume_cap))
            for t in turnarounds
            if zone.contains(t.price)
        )
        zone_density = reversal_density(weighted_count, zone.width)
        density_adv = density_advantage(zone_density, expected_density)

    score = zone_strength_score(
        concentration,
        rejection_rate,
        reaction_norm,
        confidence,
        density_advantage_value=density_adv,
        alpha=weights.get("alpha", 1.0),
        beta=weights.get("beta", 1.0),
        gamma=weights.get("gamma", 1.0),
        delta=weights.get("delta", 1.0),
    )

    return ZoneScoreBreakdown(
        concentration=concentration,
        rejection_rate=rejection_rate,
        reaction_strength=reaction_norm,
        confidence=confidence,
        density_advantage=density_adv,
        score=score,
    )