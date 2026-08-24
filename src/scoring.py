"""
Support / Resistance Zone Strength Score
=========================================

A composite, normalized score for how strong a price support/resistance
zone is, combining five ideas from the spec this was built from:

    1. Recency-Weighted Turnaround Concentration  -> C
    2. Zone Rejection Rate                        -> R
    3. Reaction Strength (normalized)              -> S
    4. Band-Width / Density Adjustment             -> D   (optional)
    5. Sample-Size Confidence Factor               -> F

Full formula:
    Zone Strength Score = 100 * C**alpha * R**beta * S**gamma * D**delta * F

Simplified (default, D omitted):
    Zone Strength Score = 100 * C * R * S * F

This is a research/backtesting heuristic: it scores zones based on
historical price action and isn't a guarantee of future behavior.

Dependency-free (standard library only) so it drops into any Python
backend. Every public function maps to one numbered idea in the spec,
so any output value can be traced back to the formula that produced it.
"""

from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
# These three classes are the raw inputs the scoring functions below
# consume. They don't do any math themselves -- they just hold the facts
# about price history that the formulas need.

@dataclass
class Turnaround:
    """
    A single significant price reversal (a swing high or swing low) found
    anywhere in the price history you're analyzing -- not just inside the
    zone being scored. The concentration functions below look at the whole
    set of turnarounds and ask "what share of them cluster in this zone?"

    price                       -- price level where the reversal happened
    age_days                    -- how many days ago this reversal happened
                                    (0 = most recent bar)
    volume                      -- traded volume at the time of the reversal
    average_volume_at_reversal  -- ROLLING average volume as of that date
                                    (e.g. a 20- or 50-day SMA of volume
                                    ending at the reversal) -- NOT an
                                    all-time/dataset-wide average. Volume
                                    regimes drift over years (splits,
                                    float growth, changing interest), so
                                    each turnaround has to be judged
                                    against what was "normal" at the time
                                    it happened, not against today's
                                    baseline.
    atr_at_reversal             -- ATR (average true range) at the time of
                                    the reversal; used to express the
                                    reaction in volatility-normalized terms
    max_favorable_move          -- biggest price move in the reversal's
                                    favor afterward (same units as
                                    price/ATR)
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
    """
    One time price approached/entered the zone and either got rejected or
    didn't. This is distinct from Turnaround: every rejection is a kind of
    turnaround, but a "test" also counts attempts that didn't produce a
    listed turnaround. Rejection Rate and the Confidence Factor are both
    measured against this list.

    entered_zone            -- True if price reached the minimum
                                entry/approach threshold (a "valid" test)
    move_away_atr           -- how far price moved away afterward, in ATRs
    rejection_threshold_atr -- ATRs of move-away required to count this
                                test as a "successful rejection"
    """
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
# 1. Recency-Weighted Turnaround Concentration  ->  C
# ---------------------------------------------------------------------------
# Idea: don't count every historical turnaround equally -- a reversal from
# months ago shouldn't carry the same weight as one from last week. Each
# turnaround gets an exponentially decaying weight based on its age. This
# section also folds in Reaction Strength and Volume Factor (originally
# "section 3" in the spec) so C ends up recency- AND volume-weighted, as
# the composite score calls for.

def recency_weight(age_days: float, half_life: float = 90.0) -> float:
    """w_i = 2 ** (-age_i / half_life). Halves every `half_life` days."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return 2 ** (-age_days / half_life)


def reaction_strength(t: Turnaround, cap: float = 5.0) -> float:
    """
    Reaction_Strength_i = max_favorable_move / ATR_at_reversal, capped so
    a single freak move can't dominate every score that uses it.
    """
    if t.atr_at_reversal <= 0:
        return 0.0
    return min(t.max_favorable_move / t.atr_at_reversal, cap)


def volume_factor(t: Turnaround, v_max: float = 3.0) -> float:
    """
    Volume_Factor_i = min(volume_i / rolling_average_volume_i, v_max)

    Compares each turnaround only to ITS OWN rolling average
    (t.average_volume_at_reversal), never to a single number for the
    whole dataset -- see the note on that field above.
    """
    if t.average_volume_at_reversal <= 0:
        return 1.0
    return min(t.volume / t.average_volume_at_reversal, v_max)


def event_weight(
    t: Turnaround,
    half_life: float = 90.0,
    reaction_cap: float = 5.0,
    volume_cap: float = 3.0,
) -> float:
    """
    Event_Weight_i = w_i * Reaction_Strength_i * Volume_Factor_i

    Recent, high-volume (relative to its own era) reversals that produced
    a big reaction count for more than quiet, low-volume ones.
    """
    return (
        recency_weight(t.age_days, half_life)
        * reaction_strength(t, reaction_cap)
        * volume_factor(t, volume_cap)
    )


def turnaround_concentration(
    turnarounds: List[Turnaround],
    zone: Zone,
    half_life: float = 90.0,
    reaction_cap: float = 5.0,
    volume_cap: float = 3.0,
) -> float:
    """
    C = sum(event_weight_i * I_i) / sum(event_weight_i), over every
    turnaround i in your dataset, where I_i = 1 if that turnaround's price
    falls inside `zone`. Already normalized to 0-1. Returns 0 if there's
    no weight to divide by.
    """
    weights = [
        event_weight(t, half_life, reaction_cap, volume_cap)
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
# Concentration alone rewards wide zones just for containing lots of price
# action. Rejection rate asks the sharper question: when price actually
# tests this zone, how often does it bounce? Already normalized to 0-1.

def zone_rejection_rate(tests: List[ZoneTest]) -> float:
    """R = successful rejections / valid zone tests. 0 if no valid tests."""
    valid = [t for t in tests if t.is_valid]
    if not valid:
        return 0.0
    rejections = sum(1 for t in valid if t.is_rejection)
    return rejections / len(valid)


# ---------------------------------------------------------------------------
# 3. Reaction Strength (normalized)  ->  S
# ---------------------------------------------------------------------------
# How big are the moves this zone actually produces, on average, relative
# to a target you consider "meaningful"? This is the "Important
# normalization" step from the spec -- capped at 1 so one exceptional zone
# can't blow the composite score past what the other factors allow.

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
# A wide zone catches more reversals just by being wide. Density asks
# "reversals per unit of width," and Density Advantage compares that to
# what an equal-width band would catch on average -- i.e. is this zone
# unusually good for its size, or just big? Skip this section entirely
# (leave expected_density=None in evaluate_zone below) if every zone in
# your system already uses the same volatility-adjusted width.

def atr_based_zone_width(atr: float, k: float = 1.0) -> float:
    """Zone_Width = k * ATR -- keeps sizing consistent across instruments."""
    return k * atr


def reversal_density(weighted_turnaround_count: float, zone_width: float) -> float:
    """Reversal_Density = weighted turnaround count / zone width."""
    if zone_width <= 0:
        return 0.0
    return weighted_turnaround_count / zone_width


def density_advantage(zone_density: float, expected_density: float) -> float:
    """
    Density_Advantage = zone_density / expected_density for equal-width
    bands. > 1 means unusually dense for its size; < 1 means just wide.
    """
    if expected_density <= 0:
        return 1.0
    return zone_density / expected_density


# ---------------------------------------------------------------------------
# 5. Sample-Size Confidence Factor  ->  F
# ---------------------------------------------------------------------------
# A zone with 2 tests and a zone with 30 tests shouldn't get equal trust
# even with the same rejection rate. F shrinks the score toward 0 when
# there isn't much evidence yet, and approaches 1 as evidence builds.
# Already normalized to 0-1.

def confidence_factor(n_valid_tests: int, k: float = 5.0) -> float:
    """F = n / (n + k). k is a smoothing constant (5-10 is a good start)."""
    if n_valid_tests < 0:
        raise ValueError("n_valid_tests can't be negative")
    return n_valid_tests / (n_valid_tests + k)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
# Combines C, R, S, F (and optionally D) into the single Zone Strength
# Score. Every input here should already be 0-1 -- that's what the
# functions above take care of.

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
# Ties every section above together so you can go from raw price data to a
# final score + breakdown in one call, without wiring the pieces yourself.

@dataclass
class ZoneScoreBreakdown:
    """Every intermediate value behind the final score, for debugging/UI."""
    concentration: float                 # C
    rejection_rate: float                # R
    reaction_strength: float             # S (normalized)
    confidence: float                    # F
    density_advantage: Optional[float]   # D (None if not used)
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
) -> ZoneScoreBreakdown:
    """
    Run all five sections against one zone and return the composite score
    plus the breakdown that produced it.

    Each Turnaround must carry its own `average_volume_at_reversal` (a
    rolling SMA of volume as of that date) -- there's no dataset-wide
    average passed in here, since one flat number can't fairly represent
    volume across years of a changing regime.

    `expected_density` is optional -- pass it (computed from neighboring
    or random bands of equal width) to include the section 4 density
    adjustment; leave it as None to use the simplified 100*C*R*S*F form.

    `weights` optionally overrides alpha/beta/gamma/delta, e.g.
    {"alpha": 1.5} to weight concentration more heavily.
    """
    weights = weights or {}

    concentration = turnaround_concentration(
        turnarounds, zone, half_life, reaction_cap, volume_cap
    )
    rejection_rate = zone_rejection_rate(tests)
    avg_reaction = average_reaction_in_zone(turnarounds, zone, reaction_cap)
    reaction_norm = normalized_reaction_strength(avg_reaction, target_reaction_atr)
    n_valid = sum(1 for t in tests if t.is_valid)
    confidence = confidence_factor(n_valid, confidence_k)

    density_adv = None
    if expected_density is not None:
        weighted_count = sum(
            event_weight(t, half_life, reaction_cap, volume_cap)
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


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    zone = Zone(low=98.0, high=102.0)

    turnarounds = [
        # volume vs. average_volume_at_reversal is each turnaround's OWN
        # rolling 20/50-day SMA at that time -- not one shared number.
        Turnaround(price=100.5, age_days=5, volume=1_500_000, average_volume_at_reversal=1_000_000, atr_at_reversal=1.2, max_favorable_move=3.6),
        Turnaround(price=99.8, age_days=20, volume=900_000, average_volume_at_reversal=950_000, atr_at_reversal=1.1, max_favorable_move=2.2),
        Turnaround(price=101.2, age_days=45, volume=1_100_000, average_volume_at_reversal=1_050_000, atr_at_reversal=1.3, max_favorable_move=1.8),
        Turnaround(price=110.0, age_days=10, volume=2_000_000, average_volume_at_reversal=1_200_000, atr_at_reversal=1.4, max_favorable_move=4.0),  # outside zone
        # from years earlier, when this instrument's typical volume was
        # much lower -- gets judged against ITS era's average, not today's
        Turnaround(price=85.0, age_days=1800, volume=500_000, average_volume_at_reversal=300_000, atr_at_reversal=1.0, max_favorable_move=1.0),
    ]

    tests = [
        ZoneTest(entered_zone=True, move_away_atr=1.5),
        ZoneTest(entered_zone=True, move_away_atr=0.4),
        ZoneTest(entered_zone=True, move_away_atr=2.1),
        ZoneTest(entered_zone=False, move_away_atr=0.0),
    ]

    result = evaluate_zone(
        zone=zone,
        turnarounds=turnarounds,
        tests=tests,
        target_reaction_atr=2.5,
    )

    print(f"Concentration (C):      {result.concentration:.3f}")
    print(f"Rejection rate (R):     {result.rejection_rate:.3f}")
    print(f"Reaction strength (S):  {result.reaction_strength:.3f}")
    print(f"Confidence (F):         {result.confidence:.3f}")
    print(f"Zone Strength Score:    {result.score:.2f} / 100")