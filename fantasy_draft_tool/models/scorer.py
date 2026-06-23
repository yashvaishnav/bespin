"""
Position-specific player scoring model.

For each position, applies the top metrics identified by the historically
most accurate FantasyPros ranker at that position. Each metric is:
  1. Normalized to a 0-1 percentile within the position pool
  2. Weighted by the metric's known predictive importance
  3. Summed to a composite SIGNAL score (0-100)

The SIGNAL score is the input to the VBD engine — it is NOT a fantasy point
projection. It is a relative quality ranking signal.
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def percentile_rank(series: pd.Series) -> pd.Series:
    """Convert a series to 0-1 percentile ranks, NaN-safe."""
    arr = series.values.astype(float)
    mask = ~np.isnan(arr)
    out = np.full_like(arr, np.nan)
    if mask.sum() > 1:
        out[mask] = (rankdata(arr[mask]) - 1) / (mask.sum() - 1)
    return pd.Series(out, index=series.index)


def score_position(df: pd.DataFrame, metric_weights: dict, min_games: int = 4) -> pd.DataFrame:
    """
    Apply metric_weights to df, return df with added 'signal_score' column.
    metric_weights: {column_name: weight}
      Positive weight  → higher is better
      Negative weight  → lower is better (e.g. int_pct, points_allowed)
    """
    df = df.copy()
    if "games_played" in df.columns:
        df = df[df["games_played"] >= min_games].copy()

    total_abs_weight = sum(abs(w) for w in metric_weights.values())
    composite = pd.Series(0.0, index=df.index)

    for col, weight in metric_weights.items():
        if col not in df.columns:
            print(f"  [warn] metric '{col}' not found — skipping")
            continue
        pct = percentile_rank(df[col])
        if weight < 0:
            # invert: lower raw value → higher contribution
            pct = 1.0 - pct
        composite += pct * (abs(weight) / total_abs_weight)

    df["signal_score"] = (composite * 100).round(2)
    df["signal_rank"] = df["signal_score"].rank(ascending=False, method="min").fillna(0).astype(int)
    return df.sort_values("signal_score", ascending=False)


# ---------------------------------------------------------------------------
# Per-position metric weight configs
# Weights reflect relative predictive importance for NEXT season performance
# in half-PPR 12-team leagues, based on published research and analyst focus.
# ---------------------------------------------------------------------------

QB_WEIGHTS = {
    # Dual-threat is the #1 separator (Del Don's known emphasis)
    "rushing_attempts_per_game":        0.18,
    "rush_fantasy_contribution":        0.12,
    # Volume baseline
    "pass_attempts_per_game":           0.15,
    # Efficiency
    "fantasy_pts_per_dropback":         0.18,
    "ypa":                              0.10,
    # TD upside
    "td_pct":                           0.12,
    # Protecting the ball
    "int_pct":                         -0.05,  # negative: lower is better
    # Rushing red zone
    "rushing_yards_per_game":           0.10,
}

RB_WEIGHTS = {
    # Opportunity share is the single strongest RB predictor
    "opportunity_share":                0.28,
    # Half-PPR specific: pass-catching value
    "target_share_backfield":           0.20,
    "receptions_per_game":              0.10,
    # Role clarity
    "carry_share":                      0.15,
    # Efficiency
    "ypc":                              0.12,
    # TD equity
    "goal_line_carries":                0.10,
    # Volume floor
    "carries_per_game":                 0.05,
}

WR_WEIGHTS = {
    # Target share is foundational
    "target_share":                     0.22,
    # WOPR: best single WR metric (combines target share + air yards)
    "wopr":                             0.20,
    # Air yards share / downfield usage
    "air_yards":                        0.15,   # weekly air_yards_share column
    # Efficiency above expectation
    "racr":                             0.13,
    # Role type / depth of target
    "adot":                             0.10,
    # YAC: separation + scheme fit
    "yac_per_reception":                0.10,
    # Volume floor
    "targets_per_game":                 0.10,
}

TE_WEIGHTS = {
    # Route participation is the gating metric — blocker or receiver?
    # Using target share as proxy since route % requires PFF
    "target_share":                     0.28,
    # TD equity — most of TE value comes from TDs
    "endzone_targets":                  0.22,
    "redzone_targets":                  0.15,
    # Efficiency proxy
    "yprr_proxy":                       0.15,
    # Floor
    "receiving_yards_per_game":         0.12,
    # Volume
    "targets_per_game":                 0.08,
}

K_WEIGHTS = {
    # Volume is king for kickers
    "fg_attempts_per_game":             0.35,
    # High-value kicks
    "fg_pct_40_49":                     0.25,
    "fg_pct_50p":                       0.15,
    # Floor
    "xp_attempts":                      0.15,
    # Overall accuracy baseline
    "fg_pct":                           0.10,
}

DST_WEIGHTS = {
    # Sack rate is most stable year-over-year DST metric
    "sack_rate":                        0.25,
    "sacks_per_game":                   0.10,
    # Turnover rate
    "turnover_rate":                    0.22,
    "turnovers_per_game":               0.08,
    # Points allowed
    "points_allowed_per_game":         -0.20,  # negative: lower is better
    # Opponent quality (lower EPA allowed = better defense)
    "opp_epa_per_dropback_allowed":    -0.15,  # negative: lower is better
}

POSITION_CONFIGS = {
    "QB":  QB_WEIGHTS,
    "RB":  RB_WEIGHTS,
    "WR":  WR_WEIGHTS,
    "TE":  TE_WEIGHTS,
    "K":   K_WEIGHTS,
    "DST": DST_WEIGHTS,
}


def score_all_positions(data: dict) -> dict:
    """
    Takes the dict returned by fetch_all(), scores each position.
    Returns dict of {position: scored_DataFrame}
    """
    scored = {}
    for pos, weights in POSITION_CONFIGS.items():
        if pos not in data:
            continue
        df = data[pos]
        print(f"[score] Scoring {pos} — {len(df)} players...")
        scored[pos] = score_position(df, weights)
    return scored
