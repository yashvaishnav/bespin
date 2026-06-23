"""
Value Based Drafting (VBD) engine.

Core idea: a player's value is not their raw signal score — it's their signal
score ABOVE the baseline replacement level at their position. Replacement level
is defined as the last starter a 12-team league would be expected to start.

Standard 12-team starter counts (used for replacement level cutoffs):
  QB:  12 starters  → replacement = QB13
  RB:  24 starters  → replacement = RB25 (2 per team)
  WR:  24 starters  → replacement = WR25 (2 per team)
  TE:  12 starters  → replacement = TE13
  K:   12 starters  → replacement = K13
  DST: 12 starters  → replacement = DST13
  FLEX (RB/WR/TE): 12 additional → adjusts RB/WR/TE replacement levels

Half-PPR 12-team typical roster: QB, 2RB, 2WR, 1TE, 1FLEX, 1K, 1DST
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Replacement level thresholds
# FLEX pushes RB/WR/TE replacement levels down by ~4 spots each
# ---------------------------------------------------------------------------

REPLACEMENT_LEVELS = {
    "QB":  12,
    "RB":  28,   # 24 starters + ~4 FLEX RBs
    "WR":  28,   # 24 starters + ~4 FLEX WRs
    "TE":  14,   # 12 starters + ~2 FLEX TEs
    # Fix 2: K replacement level dropped to 8 (rounds 14-16 in practice).
    # In a 12-team league kickers go in the final 2 rounds; treating K13 as
    # replacement inflated VBD for top kickers above RB/WR/TE streamers.
    "K":    8,
    "DST": 12,
}

# Scarcity multipliers: positions where value drops off sharply after top tier
# get a slight upward adjustment to reflect real draft-room behavior
SCARCITY_MULTIPLIER = {
    "QB":  0.90,   # deep position; slight discount
    "RB":  1.10,   # scarce at top; premium
    "WR":  1.00,   # balanced
    "TE":  1.05,   # scarce at Tier 1; slight premium
    # Fix 2 cont: halved K multiplier — kickers should never outrank skill players
    "K":   0.45,
    "DST": 0.75,   # streamable; heavy discount
}


def compute_vbd(scored: dict) -> pd.DataFrame:
    """
    Takes dict of {position: scored_DataFrame} with 'signal_score' column.
    Returns unified draft board with VBD scores and positional ranks.
    """
    frames = []

    for pos, df in scored.items():
        d = df.copy()
        d = d.sort_values("signal_score", ascending=False).reset_index(drop=True)
        d["positional_rank"] = d.index + 1

        # replacement level signal
        rep_rank = REPLACEMENT_LEVELS.get(pos, 12)
        if len(d) >= rep_rank:
            replacement_signal = d.loc[rep_rank - 1, "signal_score"]
        else:
            replacement_signal = d["signal_score"].min()

        d["replacement_signal"] = replacement_signal
        d["vbd_raw"] = d["signal_score"] - replacement_signal
        d["vbd"] = (d["vbd_raw"] * SCARCITY_MULTIPLIER.get(pos, 1.0)).round(2)

        frames.append(d)

    board = pd.concat(frames, ignore_index=True)
    board = board.sort_values("vbd", ascending=False).reset_index(drop=True)
    board["overall_rank"] = board.index + 1

    # tier assignment: natural breaks in VBD score
    board["tier"] = _assign_tiers(board)

    return board


def _assign_tiers(board: pd.DataFrame, n_tiers: int = 8) -> pd.Series:
    """
    Assign tiers using VBD score percentile breaks.
    Tier 1 = elite, Tier 8 = streaming/late-round.
    """
    vbd = board["vbd"].copy()
    # only positive VBD players get tiers 1-6; negative VBD = tiers 7-8
    positive_mask = vbd > 0
    tiers = pd.Series(8, index=board.index)

    if positive_mask.sum() > 0:
        pos_vbd = vbd[positive_mask]
        rank_pct = pos_vbd.rank(pct=True)
        # simple threshold assignment — no pd.cut label collision
        t = pd.Series(6, index=pos_vbd.index)
        t[rank_pct > 0.83] = 1
        t[(rank_pct > 0.67) & (rank_pct <= 0.83)] = 2
        t[(rank_pct > 0.50) & (rank_pct <= 0.67)] = 3
        t[(rank_pct > 0.33) & (rank_pct <= 0.50)] = 4
        t[(rank_pct > 0.17) & (rank_pct <= 0.33)] = 5
        tiers[positive_mask] = t

    # slightly negative VBD = tier 7 (handcuff/streamer)
    slight_neg = (vbd <= 0) & (vbd > vbd.quantile(0.15))
    tiers[slight_neg] = 7

    return tiers


def get_positional_tiers(board: pd.DataFrame, position: str) -> pd.DataFrame:
    """Return the draft board filtered to one position with tier breakdown."""
    return board[board["position"] == position][
        ["overall_rank", "positional_rank", "player_name", "team",
         "signal_score", "vbd", "tier"]
    ].copy()
