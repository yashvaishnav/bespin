"""
Situation delta layer (Layer 3).

Applies forward-looking adjustments to signal scores for:
  1. Rookies entering the league (2026 NFL Draft)
  2. Players in new situations (FA signings, trades, new OC, new team)
  3. Projected breakout candidates based on depth chart opportunity

Rookies have zero Wks 7-17 historical signal — this layer gives them an
estimated signal score derived from:
  - Draft capital (round + pick number)  →  strongest single predictor
  - Landing situation quality            →  team offense + depth chart role
  - Position-specific college translation rates

New-situation veterans get a multiplier adjustment to their base signal score.
"""

import pandas as pd
import numpy as np
import os

DELTA_PATH = os.path.join(os.path.dirname(__file__), "../data/situation_delta_2026.csv")

# Draft capital signal: maps (round, rough pick range) → baseline fantasy signal
# Based on historical correlation between draft position and fantasy production Yr1
DRAFT_CAPITAL_SIGNAL = {
    1: {(1, 5): 62, (6, 10): 55, (11, 16): 48, (17, 32): 42},
    2: {(33, 48): 38, (49, 64): 34},
    3: {(65, 96): 28},
    4: {(97, 128): 22},
    5: {(129, 176): 18},
    6: {(177, 220): 14},
    7: {(221, 260): 11},
}

# Position-specific draft capital multipliers:
# QB drafted early often doesn't start Yr1; RB pick 1-5 is rare but translates well
POSITION_CAPITAL_MULTIPLIER = {
    "QB": 0.60,   # most rookie QBs underperform their draft slot in fantasy yr1
    "RB": 1.15,   # RBs translate most directly from draft capital to Yr1 usage
    "WR": 0.90,   # WRs take time to develop routes but good situations help
    "TE": 0.75,   # TEs historically slowest to develop
    "K":  1.00,
    "DST": 1.00,
}

# Situation score → signal multiplier mapping
# situation_score in [0,100] from the CSV
def situation_to_multiplier(situation_score: float) -> float:
    """Converts a 0-100 situation score to a signal multiplier (0.8 to 1.3)."""
    return 0.80 + (situation_score / 100.0) * 0.50


def _draft_capital_base(draft_round: int, draft_pick: int) -> float:
    """Return baseline signal from draft capital alone."""
    rnd_map = DRAFT_CAPITAL_SIGNAL.get(int(draft_round), {(1, 999): 10})
    for (lo, hi), val in rnd_map.items():
        if lo <= draft_pick <= hi:
            return float(val)
    # fallback: interpolate by round
    return max(10.0, 65.0 - (int(draft_round) - 1) * 8.0)


def load_situation_deltas() -> pd.DataFrame:
    if not os.path.exists(DELTA_PATH):
        print(f"[situation] No delta file found at {DELTA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DELTA_PATH)
    return df


def build_rookie_rows(deltas: pd.DataFrame) -> pd.DataFrame:
    """
    For each rookie in the delta file, construct a synthetic player row
    with an estimated signal_score to insert into the draft board.
    """
    rows = []
    for _, r in deltas.iterrows():
        pick = r.get("draft_pick", 0) or 0
        rnd = r.get("draft_round", 7) or 7
        if pick == 0:
            # estimate pick from round midpoint
            pick_midpoints = {1: 20, 2: 48, 3: 80, 4: 112, 5: 150, 6: 185, 7: 225}
            pick = pick_midpoints.get(int(rnd), 200)

        capital_base = _draft_capital_base(rnd, pick)
        pos_mult = POSITION_CAPITAL_MULTIPLIER.get(r["position"], 1.0)
        sit_mult = situation_to_multiplier(float(r.get("situation_score", 50)))

        signal = capital_base * pos_mult * sit_mult

        # team win total adjustment: scale signal by offensive context
        win_total = float(r.get("team_win_total", 8.5))
        # normalized: 4.5 wins → 0.85x, 11.5 wins → 1.10x
        win_mult = 0.85 + (win_total - 4.5) / (11.5 - 4.5) * 0.25
        signal *= win_mult

        rows.append({
            "player_id": f"ROOKIE_{r['player_name'].replace(' ', '_')}",
            "player_name": r["player_name"],
            "team": r["nfl_team"],
            "position": r["position"],
            "games_played": 0,
            "signal_score": round(min(signal, 95.0), 2),
            "signal_rank": 999,
            "draft_round": rnd,
            "draft_pick": pick,
            "situation_score": r.get("situation_score", 50),
            "depth_chart_role": r.get("depth_chart_role", "unknown"),
            "opportunity_ceiling": r.get("opportunity_ceiling", "low"),
            "is_rookie": True,
            "notes": r.get("notes", ""),
        })
    return pd.DataFrame(rows)


def apply_veteran_deltas(scored: dict, deltas: pd.DataFrame) -> dict:
    """
    Apply situation multipliers to veterans in new situations.
    Currently a placeholder — to be expanded with FA/trade data.
    """
    # Filter for veterans (non-rookies) in the delta file
    vets = deltas[deltas["draft_round"].isna() | (deltas["draft_round"] == 0)].copy()
    if vets.empty:
        return scored

    for pos, df in scored.items():
        for _, vet in vets[vets["position"] == pos].iterrows():
            mask = df["player_name"].str.lower().str.contains(
                vet["player_name"].lower(), na=False
            )
            if mask.any():
                mult = situation_to_multiplier(float(vet.get("situation_score", 50)))
                scored[pos].loc[mask, "signal_score"] *= mult
                scored[pos].loc[mask, "signal_score"] = (
                    scored[pos].loc[mask, "signal_score"].clip(upper=99.0).round(2)
                )
    return scored


def inject_rookies(board: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    """
    Add rookie rows to the full draft board.
    Rookies are inserted with their estimated signal_score and their VBD
    is computed relative to the existing replacement level at their position.
    """
    rookies = build_rookie_rows(deltas)
    if rookies.empty:
        return board

    # compute VBD for rookies using current board's replacement signal per position
    rep_signals = (
        board.groupby("position")["replacement_signal"].first().to_dict()
    )
    scarcity = {
        "QB": 0.90, "RB": 1.10, "WR": 1.00, "TE": 1.05, "K": 0.70, "DST": 0.75
    }

    rookies["replacement_signal"] = rookies["position"].map(rep_signals).fillna(0)
    rookies["vbd_raw"] = rookies["signal_score"] - rookies["replacement_signal"]
    rookies["vbd"] = (
        rookies.apply(
            lambda r: r["vbd_raw"] * scarcity.get(r["position"], 1.0), axis=1
        ).round(2)
    )
    rookies["tier"] = rookies["vbd"].apply(
        lambda v: 1 if v > 20 else (2 if v > 14 else (3 if v > 8 else (4 if v > 3 else (5 if v > 0 else 7))))
    )

    combined = pd.concat([board, rookies], ignore_index=True, sort=False)
    combined = combined.sort_values("vbd", ascending=False).reset_index(drop=True)
    combined["overall_rank"] = combined.index + 1

    # recompute positional ranks
    combined["positional_rank"] = combined.groupby("position").cumcount() + 1

    return combined
