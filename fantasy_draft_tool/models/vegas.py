"""
Vegas lines integration — Layer 2 team context multiplier.

Applies 2026 implied team totals and win totals to adjust player signal scores.
Players on high-scoring offenses get a multiplier boost; low-scoring offenses
get a discount. This is applied AFTER base signal scoring, BEFORE VBD.

Data sources (manual entry / scrape targets):
  - Implied team totals: derived from O/U and spread for each game
  - Season win totals: from major sportsbooks (DraftKings, FanDuel, BetMGM avg)

Until live data is available, a template CSV is provided for manual population.
"""

import pandas as pd
import numpy as np
import os


VEGAS_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/vegas_lines_2026.csv")

# Multiplier range: team on best offense gets 1.12x, worst gets 0.88x
VEGAS_BOOST_MAX = 1.12
VEGAS_BOOST_MIN = 0.88

# Position sensitivity to team scoring environment
# (how much does a high-scoring offense matter for each position?)
POSITION_VEGAS_SENSITIVITY = {
    "QB":  1.0,    # directly tied to team scoring
    "RB":  0.7,    # partially scheme-dependent
    "WR":  0.9,    # strongly tied to pass volume
    "TE":  0.8,    # tied to pass volume, scheme-dependent
    "K":   0.6,    # good offense generates FG ops, but stalls matter more
    "DST": -0.5,   # negative: playing against high-scoring offense hurts DST
}


def load_vegas_lines() -> pd.DataFrame:
    """
    Load 2026 Vegas implied team totals.
    Expected columns: team, implied_points_per_game, season_win_total, games_total (16-17)
    """
    if not os.path.exists(VEGAS_DATA_PATH):
        print(f"[vegas] No Vegas data found at {VEGAS_DATA_PATH} — generating template...")
        _generate_template()
        print(f"[vegas] Template created. Populate {VEGAS_DATA_PATH} with real lines before running.")
        return pd.DataFrame()

    df = pd.read_csv(VEGAS_DATA_PATH)
    required = ["team", "implied_points_per_game", "season_win_total"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Vegas CSV missing columns: {missing}")
    return df


def _generate_template():
    """Create an empty template CSV for manual population."""
    nfl_teams = [
        "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE",
        "DAL","DEN","DET","GB","HOU","IND","JAX","KC",
        "LAC","LAR","LV","MIA","MIN","NE","NO","NYG",
        "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
    ]
    template = pd.DataFrame({
        "team": nfl_teams,
        "implied_points_per_game": [None] * 32,
        "season_win_total": [None] * 32,
        "spread_tendency": [None] * 32,   # avg spread; negative = expected to win
        "notes": [""] * 32,
    })
    os.makedirs(os.path.dirname(VEGAS_DATA_PATH), exist_ok=True)
    template.to_csv(VEGAS_DATA_PATH, index=False)


def apply_vegas_multiplier(scored: dict, vegas: pd.DataFrame) -> dict:
    """
    Adjusts signal_score in each position DataFrame using team implied totals.
    Returns updated scored dict.
    """
    if vegas.empty:
        print("[vegas] Skipping Vegas adjustment — no data loaded.")
        return scored

    # normalize implied points to 0-1 scale then map to multiplier range
    pts = vegas.set_index("team")["implied_points_per_game"]
    pts_norm = (pts - pts.min()) / (pts.max() - pts.min())
    # maps 0 → VEGAS_BOOST_MIN, 1 → VEGAS_BOOST_MAX
    multiplier_map = pts_norm * (VEGAS_BOOST_MAX - VEGAS_BOOST_MIN) + VEGAS_BOOST_MIN

    adjusted = {}
    for pos, df in scored.items():
        d = df.copy()
        sensitivity = POSITION_VEGAS_SENSITIVITY.get(pos, 0.5)

        team_col = "team" if "team" in d.columns else None
        if team_col is None:
            adjusted[pos] = d
            continue

        def get_multiplier(team):
            if team not in multiplier_map.index:
                return 1.0
            raw = multiplier_map[team]
            # blend toward 1.0 based on sensitivity
            return 1.0 + (raw - 1.0) * sensitivity

        d["vegas_multiplier"] = d[team_col].map(get_multiplier).fillna(1.0)
        d["signal_score_pre_vegas"] = d["signal_score"]
        d["signal_score"] = (d["signal_score"] * d["vegas_multiplier"]).round(2)
        adjusted[pos] = d

    return adjusted


def compute_playoff_schedule_weight(
    schedules: pd.DataFrame,
    vegas: pd.DataFrame,
    playoff_weeks: list = [14, 15, 16, 17],
) -> pd.DataFrame:
    """
    Compute per-team average opponent implied points during fantasy playoff weeks.
    Lower = easier playoff schedule = schedule advantage.
    Returns DataFrame with columns: team, playoff_opp_implied_pts_avg, schedule_rank
    """
    if vegas.empty or schedules.empty:
        return pd.DataFrame()

    playoff_sched = schedules[schedules["week"].isin(playoff_weeks)].copy()
    pts_map = vegas.set_index("team")["implied_points_per_game"].to_dict()

    rows = []
    for _, game in playoff_sched.iterrows():
        home, away = game["home_team"], game["away_team"]
        rows.append({"team": home, "opp": away, "opp_implied": pts_map.get(away, np.nan)})
        rows.append({"team": away, "opp": home, "opp_implied": pts_map.get(home, np.nan)})

    df = pd.DataFrame(rows)
    result = df.groupby("team")["opp_implied"].mean().rename("playoff_opp_implied_pts_avg").reset_index()
    result["schedule_rank"] = result["playoff_opp_implied_pts_avg"].rank(ascending=True).astype(int)
    return result.sort_values("schedule_rank")
