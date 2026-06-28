"""
Fetches NFL player stats for Weeks 7-17 of a given season using nflreadpy.

nflreadpy (successor to nfl_data_py) provides pre-built weekly player stats
including wopr, racr, air_yards_share, target_share — no PBP derivation needed
for skill positions. PBP is still used for DST metrics and goal-line/redzone
carries which are not in the weekly stats file.
"""

import nflreadpy as nflr
import nfl_data_py as nfl  # kept only for PBP (nflreadpy load_pbp is identical)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SEASON = 2025
WEEKS = list(range(7, 18))


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------

def load_weekly(season: int = SEASON) -> pd.DataFrame:
    print(f"[fetch] Loading {season} weekly player stats via nflreadpy...")
    df = nflr.load_player_stats(seasons=season, summary_level="week").to_pandas()
    df = df[(df["season_type"] == "REG") & (df["week"].isin(WEEKS))]
    print(f"  {len(df)} player-week rows, weeks {df['week'].min()}-{df['week'].max()}")
    return df


def load_pbp(season: int = SEASON) -> pd.DataFrame:
    print(f"[fetch] Loading {season} PBP via nflreadpy (for DST/goal-line)...")
    pbp = nflr.load_pbp(seasons=season).to_pandas()
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["week"].isin(WEEKS))]
    print(f"  {len(pbp)} plays, weeks {pbp['week'].min()}-{pbp['week'].max()}")
    return pbp


def load_schedules(season: int = SEASON) -> pd.DataFrame:
    sched = nflr.load_schedules(seasons=season).to_pandas()
    return sched[(sched["game_type"] == "REG") & (sched["week"].isin(WEEKS))]


# ---------------------------------------------------------------------------
# QB metrics (Dalton Del Don)
# ---------------------------------------------------------------------------

def compute_qb_metrics(weekly: pd.DataFrame) -> pd.DataFrame:
    qb = weekly[weekly["position"] == "QB"].copy()
    if qb.empty:
        return pd.DataFrame()

    games = qb.groupby("player_id")["week"].nunique().rename("games_played")

    agg = qb.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("team", "last"),
        pass_attempts=("attempts", "sum"),
        completions=("completions", "sum"),
        passing_yards=("passing_yards", "sum"),
        passing_tds=("passing_tds", "sum"),
        interceptions=("passing_interceptions", "sum"),
        sacks=("sacks_suffered", "sum"),
        rushing_attempts=("carries", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rushing_tds", "sum"),
        fantasy_points_half_ppr=("fantasy_points", "sum"),
    ).join(games)

    agg["pass_attempts_per_game"] = agg["pass_attempts"] / agg["games_played"]
    agg["rushing_attempts_per_game"] = agg["rushing_attempts"] / agg["games_played"]
    agg["rushing_yards_per_game"] = agg["rushing_yards"] / agg["games_played"]

    agg["ypa"] = agg["passing_yards"] / agg["pass_attempts"].replace(0, np.nan)
    agg["completion_pct"] = agg["completions"] / agg["pass_attempts"].replace(0, np.nan)
    agg["td_pct"] = agg["passing_tds"] / agg["pass_attempts"].replace(0, np.nan)
    agg["int_pct"] = agg["interceptions"] / agg["pass_attempts"].replace(0, np.nan)

    agg["dropbacks"] = agg["pass_attempts"] + agg["sacks"]
    agg["fantasy_pts_per_dropback"] = (
        agg["fantasy_points_half_ppr"] / agg["dropbacks"].replace(0, np.nan)
    )
    agg["rush_fantasy_contribution"] = (
        agg["rushing_yards"] * 0.1 + agg["rushing_tds"] * 6
    ) / agg["games_played"]

    agg["position"] = "QB"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# RB metrics (Joe Bond)
# ---------------------------------------------------------------------------

def compute_rb_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    rb = weekly[weekly["position"] == "RB"].copy()
    if rb.empty:
        return pd.DataFrame()

    games = rb.groupby("player_id")["week"].nunique().rename("games_played")

    agg = rb.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("team", "last"),
        carries=("carries", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rushing_tds", "sum"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("receiving_tds", "sum"),
        fantasy_points_half_ppr=("fantasy_points", "sum"),
    ).join(games)

    agg["carries_per_game"] = agg["carries"] / agg["games_played"]
    agg["receptions_per_game"] = agg["receptions"] / agg["games_played"]
    agg["ypc"] = agg["rushing_yards"] / agg["carries"].replace(0, np.nan)

    # team share metrics — computed from weekly rows
    team_totals = rb.groupby(["team", "week"]).agg(
        team_carries=("carries", "sum"),
        team_targets=("targets", "sum"),
    ).reset_index()
    rb2 = rb.merge(team_totals, on=["team", "week"])
    rb2["carry_share_wk"] = rb2["carries"] / rb2["team_carries"].replace(0, np.nan)
    rb2["target_share_wk"] = rb2["targets"] / rb2["team_targets"].replace(0, np.nan)
    rb2["opportunities"] = rb2["carries"] + rb2["targets"]
    team_opp = rb2.groupby(["team", "week"])["opportunities"].transform("sum")
    rb2["opp_share_wk"] = rb2["opportunities"] / team_opp.replace(0, np.nan)

    shares = rb2.groupby("player_id").agg(
        carry_share=("carry_share_wk", "mean"),
        target_share_backfield=("target_share_wk", "mean"),
        opportunity_share=("opp_share_wk", "mean"),
    )
    agg = agg.join(shares)

    # goal-line carries from PBP (not in weekly stats)
    gl = pbp[
        (pbp["play_type"] == "run") &
        (pbp["yardline_100"] <= 5) &
        pbp["rusher_player_id"].notna()
    ]
    gl_carries = gl.groupby("rusher_player_id").size().rename("goal_line_carries")
    agg = agg.join(gl_carries)
    agg["goal_line_carries"] = agg["goal_line_carries"].fillna(0)

    agg["position"] = "RB"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# WR metrics (Jeff Bell)
# ---------------------------------------------------------------------------

def compute_wr_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    wr = weekly[weekly["position"] == "WR"].copy()
    if wr.empty:
        return pd.DataFrame()

    games = wr.groupby("player_id")["week"].nunique().rename("games_played")

    agg = wr.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("team", "last"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("receiving_tds", "sum"),
        yards_after_catch=("receiving_yards_after_catch", "sum"),
        # nflreadpy provides these pre-computed per week; take mean across weeks
        target_share=("target_share", "mean"),
        air_yards=("air_yards_share", "mean"),
        wopr=("wopr", "mean"),
        racr=("racr", "mean"),
        fantasy_points_half_ppr=("fantasy_points", "sum"),
    ).join(games)

    agg["targets_per_game"] = agg["targets"] / agg["games_played"]
    agg["yac_per_reception"] = agg["yards_after_catch"] / agg["receptions"].replace(0, np.nan)

    # adot from PBP (not in weekly stats)
    pass_plays = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()]
    adot = pass_plays.groupby("receiver_player_id")["air_yards"].mean().rename("adot")
    total_air = pass_plays.groupby("receiver_player_id")["air_yards"].sum().rename("total_air_yards")
    agg = agg.join(adot).join(total_air)
    agg["racr"] = agg["receiving_yards"] / agg["total_air_yards"].replace(0, np.nan)

    agg["position"] = "WR"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# TE metrics (Scott Pianowski)
# ---------------------------------------------------------------------------

def compute_te_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    te = weekly[weekly["position"] == "TE"].copy()
    if te.empty:
        return pd.DataFrame()

    games = te.groupby("player_id")["week"].nunique().rename("games_played")

    # target_share relative to all skill players (not just TEs) — use team total
    all_skill = weekly[weekly["position"].isin(["WR", "TE", "RB"])].copy()
    team_tgts = all_skill.groupby(["team", "week"])["targets"].sum().rename("team_targets").reset_index()
    te2 = te.merge(team_tgts, on=["team", "week"])
    te2["tgt_share_wk"] = te2["targets"] / te2["team_targets"].replace(0, np.nan)
    tgt_share = te2.groupby("player_id")["tgt_share_wk"].mean().rename("target_share")

    agg = te.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("team", "last"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("receiving_tds", "sum"),
        yards_after_catch=("receiving_yards_after_catch", "sum"),
        fantasy_points_half_ppr=("fantasy_points", "sum"),
    ).join(games).join(tgt_share)

    agg["receiving_yards_per_game"] = agg["receiving_yards"] / agg["games_played"]
    agg["targets_per_game"] = agg["targets"] / agg["games_played"]
    agg["yprr_proxy"] = agg["receiving_yards"] / agg["targets"].replace(0, np.nan)

    # redzone/endzone targets from PBP
    pass_plays = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()]
    rz = pass_plays[pass_plays["yardline_100"] <= 20]
    ez = pass_plays[pass_plays["yardline_100"] <= 10]

    agg = agg.join(
        rz.groupby("receiver_player_id").size().rename("redzone_targets")
    ).join(
        ez.groupby("receiver_player_id").size().rename("endzone_targets")
    )
    agg["redzone_targets"] = agg["redzone_targets"].fillna(0)
    agg["endzone_targets"] = agg["endzone_targets"].fillna(0)

    agg["position"] = "TE"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# K metrics (Justin Sablich)
# ---------------------------------------------------------------------------

def compute_k_metrics(weekly: pd.DataFrame) -> pd.DataFrame:
    k = weekly[weekly["position"] == "K"].copy()
    if k.empty:
        return pd.DataFrame()

    games = k.groupby("player_id")["week"].nunique().rename("games_played")

    agg = k.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("team", "last"),
        fg_attempts=("fg_att", "sum"),
        fg_makes=("fg_made", "sum"),
        fg_made_40_49=("fg_made_40_49", "sum"),
        fg_att_40_49=("fg_missed_40_49", "sum"),  # made + missed = attempts
        fg_made_50p=("fg_made_50_59", "sum"),
        fg_att_50p=("fg_missed_50_59", "sum"),
        fg_made_60p=("fg_made_60_", "sum"),
        fg_att_60p=("fg_missed_60_", "sum"),
        xp_attempts=("pat_att", "sum"),
        xp_makes=("pat_made", "sum"),
    ).join(games)

    # combine 50+ tiers
    agg["fg_made_50p"] = agg["fg_made_50p"] + agg["fg_made_60p"]
    agg["fg_att_50p"] = agg["fg_att_50p"] + agg["fg_att_60p"]
    # att = made + missed
    agg["fg_att_40_49"] = agg["fg_made_40_49"] + agg["fg_att_40_49"]
    agg["fg_att_50p"] = agg["fg_made_50p"] + agg["fg_att_50p"]

    agg["fg_pct"] = agg["fg_makes"] / agg["fg_attempts"].replace(0, np.nan)
    agg["fg_pct_40_49"] = agg["fg_made_40_49"] / agg["fg_att_40_49"].replace(0, np.nan)
    agg["fg_pct_50p"] = agg["fg_made_50p"] / agg["fg_att_50p"].replace(0, np.nan)
    agg["fg_attempts_per_game"] = agg["fg_attempts"] / agg["games_played"]

    agg["position"] = "K"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# DST metrics (Tommy Garrett) — still derived from PBP
# ---------------------------------------------------------------------------

def compute_dst_metrics(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    defense = pbp[pbp["defteam"].notna()].copy()
    team_games = defense.groupby("defteam")["week"].nunique().rename("games_played")

    sacks = defense[defense["sack"] == 1].groupby("defteam").size().rename("sacks")
    ints = defense[defense["interception"] == 1].groupby("defteam").size().rename("interceptions")
    fum = defense[defense["fumble_lost"] == 1].groupby("defteam").size().rename("fumbles_recovered")

    dst = pd.DataFrame({"sacks": sacks, "interceptions": ints,
                        "fumbles_recovered": fum}).join(team_games)
    dst["turnovers"] = dst["interceptions"] + dst["fumbles_recovered"]
    dst["sacks_per_game"] = dst["sacks"] / dst["games_played"]
    dst["turnovers_per_game"] = dst["turnovers"] / dst["games_played"]

    total_pass_attempts = defense[defense["pass_attempt"] == 1].groupby("defteam")["pass_attempt"].sum()
    total_plays = defense.groupby("defteam")["play_id"].count()
    dst["sack_rate"] = dst["sacks"] / total_pass_attempts.replace(0, np.nan)
    dst["turnover_rate"] = dst["turnovers"] / total_plays.replace(0, np.nan)

    home_pts = schedules[["away_team", "week", "home_score"]].rename(
        columns={"away_team": "defteam", "home_score": "pts_allowed"})
    away_pts = schedules[["home_team", "week", "away_score"]].rename(
        columns={"home_team": "defteam", "away_score": "pts_allowed"})
    pts_pg = (
        pd.concat([home_pts, away_pts])
        .groupby("defteam")["pts_allowed"].mean()
        .rename("points_allowed_per_game")
    )
    dst = dst.join(pts_pg)

    opp_epa = (
        defense[defense["passer_player_id"].notna()]
        .groupby("defteam")["epa"].mean()
        .rename("opp_epa_per_dropback_allowed")
    )
    dst = dst.join(opp_epa)

    dst["position"] = "DST"
    dst.index.name = "team"
    dst["player_name"] = dst.index + " DST"
    dst["player_id"] = dst.index
    return dst.reset_index()


# ---------------------------------------------------------------------------
# Master fetch
# ---------------------------------------------------------------------------

def fetch_all(season: int = SEASON) -> dict:
    weekly = load_weekly(season)
    pbp = load_pbp(season)
    schedules = load_schedules(season)

    return {
        "QB":  compute_qb_metrics(weekly),
        "RB":  compute_rb_metrics(weekly, pbp),
        "WR":  compute_wr_metrics(weekly, pbp),
        "TE":  compute_te_metrics(weekly, pbp),
        "K":   compute_k_metrics(weekly),
        "DST": compute_dst_metrics(pbp, schedules),
        "_raw_weekly": weekly,
        "_raw_pbp": pbp,
        "_schedules": schedules,
    }


if __name__ == "__main__":
    import os
    data = fetch_all()
    os.makedirs("data", exist_ok=True)
    for pos, df in data.items():
        if not pos.startswith("_"):
            path = f"data/{pos}_wk7_17_{SEASON}.csv"
            df.to_csv(path, index=False)
            print(f"[saved] {path} — {len(df)} players")
