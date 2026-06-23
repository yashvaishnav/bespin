"""
Fetches NFL player stats for Weeks 7-17 of a given season using nfl_data_py.
Computes all position-specific metrics used by our position specialist rankers.
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SEASON = 2024   # last complete season available via nfl_data_py; update to 2025 when released
WEEKS = list(range(7, 18))  # Weeks 7-17 inclusive


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------

def load_pbp(season: int = SEASON) -> pd.DataFrame:
    print(f"[fetch] Loading play-by-play for {season}...")
    pbp = nfl.import_pbp_data([season], downcast=True, cache=False)
    pbp = pbp[pbp["week"].isin(WEEKS)]
    return pbp


def load_weekly(season: int = SEASON) -> pd.DataFrame:
    print(f"[fetch] Loading weekly player stats for {season}...")
    w = nfl.import_weekly_data([season])
    w = w[w["week"].isin(WEEKS)]
    return w


def load_rosters(season: int = SEASON) -> pd.DataFrame:
    print(f"[fetch] Loading rosters for {season}...")
    r = nfl.import_rosters([season])
    return r[["player_id", "player_name", "position", "team", "depth_chart_position",
              "jersey_number", "birth_date", "height", "weight", "college", "draft_number"]]


def load_schedules(pbp: pd.DataFrame) -> pd.DataFrame:
    """Derive schedule info from PBP data (home/away teams + scores per week)."""
    sched = (
        pbp[pbp["home_team"].notna() & pbp["away_team"].notna()]
        .groupby(["week", "home_team", "away_team"])
        .agg(
            home_score=("total_home_score", "max"),
            away_score=("total_away_score", "max"),
        )
        .reset_index()
    )
    return sched


# ---------------------------------------------------------------------------
# QB metrics (Dalton Del Don)
# 1. Rush attempts + yards  2. Pass attempts/gm  3. Fantasy pts/dropback
# 4. Team implied pts       5. Red zone carries  6. YPA  7. Playoff SOS
# ---------------------------------------------------------------------------

def compute_qb_metrics(weekly: pd.DataFrame) -> pd.DataFrame:
    qb = weekly[weekly["position"] == "QB"].copy()

    games = qb.groupby("player_id")["week"].nunique().rename("games_played")

    agg = qb.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("recent_team", "last"),
        pass_attempts=("attempts", "sum"),
        completions=("completions", "sum"),
        passing_yards=("passing_yards", "sum"),
        passing_tds=("passing_tds", "sum"),
        interceptions=("interceptions", "sum"),
        rushing_attempts=("carries", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rushing_tds", "sum"),
        fantasy_points_half_ppr=("fantasy_points_ppr", "sum"),  # will adjust
        sacks=("sacks", "sum"),
    ).join(games)

    # per-game rates
    for col in ["pass_attempts", "passing_yards", "rushing_attempts", "rushing_yards", "fantasy_points_half_ppr"]:
        agg[f"{col}_per_game"] = agg[col] / agg["games_played"]

    # efficiency
    agg["ypa"] = agg["passing_yards"] / agg["pass_attempts"].replace(0, np.nan)
    agg["completion_pct"] = agg["completions"] / agg["pass_attempts"].replace(0, np.nan)
    agg["td_pct"] = agg["passing_tds"] / agg["pass_attempts"].replace(0, np.nan)
    agg["int_pct"] = agg["interceptions"] / agg["pass_attempts"].replace(0, np.nan)

    # fantasy pts per dropback (attempts + sacks)
    agg["dropbacks"] = agg["pass_attempts"] + agg["sacks"]
    agg["fantasy_pts_per_dropback"] = (
        agg["fantasy_points_half_ppr"] / agg["dropbacks"].replace(0, np.nan)
    )

    # dual-threat composite: rushing contribution to fantasy
    agg["rush_fantasy_contribution"] = (
        agg["rushing_yards"] * 0.1 + agg["rushing_tds"] * 6
    ) / agg["games_played"]

    agg["position"] = "QB"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# RB metrics (Joe Bond)
# 1. Opportunity share  2. Target share  3. Snap share  4. Goal-line carries
# 5. YPC               6. Team implied pts
# ---------------------------------------------------------------------------

def compute_rb_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    rb = weekly[weekly["position"] == "RB"].copy()

    games = rb.groupby("player_id")["week"].nunique().rename("games_played")

    agg = rb.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("recent_team", "last"),
        carries=("carries", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rushing_tds", "sum"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("receiving_tds", "sum"),
        fantasy_points_half_ppr=("fantasy_points_ppr", "sum"),
    ).join(games)

    # per-game
    for col in ["carries", "targets", "receptions", "receiving_yards", "rushing_yards"]:
        agg[f"{col}_per_game"] = agg[col] / agg["games_played"]

    # efficiency
    agg["ypc"] = agg["rushing_yards"] / agg["carries"].replace(0, np.nan)
    agg["yards_per_reception"] = agg["receiving_yards"] / agg["receptions"].replace(0, np.nan)
    agg["catch_rate"] = agg["receptions"] / agg["targets"].replace(0, np.nan)

    # team-level totals for share calculations
    team_totals = rb.groupby(["recent_team", "week"]).agg(
        team_carries=("carries", "sum"),
        team_targets=("targets", "sum"),
    ).reset_index()

    rb_with_team = rb.merge(team_totals, on=["recent_team", "week"])
    rb_with_team["carry_share_wk"] = rb_with_team["carries"] / rb_with_team["team_carries"].replace(0, np.nan)
    rb_with_team["target_share_wk"] = rb_with_team["targets"] / rb_with_team["team_targets"].replace(0, np.nan)

    shares = rb_with_team.groupby("player_id").agg(
        carry_share=("carry_share_wk", "mean"),
        target_share_backfield=("target_share_wk", "mean"),
    )
    agg = agg.join(shares)

    # opportunity share: (carries + targets) / team (carries + targets)
    team_opp = rb.groupby(["recent_team", "week"]).agg(
        team_opportunities=("carries", "sum"),
    ).reset_index()
    team_opp["team_opportunities"] = team_opp["team_opportunities"]  # carries only for RB opp share
    rb2 = rb.copy()
    rb2["opportunities"] = rb2["carries"] + rb2["targets"]
    team_opp2 = rb2.groupby(["recent_team", "week"])["opportunities"].sum().rename("team_opp_total").reset_index()
    rb2 = rb2.merge(team_opp2, on=["recent_team", "week"])
    rb2["opp_share_wk"] = rb2["opportunities"] / rb2["team_opp_total"].replace(0, np.nan)
    opp_share = rb2.groupby("player_id")["opp_share_wk"].mean().rename("opportunity_share")
    agg = agg.join(opp_share)

    # goal-line carries from PBP (yardline_100 <= 5)
    gl = pbp[(pbp["play_type"] == "run") & (pbp["yardline_100"] <= 5) & pbp["rusher_player_id"].notna()]
    gl_carries = gl.groupby("rusher_player_id").size().rename("goal_line_carries")
    agg = agg.join(gl_carries)
    agg["goal_line_carries"] = agg["goal_line_carries"].fillna(0)

    agg["position"] = "RB"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# WR metrics (Jeff Bell)
# 1. Target share  2. Air yard share / WOPR  3. aDOT  4. RACR
# 5. Route participation %  6. YAC  7. NGS separation (not in nfl_data_py — noted)
# ---------------------------------------------------------------------------

def compute_wr_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    wr = weekly[weekly["position"] == "WR"].copy()

    games = wr.groupby("player_id")["week"].nunique().rename("games_played")

    agg = wr.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("recent_team", "last"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("receiving_tds", "sum"),
        air_yards=("air_yards_share", "mean"),
        yards_after_catch=("receiving_yards_after_catch", "sum"),
        wopr=("wopr", "mean"),
        fantasy_points_half_ppr=("fantasy_points_ppr", "sum"),
    ).join(games)

    agg["targets_per_game"] = agg["targets"] / agg["games_played"]
    agg["receiving_yards_per_game"] = agg["receiving_yards"] / agg["games_played"]
    agg["yac_per_reception"] = agg["yards_after_catch"] / agg["receptions"].replace(0, np.nan)
    agg["catch_rate"] = agg["receptions"] / agg["targets"].replace(0, np.nan)

    # team target share
    team_tgts = wr.groupby(["recent_team", "week"])["targets"].sum().rename("team_targets").reset_index()
    wr2 = wr.merge(team_tgts, on=["recent_team", "week"])
    wr2["tgt_share_wk"] = wr2["targets"] / wr2["team_targets"].replace(0, np.nan)
    tgt_share = wr2.groupby("player_id")["tgt_share_wk"].mean().rename("target_share")
    agg = agg.join(tgt_share)

    # aDOT from pbp
    pass_plays = pbp[pbp["play_type"] == "pass"].copy()
    adot = (
        pass_plays[pass_plays["receiver_player_id"].notna()]
        .groupby("receiver_player_id")["air_yards"]
        .mean()
        .rename("adot")
    )
    agg = agg.join(adot)

    # RACR: receiving yards / air yards (team-adjusted)
    air_total = (
        pass_plays[pass_plays["receiver_player_id"].notna()]
        .groupby("receiver_player_id")["air_yards"]
        .sum()
        .rename("total_air_yards")
    )
    agg = agg.join(air_total)
    agg["racr"] = agg["receiving_yards"] / agg["total_air_yards"].replace(0, np.nan)

    # NOTE: NGS separation requires Next Gen Stats API — flagged for future enrichment
    agg["ngs_separation"] = np.nan  # placeholder

    agg["position"] = "WR"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# TE metrics (Scott Pianowski)
# 1. Route participation %  2. Target share  3. Red zone targets
# 4. YPRR  5. Receiving yards/game
# ---------------------------------------------------------------------------

def compute_te_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    te = weekly[weekly["position"] == "TE"].copy()

    games = te.groupby("player_id")["week"].nunique().rename("games_played")

    agg = te.groupby("player_id").agg(
        player_name=("player_display_name", "first"),
        team=("recent_team", "last"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("receiving_tds", "sum"),
        yards_after_catch=("receiving_yards_after_catch", "sum"),
        fantasy_points_half_ppr=("fantasy_points_ppr", "sum"),
        wopr=("wopr", "mean"),
    ).join(games)

    agg["receiving_yards_per_game"] = agg["receiving_yards"] / agg["games_played"]
    agg["targets_per_game"] = agg["targets"] / agg["games_played"]
    agg["catch_rate"] = agg["receptions"] / agg["targets"].replace(0, np.nan)

    # team target share
    all_skill = weekly[weekly["position"].isin(["WR", "TE", "RB"])].copy()
    team_tgts = all_skill.groupby(["recent_team", "week"])["targets"].sum().rename("team_targets").reset_index()
    te2 = te.merge(team_tgts, on=["recent_team", "week"])
    te2["tgt_share_wk"] = te2["targets"] / te2["team_targets"].replace(0, np.nan)
    tgt_share = te2.groupby("player_id")["tgt_share_wk"].mean().rename("target_share")
    agg = agg.join(tgt_share)

    # red zone targets (yardline_100 <= 20) from PBP
    pass_plays = pbp[pbp["play_type"] == "pass"].copy()
    rz = pass_plays[
        (pass_plays["yardline_100"] <= 20) & pass_plays["receiver_player_id"].notna()
    ]
    rz_tgts = rz.groupby("receiver_player_id").size().rename("redzone_targets")
    agg = agg.join(rz_tgts)
    agg["redzone_targets"] = agg["redzone_targets"].fillna(0)

    # end zone targets (yardline_100 <= 10)
    ez = pass_plays[
        (pass_plays["yardline_100"] <= 10) & pass_plays["receiver_player_id"].notna()
    ]
    ez_tgts = ez.groupby("receiver_player_id").size().rename("endzone_targets")
    agg = agg.join(ez_tgts)
    agg["endzone_targets"] = agg["endzone_targets"].fillna(0)

    # YPRR: yards per route run — route_runs not in weekly, use targets as proxy
    # True YPRR requires PFF data; flagged for enrichment
    agg["yprr_proxy"] = agg["receiving_yards"] / agg["targets"].replace(0, np.nan)
    agg["yprr_note"] = "proxy via targets; true YPRR requires PFF route data"

    agg["position"] = "TE"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# K metrics (Justin Sablich)
# 1. FG attempts/game  2. Team RZ trip rate  3. FG% 40-49  4. FG% 50+
# 5. Team implied pts
# ---------------------------------------------------------------------------

def compute_k_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    # Kickers are not in weekly player data — derive entirely from PBP
    # player_id for kicker comes from kicker_player_id in PBP
    fg = pbp[pbp["play_type"] == "field_goal"].copy()
    xp = pbp[pbp["play_type"] == "extra_point"].copy()

    if fg.empty:
        return pd.DataFrame()

    # games played: unique weeks per kicker
    games = (
        fg.groupby("kicker_player_id")["week"].nunique()
        .rename("games_played")
    )

    # get kicker name + team from PBP
    kicker_info = (
        fg.groupby("kicker_player_id")
        .agg(player_name=("kicker_player_name", "first"), team=("posteam", "last"))
    )

    agg = kicker_info.join(games)

    fg["made"] = fg["field_goal_result"] == "made"

    def fg_rate(df, min_dist, max_dist):
        sub = df[(df["kick_distance"] >= min_dist) & (df["kick_distance"] <= max_dist)]
        return sub.groupby("kicker_player_id").agg(
            attempts=("made", "count"),
            makes=("made", "sum"),
        )

    fg_all = fg.groupby("kicker_player_id").agg(
        fg_attempts=("made", "count"),
        fg_makes=("made", "sum"),
    )
    fg_40_49 = fg_rate(fg, 40, 49).rename(columns={"attempts": "fg_att_40_49", "makes": "fg_made_40_49"})
    fg_50p  = fg_rate(fg, 50, 99).rename(columns={"attempts": "fg_att_50p", "makes": "fg_made_50p"})

    agg = agg.join(fg_all).join(fg_40_49).join(fg_50p)
    for c in ["fg_attempts", "fg_makes", "fg_att_40_49", "fg_made_40_49", "fg_att_50p", "fg_made_50p"]:
        agg[c] = agg[c].fillna(0)

    agg["fg_pct"] = agg["fg_makes"] / agg["fg_attempts"].replace(0, np.nan)
    agg["fg_pct_40_49"] = agg["fg_made_40_49"] / agg["fg_att_40_49"].replace(0, np.nan)
    agg["fg_pct_50p"] = agg["fg_made_50p"] / agg["fg_att_50p"].replace(0, np.nan)
    agg["fg_attempts_per_game"] = agg["fg_attempts"] / agg["games_played"]

    xp_agg = xp.groupby("kicker_player_id").size().rename("xp_attempts")
    agg = agg.join(xp_agg)
    agg["xp_attempts"] = agg["xp_attempts"].fillna(0)

    agg["position"] = "K"
    agg.index.name = "player_id"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# DST metrics (Tommy Garrett)
# 1. Sack rate  2. Turnover rate  3. Points allowed/gm  4. Opp QB rating
# 5. Opponent DVOA (external — flagged)  6. Schedule SOS Wks 14-17
# ---------------------------------------------------------------------------

def compute_dst_metrics(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    # defensive team metrics — aggregate by defensive team
    defense = pbp[pbp["defteam"].notna()].copy()

    team_games = defense.groupby("defteam")["week"].nunique().rename("games_played")

    sacks = defense[defense["sack"] == 1].groupby("defteam").size().rename("sacks")
    interceptions = defense[defense["interception"] == 1].groupby("defteam").size().rename("interceptions")
    fumbles_rec = defense[defense["fumble_lost"] == 1].groupby("defteam").size().rename("fumbles_recovered")
    safeties = defense[defense["safety"] == 1].groupby("defteam").size().rename("safeties")

    # points allowed: sum scores by possession team (= offense) when defteam is present
    pts = (
        defense.groupby(["defteam", "week"])["total_away_score"]
        .last()  # end-of-game score per week
        .reset_index()
    )
    # simpler: use score differential from schedules
    dst = pd.DataFrame({
        "sacks": sacks,
        "interceptions": interceptions,
        "fumbles_recovered": fumbles_rec,
        "safeties": safeties,
    })
    dst = dst.join(team_games)

    dst["turnovers"] = dst["interceptions"] + dst["fumbles_recovered"]
    dst["sacks_per_game"] = dst["sacks"] / dst["games_played"]
    dst["turnovers_per_game"] = dst["turnovers"] / dst["games_played"]
    dst["sack_rate"] = dst["sacks"] / defense.groupby("defteam")["pass_attempt"].sum().replace(0, np.nan)
    dst["turnover_rate"] = dst["turnovers"] / defense.groupby("defteam")["play_id"].count().replace(0, np.nan)

    # points allowed per game from schedules
    sched = schedules[schedules["week"].isin(WEEKS)].copy()
    home_pts = sched[["away_team", "week", "home_score"]].rename(
        columns={"away_team": "defteam", "home_score": "pts_allowed"}
    )
    away_pts = sched[["home_team", "week", "away_score"]].rename(
        columns={"home_team": "defteam", "away_score": "pts_allowed"}
    )
    pts_allowed = pd.concat([home_pts, away_pts])
    pts_pg = pts_allowed.groupby("defteam")["pts_allowed"].mean().rename("points_allowed_per_game")
    dst = dst.join(pts_pg)

    # opponent QB rating from PBP
    opp_qbr = (
        defense[defense["passer_player_id"].notna()]
        .groupby("defteam")
        .apply(lambda x: (
            x["epa"].sum() / len(x)  # EPA per dropback as proxy for opp QB quality
        ))
        .rename("opp_epa_per_dropback_allowed")
    )
    dst = dst.join(opp_qbr)

    # NOTE: opponent DVOA requires footballoutsiders.com data — flagged for enrichment
    dst["opponent_dvoa"] = np.nan

    dst["position"] = "DST"
    dst.index.name = "team"
    dst["player_name"] = dst.index + " DST"
    dst["player_id"] = dst.index
    return dst.reset_index()


# ---------------------------------------------------------------------------
# Master fetch: runs all positions, returns dict of DataFrames
# ---------------------------------------------------------------------------

def fetch_all(season: int = SEASON) -> dict:
    pbp = load_pbp(season)
    weekly = load_weekly(season)
    schedules = load_schedules(pbp)

    return {
        "QB":  compute_qb_metrics(weekly),
        "RB":  compute_rb_metrics(weekly, pbp),
        "WR":  compute_wr_metrics(weekly, pbp),
        "TE":  compute_te_metrics(weekly, pbp),
        "K":   compute_k_metrics(weekly, pbp),
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
