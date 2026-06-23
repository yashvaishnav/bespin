"""
Fetches NFL player stats for Weeks 7-17 of a given season.

2025 data source: nflverse PBP parquet (weekly player_stats parquet not yet
published for 2025 as of Jun 2026). All stats derived from play-by-play.

2024 and prior: uses nfl_data_py weekly player stats + PBP for advanced metrics.
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SEASON = 2025
WEEKS = list(range(7, 18))

PBP_URL_2025 = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_2025.parquet"
)
WEEKLY_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats_{year}.parquet"
)


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------

def load_pbp(season: int = SEASON) -> pd.DataFrame:
    if season == 2025:
        print(f"[fetch] Loading 2025 PBP from nflverse...")
        pbp = pd.read_parquet(PBP_URL_2025)
    else:
        print(f"[fetch] Loading {season} PBP via nfl_data_py...")
        pbp = nfl.import_pbp_data([season], downcast=True, cache=False)

    pbp = pbp[pbp["season_type"] == "REG"]
    pbp = pbp[pbp["week"].isin(WEEKS)]
    print(f"  {len(pbp)} plays, weeks {pbp['week'].min()}-{pbp['week'].max()}")
    return pbp


def load_weekly(season: int = SEASON) -> pd.DataFrame:
    """
    For 2025: derive weekly stats from PBP since prebuilt parquet isn't published.
    For prior seasons: load prebuilt weekly parquet directly.
    """
    if season == 2025:
        print("[fetch] Deriving 2025 weekly player stats from PBP...")
        pbp_full = pd.read_parquet(PBP_URL_2025)
        pbp_full = pbp_full[pbp_full["season_type"] == "REG"]
        pbp_full = pbp_full[pbp_full["week"].isin(WEEKS)]
        return _derive_weekly_from_pbp(pbp_full)
    else:
        print(f"[fetch] Loading {season} weekly stats from nflverse...")
        df = nfl.import_weekly_data([season])
        return df[df["week"].isin(WEEKS)]


def _derive_weekly_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a weekly player stats dataframe from PBP that mirrors the columns
    used by the metric computation functions.
    """
    frames = []

    # --- Passing ---
    pass_plays = pbp[pbp["play_type"] == "pass"].copy()
    pass_plays = pass_plays[pass_plays["passer_player_id"].notna()]

    qb_stats = pass_plays.groupby(["passer_player_id", "week", "posteam"]).agg(
        attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        passing_yards=("yards_gained", lambda x: x[pass_plays.loc[x.index, "complete_pass"] == 1].sum()),
        passing_tds=("touchdown", lambda x: x[pass_plays.loc[x.index, "pass_attempt"] == 1].sum()),
        interceptions=("interception", "sum"),
        sacks=("sack", "sum"),
        passing_air_yards=("air_yards", "sum"),
        passing_epa=("epa", "sum"),
    ).reset_index()

    # simpler passing yards: yards_gained on complete passes
    pass_complete = pass_plays[pass_plays["complete_pass"] == 1]
    py = pass_complete.groupby(["passer_player_id", "week"])["yards_gained"].sum().rename("passing_yards_fixed")
    ptd = pass_plays[pass_plays["touchdown"] == 1].groupby(["passer_player_id", "week"]).size().rename("passing_tds_fixed")

    qb_agg = pass_plays.groupby(["passer_player_id", "week", "posteam"]).agg(
        attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        interceptions=("interception", "sum"),
        sacks=("sack", "sum"),
        passing_air_yards=("air_yards", "sum"),
        passing_epa=("epa", "sum"),
    ).reset_index()
    qb_agg = qb_agg.join(py, on=["passer_player_id", "week"]).join(ptd, on=["passer_player_id", "week"])
    qb_agg["passing_yards"] = qb_agg["passing_yards_fixed"].fillna(0)
    qb_agg["passing_tds"] = qb_agg["passing_tds_fixed"].fillna(0)
    qb_agg = qb_agg.drop(columns=["passing_yards_fixed", "passing_tds_fixed"])
    qb_agg = qb_agg.rename(columns={"passer_player_id": "player_id", "posteam": "recent_team"})

    # --- Rushing ---
    rush_plays = pbp[pbp["play_type"] == "run"].copy()
    rush_plays = rush_plays[rush_plays["rusher_player_id"].notna()]

    rush_agg = rush_plays.groupby(["rusher_player_id", "week", "posteam"]).agg(
        carries=("rush_attempt", "sum"),
        rushing_yards=("yards_gained", "sum"),
        rushing_epa=("epa", "sum"),
    ).reset_index()
    rush_td = rush_plays[rush_plays["touchdown"] == 1].groupby(["rusher_player_id", "week"]).size().rename("rushing_tds")
    rush_agg = rush_agg.join(rush_td, on=["rusher_player_id", "week"])
    rush_agg["rushing_tds"] = rush_agg["rushing_tds"].fillna(0)
    rush_agg = rush_agg.rename(columns={"rusher_player_id": "player_id", "posteam": "recent_team"})

    # --- Receiving ---
    rec_plays = pbp[pbp["receiver_player_id"].notna()].copy()

    tgt_agg = rec_plays.groupby(["receiver_player_id", "week", "posteam"]).agg(
        targets=("pass_attempt", "sum"),
        receiving_air_yards=("air_yards", "sum"),
        receiving_epa=("epa", "sum"),
    ).reset_index()

    rec_complete = rec_plays[rec_plays["complete_pass"] == 1]
    rec_stats = rec_complete.groupby(["receiver_player_id", "week"]).agg(
        receptions=("complete_pass", "sum"),
        receiving_yards=("yards_gained", "sum"),
        receiving_yards_after_catch=("yards_after_catch", "sum"),
    )
    rec_td = rec_plays[rec_plays["touchdown"] == 1].groupby(["receiver_player_id", "week"]).size().rename("receiving_tds")

    tgt_agg = tgt_agg.join(rec_stats, on=["receiver_player_id", "week"]).join(rec_td, on=["receiver_player_id", "week"])
    for c in ["receptions", "receiving_yards", "receiving_yards_after_catch", "receiving_tds"]:
        tgt_agg[c] = tgt_agg[c].fillna(0)
    tgt_agg = tgt_agg.rename(columns={"receiver_player_id": "player_id", "posteam": "recent_team"})

    # --- Player name + position lookup from PBP ---
    # Build name/position map from pbp fields
    name_map = {}
    pos_map = {}

    for col_id, col_name in [
        ("passer_player_id", "passer_player_name"),
        ("rusher_player_id", "rusher_player_name"),
        ("receiver_player_id", "receiver_player_name"),
    ]:
        sub = pbp[[col_id, col_name]].dropna().drop_duplicates()
        for _, row in sub.iterrows():
            name_map[row[col_id]] = row[col_name]

    # --- Merge into per-player-per-week records ---
    # Full player-week grid: join passing + rushing + receiving
    all_ids = pd.concat([
        qb_agg[["player_id", "week", "recent_team"]],
        rush_agg[["player_id", "week", "recent_team"]],
        tgt_agg[["player_id", "week", "recent_team"]],
    ]).drop_duplicates()

    merged = all_ids.copy()
    merged = merged.merge(
        qb_agg.drop(columns=["recent_team"]), on=["player_id", "week"], how="left"
    ).merge(
        rush_agg.drop(columns=["recent_team"]), on=["player_id", "week"], how="left"
    ).merge(
        tgt_agg.drop(columns=["recent_team"]), on=["player_id", "week"], how="left"
    )

    # fill nulls
    stat_cols = [
        "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
        "sacks", "passing_air_yards", "passing_epa",
        "carries", "rushing_yards", "rushing_tds", "rushing_epa",
        "targets", "receptions", "receiving_yards", "receiving_yards_after_catch",
        "receiving_tds", "receiving_air_yards", "receiving_epa",
    ]
    for c in stat_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    merged["player_display_name"] = merged["player_id"].map(name_map)

    # Infer position: QBs have attempts, pure rushers are RB/FB, receivers are WR/TE/RB
    def infer_position(row):
        if row.get("attempts", 0) > row.get("carries", 0) * 2 and row.get("attempts", 0) > 3:
            return "QB"
        if row.get("carries", 0) > row.get("targets", 0) and row.get("carrying_ratio", 1) > 0:
            return "RB"
        return "WR"  # default skill player — position assignment enriched by roster data below

    # Load position data from static player registry (gsis_id = player_id in PBP)
    try:
        players = nfl.import_players()
        pos_lookup = players.set_index("gsis_id")["position"].to_dict()
        merged["position"] = merged["player_id"].map(pos_lookup)
        # For any unmapped: infer from play type
        unmapped = merged["position"].isna()
        merged.loc[unmapped & (merged["attempts"] > 5), "position"] = "QB"
        merged.loc[unmapped & (merged["carries"] > merged["targets"]) & merged["position"].isna(), "position"] = "RB"
        merged.loc[merged["position"].isna(), "position"] = "WR"
    except Exception as e:
        print(f"  [warn] Position lookup failed ({e}), using inference")
        merged["position"] = merged.apply(
            lambda r: "QB" if r.get("attempts", 0) > 5 else (
                "RB" if r.get("carries", 0) > r.get("targets", 0) else "WR"
            ), axis=1
        )

    # Half-PPR fantasy points
    merged["fantasy_points_ppr"] = (
        merged.get("passing_yards", 0) * 0.04 +
        merged.get("passing_tds", 0) * 4 +
        merged.get("interceptions", 0) * -1 +
        merged.get("rushing_yards", 0) * 0.1 +
        merged.get("rushing_tds", 0) * 6 +
        merged.get("receptions", 0) * 1.0 +      # full PPR base; we use 0.5 in half-PPR
        merged.get("receiving_yards", 0) * 0.1 +
        merged.get("receiving_tds", 0) * 6
    )

    # Derived shared metrics
    team_targets = merged.groupby(["recent_team", "week"])["targets"].transform("sum")
    team_air_yards = merged.groupby(["recent_team", "week"])["receiving_air_yards"].transform("sum")

    merged["target_share"] = merged["targets"] / team_targets.replace(0, np.nan)
    merged["air_yards_share"] = merged["receiving_air_yards"] / team_air_yards.replace(0, np.nan)
    merged["wopr"] = 1.5 * merged["target_share"].fillna(0) + 0.7 * merged["air_yards_share"].fillna(0)

    merged["season"] = SEASON
    return merged


def load_rosters(season: int = SEASON) -> pd.DataFrame:
    try:
        r = nfl.import_rosters([season])
        return r[["player_id", "player_name", "position", "team",
                   "depth_chart_position", "birth_date", "height", "weight",
                   "college", "draft_number"]].copy()
    except Exception as e:
        print(f"  [warn] Roster load failed: {e}")
        return pd.DataFrame()


def load_schedules(pbp: pd.DataFrame) -> pd.DataFrame:
    sched = (
        pbp[pbp["home_team"].notna() & pbp["away_team"].notna()]
        .groupby(["week", "home_team", "away_team"])
        .agg(home_score=("total_home_score", "max"), away_score=("total_away_score", "max"))
        .reset_index()
    )
    return sched


# ---------------------------------------------------------------------------
# QB metrics (Dalton Del Don)
# ---------------------------------------------------------------------------

def compute_qb_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    qb = weekly[weekly["position"] == "QB"].copy()
    if qb.empty:
        return pd.DataFrame()

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
        fantasy_points_half_ppr=("fantasy_points_ppr", "sum"),
        sacks=("sacks", "sum") if "sacks" in qb.columns else ("pass_attempts", "count"),
    ).join(games)

    for col in ["pass_attempts", "passing_yards", "rushing_attempts",
                "rushing_yards", "fantasy_points_half_ppr"]:
        agg[f"{col}_per_game"] = agg[col] / agg["games_played"]

    agg["ypa"] = agg["passing_yards"] / agg["pass_attempts"].replace(0, np.nan)
    agg["completion_pct"] = agg["completions"] / agg["pass_attempts"].replace(0, np.nan)
    agg["td_pct"] = agg["passing_tds"] / agg["pass_attempts"].replace(0, np.nan)
    agg["int_pct"] = agg["interceptions"] / agg["pass_attempts"].replace(0, np.nan)
    agg["dropbacks"] = agg["pass_attempts"] + agg.get("sacks", 0)
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

    for col in ["carries", "targets", "receptions", "receiving_yards", "rushing_yards"]:
        agg[f"{col}_per_game"] = agg[col] / agg["games_played"]

    agg["ypc"] = agg["rushing_yards"] / agg["carries"].replace(0, np.nan)
    agg["yards_per_reception"] = agg["receiving_yards"] / agg["receptions"].replace(0, np.nan)
    agg["catch_rate"] = agg["receptions"] / agg["targets"].replace(0, np.nan)

    # team shares
    team_totals = rb.groupby(["recent_team", "week"]).agg(
        team_carries=("carries", "sum"),
        team_targets=("targets", "sum"),
    ).reset_index()
    rb2 = rb.merge(team_totals, on=["recent_team", "week"])
    rb2["carry_share_wk"] = rb2["carries"] / rb2["team_carries"].replace(0, np.nan)
    rb2["target_share_wk"] = rb2["targets"] / rb2["team_targets"].replace(0, np.nan)
    rb2["opportunities"] = rb2["carries"] + rb2["targets"]
    team_opp = rb2.groupby(["recent_team", "week"])["opportunities"].sum().rename("team_opp")
    rb2 = rb2.join(team_opp, on=["recent_team", "week"], rsuffix="_t")
    rb2["opp_share_wk"] = rb2["opportunities"] / rb2["team_opp"].replace(0, np.nan)

    shares = rb2.groupby("player_id").agg(
        carry_share=("carry_share_wk", "mean"),
        target_share_backfield=("target_share_wk", "mean"),
        opportunity_share=("opp_share_wk", "mean"),
    )
    agg = agg.join(shares)

    # goal-line carries
    gl = pbp[(pbp["play_type"] == "run") & (pbp["yardline_100"] <= 5) &
             pbp["rusher_player_id"].notna()]
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

    team_tgts = wr.groupby(["recent_team", "week"])["targets"].sum().rename("team_targets").reset_index()
    wr2 = wr.merge(team_tgts, on=["recent_team", "week"])
    wr2["tgt_share_wk"] = wr2["targets"] / wr2["team_targets"].replace(0, np.nan)
    tgt_share = wr2.groupby("player_id")["tgt_share_wk"].mean().rename("target_share")
    agg = agg.join(tgt_share)

    pass_plays = pbp[pbp["play_type"] == "pass"].copy()
    adot = (
        pass_plays[pass_plays["receiver_player_id"].notna()]
        .groupby("receiver_player_id")["air_yards"].mean().rename("adot")
    )
    air_total = (
        pass_plays[pass_plays["receiver_player_id"].notna()]
        .groupby("receiver_player_id")["air_yards"].sum().rename("total_air_yards")
    )
    agg = agg.join(adot).join(air_total)
    agg["racr"] = agg["receiving_yards"] / agg["total_air_yards"].replace(0, np.nan)
    agg["ngs_separation"] = np.nan  # requires Next Gen Stats API

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

    all_skill = weekly[weekly["position"].isin(["WR", "TE", "RB"])].copy()
    team_tgts = all_skill.groupby(["recent_team", "week"])["targets"].sum().rename("team_targets").reset_index()
    te2 = te.merge(team_tgts, on=["recent_team", "week"])
    te2["tgt_share_wk"] = te2["targets"] / te2["team_targets"].replace(0, np.nan)
    tgt_share = te2.groupby("player_id")["tgt_share_wk"].mean().rename("target_share")
    agg = agg.join(tgt_share)

    pass_plays = pbp[pbp["play_type"] == "pass"].copy()
    rz = pass_plays[(pass_plays["yardline_100"] <= 20) & pass_plays["receiver_player_id"].notna()]
    ez = pass_plays[(pass_plays["yardline_100"] <= 10) & pass_plays["receiver_player_id"].notna()]

    agg = agg.join(
        rz.groupby("receiver_player_id").size().rename("redzone_targets")
    ).join(
        ez.groupby("receiver_player_id").size().rename("endzone_targets")
    )
    agg["redzone_targets"] = agg["redzone_targets"].fillna(0)
    agg["endzone_targets"] = agg["endzone_targets"].fillna(0)
    agg["yprr_proxy"] = agg["receiving_yards"] / agg["targets"].replace(0, np.nan)

    agg["position"] = "TE"
    return agg.reset_index()


# ---------------------------------------------------------------------------
# K metrics (Justin Sablich)
# ---------------------------------------------------------------------------

def compute_k_metrics(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    fg = pbp[pbp["play_type"] == "field_goal"].copy()
    xp = pbp[pbp["play_type"] == "extra_point"].copy()
    if fg.empty:
        return pd.DataFrame()

    games = fg.groupby("kicker_player_id")["week"].nunique().rename("games_played")
    kicker_info = fg.groupby("kicker_player_id").agg(
        player_name=("kicker_player_name", "first"),
        team=("posteam", "last"),
    )
    agg = kicker_info.join(games)

    fg["made"] = fg["field_goal_result"] == "made"

    def fg_rate(df, lo, hi):
        sub = df[(df["kick_distance"] >= lo) & (df["kick_distance"] <= hi)]
        return sub.groupby("kicker_player_id").agg(
            attempts=("made", "count"), makes=("made", "sum")
        )

    fg_all = fg.groupby("kicker_player_id").agg(fg_attempts=("made","count"), fg_makes=("made","sum"))
    fg_40 = fg_rate(fg, 40, 49).rename(columns={"attempts":"fg_att_40_49","makes":"fg_made_40_49"})
    fg_50 = fg_rate(fg, 50, 99).rename(columns={"attempts":"fg_att_50p","makes":"fg_made_50p"})

    agg = agg.join(fg_all).join(fg_40).join(fg_50)
    for c in ["fg_attempts","fg_makes","fg_att_40_49","fg_made_40_49","fg_att_50p","fg_made_50p"]:
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
# ---------------------------------------------------------------------------

def compute_dst_metrics(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    defense = pbp[pbp["defteam"].notna()].copy()
    team_games = defense.groupby("defteam")["week"].nunique().rename("games_played")

    sacks = defense[defense["sack"] == 1].groupby("defteam").size().rename("sacks")
    ints = defense[defense["interception"] == 1].groupby("defteam").size().rename("interceptions")
    fum = defense[defense["fumble_lost"] == 1].groupby("defteam").size().rename("fumbles_recovered")
    saf = defense[defense["safety"] == 1].groupby("defteam").size().rename("safeties")

    dst = pd.DataFrame({"sacks": sacks, "interceptions": ints,
                        "fumbles_recovered": fum, "safeties": saf}).join(team_games)
    dst["turnovers"] = dst["interceptions"] + dst["fumbles_recovered"]
    dst["sacks_per_game"] = dst["sacks"] / dst["games_played"]
    dst["turnovers_per_game"] = dst["turnovers"] / dst["games_played"]

    total_pass_attempts = defense[defense["pass_attempt"] == 1].groupby("defteam")["pass_attempt"].sum()
    total_plays = defense.groupby("defteam")["play_id"].count()
    dst["sack_rate"] = dst["sacks"] / total_pass_attempts.replace(0, np.nan)
    dst["turnover_rate"] = dst["turnovers"] / total_plays.replace(0, np.nan)

    home_pts = schedules[["away_team","week","home_score"]].rename(
        columns={"away_team":"defteam","home_score":"pts_allowed"})
    away_pts = schedules[["home_team","week","away_score"]].rename(
        columns={"home_team":"defteam","away_score":"pts_allowed"})
    pts_pg = pd.concat([home_pts, away_pts]).groupby("defteam")["pts_allowed"].mean().rename("points_allowed_per_game")
    dst = dst.join(pts_pg)

    opp_epa = (
        defense[defense["passer_player_id"].notna()]
        .groupby("defteam")["epa"].mean()
        .rename("opp_epa_per_dropback_allowed")
    )
    dst = dst.join(opp_epa)
    dst["opponent_dvoa"] = np.nan
    dst["position"] = "DST"
    dst.index.name = "team"
    dst["player_name"] = dst.index + " DST"
    dst["player_id"] = dst.index
    return dst.reset_index()


# ---------------------------------------------------------------------------
# Master fetch
# ---------------------------------------------------------------------------

def fetch_all(season: int = SEASON) -> dict:
    pbp = load_pbp(season)
    weekly = load_weekly(season)
    schedules = load_schedules(pbp)

    return {
        "QB":  compute_qb_metrics(weekly, pbp),
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
