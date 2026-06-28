"""
ADP Value Layer — Option C draft tool.

Pulls live FantasyPros consensus ECR (Expert Consensus Rankings) and compares
against our advanced-stats signal scores to identify:
  - Sleepers: players our model likes significantly more than ADP
  - Fades:    players our model likes significantly less than ADP
  - Positional boards: each position ranked by signal score with ADP context

ECR source: FantasyPros redraft overall rankings via nflreadpy.load_ff_rankings()
Scoring: redraft-overall reflects half-PPR consensus (FantasyPros default)
"""

import pandas as pd
import numpy as np
import nflreadpy as nflr


# Only skill positions in sleeper/fade analysis
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# ADP range for meaningful comparison
ADP_MIN_CUTOFF = 5     # ignore ADP rank 1-4 (consensus locks, not worth fading)
ADP_MAX_CUTOFF = 180   # ignore late fliers with high ADP uncertainty

# Minimum signal score — filters players we have no real data on (rookies, DNPs)
MIN_SIGNAL_SCORE = 55

# How many picks difference = meaningful signal vs ADP
SLEEPER_THRESHOLD = 12   # our rank is 12+ picks better than ADP
FADE_THRESHOLD    = 12   # our rank is 12+ picks worse than ADP


def load_adp() -> pd.DataFrame:
    """
    Load live FantasyPros consensus ECR.
    Returns DataFrame with: player, pos, team, adp, adp_best, adp_worst, adp_sd
    """
    print("[adp] Loading live FantasyPros ECR...")
    df = nflr.load_ff_rankings(type="draft").to_pandas()
    df = df[df["page_type"] == "redraft-overall"].copy()
    df = df.rename(columns={
        "player": "player_name",
        "pos":    "position",
        "ecr":    "adp",
        "best":   "adp_best",
        "worst":  "adp_worst",
        "sd":     "adp_sd",
    })
    df["position"] = df["position"].str.upper().str.strip()
    df["adp_rank"] = df["adp"].rank(method="min").astype(int)
    print(f"  {len(df)} players, scraped {df['scrape_date'].iloc[0]}")
    return df[["player_name", "position", "team", "adp", "adp_rank",
               "adp_best", "adp_worst", "adp_sd", "scrape_date"]].copy()


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    import re
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def merge_adp_with_board(board: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ADP into the draft board by player name + position fuzzy match.
    Returns board with added adp, adp_rank, value_delta columns.
    value_delta > 0  → we rank player higher than ADP (buy)
    value_delta < 0  → we rank player lower than ADP (fade)
    """
    adp_work = adp.copy()
    adp_work["_key"] = adp_work["player_name"].apply(_normalize_name)

    board_work = board.copy()
    board_work["_key"] = board_work["player_name"].apply(_normalize_name)

    # primary merge on normalized name
    merged = board_work.merge(
        adp_work[["_key", "adp", "adp_rank", "adp_best", "adp_worst", "adp_sd", "scrape_date"]],
        on="_key", how="left"
    )

    # for unmatched players try last-name match
    unmatched = merged["adp"].isna()
    if unmatched.sum() > 0:
        adp_work["_last"] = adp_work["_key"].apply(lambda x: x.split()[-1])
        board_work["_last"] = board_work["_key"].apply(lambda x: x.split()[-1])
        last_map = adp_work.set_index("_last")[["adp", "adp_rank", "adp_best", "adp_worst", "adp_sd"]]
        for col in ["adp", "adp_rank", "adp_best", "adp_worst", "adp_sd"]:
            merged.loc[unmatched, col] = board_work.loc[unmatched, "_last"].map(
                last_map[col].to_dict()
            )

    merged = merged.drop(columns=["_key"])

    # signal rank within full board (already have overall_rank)
    merged["signal_rank"] = merged["overall_rank"]

    # value delta: positive = our model likes them MORE than ADP
    merged["value_delta"] = merged["adp_rank"] - merged["signal_rank"]

    return merged


def build_value_board(board: pd.DataFrame) -> dict:
    """
    Main entry point. Returns dict with:
      - 'full':      complete board with ADP + signal + delta
      - 'sleepers':  players we like vs ADP
      - 'fades':     players ADP likes more than we do
      - 'QB' / 'RB' / 'WR' / 'TE' / 'K' / 'DST': positional boards
    """
    adp = load_adp()
    merged = merge_adp_with_board(board, adp)

    skill_mask = (
        merged["position"].isin(SKILL_POSITIONS) &
        merged["adp_rank"].between(ADP_MIN_CUTOFF, ADP_MAX_CUTOFF) &
        (merged["signal_score"] >= MIN_SIGNAL_SCORE)
    )

    sleepers = (
        merged[skill_mask & (merged["value_delta"] >= SLEEPER_THRESHOLD)]
        .sort_values("value_delta", ascending=False)
        [["player_name", "position", "team", "signal_rank", "adp_rank",
          "value_delta", "signal_score", "vbd", "tier", "adp_sd"]]
        .reset_index(drop=True)
    )

    # fades: require MIN_SIGNAL_SCORE so we only fade players we have real data on
    # (filters out rookies/injury returns where we simply have no 2025 stats)
    fade_mask = (
        merged["position"].isin(SKILL_POSITIONS) &
        merged["adp_rank"].between(ADP_MIN_CUTOFF, ADP_MAX_CUTOFF) &
        (merged["signal_score"] >= MIN_SIGNAL_SCORE) &
        (merged["value_delta"] <= -FADE_THRESHOLD)
    )
    fades = (
        merged[fade_mask]
        .sort_values("value_delta", ascending=True)
        [["player_name", "position", "team", "signal_rank", "adp_rank",
          "value_delta", "signal_score", "vbd", "tier", "adp_sd"]]
        .reset_index(drop=True)
    )

    positional = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        pos_df = merged[merged["position"] == pos].sort_values("signal_rank").copy()
        pos_df["pos_signal_rank"] = range(1, len(pos_df) + 1)
        positional[pos] = pos_df[
            ["pos_signal_rank", "player_name", "team", "signal_score",
             "vbd", "tier", "adp_rank", "value_delta", "adp_sd"]
        ].reset_index(drop=True)

    return {
        "full":     merged,
        "sleepers": sleepers,
        "fades":    fades,
        **positional,
    }


def print_value_board(results: dict, pos: str = None, n: int = 30):
    """Pretty-print the value board results."""
    if pos:
        pos = pos.upper()
        if pos in results:
            print(f"\n=== {pos} Board (signal rank | ADP | delta) ===")
            print(results[pos].head(n).to_string(index=False))
        return

    print("\n=== SLEEPERS — Our Model Likes vs ADP ===")
    print("(value_delta = how many picks earlier we'd take them vs consensus)")
    print(results["sleepers"].head(n).to_string(index=False))

    print("\n=== FADES — ADP Overrates vs Our Model ===")
    print("(value_delta = how many picks too early consensus takes them)")
    print(results["fades"].head(n).to_string(index=False))
