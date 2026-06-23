"""
Live draft board with dynamic positional scarcity recalculation.

Core behavior:
  - Starts from the VBD-sorted board
  - After each pick, removes the player and recalculates replacement levels
  - Outputs a ranked recommendation list that updates in real time
  - Applies variance-maximizing adjustment based on current roster state

Variance philosophy:
  - Early in draft (picks 1-36): weight expected value heavily
  - Mid draft (picks 37-108): balance value and variance
  - Late draft (picks 109+): weight upside/variance heavily — hitting on a
    breakout is worth more than a safe pick at this range

The objective is P(win league), not max expected points.
"""

import pandas as pd
import numpy as np
from models.vbd import REPLACEMENT_LEVELS, SCARCITY_MULTIPLIER, _assign_tiers


LEAGUE_SIZE = 12
ROSTER_SLOTS = {
    "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1
}
TOTAL_ROSTER_SPOTS = sum(ROSTER_SLOTS.values())  # 9 starters; assume ~7 bench = 16 total picks

# Variance weight schedule: (pick_threshold, variance_weight)
# variance_weight 0.0 = pure EV, 1.0 = pure upside
VARIANCE_SCHEDULE = [
    (36,  0.10),
    (72,  0.25),
    (108, 0.45),
    (144, 0.65),
    (999, 0.80),
]


class DraftBoard:
    def __init__(self, full_board: pd.DataFrame, my_pick_position: int = 1):
        """
        full_board: output of compute_vbd()
        my_pick_position: 1-indexed draft position (1 = first overall)
        """
        self.board = full_board.copy()
        self.available = full_board.copy()
        self.drafted = []
        self.my_roster = []
        self.my_pick_position = my_pick_position
        self.overall_pick_number = 0
        self.my_picks = []  # list of overall pick numbers that belong to me (snake)

        self._compute_my_picks()

    def _compute_my_picks(self, total_rounds: int = 16):
        """Precompute which overall picks belong to the user in a snake draft."""
        picks = []
        for rnd in range(1, total_rounds + 1):
            if rnd % 2 == 1:  # odd round: left to right
                pick = (rnd - 1) * LEAGUE_SIZE + self.my_pick_position
            else:             # even round: right to left
                pick = rnd * LEAGUE_SIZE - self.my_pick_position + 1
            picks.append(pick)
        self.my_picks = picks

    def pick_made(self, player_name: str, by_me: bool = False):
        """Record that a player has been drafted."""
        self.overall_pick_number += 1
        mask = self.available["player_name"].str.lower() == player_name.lower()
        if not mask.any():
            # fuzzy: try contains
            mask = self.available["player_name"].str.lower().str.contains(
                player_name.lower(), na=False
            )
        if not mask.any():
            print(f"  [warn] '{player_name}' not found on board — skipping")
            return

        player = self.available[mask].iloc[0]
        self.drafted.append(player.to_dict())
        self.available = self.available[~mask].reset_index(drop=True)

        if by_me:
            self.my_roster.append(player.to_dict())
            self.my_picks = [p for p in self.my_picks if p != self.overall_pick_number]

        self._recalculate_vbd()

    def _recalculate_vbd(self):
        """Recompute VBD scores based on who is still available."""
        av = self.available.copy()
        updated_frames = []

        for pos in av["position"].unique():
            pos_df = av[av["position"] == pos].copy()
            pos_df = pos_df.sort_values("signal_score", ascending=False).reset_index(drop=True)
            pos_df["positional_rank"] = pos_df.index + 1

            rep_rank = REPLACEMENT_LEVELS.get(pos, 12)
            if len(pos_df) >= rep_rank:
                replacement_signal = pos_df.loc[rep_rank - 1, "signal_score"]
            else:
                replacement_signal = pos_df["signal_score"].min()

            pos_df["replacement_signal"] = replacement_signal
            pos_df["vbd_raw"] = pos_df["signal_score"] - replacement_signal
            pos_df["vbd"] = (pos_df["vbd_raw"] * SCARCITY_MULTIPLIER.get(pos, 1.0)).round(2)
            updated_frames.append(pos_df)

        if updated_frames:
            av = pd.concat(updated_frames, ignore_index=True)
            av = av.sort_values("vbd", ascending=False).reset_index(drop=True)
            av["overall_rank"] = av.index + 1
            av["tier"] = _assign_tiers(av)
            self.available = av

    def recommend(self, top_n: int = 10) -> pd.DataFrame:
        """
        Return top_n recommendations for the current pick.
        Applies variance adjustment based on pick number.
        """
        av = self.available.copy()
        if av.empty:
            return pd.DataFrame()

        # variance weight for this pick
        variance_weight = 0.10
        for threshold, weight in VARIANCE_SCHEDULE:
            if self.overall_pick_number <= threshold:
                variance_weight = weight
                break

        # upside proxy: players with high signal but lower floor
        # use difference between signal_score and positional median as upside measure
        av["pos_median"] = av.groupby("position")["signal_score"].transform("median")
        av["upside_score"] = (av["signal_score"] - av["pos_median"]).clip(lower=0)

        # blended score: (1 - var_weight)*vbd + var_weight*upside
        av["draft_score"] = (
            (1 - variance_weight) * av["vbd"] +
            variance_weight * av["upside_score"]
        )

        # roster need adjustment: penalize positions we're already full on
        my_positions = [p["position"] for p in self.my_roster]
        pos_counts = pd.Series(my_positions).value_counts().to_dict()
        max_at_pos = {"QB": 2, "RB": 6, "WR": 6, "TE": 3, "K": 1, "DST": 1}

        def need_penalty(row):
            pos = row["position"]
            current = pos_counts.get(pos, 0)
            cap = max_at_pos.get(pos, 3)
            if current >= cap:
                return 0.5  # heavy penalty
            return 1.0

        av["need_multiplier"] = av.apply(need_penalty, axis=1)
        av["draft_score"] = av["draft_score"] * av["need_multiplier"]

        av = av.sort_values("draft_score", ascending=False).reset_index(drop=True)

        cols = ["player_name", "team", "position", "positional_rank",
                "signal_score", "vbd", "upside_score", "draft_score", "tier"]
        cols = [c for c in cols if c in av.columns]
        return av[cols].head(top_n)

    def roster_summary(self) -> pd.DataFrame:
        """Display current roster state."""
        if not self.my_roster:
            return pd.DataFrame([{"message": "No players drafted yet"}])
        return pd.DataFrame(self.my_roster)[
            ["player_name", "team", "position", "signal_score", "vbd", "tier"]
        ]

    def positional_scarcity(self) -> pd.DataFrame:
        """Show how many players remain above replacement level at each position."""
        rows = []
        for pos, rep_rank in REPLACEMENT_LEVELS.items():
            pos_av = self.available[self.available["position"] == pos]
            above_rep = (pos_av["vbd"] > 0).sum()
            top5_avg = pos_av.head(5)["signal_score"].mean() if len(pos_av) >= 5 else np.nan
            rows.append({
                "position": pos,
                "available": len(pos_av),
                "above_replacement": above_rep,
                "top5_avg_signal": round(top5_avg, 1) if not np.isnan(top5_avg) else None,
                "scarcity_alert": above_rep <= (rep_rank // 2),
            })
        return pd.DataFrame(rows)
