"""
Fantasy Draft Tool — entry point.

Usage:
  python main.py fetch              # Pull Wks 7-17 2025 stats, save to data/
  python main.py score              # Score all positions, save to output/
  python main.py board              # Print full draft board
  python main.py board --pos RB     # Positional board
  python main.py value              # Sleepers + fades vs live FantasyPros ADP
  python main.py value --pos WR     # WR board with ADP context
  python main.py value --sleepers   # Sleepers only
  python main.py value --fades      # Fades only
  python main.py draft              # Interactive draft session
"""

import sys
import os
import pandas as pd
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from ingestion.fetch_stats import fetch_all, SEASON
from models.scorer import score_all_positions
from models.vbd import compute_vbd, get_positional_tiers
from models.vegas import load_vegas_lines, apply_vegas_multiplier, compute_playoff_schedule_weight
from models.situation_delta import load_situation_deltas, apply_veteran_deltas, inject_rookies
from models.draft_board import DraftBoard
from models.adp_value import build_value_board, print_value_board


def cmd_fetch(args):
    data = fetch_all(SEASON)
    os.makedirs("data", exist_ok=True)
    for pos, df in data.items():
        if not pos.startswith("_"):
            path = f"data/{pos}_wk7_17_{SEASON}.csv"
            df.to_csv(path, index=False)
            print(f"  saved {path} ({len(df)} rows)")


def cmd_score(args):
    print("[score] Loading raw data...")
    data = fetch_all(SEASON)

    print("[score] Scoring positions...")
    scored = score_all_positions(data)

    print("[score] Applying veteran situation deltas...")
    deltas = load_situation_deltas()
    if not deltas.empty:
        scored = apply_veteran_deltas(scored, deltas)

    print("[score] Applying Vegas multipliers...")
    vegas = load_vegas_lines()
    if not vegas.empty:
        scored = apply_vegas_multiplier(scored, vegas)

    print("[score] Computing VBD...")
    board = compute_vbd(scored)

    print("[score] Injecting 2026 rookies...")
    if not deltas.empty:
        board = inject_rookies(board, deltas)

    os.makedirs("output", exist_ok=True)
    board.to_csv("output/draft_board.csv", index=False)
    print(f"\n[done] Draft board saved: output/draft_board.csv ({len(board)} players)")
    print(board[["overall_rank", "position", "player_name", "team",
                 "signal_score", "vbd", "tier"]].head(50).to_string(index=False))


def cmd_board(args):
    path = "output/draft_board.csv"
    if not os.path.exists(path):
        print("[error] No board found — run: python main.py score")
        return

    board = pd.read_csv(path)

    if args.pos:
        pos = args.pos.upper()
        pos_board = get_positional_tiers(board, pos)
        print(f"\n=== {pos} Board ===")
        print(pos_board.head(args.n).to_string(index=False))
    else:
        print("\n=== Overall Draft Board (Top 300) ===")
        cols = ["overall_rank", "position", "player_name", "team",
                "signal_score", "vbd", "tier"]
        cols = [c for c in cols if c in board.columns]
        print(board[cols].head(args.n).to_string(index=False))


def cmd_draft(args):
    path = "output/draft_board.csv"
    if not os.path.exists(path):
        print("[error] No board found — run: python main.py score")
        return

    board = pd.read_csv(path)

    try:
        pick_pos = int(input("Your draft position (1-12): ").strip())
    except ValueError:
        pick_pos = 1

    session = DraftBoard(board, my_pick_position=pick_pos)

    print(f"\nDraft session started. Pick position: {pick_pos}")
    print("Commands: 'pick <name>' | 'mine <name>' | 'rec' | 'roster' | 'scarcity' | 'quit'\n")

    while True:
        try:
            cmd = input(f"[Pick {session.overall_pick_number + 1}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nDraft session ended.")
            break

        if not cmd:
            continue

        parts = cmd.split(None, 1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if action == "quit":
            break
        elif action == "rec":
            recs = session.recommend(top_n=10)
            print(recs.to_string(index=False))
        elif action == "pick" and arg:
            session.pick_made(arg, by_me=False)
            print(f"  Drafted: {arg}")
        elif action == "mine" and arg:
            session.pick_made(arg, by_me=True)
            print(f"  YOUR PICK: {arg}")
        elif action == "roster":
            print(session.roster_summary().to_string(index=False))
        elif action == "scarcity":
            print(session.positional_scarcity().to_string(index=False))
        elif action == "board":
            pos_arg = arg.upper() if arg else None
            if pos_arg:
                sub = session.available[session.available["position"] == pos_arg]
                print(sub[["positional_rank", "player_name", "team", "signal_score", "vbd", "tier"]].head(20).to_string(index=False))
            else:
                print(session.available[["overall_rank", "position", "player_name", "team", "vbd", "tier"]].head(20).to_string(index=False))
        else:
            print("  Unknown command. Try: rec | pick <name> | mine <name> | roster | scarcity | board [POS]")


def cmd_value(args):
    path = "output/draft_board.csv"
    if not os.path.exists(path):
        print("[error] No board found — run: python main.py score")
        return

    board = pd.read_csv(path)
    results = build_value_board(board)

    # save full value board
    os.makedirs("output", exist_ok=True)
    results["full"].to_csv("output/value_board.csv", index=False)

    n = args.n
    if args.pos:
        print_value_board(results, pos=args.pos, n=n)
    elif args.sleepers:
        print(f"\n=== SLEEPERS — Our Model Likes vs ADP (top {n}) ===")
        print(results["sleepers"].head(n).to_string(index=False))
    elif args.fades:
        print(f"\n=== FADES — ADP Overrates vs Our Model (top {n}) ===")
        print(results["fades"].head(n).to_string(index=False))
    else:
        print_value_board(results, n=n)

    print(f"\n[saved] output/value_board.csv")


def main():
    parser = argparse.ArgumentParser(description="Fantasy Draft Tool")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("fetch")
    sub.add_parser("score")

    board_p = sub.add_parser("board")
    board_p.add_argument("--pos", type=str, default=None)
    board_p.add_argument("--n", type=int, default=300)

    value_p = sub.add_parser("value")
    value_p.add_argument("--pos", type=str, default=None)
    value_p.add_argument("--sleepers", action="store_true", default=False)
    value_p.add_argument("--fades", action="store_true", default=False)
    value_p.add_argument("--n", type=int, default=30)

    sub.add_parser("draft")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "board":
        cmd_board(args)
    elif args.command == "value":
        cmd_value(args)
    elif args.command == "draft":
        cmd_draft(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
