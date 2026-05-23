"""
C-Bet Profitability Predictor
Usage: python3 poker_model/src/predict.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_DIR, "..", "outputs")

model  = joblib.load(os.path.join(_OUT, "cbet_model.pkl"))
scaler = joblib.load(os.path.join(_OUT, "cbet_scaler.pkl"))

FEATURE_COLS = [
    "is_in_position", "position_encoded", "num_opponents", "is_heads_up",
    "cbet_size_to_pot", "is_3bet_pot", "is_4bet_pot", "stack_depth_bb",
    "is_short_stack", "pot_size_bb", "has_flush_draw", "has_straight_draw",
    "is_monotone", "is_paired_board", "board_high_card", "board_wetness_score",
    "board_connectedness", "player_vpip_history", "player_pfr_history",
    "player_winrate_history",
]

FEATURE_DESCRIPTIONS = {
    "pot_size_bb":            "Pot size at flop (in big blinds)",
    "player_vpip_history":    "Opponent's historical VPIP (0-1)",
    "is_3bet_pot":            "3-bet pot preflop (0/1)",
    "is_4bet_pot":            "4-bet pot preflop (0/1)",
    "cbet_size_to_pot":       "C-bet size as fraction of pot (e.g. 0.5 = half-pot)",
    "position_encoded":       "Position number: BTN=1 (best) … BB=9 (worst)",
    "is_in_position":         "Acting last postflop: BTN/CO/HJ=1, else 0",
    "is_heads_up":            "Heads-up to the flop (0/1)",
    "is_paired_board":        "Flop contains a pair (0/1)",
    "stack_depth_bb":         "Starting stack in big blinds",
    "board_connectedness":    "Board connectedness score (0=none, 4=max)",
    "num_opponents":          "Number of opponents seeing the flop",
    "board_wetness_score":    "Board wetness (0=dry, 5=very wet)",
    "has_straight_draw":      "Flop has straight draw potential (0/1)",
    "player_winrate_history": "Player's historical c-bet win rate (0-1)",
    "is_short_stack":         "Stack < 40 big blinds (0/1)",
    "board_high_card":        "Rank of highest flop card (2-14, A=14)",
    "player_pfr_history":     "Player's historical PFR (0-1)",
    "is_monotone":            "All three flop cards same suit (0/1)",
    "has_flush_draw":         "Flop has flush draw (two+ same suit) (0/1)",
}


# ── Coefficient table ──────────────────────────────────────────────────

def print_coefficients():
    coef_df = pd.read_csv(os.path.join(_OUT, "coefficients.csv"))

    print("=" * 72)
    print("  C-BET PROFITABILITY MODEL — FEATURE COEFFICIENTS")
    print(f"  Intercept: {model.intercept_:+.3f} bb")
    print("  (coefficients are in big blinds, after standardizing all features)")
    print("=" * 72)
    print(f"  {'#':>2}  {'Feature':<26}  {'Coeff (bb)':>10}  {'Direction':<8}  Description")
    print("  " + "-" * 70)

    for i, row in coef_df.iterrows():
        feat  = row["feature"]
        coeff = row["coefficient"]
        arrow = "▲ positive" if coeff > 0 else "▼ negative"
        desc  = FEATURE_DESCRIPTIONS.get(feat, "")
        print(f"  {i+1:>2}  {feat:<26}  {coeff:>+10.4f}  {arrow:<10}  {desc}")

    print("=" * 72)
    print()


# ── Board texture helpers ──────────────────────────────────────────────

def _parse_card(card_str):
    if not card_str or len(card_str) < 2:
        return 0, "?"
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
        "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    return rank_map.get(card_str[0].upper(), 0), card_str[1].lower()


def _board_features(flop_cards):
    (r1, s1), (r2, s2), (r3, s3) = [_parse_card(c) for c in flop_cards]
    ranks = sorted([r1, r2, r3])
    suits = [s1, s2, s3]
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit   = max(suit_counts.values())
    has_fd     = 1 if max_suit >= 2 else 0
    monotone   = 1 if max_suit == 3 else 0
    has_sd     = 1 if min(ranks[1] - ranks[0], ranks[2] - ranks[1]) <= 3 else 0
    paired     = 1 if len(set([r1, r2, r3])) < 3 else 0
    connect    = max(0, 4 - ((ranks[2] - ranks[0]) // 2))
    wetness    = has_fd * 2 + has_sd * 2 + (1 - paired)
    return {
        "has_flush_draw": has_fd, "has_straight_draw": has_sd,
        "is_monotone": monotone,  "is_paired_board": paired,
        "board_high_card": ranks[2], "board_wetness_score": wetness,
        "board_connectedness": connect,
    }


# ── Main prediction function ───────────────────────────────────────────

def evaluate_cbet(
    position: str,        # "BTN" | "CO" | "HJ" | "MP" | "UTG" | "SB" | "BB"
    num_opponents: int,   # opponents seeing the flop (1, 2, 3 …)
    cbet_size_pct: float, # c-bet as fraction of pot: 0.33, 0.5, 0.75, 1.0 …
    is_3bet_pot: int,     # 1 if there was a 3-bet preflop, else 0
    stack_depth_bb: float,# hero's starting stack in big blinds (e.g. 100)
    pot_size_bb: float,   # pot size at flop in big blinds (e.g. 7.5)
    flop_cards: list,     # three card strings: ["Ah", "6h", "7h"]
    player_vpip: float = 0.24,    # opponent's VPIP (default = population avg)
    player_pfr: float  = 0.18,    # opponent's PFR
    player_winrate: float = 0.50, # hero's historical c-bet win rate
) -> dict:
    """
    Predict the expected profitability of a c-bet.

    Returns a dict with:
      expected_profit_pct_of_pot — predicted net_profit / pot_size_at_flop (clipped ±3)
      recommendation             — label based on percentile of test-set predictions
      confidence                 — "high" (top/bottom 25%) or "low" (middle 50%)
      feature_breakdown          — each feature's contribution (coeff × scaled value)
    """
    pos_map = {
        "BTN": 1, "CO": 2, "HJ": 3, "MP+1": 4,
        "MP": 5, "UTG+1": 6, "UTG": 7, "SB": 8, "BB": 9,
    }
    board = _board_features(flop_cards)

    feature_values = [
        1 if position in ["BTN", "CO", "HJ"] else 0,  # is_in_position
        pos_map.get(position, 5),                       # position_encoded
        num_opponents,                                  # num_opponents
        1 if num_opponents == 1 else 0,                # is_heads_up
        cbet_size_pct,                                  # cbet_size_to_pot
        is_3bet_pot,                                    # is_3bet_pot
        0,                                              # is_4bet_pot (not tracked here)
        min(stack_depth_bb, 500),                       # stack_depth_bb
        1 if stack_depth_bb < 40 else 0,               # is_short_stack
        min(pot_size_bb, 200),                          # pot_size_bb
        board["has_flush_draw"],
        board["has_straight_draw"],
        board["is_monotone"],
        board["is_paired_board"],
        board["board_high_card"],
        board["board_wetness_score"],
        board["board_connectedness"],
        player_vpip,                                    # player_vpip_history
        player_pfr,                                     # player_pfr_history
        player_winrate,                                 # player_winrate_history
    ]

    scaled   = scaler.transform([feature_values])
    ev_bb    = float(model.predict(scaled)[0])

    # Per-feature contribution = coefficient × scaled value
    contributions = {
        feat: round(float(model.coef_[i] * scaled[0][i]), 4)
        for i, feat in enumerate(FEATURE_COLS)
    }
    contributions_sorted = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # Thresholds derived from actual test-set prediction distribution (n=3,337):
    #   p25=0.1664  p50=0.2127  p75=0.2646
    if ev_bb > 0.2646:
        rec, conf = "C-BET — strong spot", "high"       # top 25%
    elif ev_bb > 0.2127:
        rec, conf = "C-BET — marginal spot", "low"      # 50th–75th percentile
    elif ev_bb > 0.1664:
        rec, conf = "CHECK — marginal spot", "low"      # 25th–50th percentile
    else:
        rec, conf = "CHECK — poor c-bet spot", "high"   # bottom 25%

    return {
        "expected_profit_pct_of_pot": round(ev_bb, 4),
        "recommendation":             rec,
        "confidence":                 conf,
        "feature_breakdown":          contributions_sorted,
    }


# ── Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print_coefficients()

    print("=== C-BET DECISION TOOL ===\n")

    scenarios = [
        ("BTN, heads-up, dry board (best case)",
         dict(position="BTN", num_opponents=1, cbet_size_pct=0.5,
              is_3bet_pot=0, stack_depth_bb=100, pot_size_bb=7,
              flop_cards=["Kd", "7s", "2c"])),

        ("BB, 3 opponents, wet board (worst case)",
         dict(position="BB", num_opponents=3, cbet_size_pct=0.75,
              is_3bet_pot=0, stack_depth_bb=100, pot_size_bb=10,
              flop_cards=["9h", "8h", "7d"])),

        ("CO, heads-up, monotone flush board",
         dict(position="CO", num_opponents=1, cbet_size_pct=0.6,
              is_3bet_pot=0, stack_depth_bb=100, pot_size_bb=8,
              flop_cards=["Ah", "6h", "7h"])),

        ("BTN, 3-bet pot, dry board",
         dict(position="BTN", num_opponents=1, cbet_size_pct=0.4,
              is_3bet_pot=1, stack_depth_bb=100, pot_size_bb=22,
              flop_cards=["Kc", "4d", "2s"])),
    ]

    for label, kwargs in scenarios:
        r = evaluate_cbet(**kwargs)
        print(f"  {label}")
        print(f"    Expected profit : {r['expected_profit_pct_of_pot']:+.4f} (% of pot)")
        print(f"    Recommendation  : {r['recommendation']}")
        print(f"    Confidence      : {r['confidence']}")
        print(f"    Top drivers     : ", end="")
        top3 = list(r["feature_breakdown"].items())[:3]
        print("  |  ".join(f"{k} ({v:+.4f})" for k, v in top3))
        print()
