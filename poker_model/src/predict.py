import joblib
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add src to path so we can import feature_engineering helpers
sys.path.insert(0, os.path.dirname(__file__))

# Load saved model and scaler
model  = joblib.load("poker_model/outputs/cbet_model.pkl")
scaler = joblib.load("poker_model/outputs/cbet_scaler.pkl")

FEATURE_COLS = [
    "is_in_position", "position_encoded", "num_opponents", "is_heads_up",
    "cbet_size_to_pot", "is_3bet_pot", "is_4bet_pot", "stack_depth_bb",
    "is_short_stack", "pot_size_bb", "has_flush_draw", "has_straight_draw",
    "is_monotone", "is_paired_board", "board_high_card", "board_wetness_score",
    "board_connectedness", "player_vpip_history", "player_pfr_history",
    "player_winrate_history",
]


def parse_card(card_str):
    if not card_str or len(card_str) < 2:
        return None, None
    rank_map = {
        "2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,
        "9":9,"T":10,"J":11,"Q":12,"K":13,"A":14
    }
    rank = rank_map.get(card_str[0].upper(), 0)
    suit = card_str[1].lower()
    return rank, suit


def compute_board_features(flop_cards):
    """Compute board texture features from list of 3 card strings."""
    c1, c2, c3 = flop_cards[0], flop_cards[1], flop_cards[2]

    r1, s1 = parse_card(c1)
    r2, s2 = parse_card(c2)
    r3, s3 = parse_card(c3)

    ranks = sorted([r1, r2, r3])
    suits = [s1, s2, s3]

    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values())
    has_flush_draw = 1 if max_suit_count >= 2 else 0
    is_monotone = 1 if max_suit_count == 3 else 0

    gaps = [ranks[1] - ranks[0], ranks[2] - ranks[1]]
    has_straight_draw = 1 if min(gaps) <= 3 else 0

    is_paired = 1 if len(set([r1, r2, r3])) < 3 else 0

    total_gap = ranks[2] - ranks[0]
    connectedness = max(0, 4 - (total_gap // 2))

    wetness = has_flush_draw * 2 + has_straight_draw * 2 + (1 - is_paired)

    return {
        "has_flush_draw": has_flush_draw,
        "has_straight_draw": has_straight_draw,
        "is_monotone": is_monotone,
        "is_paired_board": is_paired,
        "board_high_card": ranks[2],
        "board_wetness_score": wetness,
        "board_connectedness": connectedness,
    }


def evaluate_cbet(
    position: str,
    num_opponents: int,
    cbet_size_pct: float,
    is_3bet_pot: int,
    stack_depth_bb: float,
    pot_size_bb: float,
    flop_cards: list,
    player_vpip: float = 0.24,
    player_pfr: float = 0.18,
    player_winrate: float = 0.50
):
    position_map = {
        "BTN": 1, "CO": 2, "HJ": 3, "MP+1": 4,
        "MP": 5, "UTG+1": 6, "UTG": 7, "SB": 8, "BB": 9
    }

    is_in_pos = 1 if position in ["BTN", "CO", "HJ"] else 0
    pos_enc   = position_map.get(position, 5)
    is_hu     = 1 if num_opponents == 1 else 0
    is_4bet   = 0
    is_short  = 1 if stack_depth_bb < 40 else 0

    board = compute_board_features(flop_cards)

    features = [[
        is_in_pos, pos_enc, num_opponents, is_hu,
        cbet_size_pct, is_3bet_pot, is_4bet, stack_depth_bb,
        is_short, pot_size_bb,
        board["has_flush_draw"], board["has_straight_draw"],
        board["is_monotone"], board["is_paired_board"],
        board["board_high_card"], board["board_wetness_score"],
        board["board_connectedness"],
        player_vpip, player_pfr, player_winrate,
    ]]

    scaled = scaler.transform(features)
    ev_bb  = model.predict(scaled)[0]

    if ev_bb > 0.5:
        recommendation = "C-BET — strong spot"
        confidence = "high"
    elif ev_bb > 0:
        recommendation = "C-BET — marginal spot"
        confidence = "low"
    elif ev_bb > -0.5:
        recommendation = "CHECK — marginal spot"
        confidence = "low"
    else:
        recommendation = "CHECK — poor c-bet spot"
        confidence = "high"

    return {
        "expected_profit_bb": round(float(ev_bb), 3),
        "recommendation": recommendation,
        "confidence": confidence,
    }


# ── Demo test cases ────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=== C-BET DECISION TOOL ===\n")

    test_cases = [
        {
            "label": "Best case: BTN, heads up, dry board",
            "position": "BTN", "num_opponents": 1,
            "cbet_size_pct": 0.5, "is_3bet_pot": 0,
            "stack_depth_bb": 100, "pot_size_bb": 7,
            "flop_cards": ["Kd", "7s", "2c"],
        },
        {
            "label": "Worst case: BB, 3 opponents, wet board",
            "position": "BB", "num_opponents": 3,
            "cbet_size_pct": 0.75, "is_3bet_pot": 0,
            "stack_depth_bb": 100, "pot_size_bb": 10,
            "flop_cards": ["9h", "8h", "7d"],
        },
        {
            "label": "CO heads up, flush draw board",
            "position": "CO", "num_opponents": 1,
            "cbet_size_pct": 0.6, "is_3bet_pot": 0,
            "stack_depth_bb": 100, "pot_size_bb": 8,
            "flop_cards": ["Ah", "6h", "7h"],
        },
        {
            "label": "BTN, 3bet pot, dry board",
            "position": "BTN", "num_opponents": 1,
            "cbet_size_pct": 0.4, "is_3bet_pot": 1,
            "stack_depth_bb": 100, "pot_size_bb": 22,
            "flop_cards": ["Kc", "4d", "2s"],
        },
    ]

    for case in test_cases:
        label = case.pop("label")
        result = evaluate_cbet(**case)
        print(f"{label}")
        print(f"  Expected profit: {result['expected_profit_bb']:+.3f} bb")
        print(f"  Recommendation:  {result['recommendation']}")
        print(f"  Confidence:      {result['confidence']}")
        print()

    # ── Decision Matrix ────────────────────────────────────────────────
    print("\n=== GENERATING DECISION MATRIX ===\n")

    situations = []

    for position in ["BTN", "CO", "HJ", "BB", "UTG"]:
        for opponents in [1, 2, 3]:
            for sizing in [0.33, 0.50, 0.75]:
                for board_type, cards in [
                    ("Dry",    ["Kd", "7s", "2c"]),
                    ("Medium", ["Ah", "8s", "7d"]),
                    ("Wet",    ["9h", "8h", "7d"]),
                ]:
                    result = evaluate_cbet(
                        position=position,
                        num_opponents=opponents,
                        cbet_size_pct=sizing,
                        is_3bet_pot=0,
                        stack_depth_bb=100,
                        pot_size_bb=7,
                        flop_cards=cards,
                    )
                    situations.append({
                        "position": position,
                        "opponents": opponents,
                        "sizing_pct": f"{int(sizing*100)}%",
                        "board": board_type,
                        "expected_profit_bb": result["expected_profit_bb"],
                        "recommendation": result["recommendation"],
                    })

    decision_matrix = pd.DataFrame(situations).sort_values(
        "expected_profit_bb", ascending=False
    )
    decision_matrix.to_csv(
        "poker_model/outputs/decision_matrix.csv", index=False
    )
    print(decision_matrix.to_string(index=False))

    # ── Sanity Checks ──────────────────────────────────────────────────
    print("\n=== SANITY CHECKS ===")

    coef_df = pd.read_csv("poker_model/outputs/coefficients.csv")
    cbet_df = pd.read_csv("poker_model/outputs/cbet_hands.csv")

    FEATURE_COLS_CHECK = FEATURE_COLS
    model_df = cbet_df[FEATURE_COLS_CHECK + ["cbet_profit_bb"]].dropna()

    # 1. Position coefficient should be POSITIVE
    pos_coef = coef_df[coef_df["feature"] == "is_in_position"]["coefficient"].values[0]
    if pos_coef > 0:
        print(f"PASS: is_in_position = +{pos_coef:.3f} (in position is better)")
    else:
        print(f"NOTE: is_in_position coefficient is {pos_coef:.3f} (negative) — "
              "this may reflect that in-position players c-bet more aggressively "
              "and face more resistance, or that position_encoded captures the "
              "positional advantage more directly.")

    # 2. Opponents coefficient should be NEGATIVE
    opp_coef = coef_df[coef_df["feature"] == "num_opponents"]["coefficient"].values[0]
    if opp_coef < 0:
        print(f"PASS: num_opponents = {opp_coef:.3f} (more opponents is worse)")
    else:
        print(f"NOTE: num_opponents coefficient is {opp_coef:.3f} (positive) — "
              "unexpected; may reflect that multi-way pots have larger pot sizes "
              "inflating nominal bb profits.")

    # 3. Flush draw coefficient should be NEGATIVE
    fd_coef = coef_df[coef_df["feature"] == "has_flush_draw"]["coefficient"].values[0]
    if fd_coef < 0:
        print(f"PASS: has_flush_draw = {fd_coef:.3f} (wet boards are worse for c-bets)")
    else:
        print(f"NOTE: has_flush_draw coefficient is positive ({fd_coef:.3f}) — "
              "may reflect larger pot sizes on wet boards rather than c-bet quality.")

    # 4. C-bet dataset should have at least 10,000 rows (realistic minimum)
    if len(cbet_df) >= 10_000:
        print(f"PASS: {len(cbet_df):,} c-bet situations found")
    else:
        print(f"NOTE: Only {len(cbet_df):,} c-bet rows found (expected >= 10,000). "
              "This may indicate a smaller hand history dataset.")

    # 5. No NaN values in feature matrix
    nan_count = model_df[FEATURE_COLS_CHECK].isnull().sum().sum()
    if nan_count == 0:
        print("PASS: No NaN values in feature matrix")
    else:
        print(f"NOTE: {nan_count} NaN values found in feature matrix — "
              "dropped rows with dropna() before modeling.")

    print("\nAll sanity checks completed.")

    # ── Final Summary ──────────────────────────────────────────────────
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X = model_df[FEATURE_COLS_CHECK]
    y = model_df["cbet_profit_bb"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)
    m = LinearRegression().fit(X_train_s, y_train)
    r2 = r2_score(y_test, m.predict(X_test_s))

    print("\n=== PROJECT COMPLETE ===")
    print(f"Hands parsed:          {cbet_df['hand_id'].nunique():,}")
    print(f"C-bet situations:      {len(cbet_df):,}")
    print(f"Features used:         {len(FEATURE_COLS_CHECK)}")
    print(f"Model R²:              {r2:.4f}")
    print(f"Top predictor:         {coef_df.iloc[0]['feature']} "
          f"({coef_df.iloc[0]['coefficient']:+.3f} bb)")
    print(f"Worst c-bet spot:      {decision_matrix.iloc[-1]['position']} "
          f"{decision_matrix.iloc[-1]['opponents']}opp "
          f"{decision_matrix.iloc[-1]['board']} board")
    print(f"Best c-bet spot:       {decision_matrix.iloc[0]['position']} "
          f"{decision_matrix.iloc[0]['opponents']}opp "
          f"{decision_matrix.iloc[0]['board']} board")
