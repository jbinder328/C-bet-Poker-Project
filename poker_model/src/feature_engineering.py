"""
Feature Engineering for Poker C-Bet Profitability Model
Task 4: Filter to c-bet situations and engineer all features
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df = pd.read_csv("poker_model/outputs/all_hands.csv")

# ─────────────────────────────────────────────
# Step 4a — Filter to C-Bet Situations
# ─────────────────────────────────────────────
cbet_df = df[
    (df["pfr"] == 1) &
    (df["saw_flop"] == 1) &
    (df["flop_bet_made"] == 1) &
    (df["flop_card_1"].notna())
].copy()

print(f"Total hands parsed: {len(df):,}")
print(f"C-bet situations: {len(cbet_df):,}")

# ─────────────────────────────────────────────
# Step 4b — Target Variable
# ─────────────────────────────────────────────
cbet_df["cbet_profit_bb"] = (
    cbet_df["net_profit"] -
    (cbet_df["preflop_raise_amount"] * -1)
) / cbet_df["big_blind"]

cbet_df["cbet_profitable"] = (cbet_df["cbet_profit_bb"] > 0).astype(int)

# ─────────────────────────────────────────────
# Step 4c — Position Features
# ─────────────────────────────────────────────
cbet_df["is_in_position"] = cbet_df["position"].isin(
    ["BTN", "CO", "HJ"]
).astype(int)

position_map = {
    "BTN": 1, "CO": 2, "HJ": 3, "MP+1": 4,
    "MP": 5, "UTG+1": 6, "UTG": 7, "SB": 8, "BB": 9
}
cbet_df["position_encoded"] = cbet_df["position"].map(position_map).fillna(5)

# ─────────────────────────────────────────────
# Step 4d — Opponent Features
# ─────────────────────────────────────────────
cbet_df["num_opponents"] = cbet_df["players_to_flop"] - 1
cbet_df["is_heads_up"] = (cbet_df["players_to_flop"] == 2).astype(int)

# ─────────────────────────────────────────────
# Step 4e — Bet Sizing Feature
# ─────────────────────────────────────────────
# If pot_size_at_flop == 0, set to NaN and drop those rows
cbet_df.loc[cbet_df["pot_size_at_flop"] == 0, "cbet_size_to_pot"] = np.nan
cbet_df = cbet_df[cbet_df["pot_size_at_flop"] != 0].copy()

cbet_df["cbet_size_to_pot"] = (
    cbet_df["flop_bet_size"] / cbet_df["pot_size_at_flop"]
).clip(0, 2)

median_sizing = cbet_df["cbet_size_to_pot"].median()
cbet_df["cbet_size_to_pot"] = cbet_df["cbet_size_to_pot"].fillna(median_sizing)

# ─────────────────────────────────────────────
# Step 4f — Pot Type Features
# ─────────────────────────────────────────────
cbet_df["is_3bet_pot"] = (cbet_df["preflop_raises_count"] >= 2).astype(int)
cbet_df["is_4bet_pot"] = (cbet_df["preflop_raises_count"] >= 3).astype(int)

# ─────────────────────────────────────────────
# Step 4g — Stack Depth Feature
# ─────────────────────────────────────────────
cbet_df["stack_depth_bb"] = (
    cbet_df["starting_stack"] / cbet_df["big_blind"]
).clip(0, 500)
cbet_df["is_short_stack"] = (cbet_df["stack_depth_bb"] < 40).astype(int)

# ─────────────────────────────────────────────
# Step 4h — Pot Size Feature
# ─────────────────────────────────────────────
cbet_df["pot_size_bb"] = (
    cbet_df["pot_size_at_flop"] / cbet_df["big_blind"]
).clip(0, 200)

# ─────────────────────────────────────────────
# Step 4i — Board Texture Features
# ─────────────────────────────────────────────
def parse_card(card_str):
    """Parse a card string like 'Ah' into (rank, suit)"""
    if not card_str or len(card_str) < 2:
        return None, None
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
        "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }
    rank = rank_map.get(card_str[0].upper(), 0)
    suit = card_str[1].lower()
    return rank, suit


def compute_board_features(row):
    """Compute all board texture features from three flop cards"""
    c1 = row["flop_card_1"]
    c2 = row["flop_card_2"]
    c3 = row["flop_card_3"]

    if not all([c1, c2, c3]):
        return {
            "has_flush_draw": 0,
            "has_straight_draw": 0,
            "is_monotone": 0,
            "is_paired_board": 0,
            "board_high_card": 7,
            "board_low_card": 7,
            "board_connectedness": 0,
            "board_wetness_score": 0,
        }

    r1, s1 = parse_card(c1)
    r2, s2 = parse_card(c2)
    r3, s3 = parse_card(c3)

    ranks = sorted([r1, r2, r3])
    suits = [s1, s2, s3]

    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values())
    has_flush_draw = 1 if max_suit_count >= 2 else 0
    is_monotone = 1 if max_suit_count == 3 else 0

    gaps = [ranks[1] - ranks[0], ranks[2] - ranks[1], ranks[2] - ranks[0]]
    has_straight_draw = 1 if min(gaps[:2]) <= 3 else 0

    is_paired = 1 if len(set([r1, r2, r3])) < 3 else 0

    total_gap = ranks[2] - ranks[0]
    connectedness = max(0, 4 - (total_gap // 2))

    wetness = (
        has_flush_draw * 2 +
        has_straight_draw * 2 +
        (1 - is_paired)
    )

    return {
        "has_flush_draw": has_flush_draw,
        "has_straight_draw": has_straight_draw,
        "is_monotone": is_monotone,
        "is_paired_board": is_paired,
        "board_high_card": ranks[2],
        "board_low_card": ranks[0],
        "board_connectedness": connectedness,
        "board_wetness_score": wetness,
    }


board_features = cbet_df.apply(compute_board_features, axis=1)
board_df = pd.DataFrame(board_features.tolist(), index=cbet_df.index)
cbet_df = pd.concat([cbet_df, board_df], axis=1)

# ─────────────────────────────────────────────
# Step 4j — Player History Features
# ─────────────────────────────────────────────
# CRITICAL: Sort by timestamp FIRST, use shift(1) to prevent data leakage.
# Compute vpip/pfr history from the FULL dataset (not cbet_df) so the features
# reflect a player's true tendencies across all hands, not just c-bet hands
# (which would always be 1.0 since pfr==1 is a c-bet requirement).

df["timestamp"] = pd.to_datetime(df["timestamp"])
df_sorted = df.sort_values("timestamp").reset_index(drop=True)


def rolling_player_stat(group, col):
    return group[col].shift(1).expanding(min_periods=5).mean()


# Compute rolling vpip/pfr from all hands per player
vpip_history = df_sorted.groupby("player_name", group_keys=False).apply(
    lambda g: rolling_player_stat(g, "vpip")
)
pfr_history = df_sorted.groupby("player_name", group_keys=False).apply(
    lambda g: rolling_player_stat(g, "pfr")
)

df_sorted["_vpip_hist"] = vpip_history
df_sorted["_pfr_hist"] = pfr_history

# Keep only the latest history value per (player, hand) — one row per hand per player
history_map = df_sorted.set_index(["hand_id", "player_name"])[["_vpip_hist", "_pfr_hist"]]

cbet_df["timestamp"] = pd.to_datetime(cbet_df["timestamp"])
cbet_df = cbet_df.sort_values("timestamp").reset_index(drop=True)

cbet_df = cbet_df.join(
    history_map,
    on=["hand_id", "player_name"],
    how="left"
)
cbet_df["player_vpip_history"] = cbet_df["_vpip_hist"].fillna(df["vpip"].mean())
cbet_df["player_pfr_history"] = cbet_df["_pfr_hist"].fillna(df["pfr"].mean())
cbet_df = cbet_df.drop(columns=["_vpip_hist", "_pfr_hist"])

# Winrate history stays in cbet_df (rolling c-bet profitability)
cbet_df["player_winrate_history"] = cbet_df.groupby(
    "player_name", group_keys=False
).apply(lambda g: rolling_player_stat(g, "cbet_profitable"))

cbet_df["player_winrate_history"] = cbet_df["player_winrate_history"].fillna(0.5)

# ─────────────────────────────────────────────
# Step 4k — Final Feature List and Save
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "is_in_position", "position_encoded",
    "num_opponents", "is_heads_up",
    "cbet_size_to_pot",
    "is_3bet_pot", "is_4bet_pot",
    "stack_depth_bb", "is_short_stack",
    "pot_size_bb",
    "has_flush_draw", "has_straight_draw",
    "is_monotone", "is_paired_board",
    "board_high_card", "board_wetness_score",
    "board_connectedness",
    "player_vpip_history", "player_pfr_history",
    "player_winrate_history",
]

TARGET_COL = "cbet_profit_bb"

cbet_df.to_csv("poker_model/outputs/cbet_hands.csv", index=False)
print(f"C-bet dataset saved: {len(cbet_df):,} rows, {len(FEATURE_COLS)} features")

# ─────────────────────────────────────────────
# Summary Stats
# ─────────────────────────────────────────────
print("\n=== Feature Summary Stats ===")
summary_cols = FEATURE_COLS + [TARGET_COL, "cbet_profitable"]
print(cbet_df[summary_cols].describe().round(4).to_string())

print("\n=== Target Variable Distribution ===")
print(f"cbet_profitable = 1 (profitable): {cbet_df['cbet_profitable'].sum():,} ({cbet_df['cbet_profitable'].mean():.1%})")
print(f"cbet_profitable = 0 (not profitable): {(1 - cbet_df['cbet_profitable']).sum():,} ({(1 - cbet_df['cbet_profitable']).mean():.1%})")
print(f"\ncbet_profit_bb mean: {cbet_df['cbet_profit_bb'].mean():.4f}")
print(f"cbet_profit_bb std:  {cbet_df['cbet_profit_bb'].std():.4f}")
print(f"cbet_profit_bb median: {cbet_df['cbet_profit_bb'].median():.4f}")

print("\n=== Key Feature Means ===")
key_features = [
    "is_in_position", "is_heads_up", "cbet_size_to_pot",
    "is_3bet_pot", "stack_depth_bb", "pot_size_bb",
    "has_flush_draw", "has_straight_draw", "is_monotone",
    "board_wetness_score", "player_winrate_history"
]
for col in key_features:
    print(f"  {col}: mean={cbet_df[col].mean():.4f}, std={cbet_df[col].std():.4f}")
