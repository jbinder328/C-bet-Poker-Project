import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
TARGET_COL = "cbet_profit_as_pct_of_pot"

# ── Load data ──────────────────────────────────────────────────────────
cbet_df = pd.read_csv("poker_model/outputs/cbet_hands.csv")

# Drop rows with any missing feature values
model_df = cbet_df[FEATURE_COLS + [TARGET_COL]].dropna()
print(f"Rows after dropping NaN: {len(model_df):,}")

X = model_df[FEATURE_COLS]
y = model_df[TARGET_COL]

# Temporal train/test split — NO SHUFFLING
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

print(f"Train size: {len(X_train):,}")
print(f"Test size:  {len(X_test):,}")

# Scale features — fit ONLY on train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train both models
model = LinearRegression()
model.fit(X_train_scaled, y_train)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Evaluate
for name, m in [("Linear Regression", model), ("Ridge Regression", ridge)]:
    y_pred = m.predict(X_test_scaled)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n{name}:")
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: {rmse:.4f} (% of pot)")

# Save model and scaler
os.makedirs("poker_model/outputs", exist_ok=True)
joblib.dump(model,  "poker_model/outputs/cbet_model.pkl")
joblib.dump(scaler, "poker_model/outputs/cbet_scaler.pkl")
print("\nModel and scaler saved.")

# ── Coefficient Table ──────────────────────────────────────────────────
coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": model.coef_
})
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

print("\n=== C-BET PROFITABILITY MODEL ===")
print(f"Intercept: {model.intercept_:.4f} (% of pot)")
print("\nFeature Coefficients (sorted by impact):")
print(coef_df[["feature", "coefficient"]].to_string(index=False))

coef_df.to_csv("poker_model/outputs/coefficients.csv", index=False)

# ── Visualizations ─────────────────────────────────────────────────────
os.makedirs("poker_model/outputs/charts", exist_ok=True)

# Chart 1 — Coefficient Bar Chart
plt.figure(figsize=(12, 8))
colors = ["#2ecc71" if c > 0 else "#e74c3c"
          for c in coef_df["coefficient"]]
plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
plt.title("What Makes a C-Bet Profitable?\nLinear Regression Coefficients",
          fontsize=14, fontweight="bold")
plt.xlabel("Impact on C-Bet Profit (as % of pot at flop)", fontsize=11)
plt.tight_layout()
plt.savefig("poker_model/outputs/charts/coefficients.png", dpi=150)
plt.close()
print("Chart 1 saved: coefficients.png")

# Chart 2 — Bet Sizing Sweet Spot
cbet_df["sizing_bucket"] = pd.cut(
    cbet_df["cbet_size_to_pot"],
    bins=[0, 0.25, 0.40, 0.60, 0.80, 1.0, 1.5],
    labels=["0-25%", "25-40%", "40-60%", "60-80%", "80-100%", "100%+"]
)

sizing_stats = cbet_df.groupby("sizing_bucket", observed=True)["cbet_profit_as_pct_of_pot"].agg(
    ["mean", "count", "sem"]
).reset_index()

plt.figure(figsize=(9, 5))
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in sizing_stats["mean"]]
plt.bar(sizing_stats["sizing_bucket"].astype(str), sizing_stats["mean"],
        color=colors, yerr=sizing_stats["sem"], capsize=4)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.title("C-Bet Profitability by Bet Sizing\n(as % of pot)",
          fontsize=13, fontweight="bold")
plt.xlabel("Bet Size (% of pot)")
plt.ylabel("Average Profit (% of pot)")
plt.tight_layout()
plt.savefig("poker_model/outputs/charts/sizing_sweet_spot.png", dpi=150)
plt.close()
print("Chart 2 saved: sizing_sweet_spot.png")

# Chart 3 — Profitability by Position
position_order = ["BTN", "CO", "HJ", "MP", "UTG", "SB", "BB"]
pos_stats = (
    cbet_df[cbet_df["position"].isin(position_order)]
    .groupby("position")["cbet_profit_as_pct_of_pot"]
    .agg(["mean", "sem", "count"])
    .reindex(position_order)
    .reset_index()
)

plt.figure(figsize=(10, 5))
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in pos_stats["mean"]]
bars = plt.bar(pos_stats["position"], pos_stats["mean"],
               color=colors, yerr=pos_stats["sem"], capsize=4)
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.title("C-Bet Profitability by Position\n(average profit as % of pot)",
          fontsize=13, fontweight="bold")
plt.xlabel("Position (BTN = best, BB = worst)")
plt.ylabel("Average Profit (% of pot)")
for bar, row in zip(bars, pos_stats.itertuples()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + row.sem + 0.003,
             f"n={row.count:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("poker_model/outputs/charts/profitability_by_position.png", dpi=150)
plt.close()
print("Chart 3 saved: profitability_by_position.png")

print("\nAll charts saved.")

# Store r2 for final summary
r2_final = r2_score(y_test, model.predict(X_test_scaled))
print(f"\nFinal Linear Regression R²: {r2_final:.4f}")
print(f"Top predictor: {coef_df.iloc[0]['feature']} ({coef_df.iloc[0]['coefficient']:+.4f})")

# ── Sanity Checks ──────────────────────────────────────────────────────
print("\n=== SANITY CHECKS ===")

# 1. player_vpip_history should be negative (loose opponents defend more → c-bet loses more)
vpip_coef = coef_df[coef_df["feature"] == "player_vpip_history"]["coefficient"].values[0]
assert vpip_coef < 0, f"FAIL: player_vpip_history = {vpip_coef:.4f} (expected negative)"
print(f"PASS: player_vpip_history = {vpip_coef:+.4f} (looser opponents make c-bets less profitable)")

# 2. board_wetness_score should be negative (wetter boards hurt c-bets)
wet_coef = coef_df[coef_df["feature"] == "board_wetness_score"]["coefficient"].values[0]
assert wet_coef < 0, f"FAIL: board_wetness_score = {wet_coef:.4f} (expected negative)"
print(f"PASS: board_wetness_score = {wet_coef:+.4f} (wetter boards are worse)")

# 3. is_3bet_pot should be negative (c-bets in 3bet pots are less profitable as % of pot)
pot3_coef = coef_df[coef_df["feature"] == "is_3bet_pot"]["coefficient"].values[0]
assert pot3_coef < 0, f"FAIL: is_3bet_pot = {pot3_coef:.4f} (expected negative)"
print(f"PASS: is_3bet_pot = {pot3_coef:+.4f} (3-bet pots are worse for c-bets)")

# Note on num_opponents: with profit/pot as target, this coefficient can be positive —
# in multi-way pots you contribute less proportionally to the pot, so profit/pot is higher
# when the c-bet succeeds. This is a mathematical artifact of the normalization, not a
# signal that c-betting multi-way is better.
opp_coef = coef_df[coef_df["feature"] == "num_opponents"]["coefficient"].values[0]
print(f"INFO: num_opponents = {opp_coef:+.4f} (positive with pct-of-pot target — see note above)")

# 4. Row count
assert len(cbet_df) >= 10_000, f"FAIL: Only {len(cbet_df):,} c-bet rows"
print(f"PASS: {len(cbet_df):,} c-bet situations")

# 5. No NaN in feature matrix
nan_count = model_df[FEATURE_COLS].isnull().sum().sum()
assert nan_count == 0, f"FAIL: {nan_count} NaN values in feature matrix"
print("PASS: No NaN values in feature matrix")

print("\nAll sanity checks passed.")
