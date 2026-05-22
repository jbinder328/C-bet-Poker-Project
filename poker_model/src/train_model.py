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
TARGET_COL = "cbet_profit_bb"

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
    print(f"  RMSE: {rmse:.4f} bb")

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
print(f"Intercept: {model.intercept_:.4f} bb")
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
plt.xlabel("Impact on C-Bet Expected Profit (big blinds)", fontsize=11)
plt.tight_layout()
plt.savefig("poker_model/outputs/charts/coefficients.png", dpi=150)
plt.close()
print("Chart 1 saved: coefficients.png")

# Chart 2 — Profitability by Position × Board Texture
cbet_df["position_type"] = cbet_df["is_in_position"].map(
    {1: "In Position", 0: "Out of Position"}
)
cbet_df["board_type"] = pd.cut(
    cbet_df["board_wetness_score"],
    bins=[-1, 1, 3, 5],
    labels=["Dry (0-1)", "Medium (2-3)", "Wet (4-5)"]
)

pivot = cbet_df.groupby(
    ["position_type", "board_type"], observed=True
)["cbet_profit_bb"].mean().unstack()

pivot.plot(kind="bar", figsize=(10, 6), colormap="RdYlGn")
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.title("C-Bet Profitability: Position × Board Texture",
          fontsize=13, fontweight="bold")
plt.xlabel("Position")
plt.ylabel("Average Profit (big blinds)")
plt.legend(title="Board Type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("poker_model/outputs/charts/position_board_texture.png", dpi=150)
plt.close()
print("Chart 2 saved: position_board_texture.png")

# Chart 3 — Profitability by Number of Opponents
opp_profits = cbet_df.groupby("num_opponents")["cbet_profit_bb"].agg(
    ["mean", "count"]
).reset_index()

fig, ax1 = plt.subplots(figsize=(9, 5))
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in opp_profits["mean"]]
bars = ax1.bar(opp_profits["num_opponents"], opp_profits["mean"], color=colors)
ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
ax1.set_xlabel("Number of Opponents Facing C-Bet")
ax1.set_ylabel("Average Profit (big blinds)")
ax1.set_title("C-Bet Profitability by Number of Opponents",
              fontsize=13, fontweight="bold")

for bar, count in zip(bars, opp_profits["count"]):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"n={count:,}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("poker_model/outputs/charts/opponents.png", dpi=150)
plt.close()
print("Chart 3 saved: opponents.png")

# Chart 4 — Bet Sizing Sweet Spot
cbet_df["sizing_bucket"] = pd.cut(
    cbet_df["cbet_size_to_pot"],
    bins=[0, 0.25, 0.40, 0.60, 0.80, 1.0, 1.5],
    labels=["0-25%", "25-40%", "40-60%", "60-80%", "80-100%", "100%+"]
)

sizing_stats = cbet_df.groupby("sizing_bucket", observed=True)["cbet_profit_bb"].agg(
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
plt.ylabel("Average Profit (big blinds)")
plt.tight_layout()
plt.savefig("poker_model/outputs/charts/sizing_sweet_spot.png", dpi=150)
plt.close()
print("Chart 4 saved: sizing_sweet_spot.png")

print("\nAll charts saved.")

# Store r2 for final summary
r2_final = r2_score(y_test, model.predict(X_test_scaled))
print(f"\nFinal Linear Regression R²: {r2_final:.4f}")
print(f"Top predictor: {coef_df.iloc[0]['feature']} ({coef_df.iloc[0]['coefficient']:+.3f} bb)")
