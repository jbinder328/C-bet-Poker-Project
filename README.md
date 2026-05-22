# C-Bet Poker Project

A linear regression model that predicts the profitability of continuation bets (c-bets) in No-Limit Hold'em poker using real PokerStars $1/$2 hand history data.

## What It Does

A **c-bet** is when the preflop raiser bets again on the flop. This project:
- Parses raw PokerStars hand histories into structured data
- Engineers features from board texture, position, bet sizing, and player tendencies
- Trains a linear regression model to predict c-bet profitability as a % of the pot
- Outputs a ranked coefficient table and decision matrix showing which spots are most/least profitable

## Results

**19,317 c-bet situations** extracted from 75,580 parsed hands.

**Top predictors of c-bet profitability (% of pot):**

| Feature | Coefficient | Interpretation |
|---|---|---|
| `player_vpip_history` | −0.070 | Looser opponents defend more → c-bets less profitable |
| `cbet_size_to_pot` | +0.069 | Larger bets capture more pot when successful |
| `is_3bet_pot` | −0.035 | 3-bet pots are harder spots to c-bet |
| `is_heads_up` | +0.034 | HU c-bets are more profitable than multiway |
| `board_wetness_score` | −0.007 | Wet boards (flush/straight draws) reduce c-bet EV |

**Best c-bet spot:** BTN, 1 opponent, dry board  
**Worst c-bet spot:** BB, 3 opponents, wet board

## Project Structure

```
poker_model/
    src/
        extract_data.py         # Extract cash game files from zip
        parse_hands.py          # Parse hand histories → all_hands.csv
        feature_engineering.py  # Engineer features → cbet_hands.csv
        train_model.py          # Train model → model files + charts
        predict.py              # Prediction function + coefficient table
    outputs/
        cbet_model.pkl          # Trained LinearRegression model
        cbet_scaler.pkl         # Fitted StandardScaler
        coefficients.csv        # Feature coefficients ranked by impact
        decision_matrix.csv     # EV table across 135 position/board situations
        charts/
            coefficients.png
            position_board_texture.png
            opponents.png
            sizing_sweet_spot.png
```

## Running the Pipeline

```bash
# 1. Extract raw hand history files from zip
python3 poker_model/src/extract_data.py

# 2. Parse all hand histories into CSV (~75k hands, takes ~25 sec)
python3 poker_model/src/parse_hands.py

# 3. Engineer features and build c-bet dataset
python3 poker_model/src/feature_engineering.py

# 4. Train model, generate charts and coefficient table
python3 poker_model/src/train_model.py

# 5. Run predictions on example hands
python3 poker_model/src/predict.py
```

## Using the Prediction Function

```python
from poker_model.src.predict import evaluate_cbet

result = evaluate_cbet(
    position="BTN",         # BTN, CO, HJ, MP, UTG, SB, BB
    num_opponents=1,        # opponents seeing the flop
    cbet_size_pct=0.5,      # bet size as fraction of pot
    is_3bet_pot=0,          # 1 if 3-bet preflop
    stack_depth_bb=100,     # starting stack in big blinds
    pot_size_bb=7.0,        # pot size at flop in big blinds
    flop_cards=["Kd", "7s", "2c"],
)

print(result["recommendation"])      # "C-BET — strong spot"
print(result["expected_profit_bb"])  # predicted profit
print(result["feature_breakdown"])   # per-feature contribution
```

## Target Variable

`cbet_profit_as_pct_of_pot = net_profit / pot_size_at_flop`, clipped to ±3.

Values beyond ±3 come from stack-off situations (e.g., shoving 100bb into a 7bb flop pot) where the ratio is not meaningful as a c-bet quality signal. The ±3 clip retains 90%+ of all hands.

## Dependencies

```
pandas numpy scikit-learn matplotlib seaborn joblib
```
