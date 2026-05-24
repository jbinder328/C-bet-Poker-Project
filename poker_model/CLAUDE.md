# Poker C-Bet Model

## Project Structure
- `data/raw/` — raw PokerStars hand history .txt files (1/2 cash game only)
- `outputs/` — generated CSVs, model files, charts
- `src/` — Python source files

## Key Files
- `src/parse_hands.py` — parses raw txt files → `outputs/all_hands.csv`
- `src/feature_engineering.py` — transforms all_hands.csv → `outputs/cbet_hands.csv`
- `src/train_model.py` — trains model → `outputs/cbet_model.pkl` + charts
- `src/predict.py` — prediction function + decision matrix

## Running
```bash
python src/parse_hands.py       # Step 1: parse raw files
python src/feature_engineering.py  # Step 2: engineer features
python src/train_model.py       # Step 3: train + evaluate + charts
python src/predict.py           # Step 4: run demo predictions
```

## Notes
- All monetary values in USD
- Target variable: `cbet_profit_as_pct_of_pot` (net_profit / pot_size_at_flop, clipped ±3)
- Model predicts c-bet profitability as a fraction of the pot at flop
