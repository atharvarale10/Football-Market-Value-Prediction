# Football Player Market Value Prediction System

A Streamlit scouting desk that predicts football player market value and projects future transfer value under different career scenarios.

## What It Includes

- A current Transfermarkt top-50 real-player market-value snapshot plus 7,500 reproducible synthetic player snapshots.
- A gradient boosting regression model trained on log market value.
- Features covering age, position, potential, fitness, injuries, minutes, goals, assists, xG/xA, wages, contract length, club reputation, league strength, form, European minutes, and sell-on potential.
- A transfer-market style Streamlit app with a market board, player valuation desk, future value forecast, comparable players, negotiation levers, and CSV batch valuation.

## Run

```powershell
python train_model.py
streamlit run app.py
```

The training script creates:

- `data/football_player_market_values.csv`
- `models/market_value_model.joblib`

## Dataset Note

The included data combines:

- `real_transfermarkt_top50`: real player names, clubs, positions, ages, and current market values from Transfermarkt's most valuable players table.
- `synthetic_training_sample`: synthetic but structurally realistic rows modeled after common football valuation signals: Transfermarkt market value as a target concept, Sofifa-style player attributes, contract context, player availability, recent production, and club/league strength.

Only the real-player names, clubs, positions, ages, and market values should be treated as directly sourced. The extra fitness/performance fields attached to those real players are model features estimated for demo/training compatibility.

For production use, replace the generated CSV with a licensed dataset from a provider such as Transfermarkt, Wyscout, StatsBomb, Opta, SkillCorner, or Sofifa/FIFA data exports, then retrain with `train_model.py`.
