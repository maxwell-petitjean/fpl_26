# FPL Prediction & Optimisation

Starter repository for the FPL expected-points model, optimiser, and Streamlit app.

## Model Pipeline

The project is designed as a sequential pipeline. Each stage creates an output used by later stages.

| Step | Stage                    | Script                                                       | Main output                                                 | Purpose                                                                            |
| ---: | ------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
|    1 | Historic data            | `src/data/build_fct_gw_historic.py`                          | `data/processed/fct_gw_historic.csv`                        | Creates the canonical historic player × gameweek dataset                           |
|    2 | Historic player features | `src/features/build_historic_features.py`                    | `data/features/fct_gw_features_historic.csv`                | Creates lagged and rolling player features used by the prediction models           |
|    3 | Opponent features        | `src/features/build_opponent_position_features.py`           | `data/features/fct_opponent_position_features_historic.csv` | Measures points historically allowed by each opponent to GK, DEF and ATT players   |
|    4 | xPP90 modelling dataset  | `src/features/build_xpp90_modelling_dataset.py`              | xPP90 modelling dataset                                     | Combines player and opponent features into the modelling grain                     |
|    5 | Current FPL data         | `src/data/fetch_current_fpl.py`                              | Current players and upcoming fixtures                       | Pulls the latest FPL player, price, availability and fixture information           |
|    6 | Future features          | `src/features/build_future_features.py`                      | Future player × fixture feature dataset                     | Applies historic player features and opponent features to upcoming fixtures        |
|    7 | Minutes prediction       | Minutes model                                                | Predicted minutes by player × fixture                       | Estimates expected playing time                                                    |
|    8 | xPP90 prediction         | xPP90 model                                                  | Predicted core PP90                                         | Estimates underlying FPL scoring rate conditional on playing                       |
|    9 | DefCon prediction        | DefCon model                                                 | DefCon probability / expected DefCon points                 | Estimates probability of receiving defensive-contribution points                   |
|   10 | Expected points          | `src/models/run_xp_pipeline.py` / future prediction pipeline | Player × fixture `xp`                                       | Combines minutes, core PP90 and DefCon into expected FPL points                    |
|   11 | Solver pool              | `src/optimizer/build_solver_pool.py`                         | `data/outputs/solver/solver_pool.csv`                       | Aggregates fixture predictions to one unique row per `player_code` with GW1–GW8 xP |
|   12 | Squad optimisation       | `src/optimizer/optimize_squad.py`                            | `optimal_squad.csv` + `optimal_lineups.csv`                 | Selects the optimal legal FPL squad and starting XI for each forecast GW           |

### Prediction logic

At a high level:

`Historic data → Features → Models → Future fixtures → Expected points → Optimisation`

The modelling components are intentionally kept separate:

* **Minutes model** — how much is the player expected to play?
* **xPP90 model** — how productive is the player expected to be while playing?
* **DefCon model** — what is the probability of earning defensive-contribution points?
* **xP pipeline** — combines these predictions into fixture-level expected points.
* **Solver** — uses the predictions but does not alter them.

### Solver grain

`player_code` is the canonical player identifier throughout the solver pipeline.

`web_name` must **not** be used as a unique key because multiple FPL players can have the same display name.

The solver pool therefore has exactly:

**1 row = 1 unique `player_code`**

with forecast columns such as:

`xp_gw1`, `xp_gw2`, ..., `xp_gw8`, `xp_8gw`, `weighted_xp_8gw`.

### Fresh Colab run order

When starting from a fresh Colab runtime, run the pipeline in this order:

1. Clone / pull repository and install requirements
2. Build historic GW data
3. Build historic player features
4. Build opponent-position features
5. Build xPP90 modelling dataset
6. Fetch current FPL data
7. Build future features
8. Generate future minutes, xPP90 and DefCon predictions
9. Calculate fixture-level expected points
10. Build the solver pool
11. Run the squad optimiser

Intermediate outputs should not be assumed to exist in a fresh Colab runtime.
