# Vaastav FPL raw data

Source: `vaastav/Fantasy-Premier-League` on GitHub.

These files are copied raw and should not be edited in place.

Each season contains:

- `players_raw.csv` — season-level player master and identifiers.
- `merged_gw.csv` — player x gameweek outcomes/statistics.
- `teams.csv` — team reference data.
- `fixtures.csv` — fixture reference/results data.

Seasons currently retained: 2021-22 through 2025-26.

Important modelling note: do not use the Vaastav `xP` field unlagged as a predictive feature. The repository documents potential post-gameweek lookahead in that field.
