from pathlib import Path
import pandas as pd

DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def load_historic_raw(base_dir="data/raw/historic", seasons=None):
    """Load Historic Vaastav raw files without joining or transforming them.

    Returns
    -------
    dict
        {season: {"players": df, "gameweeks": df, "teams": df, "fixtures": df}}
    """
    base = Path(base_dir)
    seasons = seasons or DEFAULT_SEASONS
    out = {}

    for season in seasons:
        folder = base / season
        out[season] = {
            "players": pd.read_csv(folder / "players_raw.csv"),
            "gameweeks": pd.read_csv(folder / "merged_gw.csv"),
            "teams": pd.read_csv(folder / "teams.csv"),
            "fixtures": pd.read_csv(folder / "fixtures.csv"),
        }

    return out


if __name__ == "__main__":
    data = load_historic_raw()
    for season, tables in data.items():
        print(f"\n{season}")
        for name, df in tables.items():
            print(f"  {name:10s}: {len(df):,} rows x {len(df.columns):,} cols")
