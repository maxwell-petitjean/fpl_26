from pathlib import Path
import pandas as pd

DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def build_player_season_map(base_dir="data/raw/vaastav", seasons=None):
    """Build a season-level identity table from players_raw.csv only.

    This does NOT join gameweek files together. It simply creates the bridge
    needed later to map each season's FPL element ID to the cross-season code.
    """
    base = Path(base_dir)
    seasons = seasons or DEFAULT_SEASONS
    frames = []

    for season in seasons:
        df = pd.read_csv(base / season / "players_raw.csv")

        keep = [
            c for c in [
                "id", "code", "opta_code", "first_name", "second_name",
                "web_name", "team", "team_code", "element_type"
            ] if c in df.columns
        ]
        x = df[keep].copy()
        x.insert(0, "season", season)
        x = x.rename(columns={
            "id": "fpl_element_id",
            "code": "player_code",
            "team": "team_id",
            "element_type": "position_id",
        })

        if "opta_code" not in x.columns:
            x["opta_code"] = pd.NA
        if "team_code" not in x.columns:
            x["team_code"] = pd.NA

        x["position"] = x["position_id"].map(POSITION_MAP)
        frames.append(x)

    mapping = pd.concat(frames, ignore_index=True)

    # Basic safety checks.
    if mapping.duplicated(["season", "fpl_element_id"]).any():
        raise ValueError("Duplicate season + FPL element IDs found in player mapping")

    return mapping.sort_values(["player_code", "season", "fpl_element_id"])


def add_player_code_to_gameweek(gameweek_df, player_map, season):
    """Map one season's gameweek dataframe to player_code in memory.

    Keeps the raw CSV untouched. Use this later when building processed data.
    """
    season_map = player_map.loc[
        player_map["season"].eq(season),
        ["fpl_element_id", "player_code", "opta_code"]
    ].drop_duplicates()

    return gameweek_df.merge(
        season_map,
        how="left",
        left_on="element",
        right_on="fpl_element_id",
        validate="many_to_one",
    )


if __name__ == "__main__":
    mapping = build_player_season_map()
    print(mapping.head(20).to_string(index=False))
    print(f"\nRows: {len(mapping):,}")
    print(f"Unique player_code: {mapping['player_code'].nunique():,}")
    print(f"Missing player_code: {mapping['player_code'].isna().sum():,}")
