from pathlib import Path

import pandas as pd

from src.data.load_historic import DEFAULT_SEASONS, load_historic_raw
from src.data.map_players import build_player_season_map


OUTPUT_PATH = Path("data/processed/fct_gw_historic.csv")

# These are match-level facts we want to preserve whenever Vaastav provides them.
OPTIONAL_MATCH_COLUMNS = [
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "modified",
]

CORE_MATCH_COLUMNS = [
    "element",
    "fixture",
    "GW",
    "round",
    "name",
    "position",
    "team",
    "opponent_team",
    "was_home",
    "kickoff_time",
    "value",
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "saves",
    "yellow_cards",
    "red_cards",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "team_h_score",
    "team_a_score",
]

POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]


def _normalise_gameweek_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single integer gameweek column called gameweek."""
    out = df.copy()

    if "GW" in out.columns:
        out["gameweek"] = pd.to_numeric(out["GW"], errors="coerce")
    elif "round" in out.columns:
        out["gameweek"] = pd.to_numeric(out["round"], errors="coerce")
    else:
        raise ValueError("Historic gameweek data contains neither GW nor round.")

    if out["gameweek"].isna().any():
        raise ValueError("Null/non-numeric gameweek values found.")

    out["gameweek"] = out["gameweek"].astype(int)
    return out


def _team_lookup(teams: pd.DataFrame) -> pd.DataFrame:
    """Create a season-local team lookup."""
    required = {"id", "name"}
    missing = required - set(teams.columns)
    if missing:
        raise ValueError(f"teams.csv missing required columns: {sorted(missing)}")

    keep = [c for c in ["id", "code", "name", "short_name"] if c in teams.columns]
    t = teams[keep].copy()

    rename = {
        "id": "team_id",
        "code": "team_code",
        "name": "team_name",
        "short_name": "team_short_name",
    }
    return t.rename(columns=rename).drop_duplicates("team_id")


def _build_one_season(
    season: str,
    gameweeks: pd.DataFrame,
    teams: pd.DataFrame,
    player_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build one season of the historic player-fixture fact table."""
    gw = _normalise_gameweek_column(gameweeks)

    required = {"element", "fixture", "name", "position", "team", "opponent_team"}
    missing = required - set(gw.columns)
    if missing:
        raise ValueError(f"{season}: merged_gw.csv missing columns: {sorted(missing)}")

    # Deliberately exclude Vaastav xP because it may contain post-match lookahead.
    keep = [
        c
        for c in CORE_MATCH_COLUMNS + OPTIONAL_MATCH_COLUMNS
        if c in gw.columns and c != "xP"
    ]
    gw = gw[keep].copy()
    gw = _normalise_gameweek_column(gw)

    # Identity bridge: season-specific FPL element -> stable player_code.
    season_players = (
        player_map.loc[
            player_map["season"].eq(season),
            [
                "fpl_element_id",
                "player_code",
                "opta_code",
                "web_name",
            ],
        ]
        .drop_duplicates("fpl_element_id")
    )

    gw = gw.merge(
        season_players,
        how="left",
        left_on="element",
        right_on="fpl_element_id",
        validate="many_to_one",
    )

    if gw["player_code"].isna().any():
        sample = gw.loc[gw["player_code"].isna(), ["element", "name"]].head(10)
        raise ValueError(
            f"{season}: gameweek rows failed player_code mapping.\n{sample.to_string(index=False)}"
        )

    # Map team names from the actual GW row, rather than using end-of-season
    # player master team. This preserves mid-season transfers correctly.
    teams_lu = _team_lookup(teams)

    team_by_name = teams_lu.rename(
        columns={
            "team_id": "team_id",
            "team_code": "team_code",
            "team_name": "team",
            "team_short_name": "team_short_name",
        }
    )

    gw = gw.merge(
        team_by_name,
        how="left",
        on="team",
        validate="many_to_one",
    )

    opponent_lu = teams_lu.rename(
        columns={
            "team_id": "opponent_team",
            "team_code": "opponent_team_code",
            "team_name": "opponent_team_name",
            "team_short_name": "opponent_team_short_name",
        }
    )

    gw = gw.merge(
        opponent_lu,
        how="left",
        on="opponent_team",
        validate="many_to_one",
    )

    gw.insert(0, "season", season)

    # Clean types.
    gw["kickoff_time"] = pd.to_datetime(gw.get("kickoff_time"), utc=True, errors="coerce")
    gw["was_home"] = gw["was_home"].astype("boolean")
    gw["position"] = pd.Categorical(
        gw["position"], categories=POSITION_ORDER, ordered=True
    )

    # Price is supplied in tenths of £m by FPL.
    if "value" in gw.columns:
        gw["price"] = pd.to_numeric(gw["value"], errors="coerce") / 10.0

    # 2025/26 onwards: derive whether the player earned DefCon points.
    # Defenders: 10+ CBIT. MID/FWD: 12+ CBIRT.
    # GK are not eligible.
    if "defensive_contribution" in gw.columns:
        dc = pd.to_numeric(gw["defensive_contribution"], errors="coerce")
        gw["defensive_contribution"] = dc

        gw["defcon_hit"] = pd.Series(pd.NA, index=gw.index, dtype="Int64")

        eligible = dc.notna() & gw["position"].isin(["DEF", "MID", "FWD"])
        gw.loc[eligible & gw["position"].eq("DEF"), "defcon_hit"] = (
            dc[eligible & gw["position"].eq("DEF")] >= 10
        ).astype(int)
        gw.loc[eligible & gw["position"].isin(["MID", "FWD"]), "defcon_hit"] = (
            dc[eligible & gw["position"].isin(["MID", "FWD"])] >= 12
        ).astype(int)

        gw["defcon_points"] = gw["defcon_hit"] * 2
    else:
        gw["defensive_contribution"] = pd.NA
        gw["defcon_hit"] = pd.Series(pd.NA, index=gw.index, dtype="Int64")
        gw["defcon_points"] = pd.Series(pd.NA, index=gw.index, dtype="Int64")

    # Useful modelling target later: remove separately-modelled DefCon points.
    # Before 2025/26 this simply equals total_points.
    gw["core_total_points"] = pd.to_numeric(gw["total_points"], errors="coerce")
    has_dc = gw["defcon_points"].notna()
    gw.loc[has_dc, "core_total_points"] = (
        gw.loc[has_dc, "core_total_points"]
        - gw.loc[has_dc, "defcon_points"].astype(float)
    )

    # Canonical column names.
    gw = gw.rename(
        columns={
            "element": "fpl_element_id_gw",
            "fixture": "fixture_id",
            "name": "player_name",
            "team": "team_name",
            "opponent_team": "opponent_team_id",
        }
    )

    # One row per player per fixture. This intentionally allows a player to
    # have multiple rows in the same gameweek during a DGW.
    grain = ["season", "gameweek", "player_code", "fixture_id"]
    dupes = gw.duplicated(grain, keep=False)
    if dupes.any():
        sample = gw.loc[dupes, grain + ["player_name"]].head(20)
        raise ValueError(
            f"{season}: duplicate fact-table grain detected.\n"
            f"{sample.to_string(index=False)}"
        )

    preferred_order = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "fpl_element_id_gw",
        "fpl_element_id",
        "opta_code",
        "web_name",
        "player_name",
        "position",
        "team_id",
        "team_code",
        "team_name",
        "team_short_name",
        "opponent_team_id",
        "opponent_team_code",
        "opponent_team_name",
        "opponent_team_short_name",
        "was_home",
        "price",
        "minutes",
        "starts",
        "total_points",
        "core_total_points",
        "defensive_contribution",
        "defcon_hit",
        "defcon_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "saves",
        "yellow_cards",
        "red_cards",
        "bonus",
        "bps",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "tackles",
        "recoveries",
        "clearances_blocks_interceptions",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "selected",
        "transfers_in",
        "transfers_out",
        "transfers_balance",
        "team_h_score",
        "team_a_score",
        "modified",
    ]

    cols = [c for c in preferred_order if c in gw.columns]
    remaining = [c for c in gw.columns if c not in cols and c not in {"GW", "round", "value"}]

    return gw[cols + remaining].sort_values(
        ["season", "gameweek", "fixture_id", "team_name", "player_code"]
    )


def build_fct_gw_historic(
    base_dir: str = "data/raw/historic",
    seasons=None,
    output_path: str | Path = OUTPUT_PATH,
    save: bool = True,
) -> pd.DataFrame:
    """Build the combined historic FPL player-fixture fact table."""
    seasons = seasons or DEFAULT_SEASONS

    raw = load_historic_raw(base_dir=base_dir, seasons=seasons)
    player_map = build_player_season_map(base_dir=base_dir, seasons=seasons)

    frames = []

    for season in seasons:
        frame = _build_one_season(
            season=season,
            gameweeks=raw[season]["gameweeks"],
            teams=raw[season]["teams"],
            player_map=player_map,
        )
        frames.append(frame)

        print(
            f"{season}: "
            f"{len(frame):,} rows | "
            f"{frame['player_code'].nunique():,} players | "
            f"GW {frame['gameweek'].min()}-{frame['gameweek'].max()}"
        )

    out = pd.concat(frames, ignore_index=True)

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        print(f"Rows: {len(out):,}")
        print(f"Players: {out['player_code'].nunique():,}")

    return out


if __name__ == "__main__":
    build_fct_gw_historic()
