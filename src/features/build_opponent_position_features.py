from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path("config/opponent_features.yaml")


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _rolling_sum(values: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    return (
        values.groupby(groups)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )


def _build_fixture_position_base(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    meaningful_minutes = int(config["meaningful_minutes"])
    points_col = config["points_column"]
    position_map = config["position_groups"]

    out = df.copy()
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True, errors="coerce")
    out["position_group"] = out["position"].map(position_map)

    if out["position_group"].isna().any():
        bad = out.loc[out["position_group"].isna(), ["position"]].drop_duplicates()
        raise ValueError(f"Unmapped positions found:\n{bad.to_string(index=False)}")

    out[points_col] = pd.to_numeric(out[points_col], errors="coerce")
    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce")

    meaningful = out[out["minutes"] >= meaningful_minutes].copy()

    if "opponent_team_code" in meaningful.columns and meaningful["opponent_team_code"].notna().all():
        meaningful["opponent_team_key"] = (
            "code:" + meaningful["opponent_team_code"].astype("Int64").astype(str)
        )
    else:
        meaningful["opponent_team_key"] = (
            "name:" + meaningful["opponent_team_name"].astype(str)
        )

    group_cols = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "opponent_team_key",
        "opponent_team_id",
        "opponent_team_name",
        "position_group",
    ]

    if "opponent_team_code" in meaningful.columns:
        group_cols.insert(group_cols.index("opponent_team_name"), "opponent_team_code")

    base = (
        meaningful.groupby(group_cols, as_index=False, observed=True)
        .agg(
            fixture_core_points_allowed=(points_col, "sum"),
            fixture_meaningful_appearances=("player_code", "size"),
            fixture_unique_players=("player_code", "nunique"),
        )
    )

    base["fixture_avg_core_points_allowed"] = (
        base["fixture_core_points_allowed"]
        / base["fixture_meaningful_appearances"]
    )

    return base


def _add_rolling_opponent_features(base: pd.DataFrame, config: dict) -> pd.DataFrame:
    windows = config["rolling_windows"]

    out = base.sort_values(
        [
            "opponent_team_key",
            "position_group",
            "kickoff_time",
            "season",
            "gameweek",
            "fixture_id",
        ]
    ).reset_index(drop=True)

    group_key = (
        out["opponent_team_key"].astype(str)
        + "||"
        + out["position_group"].astype(str)
    )

    grouped = out.groupby(["opponent_team_key", "position_group"], sort=False)

    shifted_points = grouped["fixture_core_points_allowed"].shift(1)
    shifted_apps = grouped["fixture_meaningful_appearances"].shift(1)

    for window in windows:
        rolling_points = _rolling_sum(
            shifted_points.fillna(0),
            group_key,
            window,
        )
        rolling_apps = _rolling_sum(
            shifted_apps.fillna(0),
            group_key,
            window,
        )

        out[f"opp_pos_core_points_avg_l{window}"] = np.where(
            rolling_apps > 0,
            rolling_points / rolling_apps,
            np.nan,
        )
        out[f"opp_pos_meaningful_apps_l{window}"] = rolling_apps

    return out


def build_opponent_position_features(
    config_path: str | Path = CONFIG_PATH,
    save: bool = True,
) -> pd.DataFrame:
    config = _load_config(config_path)

    input_path = Path(config["input_path"])
    output_path = Path(config["output_path"])

    df = pd.read_csv(input_path, low_memory=False)

    required = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "position",
        "minutes",
        "core_total_points",
        "opponent_team_id",
        "opponent_team_name",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Historic fact table missing required columns: {missing}")

    base = _build_fixture_position_base(df, config)
    out = _add_rolling_opponent_features(base, config)

    grain = [
        "season",
        "fixture_id",
        "opponent_team_key",
        "position_group",
    ]

    dupes = out.duplicated(grain, keep=False)
    if dupes.any():
        sample = out.loc[dupes, grain + ["opponent_team_name"]].head(20)
        raise ValueError(
            "Duplicate opponent feature grain detected.\n"
            f"{sample.to_string(index=False)}"
        )

    feature_cols = [c for c in out.columns if c.startswith("opp_pos_")]

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")
        print(f"Rows: {len(out):,}")
        print(f"Opponent teams: {out['opponent_team_key'].nunique():,}")
        print(f"Position groups: {sorted(out['position_group'].unique())}")
        print(f"Rolling windows: {config['rolling_windows']}")
        print(f"Opponent feature columns: {len(feature_cols):,}")

    return out


def add_position_group_for_join(
    player_df: pd.DataFrame,
    config_path: str | Path = CONFIG_PATH,
) -> pd.DataFrame:
    config = _load_config(config_path)
    out = player_df.copy()
    out["position_group"] = out["position"].map(config["position_groups"])
    return out


if __name__ == "__main__":
    build_opponent_position_features()
