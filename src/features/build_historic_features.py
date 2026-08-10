from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/fct_gw_historic.csv")
OUTPUT_PATH = Path("data/features/fct_gw_features_historic.csv")


PLAYER_FEATURE_WINDOWS = [3, 5]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def _add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add player-level historical features using only information available
    before the current fixture.

    Important:
    - All rolling metrics are shifted by 1 fixture.
    - Grain remains player x fixture.
    """
    out = df.copy()

    out = out.sort_values(
        ["player_code", "kickoff_time", "season", "gameweek", "fixture_id"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Per-match helper metrics used only to generate lagged features
    # ---------------------------------------------------------
    # Use the explicit FPL starts field where available.
    # For older rows/seasons where it is unavailable, use 60+ minutes
    # as a practical historical proxy for a start.
    minutes_numeric = pd.to_numeric(out["minutes"], errors="coerce")

    if "starts" in out.columns:
        starts_raw = pd.to_numeric(out["starts"], errors="coerce")
        out["_started"] = np.where(
            starts_raw.notna(),
            (starts_raw == 1).astype(float),
            (minutes_numeric >= 60).astype(float),
        )
    else:
        out["_started"] = (minutes_numeric >= 60).astype(float)

    out["_played"] = (
        pd.to_numeric(out["minutes"], errors="coerce") > 0
    ).astype(float)

    out["_meaningful_45"] = (
        pd.to_numeric(out["minutes"], errors="coerce") >= 45
    ).astype(float)

    out["_fullish_80"] = (
        pd.to_numeric(out["minutes"], errors="coerce") >= 80
    ).astype(float)

    out["_zero_minutes"] = (
        pd.to_numeric(out["minutes"], errors="coerce") == 0
    ).astype(float)

    # Per-90 helper measures. Only define where minutes > 0.
    minutes = pd.to_numeric(out["minutes"], errors="coerce")

    per90_sources = {
        "core_points": "core_total_points",
        "goals": "goals_scored",
        "assists": "assists",
        "bps": "bps",
        "influence": "influence",
        "creativity": "creativity",
        "threat": "threat",
        "ict": "ict_index",
        "xg": "expected_goals",
        "xa": "expected_assists",
        "xgi": "expected_goal_involvements",
        "xgc": "expected_goals_conceded",
    }

    for short_name, source_col in per90_sources.items():
        if source_col in out.columns:
            values = pd.to_numeric(out[source_col], errors="coerce")
            out[f"_{short_name}_per90"] = np.where(
                minutes > 0,
                values * 90.0 / minutes,
                np.nan,
            )

    # ---------------------------------------------------------
    # Lag-1 raw features
    # ---------------------------------------------------------
    lag_columns = [
        "minutes",
        "total_points",
        "core_total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "defensive_contribution",
    ]

    player_group = out.groupby("player_code", sort=False)

    for col in lag_columns:
        if col in out.columns:
            out[f"player_{col}_lag1"] = player_group[col].shift(1)

    # ---------------------------------------------------------
    # Rolling player form
    # ---------------------------------------------------------
    rolling_sources = {
        "minutes": "minutes",
        "total_points": "total_points",
        "core_points": "core_total_points",
        "goals": "goals_scored",
        "assists": "assists",
        "clean_sheets": "clean_sheets",
        "goals_conceded": "goals_conceded",
        "bonus": "bonus",
        "bps": "bps",
        "influence": "influence",
        "creativity": "creativity",
        "threat": "threat",
        "ict": "ict_index",
        "xg": "expected_goals",
        "xa": "expected_assists",
        "xgi": "expected_goal_involvements",
        "xgc": "expected_goals_conceded",
        "played": "_played",
        "started": "_started",
        "meaningful_45": "_meaningful_45",
        "fullish_80": "_fullish_80",
        "zero_minutes": "_zero_minutes",
    }

    for window in PLAYER_FEATURE_WINDOWS:
        for feature_name, source_col in rolling_sources.items():
            if source_col not in out.columns:
                continue

            shifted = player_group[source_col].shift(1)

            out[f"player_{feature_name}_avg_l{window}"] = (
                shifted.groupby(out["player_code"])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Sums are especially useful for goals/assists/xG.
        for feature_name, source_col in {
            "goals": "goals_scored",
            "assists": "assists",
            "xg": "expected_goals",
            "xa": "expected_assists",
            "xgi": "expected_goal_involvements",
        }.items():
            if source_col not in out.columns:
                continue

            shifted = player_group[source_col].shift(1)

            out[f"player_{feature_name}_sum_l{window}"] = (
                shifted.groupby(out["player_code"])
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )

        # Per-90 rolling metrics.
        for short_name in [
            "core_points",
            "goals",
            "assists",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict",
            "xg",
            "xa",
            "xgi",
            "xgc",
        ]:
            helper = f"_{short_name}_per90"
            if helper not in out.columns:
                continue

            shifted = player_group[helper].shift(1)

            out[f"player_{short_name}_per90_avg_l{window}"] = (
                shifted.groupby(out["player_code"])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # ---------------------------------------------------------
    # Season-to-date player metrics, all shifted
    # ---------------------------------------------------------
    season_player_group = out.groupby(["season", "player_code"], sort=False)

    season_sources = {
        "minutes": "minutes",
        "total_points": "total_points",
        "core_points": "core_total_points",
        "goals": "goals_scored",
        "assists": "assists",
        "bonus": "bonus",
        "bps": "bps",
        "xg": "expected_goals",
        "xa": "expected_assists",
        "xgi": "expected_goal_involvements",
        "played": "_played",
        "started": "_started",
    }

    for feature_name, source_col in season_sources.items():
        if source_col not in out.columns:
            continue

        shifted = season_player_group[source_col].shift(1)

        out[f"player_{feature_name}_season_avg"] = (
            shifted.groupby([out["season"], out["player_code"]])
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    # Better season-to-date per-90s using cumulative sums / cumulative minutes.
    # This avoids averaging match-level per-90 values.
    season_metric_sources = {
        "core_points": "core_total_points",
        "goals": "goals_scored",
        "assists": "assists",
        "bps": "bps",
        "xg": "expected_goals",
        "xa": "expected_assists",
        "xgi": "expected_goal_involvements",
    }

    shifted_minutes = season_player_group["minutes"].shift(1)
    cum_minutes = (
        shifted_minutes.fillna(0)
        .groupby([out["season"], out["player_code"]])
        .cumsum()
    )

    out["player_minutes_season_total"] = cum_minutes

    for feature_name, source_col in season_metric_sources.items():
        if source_col not in out.columns:
            continue

        shifted_metric = season_player_group[source_col].shift(1)
        cum_metric = (
            shifted_metric.fillna(0)
            .groupby([out["season"], out["player_code"]])
            .cumsum()
        )

        out[f"player_{feature_name}_season_total"] = cum_metric
        out[f"player_{feature_name}_per90_season"] = _safe_divide(
            cum_metric * 90.0,
            cum_minutes,
        )

    # ---------------------------------------------------------
    # Appearance / availability-style indicators
    # ---------------------------------------------------------
    # Number of prior fixtures seen in current season.
    out["player_prior_fixtures_season"] = season_player_group.cumcount()

    # Days since previous fixture for this player.
    kickoff = pd.to_datetime(out["kickoff_time"], utc=True, errors="coerce")
    prev_kickoff = player_group["kickoff_time"].shift(1)
    prev_kickoff = pd.to_datetime(prev_kickoff, utc=True, errors="coerce")

    out["player_days_since_last_fixture"] = (
        kickoff - prev_kickoff
    ).dt.total_seconds() / 86400.0

    # ---------------------------------------------------------
    # Targets
    # ---------------------------------------------------------
    out["target_minutes"] = pd.to_numeric(out["minutes"], errors="coerce")
    out["target_total_points"] = pd.to_numeric(out["total_points"], errors="coerce")
    out["target_core_points"] = pd.to_numeric(out["core_total_points"], errors="coerce")

    out["target_core_pp90"] = np.where(
        out["minutes"] >= 45,
        out["core_total_points"] * 90.0 / out["minutes"],
        np.nan,
    )

    out["target_defcon_hit"] = pd.to_numeric(
        out["defcon_hit"], errors="coerce"
    )

    # ---------------------------------------------------------
    # Remove private helper columns
    # ---------------------------------------------------------
    helper_cols = [c for c in out.columns if c.startswith("_")]
    out = out.drop(columns=helper_cols)

    return out


def build_historic_features(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the first historic feature layer.

    Scope of V1:
    - Player history only
    - No team strength features yet
    - No opponent strength features yet
    - No opponent-position PPG yet
    """
    input_path = Path(input_path)

    df = pd.read_csv(input_path, low_memory=False)

    required = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "position",
        "minutes",
        "total_points",
        "core_total_points",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Historic fact table missing required columns: {missing}"
        )

    out = _add_player_rolling_features(df)

    # Preserve the original grain.
    grain = ["season", "gameweek", "player_code", "fixture_id"]
    dupes = out.duplicated(grain, keep=False)

    if dupes.any():
        sample = out.loc[dupes, grain].head(20)
        raise ValueError(
            "Duplicate feature-table grain detected.\n"
            f"{sample.to_string(index=False)}"
        )

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)

        feature_cols = [
            c for c in out.columns
            if c.startswith("player_")
        ]

        target_cols = [
            c for c in out.columns
            if c.startswith("target_")
        ]

        print(f"Saved: {output_path}")
        print(f"Rows: {len(out):,}")
        print(f"Players: {out['player_code'].nunique():,}")
        print(f"Player feature columns: {len(feature_cols):,}")
        print(f"Target columns: {len(target_cols):,}")

    return out


if __name__ == "__main__":
    build_historic_features()
