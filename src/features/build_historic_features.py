from pathlib import Path

import numpy as np
import pandas as pd
import yaml


INPUT_PATH = Path("data/processed/fct_gw_historic.csv")
OUTPUT_PATH = Path("data/features/fct_gw_features_historic.csv")
CONFIG_PATH = Path("config/player_features.yaml")


IDENTIFIER_COLUMNS = [
    "season",
    "gameweek",
    "fixture_id",
    "kickoff_time",
    "player_code",
    "player_name",
    "web_name",
    "position",
    "team_id",
    "team_name",
    "opponent_team_id",
    "opponent_team_name",
    "was_home",
    "price",
]


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _add_kickoff_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive current-fixture kickoff context.

    Day type:
      Mon / Fri       -> primetime
      Tue / Wed / Thu -> midweek
      Sat / Sun       -> weekend

    Time type:
      before 14:00          -> lunchtime
      14:00 to before 17:30 -> afternoon
      17:30 onwards         -> evening
    """
    out = df.copy()

    kickoff = pd.to_datetime(
        out["kickoff_time"],
        utc=True,
        errors="coerce",
    )

    out["kickoff_day_of_week"] = kickoff.dt.day_name()

    out["kickoff_hour"] = (
        kickoff.dt.hour
        + kickoff.dt.minute / 60
    )

    day_map = {
        "Monday": "primetime",
        "Friday": "primetime",
        "Tuesday": "midweek",
        "Wednesday": "midweek",
        "Thursday": "midweek",
        "Saturday": "weekend",
        "Sunday": "weekend",
    }

    out["kickoff_day_type"] = (
        out["kickoff_day_of_week"]
        .map(day_map)
    )

    out["kickoff_time_type"] = np.select(
        [
            out["kickoff_hour"] < 14,
            out["kickoff_hour"] < 17.5,
        ],
        [
            "lunchtime",
            "afternoon",
        ],
        default="evening",
    )

    out.loc[
        kickoff.isna(),
        "kickoff_time_type"
    ] = pd.NA

    return out

def _add_global_match_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a chronological league-fixture index across all historic seasons.

    This is metadata/debugging only and should not be used as a model feature.
    """
    out = df.copy()
    out["kickoff_time"] = pd.to_datetime(
        out["kickoff_time"], utc=True, errors="coerce"
    )

    fixtures = (
        out[
            ["season", "fixture_id", "kickoff_time"]
        ]
        .drop_duplicates()
        .sort_values(
            ["kickoff_time", "season", "fixture_id"],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    fixtures["match_index"] = np.arange(1, len(fixtures) + 1)

    out = out.merge(
        fixtures[
            ["season", "fixture_id", "match_index"]
        ],
        on=["season", "fixture_id"],
        how="left",
        validate="many_to_one",
    )

    return out


def _rolling_mean(
    shifted: pd.Series,
    groups: pd.Series,
    window: int,
) -> pd.Series:
    return (
        shifted
        .groupby(groups)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _rolling_sum(
    shifted: pd.Series,
    groups: pd.Series,
    window: int,
) -> pd.Series:
    return (
        shifted
        .groupby(groups)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )


def _rolling_meaningful_rate(
    shifted_minutes: pd.Series,
    groups: pd.Series,
    window: int,
    threshold: float,
) -> pd.Series:
    meaningful = shifted_minutes.ge(threshold).astype(float)
    meaningful[shifted_minutes.isna()] = np.nan

    return (
        meaningful
        .groupby(groups)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _rolling_meaningful_minutes_mean(
    shifted_minutes: pd.Series,
    groups: pd.Series,
    window: int,
    threshold: float,
) -> pd.Series:
    meaningful_minutes = shifted_minutes.where(
        shifted_minutes >= threshold
    )

    return (
        meaningful_minutes
        .groupby(groups)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _rolling_per90(
    shifted_metric: pd.Series,
    shifted_minutes: pd.Series,
    groups: pd.Series,
    window: int,
) -> pd.Series:
    """
    Ratio-of-sums per90:
        sum(metric over prior N fixtures)
        / sum(minutes over prior N fixtures) * 90

    This deliberately avoids averaging individual-match per90 values.
    """
    metric_sum = _rolling_sum(
        shifted_metric.fillna(0),
        groups,
        window,
    )
    minutes_sum = _rolling_sum(
        shifted_minutes.fillna(0),
        groups,
        window,
    )

    return np.where(
        minutes_sum > 0,
        metric_sum * 90.0 / minutes_sum,
        np.nan,
    )


def _add_explicit_lags(
    out: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Add explicit lag-1 features only for categories listed in lag_categories.

    These are separate from rolling windows. For example:
      player_minutes_lag1
      player_core_points_lag1
      player_core_pp90_lag1
    """
    lag_categories = set(config.get("lag_categories", []))
    categories = config.get("categories", {})

    player_group = out.groupby("player_code", sort=False)
    shifted_minutes = player_group["minutes"].shift(1)

    used_names = set()

    for category_name in lag_categories:
        metrics = categories.get(category_name, {})

        for metric_name, spec in metrics.items():
            source = spec["source"]
            aggregation = spec["aggregation"]

            if source not in out.columns:
                continue

            feature_name = metric_name

            # Cleaner naming for the raw minutes lag.
            if metric_name == "avg_mins":
                feature_name = "minutes"

            if feature_name in used_names:
                continue

            if aggregation == "per90":
                shifted_metric = player_group[source].shift(1)

                lag = np.where(
                    shifted_minutes > 0,
                    pd.to_numeric(
                        shifted_metric, errors="coerce"
                    ) * 90.0 / pd.to_numeric(
                        shifted_minutes, errors="coerce"
                    ),
                    np.nan,
                )

            elif aggregation == "meaningful_rate":
                threshold = spec.get("meaningful_threshold", 45)

                lag = np.where(
                    shifted_minutes.notna(),
                    (shifted_minutes >= threshold).astype(float),
                    np.nan,
                )

            elif aggregation == "meaningful_minutes_mean":
                threshold = spec.get("meaningful_threshold", 45)

                lag = shifted_minutes.where(
                    shifted_minutes >= threshold
                )

            else:
                lag = player_group[source].shift(1)

            out[f"player_{feature_name}_lag1"] = lag
            used_names.add(feature_name)

    return out


def _add_configured_rolling_features(
    out: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Build all configured rolling features continuously across season boundaries.

    Each current row sees only PRIOR player fixtures because every source
    series is shifted by 1 before rolling.
    """
    windows = config["rolling_windows"]
    categories = config["categories"]

    player_group = out.groupby("player_code", sort=False)
    groups = out["player_code"]

    shifted_minutes = pd.to_numeric(
        player_group["minutes"].shift(1),
        errors="coerce",
    )

    for category_name, metrics in categories.items():
        # EXTERNAL can remain empty until a future data source is added.
        if not metrics:
            continue

        for metric_name, spec in metrics.items():
            source = spec["source"]
            aggregation = spec["aggregation"]

            if source not in out.columns:
                # Historic availability differs by season/source.
                # Missing columns are ignored at build time.
                continue

            shifted_source = pd.to_numeric(
                player_group[source].shift(1),
                errors="coerce",
            )

            for window in windows:
                col = f"player_{metric_name}_l{window}"

                if aggregation == "mean":
                    out[col] = _rolling_mean(
                        shifted_source,
                        groups,
                        window,
                    )

                elif aggregation == "sum":
                    out[col] = _rolling_sum(
                        shifted_source.fillna(0),
                        groups,
                        window,
                    )

                elif aggregation == "per90":
                    out[col] = _rolling_per90(
                        shifted_source,
                        shifted_minutes,
                        groups,
                        window,
                    )

                elif aggregation == "meaningful_rate":
                    threshold = spec.get(
                        "meaningful_threshold", 45
                    )
                    out[col] = _rolling_meaningful_rate(
                        shifted_minutes,
                        groups,
                        window,
                        threshold,
                    )

                elif aggregation == "meaningful_minutes_mean":
                    threshold = spec.get(
                        "meaningful_threshold", 45
                    )
                    out[col] = _rolling_meaningful_minutes_mean(
                        shifted_minutes,
                        groups,
                        window,
                        threshold,
                    )

                else:
                    raise ValueError(
                        f"Unsupported aggregation '{aggregation}' "
                        f"for metric '{metric_name}'"
                    )

    return out


def _add_targets(out: pd.DataFrame) -> pd.DataFrame:
    """Add modelling targets. Targets are never used as current-row features."""
    out["target_minutes"] = pd.to_numeric(
        out["minutes"], errors="coerce"
    )
    out["target_total_points"] = pd.to_numeric(
        out["total_points"], errors="coerce"
    )
    out["target_core_points"] = pd.to_numeric(
        out["core_total_points"], errors="coerce"
    )

    out["target_core_pp90"] = np.where(
        pd.to_numeric(out["minutes"], errors="coerce") >= 45,
        pd.to_numeric(
            out["core_total_points"], errors="coerce"
        )
        * 90.0
        / pd.to_numeric(out["minutes"], errors="coerce"),
        np.nan,
    )

    out["target_defcon_hit"] = pd.to_numeric(
        out.get("defcon_hit"),
        errors="coerce",
    )

    return out


def build_historic_features(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    config_path: str | Path = CONFIG_PATH,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build historic player features.

    Design:
    - Continuous history across season boundaries.
    - Rolling windows controlled in config/player_features.yaml.
    - Metric categories controlled in config.
    - Explicit lag-1 features controlled via lag_categories.
    - No season-to-date aggregates.
    - No player_history_matches model feature.
    - match_index retained as metadata/debugging only.
    """
    input_path = Path(input_path)
    config = _load_config(config_path)

    df = pd.read_csv(input_path, low_memory=False)

    required = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "player_name",
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

    df["kickoff_time"] = pd.to_datetime(
        df["kickoff_time"], utc=True, errors="coerce"
    )

    out = _add_global_match_index(df)

    # Continuous player chronology. Season is deliberately NOT a grouping key.
    out = out.sort_values(
        [
            "player_code",
            "kickoff_time",
            "season",
            "gameweek",
            "fixture_id",
        ],
        na_position="last",
    ).reset_index(drop=True)

    out = _add_explicit_lags(out, config)
    out = _add_configured_rolling_features(out, config)
    out = _add_targets(out)

    # Preserve original player-fixture grain.
    grain = [
        "season",
        "gameweek",
        "player_code",
        "fixture_id",
    ]

    dupes = out.duplicated(grain, keep=False)

    if dupes.any():
        sample = out.loc[dupes, grain].head(20)
        raise ValueError(
            "Duplicate feature-table grain detected.\n"
            f"{sample.to_string(index=False)}"
        )

    # Put identifiers/metadata first for easier human inspection.
    front = [
        c for c in IDENTIFIER_COLUMNS + ["match_index"]
        if c in out.columns
    ]

    target_cols = [
        c for c in out.columns if c.startswith("target_")
    ]

    feature_cols = [
        c for c in out.columns
        if c.startswith("player_")
        and c not in {"player_code", "player_name"}
    ]

    raw_fact_cols = [
        c for c in out.columns
        if c not in front
        and c not in feature_cols
        and c not in target_cols
    ]

    out = out[
        front
        + feature_cols
        + raw_fact_cols
        + target_cols
    ]

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        out.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")
        print(f"Rows: {len(out):,}")
        print(
            f"Players: "
            f"{out['player_code'].nunique():,}"
        )
        print(
            f"Configured rolling windows: "
            f"{config['rolling_windows']}"
        )
        print(
            f"Lag categories: "
            f"{config.get('lag_categories', [])}"
        )
        print(
            f"Player feature columns: "
            f"{len(feature_cols):,}"
        )
        print(
            f"Target columns: "
            f"{len(target_cols):,}"
        )

    return out


if __name__ == "__main__":
    build_historic_features()
