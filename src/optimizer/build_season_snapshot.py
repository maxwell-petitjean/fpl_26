from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path("config/current_predictions.yaml")

OUTPUT_PATH = Path(
    "data/outputs/solver/season_snapshot.csv"
)


def _load_config(
    path=CONFIG_PATH,
):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _safe_numeric(
    df,
    column,
):
    if column not in df.columns:
        return pd.Series(
            0.0,
            index=df.index,
        )

    return (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .fillna(0.0)
    )


def _per90(
    numerator,
    minutes,
):
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).fillna(0.0)

    minutes = pd.to_numeric(
        minutes,
        errors="coerce",
    ).fillna(0.0)

    return np.where(
        minutes > 0,
        numerator
        * 90
        / minutes,
        0.0,
    )


def build_season_snapshot(
    config_path=CONFIG_PATH,
    output_path=OUTPUT_PATH,
    save=True,
):
    """
    Build one row per player for the CURRENT season only.

    Source:
        fct_gw_all_known.csv

    Grain:
        player_code

    Only rows with known actual minutes + total_points are included,
    preventing future forecast rows from leaking into season-to-date stats.
    """

    config = _load_config(
        config_path
    )

    season = str(
        config["season"]
    )

    source_path = Path(
        config["paths"][
            "all_known_fact"
        ]
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"All-known fact not found: {source_path}"
        )

    fact = pd.read_csv(
        source_path,
        low_memory=False,
    )

    required = [
        "season",
        "player_code",
        "minutes",
        "total_points",
    ]

    missing = [
        c
        for c in required
        if c not in fact.columns
    ]

    if missing:
        raise ValueError(
            "Season snapshot source missing "
            f"required columns: {missing}"
        )

    current = (
        fact[
            fact["season"]
            .astype(str)
            .eq(season)
        ]
        .copy()
    )

    # Actual results only.
    current["minutes"] = pd.to_numeric(
        current["minutes"],
        errors="coerce",
    )

    current["total_points"] = pd.to_numeric(
        current["total_points"],
        errors="coerce",
    )

    current = current[
        current["minutes"].notna()
        & current["total_points"].notna()
    ].copy()

    # Defensive dedupe at actual player-fixture grain.
    grain = [
        c
        for c in [
            "season",
            "gameweek",
            "player_code",
            "fixture_id",
        ]
        if c in current.columns
    ]

    if grain:
        duplicate_rows = current[
            current.duplicated(
                grain,
                keep=False,
            )
        ]

        if not duplicate_rows.empty:
            raise ValueError(
                "Current-season actual fact has duplicate "
                f"rows at grain {grain}:\n"
                + duplicate_rows[
                    grain
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )

    current["_goals"] = _safe_numeric(
        current,
        "goals_scored",
    )

    current["_assists"] = _safe_numeric(
        current,
        "assists",
    )

    current["_xg"] = _safe_numeric(
        current,
        "expected_goals",
    )

    current["_xa"] = _safe_numeric(
        current,
        "expected_assists",
    )

    current["_defcon_points"] = _safe_numeric(
        current,
        "defcon_points",
    )

    snapshot = (
        current.groupby(
            "player_code",
            as_index=False,
        )
        .agg(
            season_minutes=(
                "minutes",
                "sum",
            ),
            season_total_points=(
                "total_points",
                "sum",
            ),
            season_goals=(
                "_goals",
                "sum",
            ),
            season_assists=(
                "_assists",
                "sum",
            ),
            season_xg=(
                "_xg",
                "sum",
            ),
            season_xa=(
                "_xa",
                "sum",
            ),
            season_defcon_points=(
                "_defcon_points",
                "sum",
            ),
        )
    )

    snapshot["season_pp90"] = _per90(
        snapshot[
            "season_total_points"
        ],
        snapshot[
            "season_minutes"
        ],
    )

    snapshot["season_xg90"] = _per90(
        snapshot[
            "season_xg"
        ],
        snapshot[
            "season_minutes"
        ],
    )

    snapshot["season_xa90"] = _per90(
        snapshot[
            "season_xa"
        ],
        snapshot[
            "season_minutes"
        ],
    )

    snapshot["season_defcon_per90"] = _per90(
        snapshot[
            "season_defcon_points"
        ],
        snapshot[
            "season_minutes"
        ],
    )

    snapshot.insert(
        0,
        "season",
        season,
    )

    snapshot = (
        snapshot.sort_values(
            [
                "season_total_points",
                "season_minutes",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    if (
        snapshot["player_code"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Season snapshot is not unique "
            "to player_code."
        )

    if save:
        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        snapshot.to_csv(
            output_path,
            index=False,
        )

    print("=== CURRENT-SEASON SNAPSHOT ===")
    print("Season:", season)
    print(
        "Actual rows:",
        f"{len(current):,}",
    )
    print(
        "Players:",
        f"{len(snapshot):,}",
    )

    if not snapshot.empty:
        print(
            snapshot[
                [
                    "player_code",
                    "season_minutes",
                    "season_total_points",
                    "season_pp90",
                    "season_goals",
                    "season_assists",
                    "season_xg",
                    "season_xa",
                    "season_defcon_points",
                    "season_defcon_per90",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    return snapshot


if __name__ == "__main__":
    build_season_snapshot()
