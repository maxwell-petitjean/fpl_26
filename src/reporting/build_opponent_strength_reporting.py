from pathlib import Path

import numpy as np
import pandas as pd


HISTORIC_PATH = Path(
    "data/processed/fct_gw_historic.csv"
)

ALL_KNOWN_PATH = Path(
    "data/processed/fct_gw_all_known.csv"
)

TEAM_DIM_PATH = Path(
    "data/processed/dim_team_season.csv"
)

OUTPUT_PATH = Path(
    "data/outputs/reporting/"
    "opponent_strength.csv"
)

ROLLING_WINDOWS = [
    1,
    3,
    6,
    12,
    38,
]

MEANINGFUL_MINUTES = 45

POSITION_GROUP_MAP = {
    "GK": "GK",
    "DEF": "DEF",
    "MID": "ATT",
    "FWD": "ATT",
}


def _source_path():
    """
    Prefer all-known fact so the reporting
    table automatically starts including
    current-season results once available.
    """
    if ALL_KNOWN_PATH.exists():
        return ALL_KNOWN_PATH

    return HISTORIC_PATH


def _season_start(
    season: str,
) -> int:
    return int(
        str(season).split("-")[0]
    )


def _add_team_spell(
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reset rolling history after a club has
    been absent from the PL.

    Example:
    promoted -> relegated -> promoted again
    starts a new team spell rather than
    inheriting the old PL spell.
    """
    out = fixtures.copy()

    season_presence = (
        out[
            [
                "opponent_team_code",
                "season",
            ]
        ]
        .drop_duplicates()
        .assign(
            season_start=lambda x:
            x["season"].map(
                _season_start
            )
        )
        .sort_values(
            [
                "opponent_team_code",
                "season_start",
            ]
        )
    )

    season_presence[
        "previous_start"
    ] = (
        season_presence
        .groupby(
            "opponent_team_code"
        )[
            "season_start"
        ]
        .shift(1)
    )

    season_presence[
        "new_spell"
    ] = (
        season_presence[
            "previous_start"
        ].isna()
        |
        (
            season_presence[
                "season_start"
            ]
            - season_presence[
                "previous_start"
            ]
            > 1
        )
    )

    season_presence[
        "team_spell"
    ] = (
        season_presence
        .groupby(
            "opponent_team_code"
        )[
            "new_spell"
        ]
        .cumsum()
        .astype(int)
    )

    out = out.merge(
        season_presence[
            [
                "opponent_team_code",
                "season",
                "team_spell",
            ]
        ],
        on=[
            "opponent_team_code",
            "season",
        ],
        how="left",
        validate="many_to_one",
    )

    return out


def _build_fixture_base(
    fact: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "position",
        "minutes",
        "core_total_points",
        "total_points",
        "opponent_team_id",
        "opponent_team_code",
        "opponent_team_name",
    ]

    missing = [
        c for c in required
        if c not in fact.columns
    ]

    if missing:
        raise ValueError(
            "Fact table missing required "
            f"columns: {missing}"
        )

    df = fact.copy()

    df["kickoff_time"] = (
        pd.to_datetime(
            df["kickoff_time"],
            utc=True,
            errors="coerce",
        )
    )

    df["minutes"] = (
        pd.to_numeric(
            df["minutes"],
            errors="coerce",
        )
    )

    df["core_total_points"] = (
        pd.to_numeric(
            df["core_total_points"],
            errors="coerce",
        )
    )

    df["total_points"] = (
        pd.to_numeric(
            df["total_points"],
            errors="coerce",
        )
    )

    df["position_group"] = (
        df["position"]
        .map(
            POSITION_GROUP_MAP
        )
    )

    # Match the opponent-position modelling
    # concept: compare meaningful players.
    df = df[
        df["position_group"].notna()
        & (
            df["minutes"]
            >= MEANINGFUL_MINUTES
        )
    ].copy()

    group_cols = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "opponent_team_id",
        "opponent_team_code",
        "opponent_team_name",
        "position_group",
    ]

    if (
        "opponent_team_short_name"
        in df.columns
    ):
        group_cols.insert(
            7,
            "opponent_team_short_name",
        )

    fixture = (
        df.groupby(
            group_cols,
            as_index=False,
            observed=True,
        )
        .agg(
            fixture_meaningful_apps=(
                "player_code",
                "size",
            ),
            fixture_core_points_sum=(
                "core_total_points",
                "sum",
            ),
            fixture_total_points_sum=(
                "total_points",
                "sum",
            ),
        )
    )

    fixture[
        "fixture_avg_core_points"
    ] = (
        fixture[
            "fixture_core_points_sum"
        ]
        /
        fixture[
            "fixture_meaningful_apps"
        ]
    )

    fixture[
        "fixture_avg_total_points"
    ] = (
        fixture[
            "fixture_total_points_sum"
        ]
        /
        fixture[
            "fixture_meaningful_apps"
        ]
    )

    fixture = _add_team_spell(
        fixture
    )

    return fixture


def _add_rolling_strength(
    fixture: pd.DataFrame,
) -> pd.DataFrame:

    out = (
        fixture
        .sort_values(
            [
                "opponent_team_code",
                "team_spell",
                "position_group",
                "kickoff_time",
                "fixture_id",
            ]
        )
        .reset_index(drop=True)
    )

    group_cols = [
        "opponent_team_code",
        "team_spell",
        "position_group",
    ]

    for window in ROLLING_WINDOWS:

        core_sum = (
            out.groupby(
                group_cols,
                sort=False,
            )[
                "fixture_core_points_sum"
            ]
            .rolling(
                window=window,
                min_periods=1,
            )
            .sum()
            .reset_index(
                level=group_cols,
                drop=True,
            )
        )

        total_sum = (
            out.groupby(
                group_cols,
                sort=False,
            )[
                "fixture_total_points_sum"
            ]
            .rolling(
                window=window,
                min_periods=1,
            )
            .sum()
            .reset_index(
                level=group_cols,
                drop=True,
            )
        )

        apps_sum = (
            out.groupby(
                group_cols,
                sort=False,
            )[
                "fixture_meaningful_apps"
            ]
            .rolling(
                window=window,
                min_periods=1,
            )
            .sum()
            .reset_index(
                level=group_cols,
                drop=True,
            )
        )

        out[
            f"core_points_avg_l{window}"
        ] = np.where(
            apps_sum > 0,
            core_sum / apps_sum,
            np.nan,
        )

        out[
            f"total_points_avg_l{window}"
        ] = np.where(
            apps_sum > 0,
            total_sum / apps_sum,
            np.nan,
        )

        out[
            f"meaningful_apps_l{window}"
        ] = apps_sum

    return out


def _add_league_benchmark(
    fixture: pd.DataFrame,
) -> pd.DataFrame:
    """
    League benchmark is season-to-date,
    calculated at each fixture row.

    This is descriptive reporting only,
    not a model input.
    """
    out = (
        fixture
        .sort_values(
            [
                "season",
                "position_group",
                "kickoff_time",
                "fixture_id",
                "opponent_team_code",
            ]
        )
        .copy()
    )

    season_group = (
        out.groupby(
            [
                "season",
                "position_group",
            ],
            sort=False,
        )
    )

    out[
        "league_core_points_sum_to_date"
    ] = (
        season_group[
            "fixture_core_points_sum"
        ]
        .cumsum()
    )

    out[
        "league_apps_to_date"
    ] = (
        season_group[
            "fixture_meaningful_apps"
        ]
        .cumsum()
    )

    out[
        "league_core_points_avg_to_date"
    ] = (
        out[
            "league_core_points_sum_to_date"
        ]
        /
        out[
            "league_apps_to_date"
        ]
    )

    for window in ROLLING_WINDOWS:

        avg_col = (
            f"core_points_avg_l{window}"
        )

        out[
            f"strength_index_l{window}"
        ] = (
            out[avg_col]
            /
            out[
                "league_core_points_avg_to_date"
            ]
        )

    return out


def _to_gameweek_snapshot(
    fixture: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per:
      season x gameweek x opponent team
      x position group

    In a DGW, use the final fixture state
    from that GW so the dashboard shows
    the end-of-GW strength snapshot.
    """
    snapshot = (
        fixture
        .sort_values(
            [
                "season",
                "gameweek",
                "opponent_team_code",
                "position_group",
                "kickoff_time",
                "fixture_id",
            ]
        )
        .groupby(
            [
                "season",
                "gameweek",
                "opponent_team_code",
                "position_group",
            ],
            as_index=False,
            observed=True,
        )
        .tail(1)
        .copy()
    )

    for window in ROLLING_WINDOWS:

        avg_col = (
            f"core_points_avg_l{window}"
        )

        rank_col = (
            f"rank_l{window}"
        )

        snapshot[
            rank_col
        ] = (
            snapshot
            .groupby(
                [
                    "season",
                    "gameweek",
                    "position_group",
                ]
            )[
                avg_col
            ]
            .rank(
                method="min",
                ascending=False,
            )
        )

    return snapshot


def _add_current_team_identity(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:

    if not TEAM_DIM_PATH.exists():
        return snapshot

    dim = pd.read_csv(
        TEAM_DIM_PATH,
        low_memory=False,
    )

    current = (
        dim[
            dim["is_current_season"]
            .fillna(False)
        ][
            [
                "team_code",
                "current_team_id",
                "current_team_name",
                "current_team_short_name",
            ]
        ]
        .drop_duplicates(
            "team_code"
        )
        .rename(
            columns={
                "team_code":
                    "opponent_team_code",
                "current_team_id":
                    "current_opponent_team_id",
                "current_team_name":
                    "current_opponent_team_name",
                "current_team_short_name":
                    "current_opponent_team_short_name",
            }
        )
    )

    return snapshot.merge(
        current,
        on="opponent_team_code",
        how="left",
        validate="many_to_one",
    )


def build_opponent_strength_reporting(
    output_path=OUTPUT_PATH,
    save=True,
):

    source_path = _source_path()

    fact = pd.read_csv(
        source_path,
        low_memory=False,
    )

    fixture = (
        _build_fixture_base(
            fact
        )
    )

    fixture = (
        _add_rolling_strength(
            fixture
        )
    )

    fixture = (
        _add_league_benchmark(
            fixture
        )
    )

    snapshot = (
        _to_gameweek_snapshot(
            fixture
        )
    )

    snapshot = (
        _add_current_team_identity(
            snapshot
        )
    )

    snapshot[
        "opponent_team_key"
    ] = (
        "code:"
        + snapshot[
            "opponent_team_code"
        ].astype(int).astype(str)
    )

    snapshot[
        "source"
    ] = "actual_history"

    front = [
        "season",
        "gameweek",
        "kickoff_time",
        "fixture_id",
        "opponent_team_key",
        "opponent_team_code",
        "opponent_team_id",
        "opponent_team_name",
        "opponent_team_short_name",
        "current_opponent_team_id",
        "current_opponent_team_name",
        "current_opponent_team_short_name",
        "position_group",
        "team_spell",
        "source",
        "fixture_meaningful_apps",
        "fixture_avg_core_points",
        "fixture_avg_total_points",
        "league_core_points_avg_to_date",
    ]

    rolling = []

    for window in ROLLING_WINDOWS:
        rolling += [
            f"core_points_avg_l{window}",
            f"total_points_avg_l{window}",
            f"meaningful_apps_l{window}",
            f"strength_index_l{window}",
            f"rank_l{window}",
        ]

    snapshot = snapshot[
        [
            c
            for c in front + rolling
            if c in snapshot.columns
        ]
    ].sort_values(
        [
            "season",
            "gameweek",
            "opponent_team_name",
            "position_group",
        ]
    ).reset_index(drop=True)

    grain = [
        "season",
        "gameweek",
        "opponent_team_code",
        "position_group",
    ]

    dupes = snapshot.duplicated(
        grain,
        keep=False,
    )

    if dupes.any():
        raise ValueError(
            "Opponent strength reporting "
            "grain is not unique.\n"
            + snapshot.loc[
                dupes,
                grain,
            ]
            .head(20)
            .to_string(index=False)
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

    print(
        "=== OPPONENT STRENGTH REPORTING ==="
    )
    print(
        f"Source: {source_path}"
    )
    print(
        f"Rows: {len(snapshot):,}"
    )
    print(
        f"Teams: "
        f"{snapshot['opponent_team_code'].nunique():,}"
    )
    print(
        "Position groups:",
        sorted(
            snapshot[
                "position_group"
            ].dropna().unique()
        ),
    )
    print(
        "Rolling windows:",
        ROLLING_WINDOWS,
    )
    print(
        "Interpretation: "
        "strength_index > 1 = easier opponent; "
        "< 1 = harder opponent"
    )

    return snapshot


if __name__ == "__main__":
    build_opponent_strength_reporting()
