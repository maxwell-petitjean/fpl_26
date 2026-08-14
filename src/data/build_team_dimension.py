from pathlib import Path

import pandas as pd
import yaml


HISTORIC_PATH = Path(
    "data/processed/fct_gw_historic.csv"
)

CURRENT_TEAMS_PATH = Path(
    "data/raw/current/teams.csv"
)

CURRENT_CONFIG_PATH = Path(
    "config/current_predictions.yaml"
)

OUTPUT_SEASON_PATH = Path(
    "data/processed/dim_team_season.csv"
)

OUTPUT_CURRENT_PATH = Path(
    "data/processed/dim_team_current.csv"
)


def _load_current_season(
    config_path=CURRENT_CONFIG_PATH,
):
    if not Path(config_path).exists():
        return None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("season")


def _historic_team_rows(
    historic: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "season",
        "team_id",
        "team_code",
        "team_name",
    ]

    missing = [
        c for c in required
        if c not in historic.columns
    ]

    if missing:
        raise ValueError(
            "Historic fact missing team identity "
            f"columns: {missing}"
        )

    team_cols = [
        "season",
        "team_id",
        "team_code",
        "team_name",
    ]

    if "team_short_name" in historic.columns:
        team_cols.append(
            "team_short_name"
        )

    team_side = (
        historic[team_cols]
        .copy()
    )

    opponent_required = [
        "season",
        "opponent_team_id",
        "opponent_team_code",
        "opponent_team_name",
    ]

    opponent_missing = [
        c for c in opponent_required
        if c not in historic.columns
    ]

    if opponent_missing:
        raise ValueError(
            "Historic fact missing opponent identity "
            f"columns: {opponent_missing}"
        )

    opp_cols = [
        "season",
        "opponent_team_id",
        "opponent_team_code",
        "opponent_team_name",
    ]

    if (
        "opponent_team_short_name"
        in historic.columns
    ):
        opp_cols.append(
            "opponent_team_short_name"
        )

    opponent_side = (
        historic[opp_cols]
        .copy()
        .rename(
            columns={
                "opponent_team_id":
                    "team_id",
                "opponent_team_code":
                    "team_code",
                "opponent_team_name":
                    "team_name",
                "opponent_team_short_name":
                    "team_short_name",
            }
        )
    )

    return pd.concat(
        [
            team_side,
            opponent_side,
        ],
        ignore_index=True,
        sort=False,
    )


def _current_team_rows(
    current_teams: pd.DataFrame,
    current_season: str,
) -> pd.DataFrame:

    required = [
        "id",
        "code",
        "name",
    ]

    missing = [
        c for c in required
        if c not in current_teams.columns
    ]

    if missing:
        raise ValueError(
            "Current teams file missing columns: "
            f"{missing}"
        )

    keep = [
        c for c in [
            "id",
            "code",
            "name",
            "short_name",
        ]
        if c in current_teams.columns
    ]

    out = (
        current_teams[keep]
        .copy()
        .rename(
            columns={
                "id": "team_id",
                "code": "team_code",
                "name": "team_name",
                "short_name":
                    "team_short_name",
            }
        )
    )

    out.insert(
        0,
        "season",
        current_season,
    )

    return out


def _validate_dimension(
    dim: pd.DataFrame,
):

    required = [
        "season",
        "team_id",
        "team_code",
        "team_name",
    ]

    nulls = (
        dim[required]
        .isna()
        .sum()
    )

    if nulls.any():
        raise ValueError(
            "Null team identity values found:\n"
            + nulls[
                nulls > 0
            ].to_string()
        )

    # One stable team code per season row.
    code_dupes = (
        dim.groupby(
            [
                "season",
                "team_code",
            ]
        )
        .agg(
            ids=(
                "team_id",
                "nunique",
            ),
            names=(
                "team_name",
                "nunique",
            ),
        )
        .reset_index()
    )

    bad = code_dupes[
        (code_dupes["ids"] > 1)
        | (code_dupes["names"] > 1)
    ]

    if not bad.empty:
        raise ValueError(
            "Conflicting team identity mappings:\n"
            + bad.head(20).to_string(
                index=False
            )
        )


def build_team_dimension(
    historic_path=HISTORIC_PATH,
    current_teams_path=CURRENT_TEAMS_PATH,
    current_config_path=CURRENT_CONFIG_PATH,
    output_season_path=OUTPUT_SEASON_PATH,
    output_current_path=OUTPUT_CURRENT_PATH,
    save=True,
):

    historic = pd.read_csv(
        historic_path,
        low_memory=False,
    )

    historic_rows = (
        _historic_team_rows(
            historic
        )
    )

    current_season = (
        _load_current_season(
            current_config_path
        )
    )

    current_rows = pd.DataFrame()

    if (
        current_season
        and Path(
            current_teams_path
        ).exists()
    ):
        current_teams = pd.read_csv(
            current_teams_path,
            low_memory=False,
        )

        current_rows = (
            _current_team_rows(
                current_teams,
                current_season,
            )
        )

    dim = pd.concat(
        [
            historic_rows,
            current_rows,
        ],
        ignore_index=True,
        sort=False,
    )

    # Prefer current fetch if the same season/team
    # already appears in the historic source.
    dim["is_current_season"] = (
        dim["season"]
        .eq(current_season)
        if current_season
        else False
    )

    dim = (
        dim.sort_values(
            [
                "season",
                "team_code",
                "is_current_season",
            ]
        )
        .drop_duplicates(
            [
                "season",
                "team_code",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    dim["team_code"] = (
        pd.to_numeric(
            dim["team_code"],
            errors="raise",
        )
        .astype(int)
    )

    dim["team_id"] = (
        pd.to_numeric(
            dim["team_id"],
            errors="raise",
        )
        .astype(int)
    )

    dim["team_key"] = (
        "code:"
        + dim["team_code"].astype(str)
    )

    _validate_dimension(dim)

    # Current-season bridge.
    current = (
        dim[
            dim["is_current_season"]
        ]
        .copy()
    )

    current_lookup = (
        current[
            [
                "team_code",
                "team_id",
                "team_name",
                "team_short_name",
            ]
        ]
        .rename(
            columns={
                "team_id":
                    "current_team_id",
                "team_name":
                    "current_team_name",
                "team_short_name":
                    "current_team_short_name",
            }
        )
    )

    dim = dim.merge(
        current_lookup,
        on="team_code",
        how="left",
        validate="many_to_one",
    )

    preferred = [
        "season",
        "team_key",
        "team_code",
        "team_id",
        "team_name",
        "team_short_name",
        "is_current_season",
        "current_team_id",
        "current_team_name",
        "current_team_short_name",
    ]

    dim = dim[
        [
            c for c in preferred
            if c in dim.columns
        ]
    ]

    current_out = (
        dim[
            dim["is_current_season"]
        ][
            [
                "team_key",
                "team_code",
                "team_id",
                "team_name",
                "team_short_name",
            ]
        ]
        .sort_values(
            "team_name"
        )
        .reset_index(drop=True)
    )

    if current_out[
        "team_code"
    ].duplicated().any():
        raise ValueError(
            "Current team dimension is "
            "not unique to team_code."
        )

    if save:
        output_season_path = Path(
            output_season_path
        )
        output_current_path = Path(
            output_current_path
        )

        output_season_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dim.to_csv(
            output_season_path,
            index=False,
        )

        current_out.to_csv(
            output_current_path,
            index=False,
        )

    print("=== TEAM DIMENSION ===")
    print(
        f"Season-team rows: "
        f"{len(dim):,}"
    )
    print(
        f"Unique team codes: "
        f"{dim['team_code'].nunique():,}"
    )

    if current_season:
        print(
            f"Current season: "
            f"{current_season}"
        )
        print(
            f"Current teams: "
            f"{len(current_out):,}"
        )

    print(
        "Canonical key: team_code"
    )
    print(
        "Season-local FPL key: team_id"
    )

    return {
        "season": dim,
        "current": current_out,
    }


if __name__ == "__main__":
    build_team_dimension()
