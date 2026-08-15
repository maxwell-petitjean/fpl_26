from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path(
    "config/solver.yaml"
)

# ============================================================
# EDA SCENARIO SETTINGS
#
# These DO NOT alter the canonical solver objective.
#
# form_xp / fixture_xp are exploratory views only.
# The maximum pull away from the model is 20%.
# ============================================================

SCENARIO_STRENGTH = 0.20

FORM_WEIGHTS = {
    6: 0.50,
    12: 0.30,
    38: 0.20,
}

FIXTURE_WEIGHTS = {
    6: 0.50,
    12: 0.30,
    38: 0.20,
}

FIXTURE_MULTIPLIER_MIN = 0.75
FIXTURE_MULTIPLIER_MAX = 1.25


def _load_config(
    path=CONFIG_PATH,
):
    with open(
        path,
        "r",
    ) as f:
        return yaml.safe_load(f)


def _weighted_columns(
    df,
    column_template,
    weights,
):
    """
    Weighted combination with row-level
    re-normalisation when some windows
    are missing.
    """

    numerator = pd.Series(
        0.0,
        index=df.index,
    )

    denominator = pd.Series(
        0.0,
        index=df.index,
    )

    for window, weight in (
        weights.items()
    ):
        col = column_template.format(
            window=window
        )

        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        valid = values.notna()

        numerator = (
            numerator
            + values.fillna(0)
            * float(weight)
        )

        denominator = (
            denominator
            + valid.astype(float)
            * float(weight)
        )

    return (
        numerator
        / denominator.replace(
            0,
            np.nan,
        )
    )


def _add_form_fixture_eda(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add fixture-level exploratory scenario fields.

    Canonical model:
        predicted_core_pp90
        xp

    Form view:
        raw recent form is L6/L12/L38,
        but only 20% of the difference
        from the model is applied.

    Fixture view:
        opponent strength is measured
        relative to the position-group
        average, then only 20% of the
        multiplier effect is applied.

    DefCon is unchanged in all scenarios.
    """

    out = df.copy()

    required = [
        "predicted_core_pp90",
        "final_predicted_minutes",
        "xp",
    ]

    missing = [
        c for c in required
        if c not in out.columns
    ]

    if missing:
        raise ValueError(
            "Cannot build form/fixture EDA. "
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # POSITION GROUP
    # --------------------------------------------------------

    if "position_group" not in out.columns:
        out["position_group"] = (
            out["position"]
            .map(
                {
                    "GK": "GK",
                    "DEF": "DEF",
                    "MID": "ATT",
                    "FWD": "ATT",
                }
            )
        )

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------

    out[
        "form_pp90_raw"
    ] = _weighted_columns(
        out,
        "player_core_pp90_l{window}",
        FORM_WEIGHTS,
    )

    # New / low-history players:
    # neutral scenario = model.
    out[
        "form_pp90_raw"
    ] = (
        out["form_pp90_raw"]
        .fillna(
            out[
                "predicted_core_pp90"
            ]
        )
    )

    out[
        "form_anchor_pp90"
    ] = (
        out[
            "predicted_core_pp90"
        ]
        + SCENARIO_STRENGTH
        * (
            out["form_pp90_raw"]
            - out[
                "predicted_core_pp90"
            ]
        )
    )

    # --------------------------------------------------------
    # FIXTURE
    # --------------------------------------------------------

    out[
        "opponent_points_allowed"
    ] = _weighted_columns(
        out,
        (
            "opp_pos_core_points_"
            "avg_l{window}_filled"
        ),
        FIXTURE_WEIGHTS,
    )

    # Compute the benchmark from unique
    # fixture x position-group rows so the
    # number of players at each position
    # does not bias the league average.
    fixture_benchmark = (
        out[
            [
                "fixture_id",
                "position_group",
                "opponent_points_allowed",
            ]
        ]
        .drop_duplicates(
            [
                "fixture_id",
                "position_group",
            ]
        )
        .groupby(
            "position_group"
        )[
            "opponent_points_allowed"
        ]
        .mean()
    )

    out[
        "opponent_points_allowed_avg"
    ] = (
        out["position_group"]
        .map(
            fixture_benchmark
        )
    )

    out[
        "fixture_multiplier_raw"
    ] = (
        out[
            "opponent_points_allowed"
        ]
        /
        out[
            "opponent_points_allowed_avg"
        ]
    )

    # Missing opponent information should
    # be neutral rather than null.
    out[
        "fixture_multiplier_raw"
    ] = (
        out[
            "fixture_multiplier_raw"
        ]
        .fillna(1.0)
    )

    out[
        "fixture_multiplier"
    ] = (
        out[
            "fixture_multiplier_raw"
        ]
        .clip(
            lower=(
                FIXTURE_MULTIPLIER_MIN
            ),
            upper=(
                FIXTURE_MULTIPLIER_MAX
            ),
        )
    )

    out[
        "fixture_anchor_pp90"
    ] = (
        out[
            "predicted_core_pp90"
        ]
        * (
            1
            + SCENARIO_STRENGTH
            * (
                out[
                    "fixture_multiplier"
                ]
                - 1
            )
        )
    )

    # --------------------------------------------------------
    # XP
    # --------------------------------------------------------

    mins = pd.to_numeric(
        out[
            "final_predicted_minutes"
        ],
        errors="coerce",
    ).fillna(0)

    if (
        "expected_defcon_points"
        in out.columns
    ):
        defcon = pd.to_numeric(
            out[
                "expected_defcon_points"
            ],
            errors="coerce",
        ).fillna(0)
    else:
        defcon = pd.Series(
            0.0,
            index=out.index,
        )

    out[
        "form_core_xp"
    ] = (
        out[
            "form_anchor_pp90"
        ]
        * mins
        / 90
    )

    out[
        "fixture_core_xp"
    ] = (
        out[
            "fixture_anchor_pp90"
        ]
        * mins
        / 90
    )

    out[
        "form_xp"
    ] = (
        out[
            "form_core_xp"
        ]
        + defcon
    )

    out[
        "fixture_xp"
    ] = (
        out[
            "fixture_core_xp"
        ]
        + defcon
    )

    return out


def build_solver_pool(
    config_path=CONFIG_PATH,
    save=True,
):
    config = _load_config(
        config_path
    )

    df = pd.read_csv(
        config[
            "input_predictions"
        ],
        low_memory=False,
    )

    required = [
        "player_code",
        "web_name",
        "position",
        "team_name",
        "price",
        "gameweek",
        "fixture_id",
        "final_predicted_minutes",
        "xp",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Prediction file missing "
            f"required columns: {missing}"
        )

    # ========================================================
    # ADD EXPLORATORY FORM / FIXTURE VIEWS
    # ========================================================

    df = _add_form_fixture_eda(
        df
    )

    # --------------------------------------------------------
    # Canonical grain = player_code
    # web_name is display-only and may duplicate.
    # --------------------------------------------------------

    position_check = (
        df.groupby(
            "player_code"
        )[
            "position"
        ]
        .nunique()
    )

    bad_positions = (
        position_check[
            position_check > 1
        ]
    )

    if not bad_positions.empty:
        raise ValueError(
            "Some player_codes map to "
            "multiple positions:\n"
            + bad_positions
            .head(20)
            .to_string()
        )

    # --------------------------------------------------------
    # LATEST PLAYER METADATA
    # --------------------------------------------------------

    meta_source = df.copy()

    meta_source["_gw"] = (
        pd.to_numeric(
            meta_source[
                "gameweek"
            ],
            errors="coerce",
        )
    )

    meta_source["_ko"] = (
        pd.to_datetime(
            meta_source.get(
                "kickoff_time"
            ),
            utc=True,
            errors="coerce",
        )
    )

    meta_cols = [
        "player_code",
        "web_name",
        "position",
        "team_name",
        "price",
    ] + [
        c
        for c in [
            "team_id",
            "team_code",
            "team_short_name",
            "status",
            "availability_pct",
            "is_new_player",
        ]
        if c in df.columns
    ]

    meta = (
        meta_source
        .sort_values(
            [
                "player_code",
                "_gw",
                "_ko",
            ]
        )
        .groupby(
            "player_code",
            as_index=False,
        )
        .tail(1)[
            meta_cols
        ]
        .copy()
    )

    if (
        meta[
            "player_code"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Metadata is not unique "
            "to player_code."
        )

    # --------------------------------------------------------
    # PLAYER-LEVEL EDA CONTEXT
    # --------------------------------------------------------

    player_context = (
        df.groupby(
            "player_code",
            as_index=False,
        )
        .agg(
            form_pp90_raw=(
                "form_pp90_raw",
                "first",
            ),
            model_xpp90_8gw=(
                "predicted_core_pp90",
                "mean",
            ),
            form_xpp90_8gw=(
                "form_anchor_pp90",
                "mean",
            ),
            fixture_xpp90_8gw=(
                "fixture_anchor_pp90",
                "mean",
            ),
            avg_fixture_multiplier_8gw=(
                "fixture_multiplier",
                "mean",
            ),
            min_fixture_multiplier_8gw=(
                "fixture_multiplier",
                "min",
            ),
            max_fixture_multiplier_8gw=(
                "fixture_multiplier",
                "max",
            ),
        )
    )

    meta = meta.merge(
        player_context,
        on="player_code",
        how="left",
        validate="one_to_one",
    )

    meta[
        "form_xpp90_delta"
    ] = (
        meta[
            "form_xpp90_8gw"
        ]
        - meta[
            "model_xpp90_8gw"
        ]
    )
    
    meta[
        "fixture_xpp90_delta"
    ] = (
        meta[
            "fixture_xpp90_8gw"
        ]
        - meta[
            "model_xpp90_8gw"
        ]
    )

    # --------------------------------------------------------
    # AGGREGATE FIXTURE ROWS TO PLAYER x GW
    # Handles DGWs correctly.
    # --------------------------------------------------------

    named_aggs = {
        "fixtures": (
            "fixture_id",
            "nunique",
        ),
        "xmins": (
            "final_predicted_minutes",
            "sum",
        ),
        "xp": (
            "xp",
            "sum",
        ),
        "form_xp": (
            "form_xp",
            "sum",
        ),
        "fixture_xp": (
            "fixture_xp",
            "sum",
        ),
    }

    if "core_xp" in df.columns:
        named_aggs[
            "core_xp"
        ] = (
            "core_xp",
            "sum",
        )

    if (
        "expected_defcon_points"
        in df.columns
    ):
        named_aggs[
            "defcon_xp"
        ] = (
            "expected_defcon_points",
            "sum",
        )

    player_gw = (
        df.groupby(
            [
                "player_code",
                "gameweek",
            ],
            as_index=False,
        )
        .agg(
            **named_aggs
        )
    )

    gameweeks = sorted(
        pd.to_numeric(
            player_gw[
                "gameweek"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not gameweeks:
        raise ValueError(
            "No forecast gameweeks found."
        )

    weights = config[
        "gw_weights"
    ]

    if (
        len(weights)
        < len(gameweeks)
    ):
        raise ValueError(
            f"Need {len(gameweeks)} "
            "GW weights, "
            f"but only {len(weights)} "
            "supplied."
        )

    gw_weight_map = {
        gw: float(
            weights[i]
        )
        for i, gw in enumerate(
            gameweeks
        )
    }

    # --------------------------------------------------------
    # ONE ROW PER PLAYER CODE
    # --------------------------------------------------------

    pool = meta.copy()

    metrics = [
        c
        for c in [
            "fixtures",
            "xmins",
            "xp",
            "form_xp",
            "fixture_xp",
            "core_xp",
            "defcon_xp",
        ]
        if c in player_gw.columns
    ]

    for gw in gameweeks:

        gw_df = (
            player_gw[
                pd.to_numeric(
                    player_gw[
                        "gameweek"
                    ],
                    errors="coerce",
                ).eq(gw)
            ]
            .set_index(
                "player_code"
            )
        )

        for metric in metrics:
            pool[
                f"{metric}_gw{gw}"
            ] = (
                pool[
                    "player_code"
                ]
                .map(
                    gw_df[
                        metric
                    ]
                )
                .fillna(0)
            )

    # --------------------------------------------------------
    # CANONICAL MODEL TOTALS
    # --------------------------------------------------------

    pool[
        "xp_8gw"
    ] = sum(
        pool[
            f"xp_gw{gw}"
        ]
        for gw in gameweeks
    )

    pool[
        "weighted_xp_8gw"
    ] = sum(
        pool[
            f"xp_gw{gw}"
        ]
        * gw_weight_map[gw]
        for gw in gameweeks
    )

    pool[
        "xmins_8gw"
    ] = sum(
        pool[
            f"xmins_gw{gw}"
        ]
        for gw in gameweeks
    )

    # --------------------------------------------------------
    # EDA SCENARIO TOTALS
    # --------------------------------------------------------

    pool[
        "model_xp_8gw"
    ] = pool[
        "xp_8gw"
    ]

    pool[
        "form_xp_8gw"
    ] = sum(
        pool[
            f"form_xp_gw{gw}"
        ]
        for gw in gameweeks
    )

    pool[
        "fixture_xp_8gw"
    ] = sum(
        pool[
            f"fixture_xp_gw{gw}"
        ]
        for gw in gameweeks
    )

    pool[
        "form_uplift_8gw"
    ] = (
        pool[
            "form_xp_8gw"
        ]
        - pool[
            "model_xp_8gw"
        ]
    )

    pool[
        "fixture_uplift_8gw"
    ] = (
        pool[
            "fixture_xp_8gw"
        ]
        - pool[
            "model_xp_8gw"
        ]
    )

    pool[
        "form_uplift_pct"
    ] = np.where(
        pool[
            "model_xp_8gw"
        ] != 0,
        (
            pool[
                "form_uplift_8gw"
            ]
            / pool[
                "model_xp_8gw"
            ]
        )
        * 100,
        np.nan,
    )

    pool[
        "fixture_uplift_pct"
    ] = np.where(
        pool[
            "model_xp_8gw"
        ] != 0,
        (
            pool[
                "fixture_uplift_8gw"
            ]
            / pool[
                "model_xp_8gw"
            ]
        )
        * 100,
        np.nan,
    )

    n_gws = len(
        gameweeks
    )

    pool[
        "model_xp_per_gw"
    ] = (
        pool[
            "model_xp_8gw"
        ]
        / n_gws
    )

    pool[
        "form_xp_per_gw"
    ] = (
        pool[
            "form_xp_8gw"
        ]
        / n_gws
    )

    pool[
        "fixture_xp_per_gw"
    ] = (
        pool[
            "fixture_xp_8gw"
        ]
        / n_gws
    )

    pool[
        "form_uplift_per_gw"
    ] = (
        pool[
            "form_uplift_8gw"
        ]
        / n_gws
    )

    pool[
        "fixture_uplift_per_gw"
    ] = (
        pool[
            "fixture_uplift_8gw"
        ]
        / n_gws
    )

    # --------------------------------------------------------
    # NEXT GW
    # --------------------------------------------------------

    first_gw = (
        gameweeks[0]
    )

    pool[
        "xmins_next_gw"
    ] = pool[
        f"xmins_gw{first_gw}"
    ]

    pool[
        "xp_next_gw"
    ] = pool[
        f"xp_gw{first_gw}"
    ]

    pool[
        "form_xp_next_gw"
    ] = pool[
        f"form_xp_gw{first_gw}"
    ]

    pool[
        "fixture_xp_next_gw"
    ] = pool[
        f"fixture_xp_gw{first_gw}"
    ]

    # --------------------------------------------------------
    # OPTIONAL SOLVER ELIGIBILITY FILTERS
    # --------------------------------------------------------

    pool[
        "solver_eligible"
    ] = True

    pool[
        "exclusion_reason"
    ] = ""

    min_xmins = config.get(
        "minimum_xmins_next_gw"
    )

    if min_xmins is not None:
        mask = (
            pool[
                "xmins_next_gw"
            ]
            < float(
                min_xmins
            )
        )

        pool.loc[
            mask,
            "solver_eligible",
        ] = False

        pool.loc[
            mask,
            "exclusion_reason",
        ] = "minimum_xmins"

    exclude_statuses = set(
        config.get(
            "exclude_statuses",
            [],
        )
    )

    if (
        exclude_statuses
        and "status"
        in pool.columns
    ):
        mask = (
            pool["status"]
            .isin(
                exclude_statuses
            )
        )

        pool.loc[
            mask,
            "solver_eligible",
        ] = False

        pool.loc[
            mask,
            "exclusion_reason",
        ] = "status"

    if (
        config.get(
            "exclude_new_players",
            False,
        )
        and "is_new_player"
        in pool.columns
    ):
        mask = (
            pool[
                "is_new_player"
            ]
            .fillna(False)
        )

        pool.loc[
            mask,
            "solver_eligible",
        ] = False

        pool.loc[
            mask,
            "exclusion_reason",
        ] = "new_player"

    # --------------------------------------------------------
    # HARD GRAIN VALIDATION
    # --------------------------------------------------------

    duplicate_codes = pool[
        pool[
            "player_code"
        ]
        .duplicated(
            keep=False
        )
    ]

    if (
        not duplicate_codes.empty
    ):
        raise ValueError(
            "Solver pool grain failure: "
            "player_code is not unique.\n"
            + duplicate_codes
            .head(20)
            .to_string(
                index=False
            )
        )

    pool = (
        pool.sort_values(
            [
                "solver_eligible",
                "weighted_xp_8gw",
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

    if save:
        output = Path(
            config[
                "solver_pool_output"
            ]
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pool.to_csv(
            output,
            index=False,
        )

    print(
        "=== SOLVER POOL ==="
    )

    print(
        f"Rows / players: "
        f"{len(pool):,}"
    )

    print(
        "Duplicate player_code:",
        pool[
            "player_code"
        ]
        .duplicated()
        .sum(),
    )

    print(
        "Duplicate web_name rows:",
        pool[
            "web_name"
        ]
        .duplicated(
            keep=False
        )
        .sum(),
        "(allowed)",
    )

    print(
        "Forecast GWs:",
        gameweeks,
    )

    print(
        "Eligible players:",
        int(
            pool[
                "solver_eligible"
            ]
            .sum()
        ),
    )

    print(
        "EDA scenario strength:",
        f"{SCENARIO_STRENGTH:.0%}",
    )

    print(
        "Canonical solver score "
        "remains xp / weighted_xp_8gw"
    )

    return {
        "pool": pool,
        "gameweeks": gameweeks,
        "gw_weights":
            gw_weight_map,
    }


if __name__ == "__main__":
    build_solver_pool()
