import numpy as np
import pandas as pd
import pulp


BUDGET = 100.0
MAX_PLAYERS_PER_TEAM = 3

SQUAD_REQUIREMENTS = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

XI_BOUNDS = {
    "GK": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}

GW_WEIGHTS = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
]

# Only Starting XI xP contributes to the primary objective.
BENCH_WEIGHT = 0.00

# Tiny secondary preference for a cheaper squad when Starting-XI xP
# is effectively identical.
COST_TIEBREAKER = 1e-6

FIXTURE_SENSITIVE_POSITIONS = {
    "DEF",
    "MID",
}


def apply_model_bias(
    solver_pool: pd.DataFrame,
    model_bias: float = 0.0,
) -> pd.DataFrame:
    """
    Signed scoring-bias control.

    -1.00 = fully revert core scoring to player L38
     0.00 = canonical model
    +1.00 = full extra fixture adjustment
    +1.25 = aggressive fixture chasing

    Left side:
        mean reversion applies to ALL positions.

    Right side:
        extra fixture emphasis applies to DEF / MID only.
        GK / FWD remain canonical.
    """

    bias = float(
        model_bias
    )

    if (
        bias < -1.0
        or bias > 1.25
    ):
        raise ValueError(
            "model_bias must be between "
            "-1.0 and 1.25."
        )

    pool = solver_pool.copy()

    gameweeks = sorted(
        int(
            c.replace(
                "xp_gw",
                "",
            )
        )
        for c in pool.columns
        if (
            c.startswith("xp_gw")
            and c.replace(
                "xp_gw",
                ""
            ).isdigit()
        )
    )

    if not gameweeks:
        raise ValueError(
            "No xp_gw* columns found."
        )

    if len(GW_WEIGHTS) < len(gameweeks):
        raise ValueError(
            "Not enough GW weights."
        )

    fixture_mask = (
        pool["position"]
        .isin(
            FIXTURE_SENSITIVE_POSITIONS
        )
    )

    for gw in gameweeks:

        base_col = (
            f"xp_gw{gw}"
        )

        mean_col = (
            f"mean_reversion_xp_gw{gw}"
        )

        fixture_col = (
            f"fixture_full_xp_gw{gw}"
        )

        scenario_col = (
            f"scenario_xp_gw{gw}"
        )

        # Always start from the canonical model.
        pool[
            scenario_col
        ] = pool[
            base_col
        ]

        # ----------------------------------------------------
        # LEFT: REVERT TOWARDS LONG-RUN L38
        # Applies to all positions.
        # ----------------------------------------------------

        if bias < 0:

            if mean_col not in pool.columns:
                raise ValueError(
                    f"Missing {mean_col}. "
                    "Rebuild solver pool first."
                )

            strength = abs(
                bias
            )

            pool[
                scenario_col
            ] = (
                pool[
                    base_col
                ]
                + strength
                * (
                    pool[
                        mean_col
                    ]
                    - pool[
                        base_col
                    ]
                )
            )

        # ----------------------------------------------------
        # RIGHT: EXTRA FIXTURE EMPHASIS
        # DEF / MID only.
        # ----------------------------------------------------

        elif bias > 0:

            if fixture_col not in pool.columns:
                raise ValueError(
                    f"Missing {fixture_col}. "
                    "Rebuild solver pool first."
                )

            pool.loc[
                fixture_mask,
                scenario_col,
            ] = (
                pool.loc[
                    fixture_mask,
                    base_col,
                ]
                + bias
                * (
                    pool.loc[
                        fixture_mask,
                        fixture_col,
                    ]
                    - pool.loc[
                        fixture_mask,
                        base_col,
                    ]
                )
            )

    # --------------------------------------------------------
    # HARD MODEL-BIAS INVARIANTS
    #
    # 0%:
    #   scenario MUST equal canonical xp exactly.
    #
    # -100%:
    #   scenario MUST equal the mean-reversion anchor exactly.
    #
    # +100%:
    #   DEF/MID MUST equal full fixture xP;
    #   GK/FWD MUST remain canonical.
    # --------------------------------------------------------

    if abs(bias) < 1e-12:
        for gw in gameweeks:
            if not np.allclose(
                pool[f"scenario_xp_gw{gw}"].fillna(0),
                pool[f"xp_gw{gw}"].fillna(0),
                atol=1e-9,
                rtol=1e-9,
            ):
                raise ValueError(
                    f"0% model bias invariant failed for GW{gw}."
                )

    if abs(bias + 1.0) < 1e-12:
        for gw in gameweeks:
            if not np.allclose(
                pool[f"scenario_xp_gw{gw}"].fillna(0),
                pool[f"mean_reversion_xp_gw{gw}"].fillna(0),
                atol=1e-9,
                rtol=1e-9,
            ):
                raise ValueError(
                    f"-100% model bias invariant failed for GW{gw}."
                )

    if abs(bias - 1.0) < 1e-12:
        for gw in gameweeks:
            scenario = pool[f"scenario_xp_gw{gw}"]
            canonical = pool[f"xp_gw{gw}"]
            fixture = pool[f"fixture_full_xp_gw{gw}"]

            if not np.allclose(
                scenario.loc[fixture_mask].fillna(0),
                fixture.loc[fixture_mask].fillna(0),
                atol=1e-9,
                rtol=1e-9,
            ):
                raise ValueError(
                    f"+100% fixture invariant failed for GW{gw}: "
                    "DEF/MID must equal fixture_full_xp."
                )

            if not np.allclose(
                scenario.loc[~fixture_mask].fillna(0),
                canonical.loc[~fixture_mask].fillna(0),
                atol=1e-9,
                rtol=1e-9,
            ):
                raise ValueError(
                    f"+100% fixture invariant failed for GW{gw}: "
                    "GK/FWD must remain canonical."
                )

    pool[
        "scenario_xp_8gw"
    ] = sum(
        pool[
            f"scenario_xp_gw{gw}"
        ]
        for gw in gameweeks
    )

    gw_weight_map = {
        gw: GW_WEIGHTS[i]
        for i, gw in enumerate(
            gameweeks
        )
    }

    pool[
        "scenario_weighted_xp_8gw"
    ] = sum(
        pool[
            f"scenario_xp_gw{gw}"
        ]
        * gw_weight_map[gw]
        for gw in gameweeks
    )

    pool[
        "scenario_delta_8gw"
    ] = (
        pool[
            "scenario_xp_8gw"
        ]
        - pool[
            "xp_8gw"
        ]
    )

    first_gw = (
        gameweeks[0]
    )

    pool[
        "scenario_xp_next_gw"
    ] = (
        pool[
            f"scenario_xp_gw{first_gw}"
        ]
    )

    pool[
        "model_bias"
    ] = bias

    return pool


def solve_wildcard(
    solver_pool: pd.DataFrame,
    model_bias: float = 0.0,
    horizon_weeks: int = 8,
    excluded_player_codes=(),
    excluded_teams=(),
):

    pool = apply_model_bias(
        solver_pool,
        model_bias=(
            model_bias
        ),
    )

    excluded_player_codes = {
        int(x)
        for x in excluded_player_codes
    }

    excluded_teams = {
        str(x)
        for x in excluded_teams
    }

    if excluded_player_codes:
        pool = (
            pool[
                ~pool["player_code"]
                .astype(int)
                .isin(
                    excluded_player_codes
                )
            ]
            .copy()
        )

    if excluded_teams:
        pool = (
            pool[
                ~pool["team_name"]
                .isin(
                    excluded_teams
                )
            ]
            .copy()
        )

    pool = (
        pool[
            pool["solver_eligible"]
            .fillna(False)
        ]
        .copy()
    )

    if (
        pool["player_code"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Solver pool is not unique "
            "to player_code."
        )

    all_gameweeks = sorted(
        int(
            c.replace(
                "scenario_xp_gw",
                "",
            )
        )
        for c in pool.columns
        if (
            c.startswith(
                "scenario_xp_gw"
            )
            and c.replace(
                "scenario_xp_gw",
                ""
            ).isdigit()
        )
    )

    if horizon_weeks not in {
        1,
        3,
        5,
        8,
    }:
        raise ValueError(
            "horizon_weeks must be one of {1, 3, 5, 8}."
        )

    if len(all_gameweeks) < horizon_weeks:
        raise ValueError(
            f"Requested {horizon_weeks} gameweeks, "
            f"but only {len(all_gameweeks)} are available."
        )

    gameweeks = all_gameweeks[:horizon_weeks]

    gw_weight_map = {
        gw: GW_WEIGHTS[i]
        for i, gw in enumerate(
            gameweeks
        )
    }

    for position, required in SQUAD_REQUIREMENTS.items():
        available = int(
            pool[
                pool["position"].eq(position)
            ].shape[0]
        )

        if available < required:
            raise ValueError(
                f"Not enough eligible {position} players after exclusions. "
                f"Need {required}, have {available}."
            )

    players = (
        pool["player_code"]
        .astype(int)
        .tolist()
    )

    row = (
        pool.assign(
            player_code=(
                pool["player_code"]
                .astype(int)
            )
        )
        .set_index(
            "player_code"
        )
    )

    model = pulp.LpProblem(
        "FPL_Wildcard",
        pulp.LpMaximize,
    )

    squad = {
        p: pulp.LpVariable(
            f"squad_{p}",
            cat="Binary",
        )
        for p in players
    }

    start = {
        (p, gw):
            pulp.LpVariable(
                f"start_{p}_{gw}",
                cat="Binary",
            )
        for p in players
        for gw in gameweeks
    }

    starting_xp_objective = pulp.lpSum(
        gw_weight_map[gw]
        * float(
            row.loc[
                p,
                f"scenario_xp_gw{gw}",
            ]
        )
        * start[
            (
                p,
                gw,
            )
        ]
        for p in players
        for gw in gameweeks
    )

    squad_cost_tiebreak = pulp.lpSum(
        float(
            row.loc[
                p,
                "price",
            ]
        )
        * squad[p]
        for p in players
    )

    model += (
        starting_xp_objective
        - COST_TIEBREAKER
        * squad_cost_tiebreak
    )

    model += (
        pulp.lpSum(
            squad[p]
            for p in players
        )
        == 15
    )

    model += (
        pulp.lpSum(
            float(
                row.loc[
                    p,
                    "price",
                ]
            )
            * squad[p]
            for p in players
        )
        <= BUDGET
    )

    for position, required in (
        SQUAD_REQUIREMENTS.items()
    ):
        pos_players = [
            p for p in players
            if row.loc[
                p,
                "position",
            ] == position
        ]

        model += (
            pulp.lpSum(
                squad[p]
                for p in pos_players
            )
            == required
        )

    for team_name in (
        row["team_name"]
        .dropna()
        .unique()
    ):
        team_players = [
            p for p in players
            if row.loc[
                p,
                "team_name",
            ] == team_name
        ]

        model += (
            pulp.lpSum(
                squad[p]
                for p in team_players
            )
            <= MAX_PLAYERS_PER_TEAM
        )

    for gw in gameweeks:
        for p in players:
            model += (
                start[
                    (
                        p,
                        gw,
                    )
                ]
                <= squad[p]
            )

        model += (
            pulp.lpSum(
                start[
                    (
                        p,
                        gw,
                    )
                ]
                for p in players
            )
            == 11
        )

        for position, (
            minimum,
            maximum,
        ) in XI_BOUNDS.items():
            pos_players = [
                p for p in players
                if row.loc[
                    p,
                    "position",
                ] == position
            ]

            expr = pulp.lpSum(
                start[
                    (
                        p,
                        gw,
                    )
                ]
                for p in pos_players
            )

            model += (
                expr >= minimum
            )

            model += (
                expr <= maximum
            )

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=60,
    )

    model.solve(
        solver
    )

    status = (
        pulp.LpStatus[
            model.status
        ]
    )

    if status != "Optimal":
        raise RuntimeError(
            f"Solver status: {status}"
        )

    selected_ids = [
        p for p in players
        if pulp.value(
            squad[p]
        ) > 0.5
    ]

    selected = (
        row.loc[
            selected_ids
        ]
        .reset_index()
        .copy()
    )

    selected[
        "starts"
    ] = 0

    for gw in gameweeks:
        selected[
            f"start_gw{gw}"
        ] = (
            selected[
                "player_code"
            ]
            .map(
                {
                    p: int(
                        pulp.value(
                            start[
                                (
                                    p,
                                    gw,
                                )
                            ]
                        )
                        > 0.5
                    )
                    for p in selected_ids
                }
            )
        )

        selected[
            "starts"
        ] += (
            selected[
                f"start_gw{gw}"
            ]
        )

    lineup_rows = []

    for gw in gameweeks:
        for p in selected_ids:
            scenario_xp = (
                row.loc[
                    p,
                    f"scenario_xp_gw{gw}",
                ]
            )
            base_xp = (
                row.loc[
                    p,
                    f"xp_gw{gw}",
                ]
            )

            lineup_rows.append(
                {
                    "gameweek":
                        gw,
                    "player_code":
                        p,
                    "web_name":
                        row.loc[
                            p,
                            "web_name",
                        ],
                    "position":
                        row.loc[
                            p,
                            "position",
                        ],
                    "team_name":
                        row.loc[
                            p,
                            "team_name",
                        ],
                    "price":
                        row.loc[
                            p,
                            "price",
                        ],
                    "base_xp":
                        base_xp,
                    "xp":
                        scenario_xp,
                    "fixture_uplift":
                        (
                            scenario_xp
                            - base_xp
                        ),
                    "xmins":
                        row.loc[
                            p,
                            f"xmins_gw{gw}",
                        ],
                    "starting_xi":
                        int(
                            pulp.value(
                                start[
                                    (
                                        p,
                                        gw,
                                    )
                                ]
                            )
                            > 0.5
                        ),
                }
            )

    lineups = pd.DataFrame(
        lineup_rows
    )

    return {
        "squad": selected,
        "lineups": lineups,
        "scored_pool": pool.reset_index(
            drop=True
        ),
        "status": status,
        "objective_value": float(
            pulp.value(
                model.objective
            )
        ),
        "total_cost": float(
            selected["price"]
            .sum()
        ),
        "gameweeks": gameweeks,
        "horizon_weeks": int(
            horizon_weeks
        ),
        "excluded_player_codes": tuple(
            sorted(excluded_player_codes)
        ),
        "excluded_teams": tuple(
            sorted(excluded_teams)
        ),
        "model_bias": float(
            model_bias
        ),
    }
