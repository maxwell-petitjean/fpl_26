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

BENCH_WEIGHT = 0.10

FIXTURE_SENSITIVE_POSITIONS = {
    "DEF",
    "MID",
}


def apply_fixture_sensitivity(
    solver_pool: pd.DataFrame,
    fixture_sensitivity: float = 0.0,
) -> pd.DataFrame:
    """
    0.00 = canonical model
    0.90 = 90% of full fixture effect
    1.00 = full fixture effect
    1.25 = 125% of full fixture effect

    Only DEF and MID move.
    """

    sensitivity = float(
        fixture_sensitivity
    )

    if sensitivity < 0:
        raise ValueError(
            "fixture_sensitivity must be >= 0."
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

    sensitive_mask = (
        pool["position"]
        .isin(
            FIXTURE_SENSITIVE_POSITIONS
        )
    )

    for gw in gameweeks:
        base_col = (
            f"xp_gw{gw}"
        )
        fixture_col = (
            f"fixture_full_xp_gw{gw}"
        )
        scenario_col = (
            f"scenario_xp_gw{gw}"
        )

        if fixture_col not in pool.columns:
            raise ValueError(
                f"Missing {fixture_col}. "
                "Rebuild solver pool first."
            )

        pool[
            scenario_col
        ] = pool[
            base_col
        ]

        pool.loc[
            sensitive_mask,
            scenario_col,
        ] = (
            pool.loc[
                sensitive_mask,
                base_col,
            ]
            + sensitivity
            * (
                pool.loc[
                    sensitive_mask,
                    fixture_col,
                ]
                - pool.loc[
                    sensitive_mask,
                    base_col,
                ]
            )
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
        "scenario_fixture_uplift_8gw"
    ] = (
        pool["scenario_xp_8gw"]
        - pool["xp_8gw"]
    )

    first_gw = gameweeks[0]

    pool[
        "scenario_xp_next_gw"
    ] = (
        pool[
            f"scenario_xp_gw{first_gw}"
        ]
    )

    pool[
        "fixture_sensitivity"
    ] = sensitivity

    return pool


def solve_wildcard(
    solver_pool: pd.DataFrame,
    fixture_sensitivity: float = 0.0,
):

    pool = apply_fixture_sensitivity(
        solver_pool,
        fixture_sensitivity=(
            fixture_sensitivity
        ),
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

    gameweeks = sorted(
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

    gw_weight_map = {
        gw: GW_WEIGHTS[i]
        for i, gw in enumerate(
            gameweeks
        )
    }

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

    model += pulp.lpSum(
        gw_weight_map[gw]
        * float(
            row.loc[
                p,
                f"scenario_xp_gw{gw}",
            ]
        )
        * (
            BENCH_WEIGHT
            * squad[p]
            + (
                1
                - BENCH_WEIGHT
            )
            * start[
                (
                    p,
                    gw,
                )
            ]
        )
        for p in players
        for gw in gameweeks
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
        "fixture_sensitivity": float(
            fixture_sensitivity
        ),
    }
