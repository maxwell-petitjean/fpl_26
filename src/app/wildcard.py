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


def solve_wildcard(
    solver_pool: pd.DataFrame,
):

    pool = (
        solver_pool[
            solver_pool[
                "solver_eligible"
            ]
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

    if len(
        GW_WEIGHTS
    ) < len(
        gameweeks
    ):
        raise ValueError(
            "Not enough GW weights."
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
                pool[
                    "player_code"
                ]
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

    # --------------------------------------------------------
    # Objective:
    # starters get full xP;
    # bench gets 10% xP.
    # --------------------------------------------------------

    model += pulp.lpSum(
        gw_weight_map[gw]
        * float(
            row.loc[
                p,
                f"xp_gw{gw}",
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

    # --------------------------------------------------------
    # Squad rules
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Starting XI rules for each GW
    # --------------------------------------------------------

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

    solver = (
        pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=60,
        )
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
                    for p
                    in selected_ids
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

            lineup_rows.append(
                {
                    "gameweek": gw,
                    "player_code": p,
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
                    "xp":
                        row.loc[
                            p,
                            f"xp_gw{gw}",
                        ],
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
        "status": status,
        "objective_value":
            float(
                pulp.value(
                    model.objective
                )
            ),
        "total_cost":
            float(
                selected[
                    "price"
                ]
                .sum()
            ),
        "gameweeks": gameweeks,
    }
