from pathlib import Path

import pandas as pd
import pulp
import yaml


CONFIG_PATH = Path("config/solver.yaml")


def _load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def optimize_squad(
    config_path=CONFIG_PATH,
    save=True,
):
    config = _load_config(config_path)

    pool = pd.read_csv(
        config["solver_pool_output"],
        low_memory=False,
    )

    pool = pool[
        pool["solver_eligible"].fillna(False)
    ].copy()

    if pool["player_code"].duplicated().any():
        raise ValueError(
            "Solver pool is not unique to player_code."
        )

    # --------------------------------------------------------
    # Identify forecast GWs
    # --------------------------------------------------------

    gameweeks = sorted(
        int(c.replace("xp_gw", ""))
        for c in pool.columns
        if c.startswith("xp_gw")
    )

    if not gameweeks:
        raise ValueError(
            "No xp_gw* columns found in solver pool."
        )

    weights = config["gw_weights"]

    if len(weights) < len(gameweeks):
        raise ValueError(
            f"Need at least {len(gameweeks)} GW weights."
        )

    gw_weight_map = {
        gw: float(weights[i])
        for i, gw in enumerate(gameweeks)
    }

    bench_weight = float(
        config.get("bench_weight", 0.0)
    )

    # --------------------------------------------------------
    # Canonical player IDs
    # --------------------------------------------------------

    players = (
        pool["player_code"]
        .astype(int)
        .tolist()
    )

    row = (
        pool.assign(
            player_code=pool["player_code"].astype(int)
        )
        .set_index("player_code")
    )

    # --------------------------------------------------------
    # MILP
    # --------------------------------------------------------

    model = pulp.LpProblem(
        "FPL_Initial_Squad",
        pulp.LpMaximize,
    )

    # Squad selection.
    squad = {
        p: pulp.LpVariable(
            f"squad_{p}",
            cat="Binary",
        )
        for p in players
    }

    # Starting XI for each forecast GW.
    start = {
        (p, gw): pulp.LpVariable(
            f"start_{p}_{gw}",
            cat="Binary",
        )
        for p in players
        for gw in gameweeks
    }

    # --------------------------------------------------------
    # OBJECTIVE
    #
    # XI receives full weighted xP.
    # Bench receives bench_weight * weighted xP.
    #
    # Equivalent:
    # bench_weight * squad_xP
    # + (1-bench_weight) * starter_xP
    # --------------------------------------------------------

    objective_terms = []

    for p in players:
        for gw in gameweeks:
            xp = float(
                row.loc[p, f"xp_gw{gw}"]
            )
            weight = gw_weight_map[gw]

            objective_terms.append(
                weight
                * xp
                * (
                    bench_weight * squad[p]
                    + (
                        1.0 - bench_weight
                    ) * start[(p, gw)]
                )
            )

    model += pulp.lpSum(
        objective_terms
    )

    # --------------------------------------------------------
    # SQUAD RULES
    # --------------------------------------------------------

    model += (
        pulp.lpSum(
            squad[p]
            for p in players
        )
        == 15,
        "squad_size",
    )

    model += (
        pulp.lpSum(
            float(row.loc[p, "price"])
            * squad[p]
            for p in players
        )
        <= float(config["budget"]),
        "budget",
    )

    for position, required in config["squad"].items():
        pos_players = [
            p for p in players
            if row.loc[p, "position"] == position
        ]

        model += (
            pulp.lpSum(
                squad[p]
                for p in pos_players
            )
            == int(required),
            f"squad_position_{position}",
        )

    for team_name in sorted(
        row["team_name"].dropna().unique()
    ):
        team_players = [
            p for p in players
            if row.loc[p, "team_name"] == team_name
        ]

        model += (
            pulp.lpSum(
                squad[p]
                for p in team_players
            )
            <= int(
                config["max_players_per_team"]
            ),
            f"max_team_{team_name}",
        )

    # --------------------------------------------------------
    # STARTING XI RULES BY GW
    # --------------------------------------------------------

    xi_cfg = config["starting_xi"]

    for gw in gameweeks:
        # Starter must be owned.
        for p in players:
            model += (
                start[(p, gw)]
                <= squad[p],
                f"starter_owned_{p}_{gw}",
            )

        model += (
            pulp.lpSum(
                start[(p, gw)]
                for p in players
            )
            == int(xi_cfg["total"]),
            f"xi_size_{gw}",
        )

        position_bounds = {
            "GK": (
                int(xi_cfg["GK_min"]),
                int(xi_cfg["GK_max"]),
            ),
            "DEF": (
                int(xi_cfg["DEF_min"]),
                int(xi_cfg["DEF_max"]),
            ),
            "MID": (
                int(xi_cfg["MID_min"]),
                int(xi_cfg["MID_max"]),
            ),
            "FWD": (
                int(xi_cfg["FWD_min"]),
                int(xi_cfg["FWD_max"]),
            ),
        }

        for position, (
            pos_min,
            pos_max,
        ) in position_bounds.items():

            pos_players = [
                p for p in players
                if row.loc[p, "position"] == position
            ]

            expr = pulp.lpSum(
                start[(p, gw)]
                for p in pos_players
            )

            model += (
                expr >= pos_min,
                f"xi_{position}_min_{gw}",
            )

            model += (
                expr <= pos_max,
                f"xi_{position}_max_{gw}",
            )

    # --------------------------------------------------------
    # SOLVE
    # --------------------------------------------------------

    solver_cfg = config.get(
        "solver",
        {},
    )

    solver = pulp.PULP_CBC_CMD(
        msg=bool(
            solver_cfg.get(
                "msg",
                False,
            )
        ),
        timeLimit=int(
            solver_cfg.get(
                "time_limit_seconds",
                60,
            )
        ),
    )

    status = model.solve(
        solver
    )

    status_name = pulp.LpStatus[
        model.status
    ]

    if status_name != "Optimal":
        raise RuntimeError(
            f"Solver did not return Optimal. "
            f"Status: {status_name}"
        )

    # --------------------------------------------------------
    # SQUAD OUTPUT
    # --------------------------------------------------------

    selected_ids = [
        p for p in players
        if pulp.value(
            squad[p]
        ) > 0.5
    ]

    selected = (
        row.loc[selected_ids]
        .reset_index()
        .copy()
    )

    selected["selected"] = True

    selected["weighted_starter_xp"] = 0.0
    selected["starts"] = 0

    for gw in gameweeks:
        selected[
            f"start_gw{gw}"
        ] = selected[
            "player_code"
        ].map(
            {
                p: int(
                    pulp.value(
                        start[(p, gw)]
                    ) > 0.5
                )
                for p in selected_ids
            }
        )

        selected["starts"] += (
            selected[f"start_gw{gw}"]
        )

        selected[
            "weighted_starter_xp"
        ] += (
            selected[f"start_gw{gw}"]
            * selected[f"xp_gw{gw}"]
            * gw_weight_map[gw]
        )

    selected = selected.sort_values(
        [
            "position",
            "weighted_xp_8gw",
        ],
        ascending=[
            True,
            False,
        ],
    )

    total_cost = selected[
        "price"
    ].sum()

    # --------------------------------------------------------
    # LINEUP OUTPUT
    # --------------------------------------------------------

    lineup_rows = []

    for gw in gameweeks:
        for p in selected_ids:
            is_start = int(
                pulp.value(
                    start[(p, gw)]
                ) > 0.5
            )

            lineup_rows.append(
                {
                    "gameweek": gw,
                    "player_code": p,
                    "web_name": row.loc[
                        p,
                        "web_name",
                    ],
                    "position": row.loc[
                        p,
                        "position",
                    ],
                    "team_name": row.loc[
                        p,
                        "team_name",
                    ],
                    "price": row.loc[
                        p,
                        "price",
                    ],
                    "xp": row.loc[
                        p,
                        f"xp_gw{gw}",
                    ],
                    "xmins": row.loc[
                        p,
                        f"xmins_gw{gw}",
                    ],
                    "starting_xi": is_start,
                }
            )

    lineups = pd.DataFrame(
        lineup_rows
    ).sort_values(
        [
            "gameweek",
            "starting_xi",
            "xp",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    objective_value = pulp.value(
        model.objective
    )

    if save:
        squad_output = Path(
            config["solution_output"]
        )
        lineup_output = Path(
            config["lineup_output"]
        )

        squad_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected.to_csv(
            squad_output,
            index=False,
        )

        lineups.to_csv(
            lineup_output,
            index=False,
        )

    print("=== OPTIMAL FPL SQUAD ===")
    print(f"Status: {status_name}")
    print(
        f"Cost: £{total_cost:.1f}m "
        f"/ £{float(config['budget']):.1f}m"
    )
    print(
        f"Objective value: "
        f"{objective_value:.2f}"
    )
    print()
    print(
        selected[
            [
                "player_code",
                "web_name",
                "position",
                "team_name",
                "price",
                "xp_8gw",
                "weighted_xp_8gw",
                "starts",
            ]
        ]
        .to_string(
            index=False
        )
    )

    return {
        "squad": selected,
        "lineups": lineups,
        "status": status_name,
        "objective_value": objective_value,
        "total_cost": total_cost,
        "gameweeks": gameweeks,
    }


if __name__ == "__main__":
    optimize_squad()
