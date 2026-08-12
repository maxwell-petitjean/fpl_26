from pathlib import Path
import pandas as pd
import yaml

CONFIG_PATH = Path("config/solver.yaml")


def _load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_solver_pool(config_path=CONFIG_PATH, save=True):
    config = _load_config(config_path)

    df = pd.read_csv(
        config["input_predictions"],
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

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Prediction file missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Canonical grain = player_code
    # web_name is display-only and may duplicate.
    # --------------------------------------------------------

    position_check = (
        df.groupby("player_code")["position"]
        .nunique()
    )

    bad_positions = position_check[
        position_check > 1
    ]

    if not bad_positions.empty:
        raise ValueError(
            "Some player_codes map to multiple positions:\n"
            + bad_positions.head(20).to_string()
        )

    # Latest snapshot metadata per player_code.
    meta_source = df.copy()
    meta_source["_gw"] = pd.to_numeric(
        meta_source["gameweek"],
        errors="coerce",
    )
    meta_source["_ko"] = pd.to_datetime(
        meta_source.get("kickoff_time"),
        utc=True,
        errors="coerce",
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
            "status",
            "availability_pct",
            "is_new_player",
        ]
        if c in df.columns
    ]

    meta = (
        meta_source
        .sort_values(
            ["player_code", "_gw", "_ko"]
        )
        .groupby("player_code", as_index=False)
        .tail(1)[meta_cols]
        .copy()
    )

    if meta["player_code"].duplicated().any():
        raise ValueError(
            "Metadata is not unique to player_code."
        )

    # --------------------------------------------------------
    # Aggregate fixture rows to player x gameweek.
    # This handles DGWs correctly.
    # --------------------------------------------------------

    named_aggs = {
        "fixtures": ("fixture_id", "nunique"),
        "xmins": ("final_predicted_minutes", "sum"),
        "xp": ("xp", "sum"),
    }

    if "core_xp" in df.columns:
        named_aggs["core_xp"] = ("core_xp", "sum")

    if "expected_defcon_points" in df.columns:
        named_aggs["defcon_xp"] = (
            "expected_defcon_points",
            "sum",
        )

    player_gw = (
        df.groupby(
            ["player_code", "gameweek"],
            as_index=False,
        )
        .agg(**named_aggs)
    )

    gameweeks = sorted(
        pd.to_numeric(
            player_gw["gameweek"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not gameweeks:
        raise ValueError("No forecast gameweeks found.")

    weights = config["gw_weights"]

    if len(weights) < len(gameweeks):
        raise ValueError(
            f"Need {len(gameweeks)} GW weights, "
            f"but only {len(weights)} supplied."
        )

    gw_weight_map = {
        gw: float(weights[i])
        for i, gw in enumerate(gameweeks)
    }

    # --------------------------------------------------------
    # One row per player_code.
    # --------------------------------------------------------

    pool = meta.copy()

    metrics = [
        c
        for c in [
            "fixtures",
            "xmins",
            "xp",
            "core_xp",
            "defcon_xp",
        ]
        if c in player_gw.columns
    ]

    for gw in gameweeks:
        gw_df = (
            player_gw[
                pd.to_numeric(
                    player_gw["gameweek"],
                    errors="coerce",
                ).eq(gw)
            ]
            .set_index("player_code")
        )

        for metric in metrics:
            pool[f"{metric}_gw{gw}"] = (
                pool["player_code"]
                .map(gw_df[metric])
                .fillna(0)
            )

    pool["xp_8gw"] = sum(
        pool[f"xp_gw{gw}"]
        for gw in gameweeks
    )

    pool["weighted_xp_8gw"] = sum(
        pool[f"xp_gw{gw}"] * gw_weight_map[gw]
        for gw in gameweeks
    )

    pool["xmins_8gw"] = sum(
        pool[f"xmins_gw{gw}"]
        for gw in gameweeks
    )

    first_gw = gameweeks[0]
    pool["xmins_next_gw"] = pool[f"xmins_gw{first_gw}"]
    pool["xp_next_gw"] = pool[f"xp_gw{first_gw}"]

    # --------------------------------------------------------
    # Optional solver eligibility filters.
    # --------------------------------------------------------

    pool["solver_eligible"] = True
    pool["exclusion_reason"] = ""

    min_xmins = config.get("minimum_xmins_next_gw")
    if min_xmins is not None:
        mask = (
            pool["xmins_next_gw"]
            < float(min_xmins)
        )
        pool.loc[mask, "solver_eligible"] = False
        pool.loc[
            mask,
            "exclusion_reason",
        ] = "minimum_xmins"

    exclude_statuses = set(
        config.get("exclude_statuses", [])
    )

    if exclude_statuses and "status" in pool.columns:
        mask = pool["status"].isin(exclude_statuses)
        pool.loc[mask, "solver_eligible"] = False
        pool.loc[
            mask,
            "exclusion_reason",
        ] = "status"

    if (
        config.get("exclude_new_players", False)
        and "is_new_player" in pool.columns
    ):
        mask = pool["is_new_player"].fillna(False)
        pool.loc[mask, "solver_eligible"] = False
        pool.loc[
            mask,
            "exclusion_reason",
        ] = "new_player"

    # --------------------------------------------------------
    # Hard grain validation.
    # --------------------------------------------------------

    duplicate_codes = pool[
        pool["player_code"].duplicated(
            keep=False
        )
    ]

    if not duplicate_codes.empty:
        raise ValueError(
            "Solver pool grain failure: player_code "
            "is not unique.\n"
            + duplicate_codes.head(20).to_string(index=False)
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
        .reset_index(drop=True)
    )

    if save:
        output = Path(
            config["solver_pool_output"]
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pool.to_csv(
            output,
            index=False,
        )

    print("=== SOLVER POOL ===")
    print(f"Rows / players: {len(pool):,}")
    print(
        "Duplicate player_code:",
        pool["player_code"].duplicated().sum(),
    )
    print(
        "Duplicate web_name rows:",
        pool["web_name"].duplicated(
            keep=False
        ).sum(),
        "(allowed)",
    )
    print(
        "Forecast GWs:",
        gameweeks,
    )
    print(
        "Eligible players:",
        int(pool["solver_eligible"].sum()),
    )

    return {
        "pool": pool,
        "gameweeks": gameweeks,
        "gw_weights": gw_weight_map,
    }


if __name__ == "__main__":
    build_solver_pool()
