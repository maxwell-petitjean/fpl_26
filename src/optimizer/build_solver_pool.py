from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path("config/solver.yaml")

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

FIXTURE_SENSITIVE_POSITIONS = {
    "DEF",
    "MID",
}


def _load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _weighted_columns(
    df,
    column_template,
    weights,
):
    numerator = pd.Series(
        0.0,
        index=df.index,
    )
    denominator = pd.Series(
        0.0,
        index=df.index,
    )

    for window, weight in weights.items():
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

        numerator += (
            values.fillna(0)
            * float(weight)
        )

        denominator += (
            valid.astype(float)
            * float(weight)
        )

    return (
        numerator
        / denominator.replace(
            0,
            np.nan,
        )
    )


def _add_fixture_scenario_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    required = [
        "predicted_core_pp90",
        "final_predicted_minutes",
        "xp",
        "position",
    ]

    missing = [
        c for c in required
        if c not in out.columns
    ]

    if missing:
        raise ValueError(
            "Cannot build fixture scenario. "
            f"Missing columns: {missing}"
        )

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

    # EDA-only form field retained.
    out["form_pp90_raw"] = _weighted_columns(
        out,
        "player_core_pp90_l{window}",
        FORM_WEIGHTS,
    )

    out["form_pp90_raw"] = (
        out["form_pp90_raw"]
        .fillna(
            out["predicted_core_pp90"]
        )
    )

    # Fixture strength.
    out["opponent_points_allowed"] = _weighted_columns(
        out,
        "opp_pos_core_points_avg_l{window}_filled",
        FIXTURE_WEIGHTS,
    )

    # --------------------------------------------------------
    # FIXTURE BENCHMARK GRAIN
    #
    # A fixture has two opponent perspectives:
    #   Team A players -> Team B
    #   Team B players -> Team A
    #
    # Therefore the benchmark grain must retain opponent_team_id.
    # Otherwise one side of each fixture is arbitrarily dropped.
    # --------------------------------------------------------

    fixture_benchmark = (
        out[
            [
                "fixture_id",
                "opponent_team_id",
                "position_group",
                "opponent_points_allowed",
            ]
        ]
        .drop_duplicates(
            [
                "fixture_id",
                "opponent_team_id",
                "position_group",
            ]
        )
        .groupby(
            "position_group"
        )["opponent_points_allowed"]
        .mean()
    )

    out["opponent_points_allowed_avg"] = (
        out["position_group"]
        .map(
            fixture_benchmark
        )
    )

    out["fixture_multiplier_raw"] = (
        out["opponent_points_allowed"]
        / out["opponent_points_allowed_avg"]
    )

    out["fixture_multiplier_raw"] = (
        out["fixture_multiplier_raw"]
        .fillna(1.0)
    )

    out["fixture_multiplier"] = (
        out["fixture_multiplier_raw"]
        .clip(
            lower=FIXTURE_MULTIPLIER_MIN,
            upper=FIXTURE_MULTIPLIER_MAX,
        )
    )

    sensitive = (
        out["position"]
        .isin(
            FIXTURE_SENSITIVE_POSITIONS
        )
    )

    # Full fixture xPP90.
    # FWD/GK remain canonical.
    out["fixture_full_xpp90"] = (
        out["predicted_core_pp90"]
    )

    out.loc[
        sensitive,
        "fixture_full_xpp90",
    ] = (
        out.loc[
            sensitive,
            "predicted_core_pp90",
        ]
        * out.loc[
            sensitive,
            "fixture_multiplier",
        ]
    )

    out["fixture_xpp90_delta"] = (
        out["fixture_full_xpp90"]
        - out["predicted_core_pp90"]
    )

    mins = pd.to_numeric(
        out["final_predicted_minutes"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # CANONICAL PROJECTION DIAGNOSTICS
    #
    # core_projection_90:
    #   canonical core xP assuming 90 minutes.
    #
    # core_projection_xmins:
    #   canonical core xP after FINAL expected minutes.
    #
    # final_predicted_minutes already reflects:
    #   hybrid/model minutes
    #   -> availability scaling
    #   -> manual override as final authority
    # --------------------------------------------------------

    out["core_projection_90"] = (
        out["predicted_core_pp90"]
    )

    out["model_core_xp_calc"] = (
        out["predicted_core_pp90"]
        * mins
        / 90
    )

    out["core_projection_xmins"] = (
        out["model_core_xp_calc"]
    )

    # --------------------------------------------------------
    # MEAN-REVERSION VIEW
    #
    # Full left-hand slider anchor:
    # player long-run L38 core PP90.
    #
    # Missing L38 falls back to the canonical model, so new /
    # low-history players are neutral rather than dropped.
    # Applies to ALL positions.
    # --------------------------------------------------------

    if "player_core_pp90_l38" in out.columns:
        out["mean_reversion_xpp90"] = pd.to_numeric(
            out["player_core_pp90_l38"],
            errors="coerce",
        )
    else:
        out["mean_reversion_xpp90"] = np.nan

    out["mean_reversion_xpp90"] = (
        out["mean_reversion_xpp90"]
        .fillna(
            out["predicted_core_pp90"]
        )
    )

    out["mean_reversion_xpp90_delta"] = (
        out["mean_reversion_xpp90"]
        - out["predicted_core_pp90"]
    )

    out["mean_reversion_core_xp"] = (
        out["mean_reversion_xpp90"]
        * mins
        / 90
    )

    # Preserve DefCon / any other non-core component in canonical xp.
    out["mean_reversion_xp"] = (
        out["xp"]
        + (
            out["mean_reversion_core_xp"]
            - out["model_core_xp_calc"]
        )
    )

    out["mean_reversion_xp_delta"] = (
        out["mean_reversion_xp"]
        - out["xp"]
    )

    out["fixture_full_core_xp"] = (
        out["fixture_full_xpp90"]
        * mins
        / 90
    )

    # Preserve everything already included in canonical xp.
    out["fixture_full_xp"] = (
        out["xp"]
        + (
            out["fixture_full_core_xp"]
            - out["model_core_xp_calc"]
        )
    )

    out["fixture_full_xp_delta"] = (
        out["fixture_full_xp"]
        - out["xp"]
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

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Prediction file missing "
            f"required columns: {missing}"
        )

    df = _add_fixture_scenario_fields(
        df
    )

    position_check = (
        df.groupby(
            "player_code"
        )["position"]
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

    meta_source = df.copy()

    meta_source["_gw"] = pd.to_numeric(
        meta_source["gameweek"],
        errors="coerce",
    )

    meta_source["_ko"] = pd.to_datetime(
        meta_source.get(
            "kickoff_time"
        ),
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
            "team_id",
            "team_code",
            "team_short_name",
            "fpl_element_id",
            "fpl_element_id_gw",
            "status",
            "availability_pct",
            "is_new_player",
        ]
        if c in df.columns
    ]

    # FPL's public picks endpoint returns the current-season
    # element id. Keep a canonical fpl_element_id in the solver pool
    # so a user's FPL team can be mapped to player_code without names.
    if (
        "fpl_element_id" not in meta_source.columns
        and "fpl_element_id_gw" in meta_source.columns
    ):
        meta_source["fpl_element_id"] = (
            pd.to_numeric(
                meta_source["fpl_element_id_gw"],
                errors="coerce",
            )
        )

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
        .tail(1)[meta_cols]
        .copy()
    )

    if (
        meta["player_code"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Metadata is not unique "
            "to player_code."
        )

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
            mean_reversion_xpp90_8gw=(
                "mean_reversion_xpp90",
                "mean",
            ),
            fixture_xpp90_full_8gw=(
                "fixture_full_xpp90",
                "mean",
            ),
            avg_final_predicted_minutes_8gw=(
                "final_predicted_minutes",
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

    player_context["mean_reversion_xpp90_delta_8gw"] = (
        player_context["mean_reversion_xpp90_8gw"]
        - player_context["model_xpp90_8gw"]
    )

    player_context["fixture_xpp90_delta_8gw"] = (
        player_context["fixture_xpp90_full_8gw"]
        - player_context["model_xpp90_8gw"]
    )

    meta = meta.merge(
        player_context,
        on="player_code",
        how="left",
        validate="one_to_one",
    )

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
        "core_projection_90": (
            "core_projection_90",
            "sum",
        ),
        "core_projection_xmins": (
            "core_projection_xmins",
            "sum",
        ),
        "mean_reversion_xp": (
            "mean_reversion_xp",
            "sum",
        ),
        "fixture_full_xp": (
            "fixture_full_xp",
            "sum",
        ),
    }

    if "core_xp" in df.columns:
        named_aggs["core_xp"] = (
            "core_xp",
            "sum",
        )

    if (
        "expected_defcon_points"
        in df.columns
    ):
        named_aggs["defcon_xp"] = (
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
            player_gw["gameweek"],
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
            f"Need {len(gameweeks)} GW weights, "
            f"but only {len(weights)} supplied."
        )

    gw_weight_map = {
        gw: float(
            weights[i]
        )
        for i, gw in enumerate(
            gameweeks
        )
    }

    pool = meta.copy()

    metrics = [
        c
        for c in [
            "fixtures",
            "xmins",
            "xp",
            "core_projection_90",
            "core_projection_xmins",
            "mean_reversion_xp",
            "fixture_full_xp",
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
            .set_index(
                "player_code"
            )
        )

        for metric in metrics:
            pool[
                f"{metric}_gw{gw}"
            ] = (
                pool["player_code"]
                .map(
                    gw_df[metric]
                )
                .fillna(0)
            )

    pool["xp_8gw"] = sum(
        pool[
            f"xp_gw{gw}"
        ]
        for gw in gameweeks
    )

    pool["weighted_xp_8gw"] = sum(
        pool[
            f"xp_gw{gw}"
        ]
        * gw_weight_map[gw]
        for gw in gameweeks
    )

    pool["xmins_8gw"] = sum(
        pool[
            f"xmins_gw{gw}"
        ]
        for gw in gameweeks
    )

    # --------------------------------------------------------
    # PROJECTION DIAGNOSTICS
    # --------------------------------------------------------

    pool["core_projection_90_8gw"] = sum(
        pool[
            f"core_projection_90_gw{gw}"
        ]
        for gw in gameweeks
    )

    pool["core_projection_xmins_8gw"] = sum(
        pool[
            f"core_projection_xmins_gw{gw}"
        ]
        for gw in gameweeks
    )

    pool["xmins_projection_factor_8gw"] = np.where(
        pool["core_projection_90_8gw"].abs() > 1e-12,
        (
            pool["core_projection_xmins_8gw"]
            / pool["core_projection_90_8gw"]
        ),
        np.nan,
    )

    # --------------------------------------------------------
    # FULL MEAN-REVERSION TOTALS
    # --------------------------------------------------------

    pool["mean_reversion_xp_8gw"] = sum(
        pool[
            f"mean_reversion_xp_gw{gw}"
        ]
        for gw in gameweeks
    )

    pool[
        "mean_reversion_weighted_xp_8gw"
    ] = sum(
        pool[
            f"mean_reversion_xp_gw{gw}"
        ]
        * gw_weight_map[gw]
        for gw in gameweeks
    )

    pool["mean_reversion_uplift_8gw"] = (
        pool["mean_reversion_xp_8gw"]
        - pool["xp_8gw"]
    )

    # --------------------------------------------------------
    # FULL FIXTURE TOTALS
    # --------------------------------------------------------

    pool["fixture_full_xp_8gw"] = sum(
        pool[
            f"fixture_full_xp_gw{gw}"
        ]
        for gw in gameweeks
    )

    pool[
        "fixture_full_weighted_xp_8gw"
    ] = sum(
        pool[
            f"fixture_full_xp_gw{gw}"
        ]
        * gw_weight_map[gw]
        for gw in gameweeks
    )

    pool["fixture_full_uplift_8gw"] = (
        pool["fixture_full_xp_8gw"]
        - pool["xp_8gw"]
    )

    pool[
        "fixture_full_weighted_uplift_8gw"
    ] = (
        pool[
            "fixture_full_weighted_xp_8gw"
        ]
        - pool[
            "weighted_xp_8gw"
        ]
    )

    first_gw = gameweeks[0]

    pool["xmins_next_gw"] = (
        pool[
            f"xmins_gw{first_gw}"
        ]
    )

    pool["xp_next_gw"] = (
        pool[
            f"xp_gw{first_gw}"
        ]
    )

    pool["mean_reversion_xp_next_gw"] = (
        pool[
            f"mean_reversion_xp_gw{first_gw}"
        ]
    )

    pool["fixture_full_xp_next_gw"] = (
        pool[
            f"fixture_full_xp_gw{first_gw}"
        ]
    )

    pool["solver_eligible"] = True
    pool["exclusion_reason"] = ""

    min_xmins = config.get(
        "minimum_xmins_next_gw"
    )

    if min_xmins is not None:
        mask = (
            pool["xmins_next_gw"]
            < float(min_xmins)
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
            pool["is_new_player"]
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
    # HARD SCORING-ANCHOR VALIDATION
    #
    # All anchors must use FINAL predicted minutes.
    # These checks prevent pre-availability / pre-manual minutes
    # from leaking back into the solver scenarios.
    # --------------------------------------------------------

    expected_model_core = (
        pd.to_numeric(
            df["predicted_core_pp90"],
            errors="coerce",
        )
        * pd.to_numeric(
            df["final_predicted_minutes"],
            errors="coerce",
        )
        / 90
    )

    expected_mean_core = (
        pd.to_numeric(
            df["mean_reversion_xpp90"],
            errors="coerce",
        )
        * pd.to_numeric(
            df["final_predicted_minutes"],
            errors="coerce",
        )
        / 90
    )

    expected_fixture_core = (
        pd.to_numeric(
            df["fixture_full_xpp90"],
            errors="coerce",
        )
        * pd.to_numeric(
            df["final_predicted_minutes"],
            errors="coerce",
        )
        / 90
    )

    if not np.allclose(
        df["model_core_xp_calc"].fillna(0),
        expected_model_core.fillna(0),
        atol=1e-9,
        rtol=1e-9,
    ):
        raise ValueError(
            "Canonical core xP is not using final_predicted_minutes."
        )

    if not np.allclose(
        df["mean_reversion_core_xp"].fillna(0),
        expected_mean_core.fillna(0),
        atol=1e-9,
        rtol=1e-9,
    ):
        raise ValueError(
            "Mean-reversion xP is not using final_predicted_minutes."
        )

    if not np.allclose(
        df["fixture_full_core_xp"].fillna(0),
        expected_fixture_core.fillna(0),
        atol=1e-9,
        rtol=1e-9,
    ):
        raise ValueError(
            "Fixture-biased xP is not using final_predicted_minutes."
        )

    duplicate_codes = pool[
        pool[
            "player_code"
        ]
        .duplicated(
            keep=False
        )
    ]

    if not duplicate_codes.empty:
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

    print("=== SOLVER POOL ===")
    print(
        f"Rows / players: "
        f"{len(pool):,}"
    )
    print(
        "Forecast GWs:",
        gameweeks,
    )
    print(
        "Fixture-sensitive positions:",
        sorted(
            FIXTURE_SENSITIVE_POSITIONS
        ),
    )
    print(
        "Canonical xp unchanged; "
        "mean-reversion and full-fixture anchors added."
    )

    return {
        "pool": pool,
        "gameweeks": gameweeks,
        "gw_weights": gw_weight_map,
    }


if __name__ == "__main__":
    build_solver_pool()