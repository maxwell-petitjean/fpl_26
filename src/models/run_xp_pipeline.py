from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

CONFIG_PATH = Path("config/model_xp_pipeline.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _require(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def run_xp_pipeline_backtest(config_path=CONFIG_PATH):
    config = load_config(config_path)

    xpp = pd.read_csv(config["xpp_predictions_path"], low_memory=False)
    minutes = pd.read_csv(config["minutes_predictions_path"], low_memory=False)
    defcon = pd.read_csv(config["defcon_predictions_path"], low_memory=False)

    join_keys = ["season", "gameweek", "fixture_id", "player_code"]

    _require(
        xpp,
        join_keys + [
            "player_name",
            "position",
            "target_core_pp90",
            "predicted_core_pp90",
            "total_points",
        ],
        "xPP90 predictions",
    )

    _require(
        minutes,
        join_keys + ["final_predicted_minutes"],
        "Minutes predictions",
    )

    _require(
        defcon,
        join_keys + ["pred_defcon_probability"],
        "DefCon predictions",
    )

    xpp = xpp[xpp["season"].isin(config["test_seasons"])].copy()

    # Apply season-specific calibration to raw xPP90 holdout predictions.
    xpp["calibrated_core_pp90"] = np.nan

    for season, params in config["xpp90_calibration"].items():
        mask = xpp["season"].eq(season)

        xpp.loc[mask, "calibrated_core_pp90"] = (
            float(params["intercept"])
            + float(params["slope"])
            * pd.to_numeric(
                xpp.loc[mask, "predicted_core_pp90"],
                errors="coerce",
            )
        )

    # Keep minutes fields useful for diagnostics.
    minute_keep = join_keys + [
        c for c in [
            "final_predicted_minutes",
            "prediction_source",
            "manual_override_applied",
            "player_minutes_std_l6",
        ]
        if c in minutes.columns
    ]

    # DefCon currently only has a holdout for part of 2025-26.
    # Missing rows get zero for historic end-to-end backtesting.
    defcon_keep = join_keys + [
        c for c in [
            "pred_defcon_probability",
            "defcon_hit",
        ]
        if c in defcon.columns
    ]

    out = xpp.merge(
        minutes[minute_keep],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        defcon[defcon_keep],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )

    out["final_predicted_minutes"] = pd.to_numeric(
        out["final_predicted_minutes"], errors="coerce"
    )

    out["pred_defcon_probability"] = pd.to_numeric(
        out["pred_defcon_probability"], errors="coerce"
    )

    # Pre-DefCon season = zero.
    out.loc[
        out["season"] < config["defcon_start_season"],
        "pred_defcon_probability",
    ] = 0.0

    # For 2025-26 rows outside the DefCon holdout window, probability is unknown.
    # Keep this flag explicit rather than pretending these are true zeroes.
    out["defcon_prediction_available"] = (
        out["pred_defcon_probability"].notna()
    )

    out["expected_defcon_points"] = (
        float(config["defcon_points_per_hit"])
        * out["pred_defcon_probability"]
    )

    out["core_xp"] = (
        out["calibrated_core_pp90"]
        * out["final_predicted_minutes"]
        / 90.0
    )

    # xp_core_only is available for all joined rows.
    out["xp_core_only"] = out["core_xp"]

    # Full xp is only available where DefCon probability is genuinely known,
    # or where DefCon did not exist yet.
    full_xp_mask = (
        out["season"].lt(config["defcon_start_season"])
        | out["defcon_prediction_available"]
    )

    out["xp"] = np.where(
        full_xp_mask,
        out["core_xp"] + out["expected_defcon_points"].fillna(0),
        np.nan,
    )

    out["actual_points"] = pd.to_numeric(
        out["total_points"], errors="coerce"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = []

    for season, season_df in out.groupby("season"):

        for metric_name in ["xp_core_only", "xp"]:
            valid = (
                season_df[metric_name].notna()
                & season_df["actual_points"].notna()
            )

            if not valid.any():
                continue

            d = season_df.loc[valid].copy()
            m = _metrics(d["actual_points"], d[metric_name])

            summary_rows.append(
                {
                    "season": season,
                    "prediction": metric_name,
                    "rows": len(d),
                    "actual_avg_points": d["actual_points"].mean(),
                    "predicted_avg_points": d[metric_name].mean(),
                    "bias": (
                        d[metric_name].mean()
                        - d["actual_points"].mean()
                    ),
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "spearman": spearmanr(
                        d[metric_name],
                        d["actual_points"],
                        nan_policy="omit",
                    ).statistic,
                }
            )

    summary = pd.DataFrame(summary_rows)

    # --------------------------------------------------------
    # POSITION SUMMARY
    # --------------------------------------------------------

    valid_full = (
        out["xp"].notna()
        & out["actual_points"].notna()
    )

    position_summary = (
        out.loc[valid_full]
        .groupby(
            ["season", "position"],
            as_index=False,
            observed=True,
        )
        .agg(
            rows=("player_code", "size"),
            actual_avg_points=("actual_points", "mean"),
            predicted_avg_xp=("xp", "mean"),
        )
    )

    position_summary["bias"] = (
        position_summary["predicted_avg_xp"]
        - position_summary["actual_avg_points"]
    )

    # --------------------------------------------------------
    # GAMEWEEK SUMMARY
    # --------------------------------------------------------

    gameweek_summary = (
        out.loc[valid_full]
        .groupby(
            ["season", "gameweek"],
            as_index=False,
        )
        .agg(
            rows=("player_code", "size"),
            actual_avg_points=("actual_points", "mean"),
            predicted_avg_xp=("xp", "mean"),
        )
    )

    gameweek_summary["bias"] = (
        gameweek_summary["predicted_avg_xp"]
        - gameweek_summary["actual_avg_points"]
    )

    # --------------------------------------------------------
    # XP DECILES
    # --------------------------------------------------------

    decile_frames = []

    for season, d in out.loc[valid_full].groupby("season"):
        tmp = d.copy()

        tmp["xp_decile"] = pd.qcut(
            tmp["xp"],
            q=10,
            labels=False,
            duplicates="drop",
        ) + 1

        dec = (
            tmp.groupby("xp_decile", as_index=False)
            .agg(
                rows=("player_code", "size"),
                predicted_xp=("xp", "mean"),
                actual_points=("actual_points", "mean"),
            )
        )

        dec["season"] = season
        decile_frames.append(dec)

    deciles = (
        pd.concat(decile_frames, ignore_index=True)
        if decile_frames
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        output_dir / "xp_backtest_predictions.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "xp_backtest_summary.csv",
        index=False,
    )
    position_summary.to_csv(
        output_dir / "xp_backtest_position_summary.csv",
        index=False,
    )
    gameweek_summary.to_csv(
        output_dir / "xp_backtest_gameweek_summary.csv",
        index=False,
    )
    deciles.to_csv(
        output_dir / "xp_backtest_deciles.csv",
        index=False,
    )

    print("=== EXPECTED POINTS BACKTEST ===")
    print(summary.to_string(index=False))
    print()
    print(
        "DefCon prediction coverage:",
        f"{out['defcon_prediction_available'].mean() * 100:.1f}% of joined rows",
    )

    return {
        "predictions": out,
        "summary": summary,
        "position_summary": position_summary,
        "gameweek_summary": gameweek_summary,
        "deciles": deciles,
    }


if __name__ == "__main__":
    run_xp_pipeline_backtest()
