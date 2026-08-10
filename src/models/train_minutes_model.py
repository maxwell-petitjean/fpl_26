from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_PATH = Path("data/features/fct_gw_features_historic.csv")
CONFIG_PATH = Path("config/model_minutes.yaml")
OUTPUT_DIR = Path("data/outputs/minutes_model")


POSITION_LEVELS = ["GK", "DEF", "MID", "FWD"]
CONTEXT_LEVELS = {
    "kickoff_day_type": ["primetime", "midweek", "weekend"],
    "kickoff_time_type": ["lunchtime", "afternoon", "evening"],
}


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _select_features(df: pd.DataFrame, config: dict):
    features = []

    for prefix in config["minutes_feature_prefixes"]:
        features.extend(
            sorted(c for c in df.columns if c.startswith(prefix))
        )

    chance_col = config.get("chance_of_playing_feature")
    if (
        config.get("use_chance_of_playing", False)
        and chance_col in df.columns
    ):
        features.append(chance_col)

    return list(dict.fromkeys(features))


def _encode_inputs(
    df: pd.DataFrame,
    numeric_features: list[str],
    config: dict,
) -> pd.DataFrame:
    X = df[numeric_features].copy()

    if config.get("use_position", True):
        for level in POSITION_LEVELS:
            X[f"position__{level}"] = (
                df["position"].astype(str).eq(level).astype(int)
            )

    if config.get("use_context", False):
        for col in config.get("context_features", []):
            if col not in df.columns:
                continue

            levels = CONTEXT_LEVELS.get(
                col,
                sorted(df[col].dropna().astype(str).unique()),
            )

            for level in levels:
                X[f"{col}__{level}"] = (
                    df[col].astype(str).eq(level).astype(int)
                )

    return X


def _metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def _add_last6_std(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate std dev of prior 6 fixture minutes, continuously across seasons.
    Current fixture is excluded via shift(1).
    """
    out = df.copy()

    out["kickoff_time"] = pd.to_datetime(
        out["kickoff_time"], utc=True, errors="coerce"
    )

    out = out.sort_values(
        ["player_code", "kickoff_time", "season", "gameweek", "fixture_id"]
    ).reset_index(drop=True)

    shifted_minutes = (
        out.groupby("player_code", sort=False)["minutes"]
        .shift(1)
    )

    out["player_minutes_std_l6"] = (
        shifted_minutes
        .groupby(out["player_code"])
        .rolling(window=6, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    return out


def _load_manual_overrides(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "player_code",
                "web_name",
                "override_minutes",
                "reason",
                "active",
            ]
        )

    overrides = pd.read_csv(path, low_memory=False)

    required = {
        "player_code",
        "override_minutes",
        "active",
    }
    missing = required - set(overrides.columns)

    if missing:
        raise ValueError(
            f"Manual override file missing columns: {sorted(missing)}"
        )

    overrides["active"] = (
        overrides["active"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "Y"])
    )

    overrides["override_minutes"] = pd.to_numeric(
        overrides["override_minutes"], errors="coerce"
    )

    overrides = overrides[
        overrides["active"]
        & overrides["player_code"].notna()
        & overrides["override_minutes"].notna()
    ].copy()

    overrides["player_code"] = pd.to_numeric(
        overrides["player_code"], errors="coerce"
    )

    overrides["override_minutes"] = overrides[
        "override_minutes"
    ].clip(0, 90)

    return overrides


def _apply_manual_overrides(
    pred_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Apply active manual overrides on top of hybrid predictions.

    Intended primarily for current/live predictions.
    During historic validation this usually does nothing unless the user
    deliberately supplies matching player codes.
    """
    overrides = _load_manual_overrides(
        config["manual_override_path"]
    )

    out = pred_df.copy()
    out["manual_override_applied"] = False
    out["manual_override_reason"] = pd.NA

    if overrides.empty:
        return out

    keep = [
        c for c in [
            "player_code",
            "override_minutes",
            "reason",
        ]
        if c in overrides.columns
    ]

    overrides = overrides[keep].drop_duplicates(
        "player_code", keep="last"
    )

    out = out.merge(
        overrides,
        on="player_code",
        how="left",
        validate="many_to_one",
    )

    mask = out["override_minutes"].notna()

    out.loc[mask, "final_predicted_minutes"] = (
        out.loc[mask, "override_minutes"]
    )
    out.loc[mask, "manual_override_applied"] = True

    if "reason" in out.columns:
        out.loc[mask, "manual_override_reason"] = (
            out.loc[mask, "reason"]
        )
        out = out.drop(columns=["reason"])

    return out


def run_minutes_model_test(
    feature_path=FEATURE_PATH,
    config_path=CONFIG_PATH,
    output_dir=OUTPUT_DIR,
):
    config = load_config(config_path)
    df = pd.read_csv(feature_path, low_memory=False)

    target = config["target"]
    baseline_feature = config["baseline_feature"]
    stable_threshold = float(config["stable_std_threshold"])

    required = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "player_name",
        "position",
        "minutes",
        target,
        baseline_feature,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature table missing required columns: {missing}"
        )

    df = _add_last6_std(df)

    numeric_features = _select_features(df, config)

    print("=== HYBRID MINUTES MODEL TEST ===")
    print(f"Rows: {len(df):,}")
    print(f"Rolling model features: {len(numeric_features):,}")
    print(f"Baseline: {baseline_feature}")
    print(f"Stable std threshold: <= {stable_threshold:.2f}")
    print(f"Test seasons: {config['test_seasons']}")

    all_predictions = []
    summary_rows = []
    split_rows = []
    importance_rows = []

    for test_season in config["test_seasons"]:
        train = df[df["season"] < test_season].copy()
        test = df[df["season"] == test_season].copy()

        train = train[train[target].notna()].copy()

        # LightGBM is only trained on historically volatile rows.
        volatile_train = train[
            train["player_minutes_std_l6"].isna()
            | (train["player_minutes_std_l6"] > stable_threshold)
        ].copy()

        test_eval = test[
            test[target].notna()
            & test[baseline_feature].notna()
        ].copy()

        if volatile_train.empty or test_eval.empty:
            print(
                f"{test_season}: skipped "
                f"(volatile train={len(volatile_train):,}, "
                f"test={len(test_eval):,})"
            )
            continue

        X_train = _encode_inputs(
            volatile_train, numeric_features, config
        )
        y_train = volatile_train[target].astype(float)

        X_test = _encode_inputs(
            test_eval, numeric_features, config
        )
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0,
        )

        lgbm_model = LGBMRegressor(
            **config["lightgbm"]
        )
        lgbm_model.fit(X_train, y_train)

        lgbm_pred = np.clip(
            lgbm_model.predict(X_test),
            0,
            90,
        )

        baseline_pred = (
            test_eval[baseline_feature]
            .astype(float)
            .clip(0, 90)
            .to_numpy()
        )

        stable_mask = (
            test_eval["player_minutes_std_l6"].notna()
            & (
                test_eval["player_minutes_std_l6"]
                <= stable_threshold
            )
        ).to_numpy()

        hybrid_pred = np.where(
            stable_mask,
            baseline_pred,
            lgbm_pred,
        )

        pred_frame = test_eval[
            [
                "season",
                "gameweek",
                "fixture_id",
                "player_code",
                "player_name",
                "position",
                target,
                baseline_feature,
                "player_minutes_std_l6",
            ]
        ].copy()

        pred_frame = pred_frame.rename(
            columns={target: "actual_minutes"}
        )

        pred_frame["pred_avg_mins_l6"] = baseline_pred
        pred_frame["pred_lightgbm"] = lgbm_pred
        pred_frame["stable_last6"] = stable_mask
        pred_frame["prediction_source"] = np.where(
            stable_mask,
            "avg_mins_l6",
            "lightgbm",
        )
        pred_frame["hybrid_predicted_minutes"] = hybrid_pred
        pred_frame["final_predicted_minutes"] = hybrid_pred

        pred_frame = _apply_manual_overrides(
            pred_frame, config
        )

        model_predictions = {
            "avg_mins_l6": baseline_pred,
            "lightgbm_all_test_rows": lgbm_pred,
            "hybrid": hybrid_pred,
            "final_with_manual_overrides": pred_frame[
                "final_predicted_minutes"
            ].to_numpy(),
        }

        for model_name, pred in model_predictions.items():
            m = _metrics(
                pred_frame["actual_minutes"],
                pred,
            )

            summary_rows.append(
                {
                    "test_season": test_season,
                    "model": model_name,
                    "rows": len(pred_frame),
                    **m,
                }
            )

        split_summary = (
            pred_frame
            .assign(
                abs_error=lambda x: (
                    x["actual_minutes"]
                    - x["hybrid_predicted_minutes"]
                ).abs()
            )
            .groupby(
                ["prediction_source"],
                as_index=False,
            )
            .agg(
                rows=("player_code", "size"),
                actual_avg=("actual_minutes", "mean"),
                predicted_avg=(
                    "hybrid_predicted_minutes", "mean"
                ),
                mae=("abs_error", "mean"),
                avg_std_l6=("player_minutes_std_l6", "mean"),
            )
        )

        split_summary["test_season"] = test_season
        split_rows.append(split_summary)

        importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": lgbm_model.feature_importances_,
                "test_season": test_season,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )
        importance_rows.append(importance)

        all_predictions.append(pred_frame)

        print(f"\n--- {test_season} ---")
        season_results = pd.DataFrame(
            [
                r for r in summary_rows
                if r["test_season"] == test_season
            ]
        ).sort_values("mae")

        print(
            season_results[
                ["model", "mae", "rmse", "rows"]
            ].to_string(index=False)
        )

        print("\nPrediction source split:")
        print(split_summary.to_string(index=False))

    summary = pd.DataFrame(summary_rows)
    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    split_summary = (
        pd.concat(split_rows, ignore_index=True)
        if split_rows
        else pd.DataFrame()
    )
    importance = (
        pd.concat(importance_rows, ignore_index=True)
        if importance_rows
        else pd.DataFrame()
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "minutes_hybrid_summary.csv",
        index=False,
    )
    predictions.to_csv(
        output_dir / "minutes_hybrid_predictions.csv",
        index=False,
    )
    split_summary.to_csv(
        output_dir / "minutes_hybrid_source_split.csv",
        index=False,
    )
    importance.to_csv(
        output_dir / "minutes_hybrid_lightgbm_importance.csv",
        index=False,
    )

    print("\n=== OVERALL ===")

    if not summary.empty:
        overall = (
            summary
            .groupby("model", as_index=False)
            .agg(
                avg_mae=("mae", "mean"),
                avg_rmse=("rmse", "mean"),
            )
            .sort_values("avg_mae")
        )
        print(overall.to_string(index=False))

    print(f"\nSaved outputs to: {output_dir}")

    return {
        "summary": summary,
        "predictions": predictions,
        "source_split": split_summary,
        "importance": importance,
        "numeric_features": numeric_features,
    }


if __name__ == "__main__":
    run_minutes_model_test()
