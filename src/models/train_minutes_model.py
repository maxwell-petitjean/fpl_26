from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
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
            sorted(
                c for c in df.columns
                if c.startswith(prefix)
            )
        )

    # Optional live/historic injury field.
    chance_col = config.get("chance_of_playing_feature")
    if (
        config.get("use_chance_of_playing", False)
        and chance_col in df.columns
    ):
        features.append(chance_col)

    # Remove duplicates while keeping order.
    features = list(dict.fromkeys(features))
    return features


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
                sorted(df[col].dropna().astype(str).unique())
            )

            for level in levels:
                X[f"{col}__{level}"] = (
                    df[col].astype(str).eq(level).astype(int)
                )

    return X


def _metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(
            y_true, y_pred
        ) ** 0.5,
    }


def _minutes_bucket(minutes):
    if minutes == 0:
        return "0"
    if minutes < 45:
        return "1-44"
    if minutes < 60:
        return "45-59"
    if minutes < 90:
        return "60-89"
    return "90+"


def run_minutes_model_test(
    feature_path=FEATURE_PATH,
    config_path=CONFIG_PATH,
    output_dir=OUTPUT_DIR,
):
    config = load_config(config_path)
    df = pd.read_csv(feature_path, low_memory=False)

    target = config["target"]
    baseline_feature = config["baseline_feature"]

    required = [
        "season",
        "player_code",
        "player_name",
        "position",
        target,
        baseline_feature,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature table missing required columns: {missing}"
        )

    numeric_features = _select_features(df, config)

    if not numeric_features:
        raise ValueError("No minutes features selected.")

    print("=== MINUTES MODEL TEST ===")
    print(f"Rows: {len(df):,}")
    print(f"Numeric rolling features: {len(numeric_features):,}")
    print(f"Baseline: {baseline_feature}")
    print(f"Test seasons: {config['test_seasons']}")

    all_predictions = []
    summary_rows = []
    bucket_rows = []
    importance_rows = []

    for test_season in config["test_seasons"]:
        train = df[df["season"] < test_season].copy()
        test = df[df["season"] == test_season].copy()

        # Target must exist.
        train = train[train[target].notna()].copy()
        test = test[test[target].notna()].copy()

        # Baseline comparison needs l6 available.
        test_eval = test[test[baseline_feature].notna()].copy()

        if train.empty or test_eval.empty:
            print(
                f"{test_season}: skipped "
                f"(train={len(train):,}, test={len(test_eval):,})"
            )
            continue

        X_train = _encode_inputs(
            train, numeric_features, config
        )
        X_test = _encode_inputs(
            test_eval, numeric_features, config
        )

        # Ensure identical columns/order.
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0,
        )

        y_train = train[target].astype(float)
        y_test = test_eval[target].astype(float)

        baseline_pred = (
            test_eval[baseline_feature]
            .astype(float)
            .clip(0, 90)
            .to_numpy()
        )

        # HistGradientBoosting cannot accept +/- inf.
        X_train_hgb = X_train.replace(
            [np.inf, -np.inf], np.nan
        )
        X_test_hgb = X_test.replace(
            [np.inf, -np.inf], np.nan
        )

        hist_model = HistGradientBoostingRegressor(
            **config["hist_gradient_boosting"]
        )
        hist_model.fit(X_train_hgb, y_train)
        hist_pred = np.clip(
            hist_model.predict(X_test_hgb),
            0,
            90,
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

        model_predictions = {
            "avg_mins_l6": baseline_pred,
            "hist_gradient_boosting": hist_pred,
            "lightgbm": lgbm_pred,
        }

        for model_name, pred in model_predictions.items():
            m = _metrics(y_test, pred)

            summary_rows.append(
                {
                    "test_season": test_season,
                    "model": model_name,
                    "rows": len(y_test),
                    **m,
                }
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
            ]
        ].copy()

        pred_frame = pred_frame.rename(
            columns={target: "actual_minutes"}
        )
        pred_frame["pred_avg_mins_l6"] = baseline_pred
        pred_frame["pred_hist_gradient_boosting"] = hist_pred
        pred_frame["pred_lightgbm"] = lgbm_pred

        all_predictions.append(pred_frame)

        # Error diagnostics by actual minutes bucket.
        for model_name, pred in model_predictions.items():
            tmp = pred_frame[
                [
                    "actual_minutes",
                    "position",
                ]
            ].copy()
            tmp["prediction"] = pred
            tmp["minutes_bucket"] = tmp[
                "actual_minutes"
            ].apply(_minutes_bucket)
            tmp["abs_error"] = (
                tmp["actual_minutes"]
                - tmp["prediction"]
            ).abs()

            grouped = (
                tmp
                .groupby(
                    ["minutes_bucket"],
                    as_index=False,
                    observed=True,
                )
                .agg(
                    rows=("actual_minutes", "size"),
                    actual_avg=("actual_minutes", "mean"),
                    predicted_avg=("prediction", "mean"),
                    mae=("abs_error", "mean"),
                )
            )
            grouped["test_season"] = test_season
            grouped["model"] = model_name
            bucket_rows.append(grouped)

        # LightGBM feature importance only.
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

    summary = pd.DataFrame(summary_rows)
    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    buckets = (
        pd.concat(bucket_rows, ignore_index=True)
        if bucket_rows
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
        output_dir / "minutes_model_summary.csv",
        index=False,
    )
    predictions.to_csv(
        output_dir / "minutes_model_predictions.csv",
        index=False,
    )
    buckets.to_csv(
        output_dir / "minutes_model_buckets.csv",
        index=False,
    )
    importance.to_csv(
        output_dir / "minutes_model_lightgbm_importance.csv",
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
        "buckets": buckets,
        "importance": importance,
        "numeric_features": numeric_features,
    }


if __name__ == "__main__":
    run_minutes_model_test()
