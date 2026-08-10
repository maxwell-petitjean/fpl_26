from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

CONFIG_PATH = Path("config/model_defcon.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _prepare(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()

    out["kickoff_time"] = pd.to_datetime(
        out["kickoff_time"], utc=True, errors="coerce"
    )

    out = out[out["season"].eq(config["season"])].copy()

    if "defcon_points" in out.columns:
        out["defcon_hit"] = (
            pd.to_numeric(out["defcon_points"], errors="coerce")
            .fillna(0)
            .gt(0)
            .astype(int)
        )
    elif "target_defcon_hit" in out.columns:
        out["defcon_hit"] = (
            pd.to_numeric(out["target_defcon_hit"], errors="coerce")
            .fillna(0)
            .gt(0)
            .astype(int)
        )
    else:
        raise ValueError(
            "Need defcon_points or target_defcon_hit to define the target."
        )

    out["defensive_contribution"] = pd.to_numeric(
        out["defensive_contribution"], errors="coerce"
    ).fillna(0)

    out["minutes"] = pd.to_numeric(
        out["minutes"], errors="coerce"
    ).fillna(0)

    out = out.sort_values(
        ["player_code", "kickoff_time", "gameweek", "fixture_id"]
    ).reset_index(drop=True)

    g = out.groupby("player_code", sort=False)

    shifted_dc = g["defensive_contribution"].shift(1)
    shifted_hit = g["defcon_hit"].shift(1)
    shifted_minutes = g["minutes"].shift(1)

    for w in config["rolling_windows"]:
        out[f"defcon_avg_l{w}"] = (
            shifted_dc
            .groupby(out["player_code"])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        out[f"defcon_hit_rate_l{w}"] = (
            shifted_hit
            .groupby(out["player_code"])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        out[f"minutes_avg_l{w}"] = (
            shifted_minutes
            .groupby(out["player_code"])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return out


def _build_X(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    feature_cols = []

    for w in config["rolling_windows"]:
        feature_cols += [
            f"defcon_avg_l{w}",
            f"defcon_hit_rate_l{w}",
            f"minutes_avg_l{w}",
        ]

    X = df[feature_cols].copy()

    for pos in ["GK", "DEF", "MID", "FWD"]:
        X[f"position__{pos}"] = (
            df["position"].astype(str).eq(pos).astype(int)
        )

    return X


def run_defcon_model_test(config_path=CONFIG_PATH):
    config = load_config(config_path)

    df = pd.read_csv(
        config["input_path"],
        low_memory=False,
    )

    required = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "position",
        "minutes",
        "defensive_contribution",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input missing required columns: {missing}"
        )

    work = _prepare(df, config)

    train = work[
        work["gameweek"] <= config["train_max_gameweek"]
    ].copy()

    test = work[
        work["gameweek"] >= config["test_min_gameweek"]
    ].copy()

    train = train[train["defcon_avg_l1"].notna()].copy()
    test = test[test["defcon_avg_l1"].notna()].copy()

    X_train = _build_X(train, config)
    X_test = _build_X(test, config)

    y_train = train["defcon_hit"]
    y_test = test["defcon_hit"]

    model = LGBMClassifier(
        objective="binary",
        **config["lightgbm"],
    )

    model.fit(X_train, y_train)

    pred_prob = model.predict_proba(X_test)[:, 1]

    baseline = (
        test["defcon_hit_rate_l6"]
        .fillna(train["defcon_hit"].mean())
        .clip(0, 1)
        .to_numpy()
    )

    rows = []

    for name, pred in {
        "hit_rate_l6": baseline,
        "lightgbm": pred_prob,
    }.items():
        row = {
            "model": name,
            "rows": len(test),
            "actual_hit_rate": y_test.mean(),
            "predicted_hit_rate": pred.mean(),
            "brier": brier_score_loss(y_test, pred),
            "log_loss": log_loss(y_test, pred, labels=[0, 1]),
        }

        row["roc_auc"] = (
            roc_auc_score(y_test, pred)
            if y_test.nunique() == 2
            else np.nan
        )

        rows.append(row)

    summary = pd.DataFrame(rows)

    predictions = test[
        [
            "season",
            "gameweek",
            "fixture_id",
            "player_code",
            "player_name",
            "position",
            "minutes",
            "defensive_contribution",
            "defcon_hit",
        ]
    ].copy()

    predictions["pred_hit_rate_l6"] = baseline
    predictions["pred_defcon_probability"] = pred_prob
    predictions["expected_defcon_points"] = 2 * pred_prob

    predictions["probability_decile"] = pd.qcut(
        predictions["pred_defcon_probability"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    calibration = (
        predictions
        .groupby("probability_decile", as_index=False)
        .agg(
            rows=("player_code", "size"),
            predicted_probability=("pred_defcon_probability", "mean"),
            actual_hit_rate=("defcon_hit", "mean"),
            avg_defensive_contribution=("defensive_contribution", "mean"),
        )
    )

    importance = (
        pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "defcon_model_summary.csv",
        index=False,
    )
    predictions.to_csv(
        output_dir / "defcon_model_predictions.csv",
        index=False,
    )
    calibration.to_csv(
        output_dir / "defcon_probability_calibration.csv",
        index=False,
    )
    importance.to_csv(
        output_dir / "defcon_feature_importance.csv",
        index=False,
    )

    print("=== DEFCON MODEL TEST ===")
    print(
        f"Season: {config['season']} | "
        f"Train GW <= {config['train_max_gameweek']} | "
        f"Test GW >= {config['test_min_gameweek']}"
    )
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Train hit rate: {y_train.mean():.3f}")
    print(f"Test hit rate: {y_test.mean():.3f}")
    print()
    print(summary.to_string(index=False))

    return {
        "summary": summary,
        "predictions": predictions,
        "calibration": calibration,
        "importance": importance,
        "work": work,
        "model": model,
    }


if __name__ == "__main__":
    run_defcon_model_test()
