from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb

import src.models.train_minutes_model as minutes_model
import src.models.train_defcon_model as defcon_model


CONFIG_PATH = Path("config/current_predictions.yaml")
MINUTES_CONFIG_PATH = Path("config/model_minutes.yaml")
DEFCON_CONFIG_PATH = Path("config/model_defcon.yaml")

CATEGORICAL_FEATURES = [
    "position",
    "kickoff_day_type",
    "kickoff_time_type",
]


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_columns(df, columns):
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _fit_minutes(historic_features, future, config):
    minutes_cfg = _load_yaml(MINUTES_CONFIG_PATH)

    train = minutes_model._add_last6_std(
        historic_features
    )

    target = minutes_cfg["target"]
    baseline_feature = minutes_cfg["baseline_feature"]
    threshold = float(minutes_cfg["stable_std_threshold"])

    # Select exactly the feature family used by the validated minutes model.
    numeric_features = minutes_model._select_features(
        train,
        minutes_cfg,
    )

    # chance_of_playing was not historically available in the training table,
    # so current availability is applied post-model rather than introducing
    # an untrained feature at prediction time.
    numeric_features = [
        c for c in numeric_features
        if c in train.columns
    ]

    volatile_train = train[
        train[target].notna()
        & (
            train["player_minutes_std_l6"].isna()
            | (
                train["player_minutes_std_l6"]
                > threshold
            )
        )
    ].copy()

    future = _ensure_columns(
        future,
        numeric_features
        + [
            baseline_feature,
            "player_minutes_std_l6",
        ],
    )

    X_train = minutes_model._encode_inputs(
        volatile_train,
        numeric_features,
        minutes_cfg,
    )

    X_future = minutes_model._encode_inputs(
        future,
        numeric_features,
        minutes_cfg,
    ).reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    model = lgb.LGBMRegressor(
        **minutes_cfg["lightgbm"]
    )

    model.fit(
        X_train,
        volatile_train[target].astype(float),
    )

    lgb_pred = np.clip(
        model.predict(X_future),
        0,
        90,
    )

    baseline = pd.to_numeric(
        future[baseline_feature],
        errors="coerce",
    ).clip(0, 90)

    stable = (
        pd.to_numeric(
            future["player_minutes_std_l6"],
            errors="coerce",
        ).notna()
        & (
            pd.to_numeric(
                future["player_minutes_std_l6"],
                errors="coerce",
            )
            <= threshold
        )
        & baseline.notna()
    )

    hybrid = np.where(
        stable,
        baseline,
        lgb_pred,
    )

    availability = (
        pd.to_numeric(
            future["availability_pct"],
            errors="coerce",
        )
        .fillna(100)
        .clip(0, 100)
        / 100.0
    )

    out = future[
        [
            "season",
            "gameweek",
            "fixture_id",
            "player_code",
            "web_name",
        ]
    ].copy()

    out["pred_avg_mins_l6"] = baseline
    out["pred_lightgbm_minutes"] = lgb_pred
    out["stable_last6"] = stable
    out["minutes_prediction_source"] = np.where(
        stable,
        "avg_mins_l6",
        "lightgbm",
    )
    out["pre_availability_minutes"] = hybrid
    out["availability_pct"] = future["availability_pct"].to_numpy()
    out["final_predicted_minutes"] = (
        hybrid * availability
    ).clip(0, 90)

    # Existing manual override file remains the final authority.
    override_frame = out.rename(
        columns={
            "minutes_prediction_source": "prediction_source",
        }
    )

    override_frame = minutes_model._apply_manual_overrides(
        override_frame,
        minutes_cfg,
    )

    override_frame = override_frame.rename(
        columns={
            "prediction_source": "minutes_prediction_source",
        }
    )

    return override_frame, model


def _xpp_feature_lists(df):
    player_features = [
        c for c in df.columns
        if (
            c.startswith("player_")
            and c not in [
                "player_code",
                "player_name",
            ]
            and pd.api.types.is_numeric_dtype(df[c])
        )
    ]

    opponent_features = [
        c for c in df.columns
        if (
            c.startswith("opp_pos_")
            and c.endswith("_filled")
            and pd.api.types.is_numeric_dtype(df[c])
        )
    ]

    numeric = sorted(
        set(player_features + opponent_features)
    )

    categorical = [
        c for c in CATEGORICAL_FEATURES
        if c in df.columns
    ]

    return numeric, categorical


def _fit_xpp(historic_xpp, future, config):
    target = "target_core_pp90"

    train = historic_xpp[
        historic_xpp[target].notna()
    ].copy()

    numeric_features, categorical = (
        _xpp_feature_lists(train)
    )

    future = _ensure_columns(
        future,
        numeric_features + categorical,
    )

    X_train = train[
        numeric_features + categorical
    ].copy()

    X_future = future[
        numeric_features + categorical
    ].copy()

    for col in categorical:
        train_values = (
            X_train[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

        future_values = (
            X_future[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

        categories = sorted(
            set(train_values)
            | set(future_values)
        )

        X_train[col] = pd.Categorical(
            train_values,
            categories=categories,
        )

        X_future[col] = pd.Categorical(
            future_values,
            categories=categories,
        )

    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train,
        train[target].astype(float),
        categorical_feature=categorical,
    )

    raw = model.predict(X_future)

    calibration = config["xpp90_calibration"]

    calibrated = (
        float(calibration["intercept"])
        + float(calibration["slope"])
        * raw
    )

    # PP90 cannot sensibly be negative for the planning layer.
    calibrated = np.clip(
        calibrated,
        0,
        None,
    )

    out = future[
        [
            "season",
            "gameweek",
            "fixture_id",
            "player_code",
            "web_name",
        ]
    ].copy()

    out["raw_predicted_core_pp90"] = raw
    out["predicted_core_pp90"] = calibrated

    return out, model, numeric_features, categorical


def _fit_defcon(historic_fact, future, config):
    defcon_cfg = _load_yaml(
        DEFCON_CONFIG_PATH
    )

    work = defcon_model._prepare(
        historic_fact,
        defcon_cfg,
    )

    train = work[
        work["defcon_avg_l1"].notna()
    ].copy()

    X_train = defcon_model._build_X(
        train,
        defcon_cfg,
    )

    future = _ensure_columns(
        future,
        [
            f"{prefix}_l{w}"
            for w in defcon_cfg["rolling_windows"]
            for prefix in [
                "defcon_avg",
                "defcon_hit_rate",
                "minutes_avg",
            ]
        ],
    )

    X_future = defcon_model._build_X(
        future,
        defcon_cfg,
    ).reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        **defcon_cfg["lightgbm"],
    )

    model.fit(
        X_train,
        train["defcon_hit"],
    )

    probability = model.predict_proba(
        X_future
    )[:, 1]

    # GK cannot earn DefCon points under the current rules.
    probability = np.where(
        future["position"].eq("GK"),
        0.0,
        probability,
    )

    # Immediate injury/availability information should also affect the
    # chance of earning the threshold bonus.
    availability = (
        pd.to_numeric(
            future["availability_pct"],
            errors="coerce",
        )
        .fillna(100)
        .clip(0, 100)
        / 100.0
    )

    probability = (
        probability
        * availability.to_numpy()
    )

    probability = np.clip(
        probability,
        0,
        1,
    )

    out = future[
        [
            "season",
            "gameweek",
            "fixture_id",
            "player_code",
            "web_name",
        ]
    ].copy()

    out["pred_defcon_probability"] = probability
    out["expected_defcon_points"] = (
        float(config["defcon"]["points_per_hit"])
        * probability
    )

    return out, model


def predict_future_8gw(
    config_path=CONFIG_PATH,
    save=True,
):
    config = _load_yaml(config_path)

    future = pd.read_csv(
        config["paths"]["future_features"],
        low_memory=False,
    )

    historic_features = pd.read_csv(
        config["paths"]["historic_player_features"],
        low_memory=False,
    )

    historic_xpp = pd.read_csv(
        config["paths"]["historic_xpp"],
        low_memory=False,
    )

    historic_fact = pd.read_csv(
        config["paths"]["historic_fact"],
        low_memory=False,
    )

    minutes_pred, minutes_fit = _fit_minutes(
        historic_features,
        future,
        config,
    )

    xpp_pred, xpp_fit, xpp_numeric, xpp_categorical = _fit_xpp(
        historic_xpp,
        future,
        config,
    )

    defcon_pred, defcon_fit = _fit_defcon(
        historic_fact,
        future,
        config,
    )

    keys = [
        "season",
        "gameweek",
        "fixture_id",
        "player_code",
        "web_name",
    ]

    result = future.merge(
        minutes_pred,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_mins"),
    )

    result = result.merge(
        xpp_pred,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_xpp"),
    )

    result = result.merge(
        defcon_pred,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_defcon"),
    )

    result["core_xp"] = (
        result["predicted_core_pp90"]
        * result["final_predicted_minutes"]
        / 90.0
    )

    result["xp"] = (
        result["core_xp"]
        + result["expected_defcon_points"]
    )

    result["fixture"] = np.where(
        result["was_home"].fillna(False),
        "vs " + result["opponent_team_name"].astype(str),
        "@ " + result["opponent_team_name"].astype(str),
    )

    front = [
        "season",
        "gameweek",
        "kickoff_time",
        "fixture_id",
        "player_code",
        "web_name",
        "position",
        "team_name",
        "fixture",
        "price",
        "status",
        "availability_pct",
        "player_history_fixtures",
        "is_new_player",
        "opponent_source",
        "final_predicted_minutes",
        "predicted_core_pp90",
        "pred_defcon_probability",
        "core_xp",
        "expected_defcon_points",
        "xp",
    ]

    result = result[
        front
        + [c for c in result.columns if c not in front]
    ]

    output_dir = Path(
        config["paths"]["output_dir"]
    )

    if save:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_csv(
            output_dir / "player_fixture_xp_8gw.csv",
            index=False,
        )

        player_gw = (
            result.groupby(
                [
                    "season",
                    "gameweek",
                    "player_code",
                    "web_name",
                    "position",
                    "team_name",
                    "price",
                ],
                as_index=False,
            )
            .agg(
                fixtures=("fixture_id", "size"),
                xmins=("final_predicted_minutes", "sum"),
                core_xp=("core_xp", "sum"),
                defcon_xp=("expected_defcon_points", "sum"),
                xp=("xp", "sum"),
            )
        )

        player_gw.to_csv(
            output_dir / "player_gw_xp_8gw.csv",
            index=False,
        )

        horizon = (
            player_gw.groupby(
                [
                    "player_code",
                    "web_name",
                    "position",
                    "team_name",
                    "price",
                ],
                as_index=False,
            )
            .agg(
                fixtures=("fixtures", "sum"),
                xmins_8gw=("xmins", "sum"),
                core_xp_8gw=("core_xp", "sum"),
                defcon_xp_8gw=("defcon_xp", "sum"),
                xp_8gw=("xp", "sum"),
            )
            .sort_values(
                "xp_8gw",
                ascending=False,
            )
        )

        horizon.to_csv(
            output_dir / "player_horizon_xp_8gw.csv",
            index=False,
        )

    print("=== FUTURE 8-GW PREDICTIONS ===")
    print(f"Player-fixture rows: {len(result):,}")
    print(
        "GWs:",
        sorted(result["gameweek"].dropna().astype(int).unique()),
    )
    print(
        f"Players: {result['player_code'].nunique():,}"
    )
    print(
        f"New players: "
        f"{result.loc[result['is_new_player'], 'player_code'].nunique():,}"
    )

    return {
        "player_fixture": result,
        "minutes_model": minutes_fit,
        "xpp_model": xpp_fit,
        "defcon_model": defcon_fit,
        "xpp_numeric_features": xpp_numeric,
        "xpp_categorical_features": xpp_categorical,
    }


if __name__ == "__main__":
    predict_future_8gw()
