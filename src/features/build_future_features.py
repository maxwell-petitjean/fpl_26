from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import src.features.build_historic_features as historic_feature_builder


CONFIG_PATH = Path("config/current_predictions.yaml")

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

POSITION_GROUP_MAP = {
    "GK": "GK",
    "DEF": "DEF",
    "MID": "ATT",
    "FWD": "ATT",
}


def _load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _derive_defcon_hit(position, defensive_contribution, config):
    dc = pd.to_numeric(defensive_contribution, errors="coerce").fillna(0)

    pos = position.astype(str)

    defender_threshold = int(
        config["defcon"]["defender_threshold"]
    )
    attacker_threshold = int(
        config["defcon"]["attacker_threshold"]
    )

    return np.select(
        [
            pos.eq("DEF") & (dc >= defender_threshold),
            pos.isin(["MID", "FWD"]) & (dc >= attacker_threshold),
        ],
        [1, 1],
        default=0,
    ).astype(int)


def _current_history_to_fact(
    history,
    players,
    teams,
    config,
):
    if history.empty:
        return pd.DataFrame()

    player_cols = [
        "id",
        "code",
        "first_name",
        "second_name",
        "web_name",
        "team",
        "element_type",
    ]

    meta = players[player_cols].copy().rename(
        columns={
            "id": "fpl_element_id",
            "code": "player_code",
            "team": "team_id",
        }
    )

    hist = history.merge(
        meta,
        on="fpl_element_id",
        how="left",
        validate="many_to_one",
    )

    team_name = teams.set_index("id")["name"]
    team_code = teams.set_index("id")["code"]

    hist["season"] = config["season"]
    hist["gameweek"] = pd.to_numeric(
        hist["round"],
        errors="coerce",
    ).astype("Int64")

    hist["fixture_id"] = pd.to_numeric(
        hist["fixture"],
        errors="coerce",
    ).astype("Int64")

    hist["player_name"] = (
        hist["first_name"].fillna("").astype(str)
        + " "
        + hist["second_name"].fillna("").astype(str)
    ).str.strip()

    hist["position"] = hist["element_type"].map(POSITION_MAP)

    hist["team_name"] = hist["team_id"].map(team_name)
    hist["team_code"] = hist["team_id"].map(team_code)

    hist["opponent_team_id"] = pd.to_numeric(
        hist["opponent_team"],
        errors="coerce",
    ).astype("Int64")

    hist["opponent_team_name"] = (
        hist["opponent_team_id"].map(team_name)
    )

    hist["opponent_team_code"] = (
        hist["opponent_team_id"].map(team_code)
    )

    hist["price"] = (
        pd.to_numeric(hist.get("value"), errors="coerce")
        / 10.0
    )

    hist["kickoff_time"] = pd.to_datetime(
        hist["kickoff_time"],
        utc=True,
        errors="coerce",
    )

    numeric_sources = [
        "minutes",
        "total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "defensive_contribution",
    ]

    for col in numeric_sources:
        if col not in hist.columns:
            hist[col] = np.nan
        hist[col] = pd.to_numeric(
            hist[col],
            errors="coerce",
        )

    hist["defcon_hit"] = _derive_defcon_hit(
        hist["position"],
        hist["defensive_contribution"],
        config,
    )

    hist["defcon_points"] = (
        float(config["defcon"]["points_per_hit"])
        * hist["defcon_hit"]
    )

    hist["core_total_points"] = (
        hist["total_points"]
        - hist["defcon_points"]
    )

    fact_cols = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "player_name",
        "web_name",
        "position",
        "team_id",
        "team_code",
        "team_name",
        "opponent_team_id",
        "opponent_team_code",
        "opponent_team_name",
        "was_home",
        "price",
        "minutes",
        "total_points",
        "core_total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "defensive_contribution",
        "defcon_hit",
        "defcon_points",
    ]

    for col in fact_cols:
        if col not in hist.columns:
            hist[col] = np.nan

    return hist[fact_cols].copy()


def _build_snapshot_rows(
    players,
    teams,
    next_gw,
    snapshot_time,
    historic_columns,
):
    team_name = teams.set_index("id")["name"]
    team_code = teams.set_index("id")["code"]

    snap = pd.DataFrame()

    snap["season"] = pd.Series(
        [None] * len(players),
        dtype="object",
    )

    snap["gameweek"] = next_gw
    snap["fixture_id"] = (
        -pd.to_numeric(players["id"], errors="coerce").astype(int)
    )
    snap["kickoff_time"] = snapshot_time

    snap["player_code"] = pd.to_numeric(
        players["code"],
        errors="coerce",
    )

    snap["player_name"] = (
        players["first_name"].fillna("").astype(str)
        + " "
        + players["second_name"].fillna("").astype(str)
    ).str.strip()

    snap["web_name"] = players["web_name"]
    snap["position"] = players["element_type"].map(POSITION_MAP)
    snap["team_id"] = players["team"]
    snap["team_code"] = players["team"].map(team_code)
    snap["team_name"] = players["team"].map(team_name)

    snap["opponent_team_id"] = np.nan
    snap["opponent_team_code"] = np.nan
    snap["opponent_team_name"] = pd.NA
    snap["was_home"] = np.nan

    snap["price"] = (
        pd.to_numeric(players["now_cost"], errors="coerce")
        / 10.0
    )

    stat_cols = [
        "minutes",
        "total_points",
        "core_total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "defensive_contribution",
        "defcon_hit",
        "defcon_points",
    ]

    for col in stat_cols:
        snap[col] = np.nan

    for col in historic_columns:
        if col not in snap.columns:
            snap[col] = np.nan

    return snap[list(historic_columns)].copy()


def _add_last6_std_from_fact(snapshot_features, all_known_fact):
    fact = all_known_fact.copy()

    fact["kickoff_time"] = pd.to_datetime(
        fact["kickoff_time"],
        utc=True,
        errors="coerce",
    )

    fact = fact.sort_values(
        ["player_code", "kickoff_time", "season", "gameweek", "fixture_id"]
    )

    last6 = (
        fact[
            fact["minutes"].notna()
            & (pd.to_numeric(fact["fixture_id"], errors="coerce") >= 0)
        ]
        .groupby("player_code", as_index=False)
        .tail(6)
    )

    std = (
        last6.groupby("player_code")["minutes"]
        .std()
        .rename("player_minutes_std_l6")
        .reset_index()
    )

    counts = (
        fact[
            fact["minutes"].notna()
            & (pd.to_numeric(fact["fixture_id"], errors="coerce") >= 0)
        ]
        .groupby("player_code")
        .size()
        .rename("player_history_fixtures")
        .reset_index()
    )

    out = snapshot_features.merge(
        std,
        on="player_code",
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        counts,
        on="player_code",
        how="left",
        validate="one_to_one",
    )

    out["player_history_fixtures"] = (
        out["player_history_fixtures"]
        .fillna(0)
        .astype(int)
    )

    out["is_new_player"] = (
        out["player_history_fixtures"] == 0
    )

    return out


def _add_defcon_snapshot(snapshot_features, all_known_fact, config):
    out = snapshot_features.copy()

    fact = all_known_fact.copy()
    fact["kickoff_time"] = pd.to_datetime(
        fact["kickoff_time"],
        utc=True,
        errors="coerce",
    )

    fact["defensive_contribution"] = pd.to_numeric(
        fact.get("defensive_contribution"),
        errors="coerce",
    ).fillna(0)

    fact["minutes"] = pd.to_numeric(
        fact["minutes"],
        errors="coerce",
    )

    if "defcon_hit" not in fact.columns:
        fact["defcon_hit"] = _derive_defcon_hit(
            fact["position"],
            fact["defensive_contribution"],
            config,
        )

    fact = fact[
        fact["minutes"].notna()
    ].sort_values(
        ["player_code", "kickoff_time", "season", "gameweek", "fixture_id"]
    )

    rows = []

    for player_code, player_df in fact.groupby("player_code"):
        player_df = player_df.sort_values("kickoff_time")

        row = {"player_code": player_code}

        for w in [1, 3, 6, 12]:
            tail = player_df.tail(w)

            row[f"defcon_avg_l{w}"] = (
                tail["defensive_contribution"].mean()
                if not tail.empty else np.nan
            )

            row[f"defcon_hit_rate_l{w}"] = (
                tail["defcon_hit"].mean()
                if not tail.empty else np.nan
            )

            row[f"minutes_avg_l{w}"] = (
                tail["minutes"].mean()
                if not tail.empty else np.nan
            )

        rows.append(row)

    defcon_snapshot = pd.DataFrame(rows)

    return out.merge(
        defcon_snapshot,
        on="player_code",
        how="left",
        validate="one_to_one",
    )


def _fixture_position_base(fact):
    meaningful = fact[
        pd.to_numeric(fact["minutes"], errors="coerce") >= 45
    ].copy()

    meaningful["position_group"] = (
        meaningful["position"].map(POSITION_GROUP_MAP)
    )

    base = (
        meaningful.groupby(
            [
                "season",
                "gameweek",
                "fixture_id",
                "kickoff_time",
                "opponent_team_code",
                "position_group",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            points=("core_total_points", "sum"),
            apps=("player_code", "size"),
        )
    )

    return base


def _rolling_opponent_value(
    base,
    team_code,
    position_group,
    window,
    season_filter=None,
):
    d = base[
        base["opponent_team_code"].eq(team_code)
        & base["position_group"].eq(position_group)
    ].copy()

    if season_filter is not None:
        d = d[d["season"].isin(season_filter)]

    d = d.sort_values("kickoff_time").tail(window)

    if d.empty or d["apps"].sum() <= 0:
        return np.nan

    return d["points"].sum() / d["apps"].sum()


def _build_opponent_snapshots(
    all_known_fact,
    teams,
    current_season,
    windows,
):
    fact = all_known_fact.copy()
    base = _fixture_position_base(fact)

    season_order = (
        fact[["season", "kickoff_time"]]
        .dropna(subset=["season", "kickoff_time"])
        .groupby("season", as_index=False)["kickoff_time"]
        .min()
        .sort_values("kickoff_time")["season"]
        .tolist()
    )

    prev_season = season_order[-1]

    teams_by_season = {
        s: set(
            pd.to_numeric(
                fact.loc[fact["season"].eq(s), "team_code"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )
        for s in season_order
    }

    current_codes = set(
        pd.to_numeric(teams["code"], errors="coerce")
        .dropna()
        .astype(int)
    )

    prev_codes = teams_by_season[prev_season]
    promoted_codes = current_codes - prev_codes

    # Build current transition plus the two preceding transitions.
    transitions = []

    historical_transitions = []
    for i in range(1, len(season_order)):
        relegated = (
            teams_by_season[season_order[i - 1]]
            - teams_by_season[season_order[i]]
        )
        historical_transitions.append(
            (season_order[i], relegated)
        )

    current_relegated = prev_codes - current_codes
    transitions = historical_transitions[-2:] + [
        (current_season, current_relegated)
    ]

    relegated_3yr = set()
    for _, codes in transitions:
        relegated_3yr |= set(codes)

    proxy_values = {}

    for pos_group in ["GK", "DEF", "ATT"]:
        for w in windows:
            vals = []

            for code in relegated_3yr:
                value = _rolling_opponent_value(
                    base,
                    code,
                    pos_group,
                    w,
                )
                if pd.notna(value):
                    vals.append(value)

            proxy_values[(pos_group, w)] = (
                float(np.mean(vals))
                if vals else np.nan
            )

    rows = []

    current_history_seasons = set(
        fact.loc[
            fact["season"].eq(current_season),
            "team_code",
        ]
        .dropna()
        .astype(int)
    )

    for code in sorted(current_codes):
        is_promoted = code in promoted_codes

        for pos_group in ["GK", "DEF", "ATT"]:
            row = {
                "opponent_team_code": code,
                "position_group": pos_group,
                "opponent_source": (
                    "promoted_proxy_3yr"
                    if is_promoted and code not in current_history_seasons
                    else "team_history"
                ),
            }

            for w in windows:
                col = f"opp_pos_core_points_avg_l{w}_filled"

                if is_promoted:
                    # Ignore any old PL spell. Once the new spell has actual
                    # current-season fixtures, only use that new spell.
                    if code in current_history_seasons:
                        value = _rolling_opponent_value(
                            base,
                            code,
                            pos_group,
                            w,
                            season_filter=[current_season],
                        )
                    else:
                        value = proxy_values[(pos_group, w)]
                else:
                    value = _rolling_opponent_value(
                        base,
                        code,
                        pos_group,
                        w,
                    )

                row[col] = value

            rows.append(row)

    return pd.DataFrame(rows), promoted_codes


def build_future_features(
    config_path=CONFIG_PATH,
    save=True,
):
    config = _load_config(config_path)

    current_dir = Path(config["paths"]["current_dir"])

    historic = pd.read_csv(
        config["paths"]["historic_fact"],
        low_memory=False,
    )

    players = pd.read_csv(
        current_dir / "players.csv",
        low_memory=False,
    )

    teams = pd.read_csv(
        current_dir / "teams.csv",
        low_memory=False,
    )

    fixtures = pd.read_csv(
        current_dir / "future_fixtures_8gw.csv",
        low_memory=False,
    )

    history_path = current_dir / "player_history.csv"
    history = (
        pd.read_csv(history_path, low_memory=False)
        if history_path.exists() and history_path.stat().st_size > 0
        else pd.DataFrame()
    )

    current_history_fact = _current_history_to_fact(
        history,
        players,
        teams,
        config,
    )

    historic["kickoff_time"] = pd.to_datetime(
        historic["kickoff_time"],
        utc=True,
        errors="coerce",
    )

    if not current_history_fact.empty:
        current_history_fact["kickoff_time"] = pd.to_datetime(
            current_history_fact["kickoff_time"],
            utc=True,
            errors="coerce",
        )

        # Avoid duplicate current rows if this script is rerun later and
        # current-season data has already been appended upstream.
        existing_grain = set(
            zip(
                historic["season"].astype(str),
                historic["fixture_id"].astype(str),
                historic["player_code"].astype(str),
            )
        )

        mask_new = [
            (
                str(r.season),
                str(r.fixture_id),
                str(r.player_code),
            )
            not in existing_grain
            for r in current_history_fact[
                ["season", "fixture_id", "player_code"]
            ].itertuples(index=False)
        ]

        current_history_fact = current_history_fact.loc[
            mask_new
        ].copy()

    all_known = pd.concat(
        [historic, current_history_fact],
        ignore_index=True,
        sort=False,
    )

    next_gw = int(
        pd.to_numeric(fixtures["event"], errors="coerce")
        .dropna()
        .min()
    )

    first_kickoff = pd.to_datetime(
        fixtures["kickoff_time"],
        utc=True,
        errors="coerce",
    ).min()

    snapshot_time = first_kickoff - pd.Timedelta(seconds=1)

    snapshot = _build_snapshot_rows(
        players,
        teams,
        next_gw,
        snapshot_time,
        all_known.columns,
    )

    snapshot["season"] = config["season"]

    combined_for_features = pd.concat(
        [all_known, snapshot],
        ignore_index=True,
        sort=False,
    )

    all_known_path = Path(config["paths"]["all_known_fact"])
    all_known_path.parent.mkdir(parents=True, exist_ok=True)
    combined_for_features.to_csv(
        all_known_path,
        index=False,
    )

    feature_all = historic_feature_builder.build_historic_features(
        input_path=all_known_path,
        save=False,
    )

    snapshot_features = feature_all[
        pd.to_numeric(
            feature_all["fixture_id"],
            errors="coerce",
        ) < 0
    ].copy()

    snapshot_features = _add_last6_std_from_fact(
        snapshot_features,
        all_known,
    )

    snapshot_features = _add_defcon_snapshot(
        snapshot_features,
        all_known,
        config,
    )

    team_name = teams.set_index("id")["name"]
    team_code = teams.set_index("id")["code"]

    player_meta = players[
        [
            "id",
            "code",
            "web_name",
            "first_name",
            "second_name",
            "team",
            "element_type",
            "now_cost",
            "status",
            "chance_of_playing_next_round",
        ]
    ].copy()

    player_meta["position"] = (
        player_meta["element_type"].map(POSITION_MAP)
    )

    future_rows = []

    for fixture in fixtures.itertuples(index=False):
        for is_home, team_id, opp_id in [
            (True, fixture.team_h, fixture.team_a),
            (False, fixture.team_a, fixture.team_h),
        ]:
            team_players = player_meta[
                player_meta["team"].eq(team_id)
            ].copy()

            rows = team_players.merge(
                snapshot_features,
                left_on="code",
                right_on="player_code",
                how="left",
                validate="one_to_one",
                suffixes=("", "_snapshot"),
            )

            rows["season"] = config["season"]
            rows["gameweek"] = int(fixture.event)
            rows["fixture_id"] = int(fixture.id)
            rows["kickoff_time"] = fixture.kickoff_time
            rows["was_home"] = is_home

            rows["team_id"] = int(team_id)
            rows["team_code"] = int(team_code.loc[team_id])
            rows["team_name"] = team_name.loc[team_id]

            rows["opponent_team_id"] = int(opp_id)
            rows["opponent_team_code"] = int(team_code.loc[opp_id])
            rows["opponent_team_name"] = team_name.loc[opp_id]

            rows["price"] = (
                pd.to_numeric(rows["now_cost"], errors="coerce")
                / 10.0
            )

            rows["position"] = rows["element_type"].map(
                POSITION_MAP
            )

            rows["position_group"] = rows["position"].map(
                POSITION_GROUP_MAP
            )

            # Current injury status is immediate-next-GW information.
            chance = pd.to_numeric(
                rows["chance_of_playing_next_round"],
                errors="coerce",
            ).fillna(100.0)

            if (
                config.get("availability_next_gw_only", True)
                and int(fixture.event) != next_gw
            ):
                chance = pd.Series(
                    100.0,
                    index=rows.index,
                )

            rows["availability_pct"] = chance.clip(0, 100)

            future_rows.append(rows)

    future = pd.concat(
        future_rows,
        ignore_index=True,
        sort=False,
    )

    # Rebuild current-fixture kickoff context after duplicating the snapshot.
    kickoff = pd.to_datetime(
        future["kickoff_time"],
        utc=True,
        errors="coerce",
    )

    future["kickoff_day_of_week"] = kickoff.dt.day_name()
    future["kickoff_hour"] = (
        kickoff.dt.hour
        + kickoff.dt.minute / 60
    )

    future["kickoff_day_type"] = (
        future["kickoff_day_of_week"]
        .map(
            {
                "Monday": "primetime",
                "Friday": "primetime",
                "Tuesday": "midweek",
                "Wednesday": "midweek",
                "Thursday": "midweek",
                "Saturday": "weekend",
                "Sunday": "weekend",
            }
        )
    )

    future["kickoff_time_type"] = np.select(
        [
            future["kickoff_hour"] < 14,
            future["kickoff_hour"] < 17.5,
        ],
        [
            "lunchtime",
            "afternoon",
        ],
        default="evening",
    )

    windows = [1, 3, 6, 12, 38, 76, 114]

    opponent_snapshot, promoted_codes = _build_opponent_snapshots(
        all_known,
        teams,
        config["season"],
        windows,
    )

    future = future.merge(
        opponent_snapshot,
        on=["opponent_team_code", "position_group"],
        how="left",
        validate="many_to_one",
    )

    future["opponent_is_promoted"] = (
        future["opponent_team_code"]
        .isin(promoted_codes)
    )

    keep_first = [
        "season",
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "player_code",
        "web_name",
        "position",
        "team_id",
        "team_code",
        "team_name",
        "opponent_team_id",
        "opponent_team_code",
        "opponent_team_name",
        "was_home",
        "price",
        "status",
        "availability_pct",
        "player_history_fixtures",
        "is_new_player",
        "position_group",
        "opponent_source",
        "opponent_is_promoted",
        "kickoff_day_type",
        "kickoff_time_type",
    ]

    remaining = [
        c for c in future.columns
        if c not in keep_first
    ]

    future = future[
        keep_first + remaining
    ]

    if save:
        output_path = Path(config["paths"]["future_features"])
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        future.to_csv(output_path, index=False)

    print("=== FUTURE FEATURE BUILD ===")
    print(f"Rows: {len(future):,}")
    print(f"Players: {future['player_code'].nunique():,}")
    print(
        "GWs:",
        sorted(future["gameweek"].dropna().astype(int).unique()),
    )
    print(
        f"New players with no PL fixture history: "
        f"{future.loc[future['is_new_player'], 'player_code'].nunique():,}"
    )
    print(
        "Promoted opponent codes:",
        sorted(promoted_codes),
    )

    return future


if __name__ == "__main__":
    build_future_features()
