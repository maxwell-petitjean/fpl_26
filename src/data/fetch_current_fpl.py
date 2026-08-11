from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yaml

CONFIG_PATH = Path("config/current_predictions.yaml")
BASE_URL = "https://fantasy.premierleague.com/api"


def _load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_json(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _next_gw(events, fixtures):
    if "is_next" in events.columns:
        nxt = events[events["is_next"].fillna(False)]
        if len(nxt) == 1:
            return int(nxt.iloc[0]["id"])

    unfinished = fixtures[
        fixtures["event"].notna()
        & ~fixtures["finished"].fillna(False)
    ].copy()

    if unfinished.empty:
        raise ValueError("Could not determine next gameweek.")

    return int(
        pd.to_numeric(unfinished["event"], errors="coerce")
        .dropna()
        .min()
    )


def _fetch_one_summary(element_id, timeout):
    data = _get_json(
        f"{BASE_URL}/element-summary/{int(element_id)}/",
        timeout=timeout,
    )
    history = pd.DataFrame(data.get("history", []))
    if history.empty:
        return history
    history["fpl_element_id"] = int(element_id)
    return history


def fetch_current_fpl(
    config_path=CONFIG_PATH,
    fetch_history=True,
    save=True,
):
    config = _load_config(config_path)
    timeout = int(config["fetch"].get("timeout_seconds", 30))
    max_workers = int(config["fetch"].get("max_workers", 12))
    current_dir = Path(config["paths"]["current_dir"])

    bootstrap = _get_json(
        f"{BASE_URL}/bootstrap-static/",
        timeout=timeout,
    )
    fixture_json = _get_json(
        f"{BASE_URL}/fixtures/",
        timeout=timeout,
    )

    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    events = pd.DataFrame(bootstrap["events"])
    element_types = pd.DataFrame(bootstrap["element_types"])
    fixtures = pd.DataFrame(fixture_json)

    next_gw = _next_gw(events, fixtures)

    event_num = pd.to_numeric(fixtures["event"], errors="coerce")
    future_events = sorted(
        event_num[
            event_num.notna()
            & (event_num >= next_gw)
        ]
        .astype(int)
        .unique()
    )[: int(config["horizon_gws"])]

    future_fixtures = fixtures[
        event_num.isin(future_events)
    ].copy()

    team_name = teams.set_index("id")["name"]
    team_short = teams.set_index("id")["short_name"]
    team_code = teams.set_index("id")["code"]

    future_fixtures["home_team_name"] = future_fixtures["team_h"].map(team_name)
    future_fixtures["away_team_name"] = future_fixtures["team_a"].map(team_name)
    future_fixtures["home_team_short"] = future_fixtures["team_h"].map(team_short)
    future_fixtures["away_team_short"] = future_fixtures["team_a"].map(team_short)
    future_fixtures["home_team_code"] = future_fixtures["team_h"].map(team_code)
    future_fixtures["away_team_code"] = future_fixtures["team_a"].map(team_code)

    any_finished = fixtures["finished"].fillna(False).any()
    history = pd.DataFrame()

    if fetch_history and any_finished:
        frames = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _fetch_one_summary,
                    element_id,
                    timeout,
                ): element_id
                for element_id in players["id"].tolist()
            }

            for future in as_completed(futures):
                element_id = futures[future]
                try:
                    frame = future.result()
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:
                    print(
                        f"Warning: element-summary failed for "
                        f"{element_id}: {exc}"
                    )

        if frames:
            history = pd.concat(frames, ignore_index=True)

    if save:
        current_dir.mkdir(parents=True, exist_ok=True)
        players.to_csv(current_dir / "players.csv", index=False)
        teams.to_csv(current_dir / "teams.csv", index=False)
        events.to_csv(current_dir / "events.csv", index=False)
        fixtures.to_csv(current_dir / "fixtures.csv", index=False)
        element_types.to_csv(
            current_dir / "element_types.csv",
            index=False,
        )
        future_fixtures.to_csv(
            current_dir / "future_fixtures_8gw.csv",
            index=False,
        )
        history.to_csv(
            current_dir / "player_history.csv",
            index=False,
        )

    print("=== CURRENT FPL FETCH ===")
    print(f"Players: {len(players):,}")
    print(f"Teams: {len(teams):,}")
    print(f"Next GW: {next_gw}")
    print(f"Forecast GWs: {future_events}")
    print(f"Future fixtures: {len(future_fixtures):,}")
    print(f"Current-season history rows: {len(history):,}")

    return {
        "players": players,
        "teams": teams,
        "events": events,
        "fixtures": fixtures,
        "future_fixtures": future_fixtures,
        "history": history,
        "element_types": element_types,
        "next_gw": next_gw,
        "future_events": future_events,
    }


if __name__ == "__main__":
    fetch_current_fpl()
