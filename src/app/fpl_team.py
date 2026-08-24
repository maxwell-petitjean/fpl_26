import requests


FPL_BASE_URL = "https://fantasy.premierleague.com/api"


class FPLTeamError(RuntimeError):
    pass


def _get_json(path: str, timeout: int = 10):
    url = f"{FPL_BASE_URL}/{path.lstrip('/')}"
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "fpldataguy/1.0",
            "Accept": "application/json",
        },
    )

    if response.status_code == 404:
        return None

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise FPLTeamError(
            f"FPL API request failed ({response.status_code})."
        ) from exc

    return response.json()


def _current_event_id(bootstrap: dict) -> int:
    events = bootstrap.get("events", [])

    current = [
        int(event["id"])
        for event in events
        if event.get("is_current")
    ]
    if current:
        return current[0]

    previous = [
        int(event["id"])
        for event in events
        if event.get("is_previous")
    ]
    if previous:
        return previous[0]

    finished = [
        int(event["id"])
        for event in events
        if event.get("finished")
    ]
    if finished:
        return max(finished)

    return 1


def load_fpl_team(team_id: int) -> dict:
    """
    Load a public FPL entry and its latest available 15-player squad.

    The picks endpoint returns FPL element ids. Those are mapped to
    player_code in Streamlit using the solver pool's fpl_element_id.

    Finance:
      last_deadline_value and last_deadline_bank are returned in tenths.
      available_budget is their sum in £m and is used as the transfer
      solver's public-data budget envelope.
    """
    team_id = int(team_id)

    if team_id <= 0:
        raise FPLTeamError("Team ID must be a positive integer.")

    entry = _get_json(f"entry/{team_id}/")
    if not entry:
        raise FPLTeamError(
            f"FPL team {team_id} was not found."
        )

    bootstrap = _get_json("bootstrap-static/")
    if not bootstrap:
        raise FPLTeamError(
            "Could not load FPL gameweek metadata."
        )

    current_event = _current_event_id(bootstrap)

    # During a live/transitioning GW, try the current event first,
    # then walk backwards until a public picks snapshot is available.
    picks_data = None
    picks_event = None

    for event_id in range(current_event, 0, -1):
        candidate = _get_json(
            f"entry/{team_id}/event/{event_id}/picks/"
        )
        if candidate and candidate.get("picks"):
            picks_data = candidate
            picks_event = event_id
            break

    if not picks_data:
        raise FPLTeamError(
            "No public Gameweek squad could be found for this team yet."
        )

    picks = picks_data["picks"]
    element_ids = [
        int(pick["element"])
        for pick in picks
    ]

    if len(element_ids) != 15:
        raise FPLTeamError(
            f"Expected 15 players, received {len(element_ids)}."
        )

    bank = entry.get("last_deadline_bank")
    value = entry.get("last_deadline_value")

    bank_m = (
        float(bank) / 10
        if bank is not None
        else 0.0
    )
    squad_value_m = (
        float(value) / 10
        if value is not None
        else None
    )
    available_budget_m = (
        squad_value_m + bank_m
        if squad_value_m is not None
        else None
    )

    return {
        "team_id": team_id,
        "team_name": entry.get("name") or f"Team {team_id}",
        "manager_name": " ".join(
            x for x in [
                entry.get("player_first_name"),
                entry.get("player_last_name"),
            ]
            if x
        ).strip(),
        "picks_event": int(picks_event),
        "current_event": int(current_event),
        "element_ids": tuple(element_ids),
        "bank": bank_m,
        "squad_value": squad_value_m,
        "available_budget": available_budget_m,
        "overall_points": entry.get("summary_overall_points"),
        "overall_rank": entry.get("summary_overall_rank"),
    }


def map_fpl_team_to_solver(
    team: dict,
    solver_pool,
) -> dict:
    """
    Map FPL element ids to the model's stable player_code.
    """
    if "fpl_element_id" not in solver_pool.columns:
        raise FPLTeamError(
            "Solver pool is missing fpl_element_id. "
            "Rebuild it with the updated build_solver_pool.py."
        )

    mapping = (
        solver_pool[
            [
                "fpl_element_id",
                "player_code",
                "web_name",
                "position",
                "team_name",
                "price",
            ]
        ]
        .dropna(
            subset=[
                "fpl_element_id",
                "player_code",
            ]
        )
        .drop_duplicates(
            "fpl_element_id"
        )
        .copy()
    )

    mapping["fpl_element_id"] = (
        mapping["fpl_element_id"]
        .astype(int)
    )
    mapping["player_code"] = (
        mapping["player_code"]
        .astype(int)
    )

    by_element = (
        mapping.set_index(
            "fpl_element_id"
        )
    )

    missing = [
        element_id
        for element_id in team["element_ids"]
        if element_id not in by_element.index
    ]

    if missing:
        raise FPLTeamError(
            "Some players in this FPL squad are missing from the "
            f"solver pool. Missing FPL element ids: {missing}"
        )

    player_codes = tuple(
        int(
            by_element.loc[
                element_id,
                "player_code",
            ]
        )
        for element_id in team["element_ids"]
    )

    squad = (
        by_element.loc[
            list(team["element_ids"])
        ]
        .reset_index()
        .copy()
    )

    return {
        **team,
        "player_codes": player_codes,
        "squad": squad,
    }
