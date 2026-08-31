import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

from src.app.data import load_solver_pool
from src.app.wildcard import solve_wildcard
from src.app.fpl_team import (
    FPLTeamError,
    load_fpl_team,
    map_fpl_team_to_solver,
)


@st.cache_data(
    ttl=300,
    show_spinner="Loading FPL team..."
)
def load_fpl_team_cached(
    team_id,
):
    return load_fpl_team(
        int(team_id)
    )


@st.cache_data(
    show_spinner="Optimising squad..."
)
def solve_wildcard_cached(
    solver_pool,
    model_bias,
    horizon_weeks,
    locked_player_codes,
    excluded_player_codes,
    excluded_teams,
    current_player_codes,
    max_transfers,
    budget,
):
    return solve_wildcard(
        solver_pool,
        model_bias=model_bias,
        horizon_weeks=horizon_weeks,
        included_player_codes=(
            locked_player_codes
        ),
        excluded_player_codes=(
            excluded_player_codes
        ),
        excluded_teams=(
            excluded_teams
        ),
        current_player_codes=(
            current_player_codes
        ),
        max_transfers=max_transfers,
        budget=budget,
    )



# ============================================================
# FIXTURE STRENGTH DATA
# ============================================================

@st.cache_resource
def get_bq_client():
    """
    Reuse the same Streamlit service-account secret already
    used for the deployed app.

    Expected Streamlit secret:
        [gcp_service_account]
    """

    credentials = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"])
    )

    return bigquery.Client(
        credentials=credentials,
        project=credentials.project_id,
    )


@st.cache_data(ttl=3600)
def load_fixture_strength_data():
    """
    Build one row per upcoming team fixture and attach the latest
    available opponent-strength snapshot for GK / DEF / ATT.

    The historic opponent_strength table is used only to obtain the
    latest trailing L3 / L6 / L12 values for each opponent.
    """

    client = get_bq_client()

    sql = """
    WITH future_fixtures AS (
        SELECT DISTINCT
            season,
            gameweek,
            fixture_id,
            team_name,
            opponent_team_name,
            home_away,
            fixture_label,
            opponent_is_promoted
        FROM `mptestproject-489015.fpl.predictions`
        WHERE gameweek IS NOT NULL
    ),

    latest_strength AS (
        SELECT
            current_opponent_team_name AS opponent_team_name,
            position_group,
            core_points_avg_l3,
            core_points_avg_l6,
            core_points_avg_l12
        FROM `mptestproject-489015.fpl.opponent_strength`
        WHERE current_opponent_team_name IS NOT NULL
          AND position_group IN ('GK', 'DEF', 'ATT')
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY
                current_opponent_team_name,
                position_group
            ORDER BY
                kickoff_time DESC,
                gameweek DESC
        ) = 1
    ),

    position_group_average AS (
        SELECT
            position_group,
            AVG(core_points_avg_l3) AS avg_core_points_avg_l3,
            AVG(core_points_avg_l6) AS avg_core_points_avg_l6,
            AVG(core_points_avg_l12) AS avg_core_points_avg_l12
        FROM latest_strength
        GROUP BY
            position_group
    )

    SELECT
        f.season,
        f.gameweek,
        f.fixture_id,
        f.team_name,
        f.opponent_team_name,
        f.home_away,
        f.fixture_label,
        f.opponent_is_promoted,
        position_group,

        CASE
            WHEN f.opponent_is_promoted
                THEN a.avg_core_points_avg_l3
            ELSE s.core_points_avg_l3
        END AS core_points_avg_l3,

        CASE
            WHEN f.opponent_is_promoted
                THEN a.avg_core_points_avg_l6
            ELSE s.core_points_avg_l6
        END AS core_points_avg_l6,

        CASE
            WHEN f.opponent_is_promoted
                THEN a.avg_core_points_avg_l12
            ELSE s.core_points_avg_l12
        END AS core_points_avg_l12,

        CASE
            WHEN f.opponent_is_promoted
                THEN 'promoted_league_average'
            WHEN s.opponent_team_name IS NOT NULL
                THEN 'team_history'
            ELSE 'missing'
        END AS strength_source

    FROM future_fixtures f

    CROSS JOIN UNNEST(
        ['GK', 'DEF', 'ATT']
    ) AS position_group

    LEFT JOIN latest_strength s
        ON f.opponent_team_name = s.opponent_team_name
       AND position_group = s.position_group

    LEFT JOIN position_group_average a
        ON position_group = a.position_group

    ORDER BY
        f.gameweek,
        f.team_name,
        f.fixture_id,
        position_group
    """

    return (
        client.query(sql)
        .to_dataframe(
            create_bqstorage_client=False
        )
    )


def build_fixture_matrix(
    fixture_data: pd.DataFrame,
    position_group: str,
    lookback: int,
):
    """
    Return:
      display_matrix: opponent + score text
      score_matrix: numeric matrix used for heatmap colouring

    DGWs are retained by joining multiple opponents in the same cell.
    """

    score_col = (
        f"core_points_avg_l{lookback}"
    )

    view = (
        fixture_data[
            fixture_data[
                "position_group"
            ].eq(position_group)
        ]
        .copy()
    )

    view["gameweek"] = pd.to_numeric(
        view["gameweek"],
        errors="coerce",
    )

    view = view[
        view["gameweek"].notna()
    ].copy()

    view["gameweek"] = (
        view["gameweek"]
        .astype(int)
    )

    view["score"] = pd.to_numeric(
        view[score_col],
        errors="coerce",
    )

    # Short display label while keeping the real team name.
    view["opponent_label"] = (
        view["opponent_team_name"]
    )

    promoted_mask = (
        view["strength_source"]
        .eq(
            "promoted_league_average"
        )
    )

    view.loc[
        promoted_mask,
        "opponent_label",
    ] = (
        view.loc[
            promoted_mask,
            "opponent_team_name",
        ]
        + "*"
    )

    view["cell"] = np.where(
        view["score"].notna(),
        (
            view["opponent_label"]
            + " "
            + view["score"].map(
                lambda x: f"{x:.2f}"
            )
        ),
        (
            view["opponent_label"]
            + " —"
        ),
    )

    # Supports DGWs without assuming they will not occur.
    display_long = (
        view.groupby(
            [
                "team_name",
                "gameweek",
            ],
            as_index=False,
        )
        .agg(
            cell=(
                "cell",
                " / ".join,
            )
        )
    )

    score_long = (
        view.groupby(
            [
                "team_name",
                "gameweek",
            ],
            as_index=False,
        )
        .agg(
            score=(
                "score",
                "mean",
            )
        )
    )

    display_matrix = (
        display_long
        .pivot(
            index="team_name",
            columns="gameweek",
            values="cell",
        )
    )

    score_matrix = (
        score_long
        .pivot(
            index="team_name",
            columns="gameweek",
            values="score",
        )
    )

    gameweeks = sorted(
        set(
            display_matrix.columns
        )
    )

    display_matrix = (
        display_matrix
        .reindex(
            columns=gameweeks
        )
        .sort_index()
    )

    score_matrix = (
        score_matrix
        .reindex(
            index=display_matrix.index,
            columns=gameweeks,
        )
    )

    display_matrix.columns = [
        f"GW{gw}"
        for gw in gameweeks
    ]

    score_matrix.columns = (
        display_matrix.columns
    )

    display_matrix.index.name = (
        "Team"
    )

    return (
        display_matrix,
        score_matrix,
    )


def style_fixture_matrix(
    display_matrix: pd.DataFrame,
    score_matrix: pd.DataFrame,
):
    """
    Darker green = opponent has allowed more core FPL points
    to the selected position group = more attractive fixture.
    """

    all_scores = (
        score_matrix
        .stack()
        .dropna()
    )

    if all_scores.empty:
        return (
            display_matrix
            .style
        )

    vmin = float(
        all_scores.min()
    )

    vmax = float(
        all_scores.max()
    )

    def colour_cell(
        value,
        row_name,
        col_name,
    ):
        score = (
            score_matrix.loc[
                row_name,
                col_name,
            ]
        )

        if pd.isna(score):
            return (
                "background-color: #eeeeee; "
                "color: #777777;"
            )

        if vmax <= vmin:
            strength = 0.5
        else:
            strength = (
                float(score) - vmin
            ) / (
                vmax - vmin
            )

        # Deliberately simple custom green scale so this does not
        # depend on pandas Styler's matplotlib gradient support.
        light = np.array(
            [232, 245, 233]
        )
        dark = np.array(
            [27, 94, 32]
        )

        rgb = (
            light
            + strength
            * (
                dark - light
            )
        ).astype(int)

        text_colour = (
            "#ffffff"
            if strength >= 0.58
            else "#111111"
        )

        return (
            f"background-color: "
            f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); "
            f"color: {text_colour}; "
            "font-weight: 650;"
        )

    styles = pd.DataFrame(
        "",
        index=display_matrix.index,
        columns=display_matrix.columns,
    )

    for row_name in (
        display_matrix.index
    ):
        for col_name in (
            display_matrix.columns
        ):
            styles.loc[
                row_name,
                col_name,
            ] = colour_cell(
                display_matrix.loc[
                    row_name,
                    col_name,
                ],
                row_name,
                col_name,
            )

    return (
        display_matrix
        .style
        .apply(
            lambda _: styles,
            axis=None,
        )
    )

st.set_page_config(
    page_title="FPL Solver",
    page_icon="⚽",
    layout="wide",
)

st.title("FPL Solver")
st.caption(
    "Predictive model player pool and optimal squad builder.\n"
    "GW2 is (nearly) done. Ready to optimise for GW3"
)

solver_pool = load_solver_pool()

if solver_pool.empty:
    st.error(
        "Solver pool is empty."
    )
    st.stop()

tab_pool, tab_fixtures, tab_wildcard = st.tabs(
    [
        "Player pool",
        "Fixture strength",
        "Optimal wildcard",
    ]
)


# ============================================================
# PLAYER POOL
# ============================================================

with tab_pool:

    st.subheader("Solver pool")

    st.caption(
        "Explore the modelled player pool, expected points and fixture context. "
        "Use the filters below to compare players before optimisation."
    )

    positions = (
        ["All"]
        + sorted(
            solver_pool[
                "position"
            ]
            .dropna()
            .unique()
            .tolist()
        )
    )

    teams = (
        ["All"]
        + sorted(
            solver_pool[
                "team_name"
            ]
            .dropna()
            .unique()
            .tolist()
        )
    )

    f1, f2, f3 = st.columns(
        [1, 2, 2]
    )

    with f1:
        selected_position = st.selectbox(
            "Position",
            positions,
        )

    with f2:
        selected_team = st.selectbox(
            "Team",
            teams,
        )

    with f3:
        search = st.text_input(
            "Player search",
            "",
        )

    view = solver_pool.copy()

    if selected_position != "All":
        view = view[
            view["position"].eq(
                selected_position
            )
        ]

    if selected_team != "All":
        view = view[
            view["team_name"].eq(
                selected_team
            )
        ]

    if search:
        view = view[
            view["web_name"]
            .str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    preferred_columns = [
        "player_code",
        "web_name",
        "position",
        "team_name",
        "price",
        "xmins_next_gw",
        "xp_next_gw",
        "xp_8gw",
        "weighted_xp_8gw",
        "model_xpp90_8gw",
        "mean_reversion_xpp90_8gw",
        "mean_reversion_xpp90_delta_8gw",
        "mean_reversion_xp_8gw",
        "mean_reversion_uplift_8gw",
        "fixture_xpp90_full_8gw",
        "fixture_xpp90_delta_8gw",
        "fixture_full_xp_8gw",
        "fixture_full_uplift_8gw",
        "avg_fixture_multiplier_8gw",
        "solver_eligible",
        "season_minutes",
        "season_total_points",
        "season_pp90",
        "season_goals",
        "season_assists",
        "season_xg",
        "season_xa",
        "season_xg90",
        "season_xa90",
        "season_defcon_points",
        "season_defcon_per90",
    ]

    display_columns = [
        c for c in preferred_columns
        if c in view.columns
    ]

    solver_default_columns = [
        c for c in [
            "web_name",
            "position",
            "team_name",
            "price",
            "xp_8gw",
            "xmins_next_gw",
            "xp_next_gw",
            "model_xpp90_8gw",
            "mean_reversion_xpp90_8gw",
            "season_pp90",
            "season_total_points",
            "season_goals",
            "season_assists",
            "season_xg",
            "season_xa",
            "season_xg90",
            "season_xa90",
            "season_defcon_points",
            "season_defcon_per90"
        ]
        if c in display_columns
    ]

    sort_col = (
        "weighted_xp_8gw"
        if "weighted_xp_8gw"
        in view.columns
        else "xp_8gw"
    )

    view = (
        view.sort_values(
            sort_col,
            ascending=False,
        )
    )

    st.dataframe(
        view[display_columns],
        width="stretch",
        hide_index=True,
        height=600,
        column_order=solver_default_columns,
        column_config={
            "price":
                st.column_config.NumberColumn(
                    "Price",
                    format="£%.1fm",
                ),
            "xp_next_gw":
                st.column_config.NumberColumn(
                    "Next GW xP",
                    format="%.2f",
                ),
            "xp_8gw":
                st.column_config.NumberColumn(
                    "8GW xP",
                    format="%.2f",
                ),
            "weighted_xp_8gw":
                st.column_config.NumberColumn(
                    "Weighted xP",
                    format="%.2f",
                ),
            "model_xpp90_8gw":
                st.column_config.NumberColumn(
                    "Model xPP90",
                    format="%.2f",
                ),
            "mean_reversion_xpp90_8gw":
                st.column_config.NumberColumn(
                    "L38 xPP90",
                    format="%.2f",
                ),
            "mean_reversion_xpp90_delta_8gw":
                st.column_config.NumberColumn(
                    "L38 Δ xPP90",
                    format="%+.2f",
                ),
            "mean_reversion_xp_8gw":
                st.column_config.NumberColumn(
                    "L38 8GW xP",
                    format="%.2f",
                ),
            "mean_reversion_uplift_8gw":
                st.column_config.NumberColumn(
                    "L38 Δ xP",
                    format="%+.2f",
                ),
            "season_minutes":
                st.column_config.NumberColumn(
                    "Season mins",
                    format="%.0f",
                ),
            "season_total_points":
                st.column_config.NumberColumn(
                    "Season pts",
                    format="%.0f",
                ),
            "season_pp90":
                st.column_config.NumberColumn(
                    "Season PP90",
                    format="%.2f",
                ),
            "season_goals":
                st.column_config.NumberColumn(
                    "Goals",
                    format="%.0f",
                ),
            "season_assists":
                st.column_config.NumberColumn(
                    "Assists",
                    format="%.0f",
                ),
            "season_xg":
                st.column_config.NumberColumn(
                    "xG",
                    format="%.2f",
                ),
            "season_xa":
                st.column_config.NumberColumn(
                    "xA",
                    format="%.2f",
                ),
            "season_xg90":
                st.column_config.NumberColumn(
                    "xG90",
                    format="%.2f",
                ),
            "season_xa90":
                st.column_config.NumberColumn(
                    "xA90",
                    format="%.2f",
                ),
            "season_defcon_points":
                st.column_config.NumberColumn(
                    "DefCon pts",
                    format="%.0f",
                ),
            "season_defcon_per90":
                st.column_config.NumberColumn(
                    "DefCon/90",
                    format="%.2f",
                ),
            "fixture_xpp90_full_8gw":
                st.column_config.NumberColumn(
                    "Fixture xPP90",
                    format="%.2f",
                ),
            "fixture_xpp90_delta_8gw":
                st.column_config.NumberColumn(
                    "Fixture Δ xPP90",
                    format="%+.2f",
                ),
            "fixture_full_xp_8gw":
                st.column_config.NumberColumn(
                    "Full fixture 8GW xP",
                    format="%.2f",
                ),
            "fixture_full_uplift_8gw":
                st.column_config.NumberColumn(
                    "Fixture uplift",
                    format="%+.2f",
                ),
        },
    )



# ============================================================
# FIXTURE STRENGTH
# ============================================================

with tab_fixtures:

    st.subheader(
        "Fixture strength"
    )

    st.caption(
        "Upcoming fixtures scored by how many core FPL points "
        "the opponent has recently allowed to the selected "
        "position group. Higher = more attractive fixture."
    )

    fixture_data = (
        load_fixture_strength_data()
    )

    if fixture_data.empty:
        st.info(
            "No upcoming fixture-strength data found."
        )
    else:

        f1, f2 = st.columns(
            [1, 1]
        )

        with f1:
            fixture_position = (
                st.segmented_control(
                    "Position group",
                    options=[
                        "GK",
                        "DEF",
                        "ATT",
                    ],
                    default="ATT",
                    key=(
                        "fixture_position_group"
                    ),
                )
            )

        with f2:
            fixture_lookback = (
                st.segmented_control(
                    "Lookback",
                    options=[
                        3,
                        6,
                        12,
                    ],
                    default=6,
                    format_func=(
                        lambda x:
                        f"Last {x}"
                    ),
                    key=(
                        "fixture_lookback"
                    ),
                )
            )

        if fixture_position is None:
            fixture_position = "ATT"

        if fixture_lookback is None:
            fixture_lookback = 6

        display_matrix, score_matrix = (
            build_fixture_matrix(
                fixture_data,
                position_group=(
                    fixture_position
                ),
                lookback=int(
                    fixture_lookback
                ),
            )
        )

        if display_matrix.empty:
            st.info(
                "No fixture-strength values "
                "for this selection."
            )
        else:
            styled_fixtures = (
                style_fixture_matrix(
                    display_matrix,
                    score_matrix,
                )
            )

            st.dataframe(
                styled_fixtures,
                width="stretch",
                height=760,
            )

            st.caption(
                "Cell = upcoming opponent + trailing average "
                "core points allowed per meaningful appearance. "
                "Darker green means the opponent has allowed "
                "more points to that position group. "
                "* Newly promoted opponents use the current "
                "position-group league average because no comparable "
                "Premier League history is available."
            )

# ============================================================
# OPTIMAL WILDCARD
# ============================================================

with tab_wildcard:

    st.subheader(
        "Optimal wildcard"
    )

    st.caption(
        "Independent of your current FPL team. "
        "0% uses the canonical model. Move left to pull scoring "
        "towards long-run player ability; move right to place extra "
        "emphasis on favourable fixtures."
    )

    st.markdown(
        "### Scoring strategy"
    )

    if (
        "applied_optimisation_mode"
        not in st.session_state
    ):
        st.session_state[
            "applied_optimisation_mode"
        ] = "Wildcard"

    if (
        "applied_fpl_team_id"
        not in st.session_state
    ):
        st.session_state[
            "applied_fpl_team_id"
        ] = None

    if (
        "applied_max_transfers"
        not in st.session_state
    ):
        st.session_state[
            "applied_max_transfers"
        ] = 1

    if (
        "applied_model_bias_pct"
        not in st.session_state
    ):
        st.session_state[
            "applied_model_bias_pct"
        ] = 0

    if (
        "applied_solve_horizon"
        not in st.session_state
    ):
        st.session_state[
            "applied_solve_horizon"
        ] = 8

    if (
        "applied_locked_player_codes"
        not in st.session_state
    ):
        st.session_state[
            "applied_locked_player_codes"
        ] = ()

    if (
        "applied_excluded_player_codes"
        not in st.session_state
    ):
        st.session_state[
            "applied_excluded_player_codes"
        ] = ()

    if (
        "applied_excluded_teams"
        not in st.session_state
    ):
        st.session_state[
            "applied_excluded_teams"
        ] = ()

    with st.form(
        "scoring_strategy_form"
    ):

        selected_optimisation_mode = (
            st.segmented_control(
                "Optimisation mode",
                options=[
                    "Wildcard",
                    "Transfers",
                ],
                default=st.session_state[
                    "applied_optimisation_mode"
                ],
                help=(
                    "Wildcard builds the best squad from scratch. "
                    "Transfers starts from your current FPL squad and allows "
                    "a limited number of changes, including 0 transfers."
                ),
            )
        )

        selected_fpl_team_id = st.text_input(
            "FPL Team ID",
            value="",
            placeholder="e.g. 2403195",
            help=(
                "Enter your FPL Team ID. "
                "The app loads your latest publicly available 15-player squad."
            ),
        )
        
        if selected_fpl_team_id:
            try:
                selected_fpl_team_id = int(
                    selected_fpl_team_id
                )
            except ValueError:
                st.error(
                    "FPL Team ID must be a number."
                )
                selected_fpl_team_id = None
        else:
            selected_fpl_team_id = None

        selected_max_transfers = (
            st.segmented_control(
                "Maximum transfers",
                options=[
                    0,
                    1,
                    2,
                    3,
                ],
                default=st.session_state[
                    "applied_max_transfers"
                ],
                help=(
                    "Used only in Transfers mode. "
                    "0 keeps your existing 15 and optimises the Starting XI. "
                    "For 1–3, the optimiser may use fewer transfers if that "
                    "scores better."
                ),
            )
        )

        selected_model_bias_pct = (
            st.slider(
                "Model bias",
                min_value=-100,
                max_value=125,
                value=st.session_state[
                    "applied_model_bias_pct"
                ],
                step=5,
                format="%d%%",
                help=(
                    "0% = canonical model. "
                    "Move left to pull predictions towards each player's "
                    "long-run L38 core xPP90. "
                    "Move right to add extra fixture emphasis for DEF and MID. "
                    "Positive settings do not change GK or FWD."
                ),
            )
        )

        selected_solve_horizon = (
            st.segmented_control(
                "Optimisation horizon",
                options=[
                    1,
                    3,
                    5,
                    8,
                ],
                default=st.session_state[
                    "applied_solve_horizon"
                ],
                format_func=lambda x:
                    f"{x} GW",
            )
        )

        st.markdown(
            "#### Squad controls"
        )

        player_options = (
            solver_pool[
                [
                    "player_code",
                    "web_name",
                    "position",
                    "team_name",
                    "price",
                ]
            ]
            .drop_duplicates(
                "player_code"
            )
            .sort_values(
                [
                    "web_name",
                    "team_name",
                    "position",
                ]
            )
        )

        player_label_map = {
            int(row.player_code): (
                f"{row.web_name} · "
                f"{row.position} · "
                f"{row.team_name} · "
                f"£{row.price:.1f}m"
            )
            for row
            in player_options.itertuples()
        }

        selected_locked_player_codes = (
            st.multiselect(
                "Lock players",
                options=list(
                    player_label_map.keys()
                ),
                default=list(
                    st.session_state[
                        "applied_locked_player_codes"
                    ]
                ),
                format_func=lambda player_code:
                    player_label_map[
                        player_code
                    ],
                placeholder=(
                    "Search player name..."
                ),
                help=(
                    "Locked players must be included in the 15-player squad. "
                    "The optimiser still decides whether they start each gameweek."
                ),
                key=(
                    "locked_players_input"
                ),
            )
        )

        selected_excluded_player_codes = (
            st.multiselect(
                "Exclude players",
                options=list(
                    player_label_map.keys()
                ),
                default=list(
                    st.session_state[
                        "applied_excluded_player_codes"
                    ]
                ),
                format_func=lambda player_code:
                    player_label_map[
                        player_code
                    ],
                placeholder=(
                    "Search player name..."
                ),
                key=(
                    "excluded_players_input"
                ),
            )
        )

        team_options = sorted(
            solver_pool[
                "team_name"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_excluded_teams = (
            st.multiselect(
                "Exclude teams",
                options=team_options,
                default=list(
                    st.session_state[
                        "applied_excluded_teams"
                    ]
                ),
                placeholder=(
                    "Select teams..."
                ),
                key=(
                    "excluded_teams_input"
                ),
            )
        )

        bias_left, bias_mid, bias_right = (
            st.columns(
                [1, 1, 1]
            )
        )

        with bias_left:
            st.caption(
                "← **Revert to mean**  \n"
                "Long-run L38 xPP90"
            )

        with bias_mid:
            st.caption(
                "**Model**  \n"
                "Canonical prediction"
            )

        with bias_right:
            st.caption(
                "**Chase fixtures →**  \n"
                "Extra DEF/MID fixture emphasis"
            )

        run_solver = (
            st.form_submit_button(
                "Run optimisation",
                type="primary",
                width="stretch",
            )
        )

    if run_solver:
        st.session_state[
            "applied_optimisation_mode"
        ] = (
            selected_optimisation_mode
            or "Wildcard"
        )

        st.session_state[
            "applied_fpl_team_id"
        ] = (
            int(selected_fpl_team_id)
            if selected_fpl_team_id
            else None
        )

        st.session_state[
            "applied_max_transfers"
        ] = int(
            selected_max_transfers
            or 1
        )

        st.session_state[
            "applied_model_bias_pct"
        ] = (
            selected_model_bias_pct
        )

        st.session_state[
            "applied_solve_horizon"
        ] = int(
            selected_solve_horizon
        )

        st.session_state[
            "applied_locked_player_codes"
        ] = tuple(
            int(x)
            for x
            in selected_locked_player_codes
        )

        st.session_state[
            "applied_excluded_player_codes"
        ] = tuple(
            int(x)
            for x
            in selected_excluded_player_codes
        )

        st.session_state[
            "applied_excluded_teams"
        ] = tuple(
            str(x)
            for x
            in selected_excluded_teams
        )

    optimisation_mode = (
        st.session_state[
            "applied_optimisation_mode"
        ]
    )

    fpl_team_id = (
        st.session_state[
            "applied_fpl_team_id"
        ]
    )

    applied_max_transfers = int(
        st.session_state[
            "applied_max_transfers"
        ]
    )

    model_bias_pct = (
        st.session_state[
            "applied_model_bias_pct"
        ]
    )

    model_bias = (
        model_bias_pct
        / 100
    )

    solve_horizon = int(
        st.session_state[
            "applied_solve_horizon"
        ]
    )

    locked_player_codes = tuple(
        st.session_state[
            "applied_locked_player_codes"
        ]
    )

    excluded_player_codes = tuple(
        st.session_state[
            "applied_excluded_player_codes"
        ]
    )

    excluded_teams = tuple(
        st.session_state[
            "applied_excluded_teams"
        ]
    )

    current_player_codes = ()
    transfer_budget = None
    loaded_team = None
    effective_max_transfers = None

    if optimisation_mode == "Transfers":
        if not fpl_team_id:
            st.warning(
                "Enter an FPL Team ID and click Run optimisation."
            )
            st.stop()

        try:
            raw_team = load_fpl_team_cached(
                int(fpl_team_id)
            )

            loaded_team = map_fpl_team_to_solver(
                raw_team,
                solver_pool,
            )

        except FPLTeamError as exc:
            st.error(str(exc))
            st.stop()

        current_player_codes = tuple(
            loaded_team[
                "player_codes"
            ]
        )

        # Public FPL data does not expose each player's exact live
        # transfer selling price. Use the current model price of the
        # owned 15 plus the public bank as the transfer budget envelope.
        current_squad_price = float(
            loaded_team[
                "squad"
            ]["price"]
            .sum()
        )

        transfer_budget = (
            current_squad_price
            + float(
                loaded_team.get(
                    "bank",
                    0.0,
                )
            )
        )

        effective_max_transfers = (
            applied_max_transfers
        )

        st.success(
            f"Loaded {loaded_team['team_name']} · "
            f"GW{loaded_team['picks_event']} squad · "
            f"£{loaded_team['bank']:.1f}m bank"
        )

        with st.expander(
            "Current FPL squad",
            expanded=False,
        ):
            current_view = (
                loaded_team[
                    "squad"
                ][
                    [
                        "web_name",
                        "position",
                        "team_name",
                        "price",
                    ]
                ]
                .copy()
            )

            st.dataframe(
                current_view,
                hide_index=True,
                width="stretch",
                column_config={
                    "web_name":
                        st.column_config.TextColumn(
                            "Player"
                        ),
                    "position":
                        st.column_config.TextColumn(
                            "Pos"
                        ),
                    "team_name":
                        st.column_config.TextColumn(
                            "Team"
                        ),
                    "price":
                        st.column_config.NumberColumn(
                            "Price",
                            format="£%.1fm",
                        ),
                },
            )

        if optimisation_mode == "Transfers":
            st.caption(
                "Transfer budget uses current model prices for your owned "
                "players plus the public FPL bank. Exact FPL selling prices "
                "are not exposed by the public picks endpoint, so price-profit "
                "edge cases can differ slightly from the live game."
            )

    if model_bias_pct < 0:
        st.info(
            f"Current solve: {abs(model_bias_pct)}% towards "
            "each player's long-run L38 core xPP90. "
            "This applies to all positions."
        )

    elif model_bias_pct > 0:
        st.info(
            f"Current solve: +{model_bias_pct}% extra fixture emphasis "
            "for DEF and MID. GK and FWD remain on canonical model xP."
        )

    else:
        st.info(
            "Current solve: canonical model."
        )

    context_bits = [
        optimisation_mode,
        f"{solve_horizon} GW horizon",
    ]

    if effective_max_transfers is not None:
        context_bits.append(
            f"{effective_max_transfers} max transfer"
            + (
                "s"
                if effective_max_transfers != 1
                else ""
            )
        )

    if locked_player_codes:
        context_bits.append(
            f"{len(locked_player_codes)} locked player"
            + (
                "s"
                if len(locked_player_codes) != 1
                else ""
            )
        )

    if excluded_player_codes:
        context_bits.append(
            f"{len(excluded_player_codes)} player exclusion"
            + (
                "s"
                if len(excluded_player_codes) != 1
                else ""
            )
        )

    if excluded_teams:
        context_bits.append(
            f"{len(excluded_teams)} team exclusion"
            + (
                "s"
                if len(excluded_teams) != 1
                else ""
            )
        )

    st.caption(
        "Current optimisation: "
        + " · ".join(
            context_bits
        )
    )

    result = solve_wildcard_cached(
        solver_pool,
        model_bias=model_bias,
        horizon_weeks=solve_horizon,
        locked_player_codes=(
            locked_player_codes
        ),
        excluded_player_codes=(
            excluded_player_codes
        ),
        excluded_teams=(
            excluded_teams
        ),
        current_player_codes=(
            current_player_codes
        ),
        max_transfers=(
            effective_max_transfers
        ),
        budget=transfer_budget,
    )

    squad = result["squad"]
    lineups = result["lineups"]
    scored_pool = result["scored_pool"]

    if optimisation_mode == "Transfers":
        transfers_in = set(
            result.get(
                "transfers_in",
                (),
            )
        )
        transfers_out = set(
            result.get(
                "transfers_out",
                (),
            )
        )

        st.markdown(
            "### Recommended transfers"
        )

        if not transfers_in:
            st.info(
                "No transfer improves the selected objective enough "
                "to be required."
            )
        else:
            transfer_rows = []

            current_lookup = (
                loaded_team["squad"]
                .set_index(
                    "player_code"
                )
            )

            solved_lookup = (
                scored_pool
                .assign(
                    player_code=(
                        scored_pool[
                            "player_code"
                        ]
                        .astype(int)
                    )
                )
                .set_index(
                    "player_code"
                )
            )

            outs = sorted(
                transfers_out
            )
            ins = sorted(
                transfers_in
            )

            for i in range(
                max(
                    len(outs),
                    len(ins),
                )
            ):
                out_code = (
                    outs[i]
                    if i < len(outs)
                    else None
                )
                in_code = (
                    ins[i]
                    if i < len(ins)
                    else None
                )

                transfer_rows.append(
                    {
                        "Out": (
                            current_lookup.loc[
                                out_code,
                                "web_name",
                            ]
                            if out_code is not None
                            else ""
                        ),
                        "Out team": (
                            current_lookup.loc[
                                out_code,
                                "team_name",
                            ]
                            if out_code is not None
                            else ""
                        ),
                        "In": (
                            solved_lookup.loc[
                                in_code,
                                "web_name",
                            ]
                            if in_code is not None
                            else ""
                        ),
                        "In team": (
                            solved_lookup.loc[
                                in_code,
                                "team_name",
                            ]
                            if in_code is not None
                            else ""
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    transfer_rows
                ),
                hide_index=True,
                width="stretch",
            )

    # ========================================================
    # SUMMARY CARDS
    # ========================================================

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Squad cost",
        f"£{result['total_cost']:.1f}m",
    )

    m2.metric(
        "Budget left",
        f"£{result['budget'] - result['total_cost']:.1f}m",
    )

    m3.metric(
        "8GW squad xP",
        f"{squad['scenario_xp_8gw'].sum():.1f}",
    )

    m4.metric(
        "Objective",
        f"{result['objective_value']:.1f}",
    )

    if model_bias_pct < 0:
        bias_label = (
            f"Mean {model_bias_pct}%"
        )
    elif model_bias_pct > 0:
        bias_label = (
            f"Fixtures +{model_bias_pct}%"
        )
    else:
        bias_label = "Model"

    m5.metric(
        "Scoring bias",
        bias_label,
    )

    # ========================================================
    # STARTING XI MATRIX
    # ========================================================

    st.markdown(
        "### Starting XI by gameweek"
    )

    st.caption(
        "Heatmap shows predicted xP. "
        "Light cells indicate the player is benched."
    )

    lineup_matrix = (
        lineups.copy()
    )

    lineup_matrix[
        "starter_xp"
    ] = np.where(
        lineup_matrix[
            "starting_xi"
        ].eq(1),
        lineup_matrix["xp"],
        0,
    )

    gameweeks = sorted(
        lineup_matrix[
            "gameweek"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    base = (
        lineup_matrix[
            [
                "player_code",
                "web_name",
                "position",
                "price",
            ]
        ]
        .drop_duplicates(
            "player_code"
        )
        .copy()
    )

    xp_pivot = (
        lineup_matrix
        .pivot(
            index="player_code",
            columns="gameweek",
            values="xp",
        )
    )

    starter_pivot = (
        lineup_matrix
        .pivot(
            index="player_code",
            columns="gameweek",
            values="starting_xi",
        )
    )

    starter_total = (
        lineup_matrix
        .groupby(
            "player_code"
        )[
            "starter_xp"
        ]
        .sum()
    )

    matrix = (
        base
        .set_index(
            "player_code"
        )
        .join(
            xp_pivot
        )
    )

    matrix = matrix[
        [
            "web_name",
            "position",
            "price",
            *gameweeks,
        ]
    ]

    xp_total = (
        lineup_matrix
        .groupby(
            "player_code"
        )["xp"]
        .sum()
    )

    bench_total = (
        xp_total
        - starter_total
    )

    matrix[
        "Total"
    ] = (
        xp_total
    )

    matrix[
        "Starting"
    ] = (
        starter_total
    )

    matrix[
        "Bench"
    ] = (
        bench_total
    )

    position_order = {
        "GK": 1,
        "DEF": 2,
        "MID": 3,
        "FWD": 4,
    }

    matrix[
        "_position_order"
    ] = (
        matrix["position"]
        .map(
            position_order
        )
    )

    matrix = (
        matrix
        .sort_values(
            [
                "_position_order",
                "Total",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(
            columns=[
                "_position_order"
            ]
        )
    )

    def style_lineup_matrix(
        row
    ):

        styles = [
            "",  # web_name
            "",  # position
            "",  # price
        ]

        player_code = (
            row.name
        )

        for gw in gameweeks:

            is_start = (
                starter_pivot.loc[
                    player_code,
                    gw
                ]
                == 1
            )

            if is_start:
                styles.append(
                    "font-weight: 700;"
                )
            else:
                styles.append(
                    "background-color: #e8e8e8; "
                    "color: #777777; "
                    "font-weight: 500;"
                )

        styles.extend(
            [
                "font-weight: 700;",  # Total
                "font-weight: 700;",  # Starting
                "color: #777777;",    # Bench
            ]
        )

        return styles

    styled_matrix = (
        matrix
        .style
        .background_gradient(
            subset=gameweeks,
            cmap="Greens",
        )
        .apply(
            style_lineup_matrix,
            axis=1,
        )
        .format(
            {
                gw: "{:.2f}"
                for gw in gameweeks
            }
            | {
                "price": "£{:.1f}m",
                "Total": "{:.2f}",
                "Starting": "{:.2f}",
                "Bench": "{:.2f}",
            }
        )
    )

    st.dataframe(
        styled_matrix,
        width="stretch",
        height=650,
    )

    # ========================================================
    # TOP THREATS
    # ========================================================

    st.markdown(
        "### ⚠️ Top threats (not selected)"
    )

    st.caption(
        "Highest-scoring players outside the wildcard squad."
    )

    selected_codes = set(
        squad[
            "player_code"
        ]
        .astype(int)
        .tolist()
    )

    threats = (
        scored_pool[
            ~scored_pool["player_code"]
            .astype(int)
            .isin(selected_codes)
        ]
        .copy()
    )

    threats = (
        threats[
            threats[
                "solver_eligible"
            ]
            .fillna(False)
        ]
    )

    threat_limits = {
        "GK": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }

    threat_cols = (
        st.columns(4)
    )

    for i, position in enumerate(
        [
            "GK",
            "DEF",
            "MID",
            "FWD",
        ]
    ):

        with threat_cols[i]:

            st.markdown(
                f"#### {position}"
            )

            pos_threats = (
                threats[
                    threats[
                        "position"
                    ].eq(
                        position
                    )
                ]
                .sort_values(
                    "scenario_weighted_xp_8gw",
                    ascending=False,
                )
                .head(
                    threat_limits[
                        position
                    ]
                )
                .copy()
            )

            pos_threats = (
                pos_threats[
                    [
                        c for c in [
                            "web_name",
                            "team_name",
                            "price",
                            "scenario_xp_next_gw",
                            "scenario_xp_8gw",
                            "scenario_weighted_xp_8gw",
                            "scenario_delta_8gw",
                        ]
                        if c
                        in pos_threats.columns
                    ]
                ]
            )

            threat_default_columns = [
                c for c in [
                    "web_name",
                    "team_name",
                    "price",
                    "scenario_xp_next_gw",
                    "scenario_xp_8gw",
                    "scenario_weighted_xp_8gw",
                ]
                if c in pos_threats.columns
            ]

            st.dataframe(
                pos_threats,
                hide_index=True,
                width="stretch",
                column_order=threat_default_columns,
                column_config={
                    "web_name":
                        st.column_config.TextColumn(
                            "Player"
                        ),
                    "team_name":
                        st.column_config.TextColumn(
                            "Team"
                        ),
                    "price":
                        st.column_config.NumberColumn(
                            "Price",
                            format="£%.1fm",
                        ),
                    "scenario_xp_next_gw":
                        st.column_config.NumberColumn(
                            "xP next",
                            format="%.2f",
                        ),
                    "scenario_xp_8gw":
                        st.column_config.NumberColumn(
                            "8GW xP",
                            format="%.2f",
                        ),
                    "scenario_weighted_xp_8gw":
                        st.column_config.NumberColumn(
                            "Wtd xP",
                            format="%.2f",
                        ),
                    "scenario_delta_8gw":
                        st.column_config.NumberColumn(
                            "Fixture Δ",
                            format="%+.2f",
                        ),
                },
            )

    st.caption(
        "Threat ranking uses the selected scoring-bias view, "
        "the same scoring view used by the wildcard solver."
    )

    # ========================================================
    # SELECTED SQUAD DETAIL
    # ========================================================

    st.markdown(
        "### Selected squad"
    )

    for position in [
        "GK",
        "DEF",
        "MID",
        "FWD",
    ]:

        pos = (
            squad[
                squad[
                    "position"
                ].eq(
                    position
                )
            ]
            .sort_values(
                "scenario_weighted_xp_8gw",
                ascending=False,
            )
        )

        st.markdown(
            f"#### {position}"
        )

        st.dataframe(
            pos[
                [
                    c for c in [
                        "player_code",
                        "web_name",
                        "team_name",
                        "price",
                        "scenario_xp_next_gw",
                        "scenario_xp_8gw",
                        "scenario_weighted_xp_8gw",
                        "scenario_delta_8gw",
                        "starts",
                    ]
                    if c in pos.columns
                ]
            ],
            width="stretch",
            hide_index=True,
        )
