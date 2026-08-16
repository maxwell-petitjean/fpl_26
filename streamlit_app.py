import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

from src.app.data import load_solver_pool
from src.app.wildcard import solve_wildcard



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
            fixture_label
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
    )

    SELECT
        f.season,
        f.gameweek,
        f.fixture_id,
        f.team_name,
        f.opponent_team_name,
        f.home_away,
        f.fixture_label,
        s.position_group,
        s.core_points_avg_l3,
        s.core_points_avg_l6,
        s.core_points_avg_l12
    FROM future_fixtures f
    CROSS JOIN UNNEST(['GK', 'DEF', 'ATT']) AS position_group
    LEFT JOIN latest_strength s
        ON f.opponent_team_name = s.opponent_team_name
       AND position_group = s.position_group
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
    view["cell"] = np.where(
        view["score"].notna(),
        (
            view["opponent_team_name"]
            + " "
            + view["score"].map(
                lambda x: f"{x:.2f}"
            )
        ),
        (
            view["opponent_team_name"]
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
    "Model-led player pool and optimal wildcard squad."
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
        "fixture_xpp90_full_8gw",
        "fixture_xpp90_delta_8gw",
        "fixture_full_xp_8gw",
        "fixture_full_uplift_8gw",
        "avg_fixture_multiplier_8gw",
        "solver_eligible",
    ]

    solver_default_columns = [
        "web_name",
        "position",
        "team_name",
        "price",
        "model_xpp90_8gw",
        "fixture_xpp90_full_8gw",
        "xmins_next_gw",
        "xp_next_gw",
        "xp_8gw",
    ]

    display_columns = [
        c for c in preferred_columns
        if c in view.columns
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
        use_container_width=True,
        hide_index=True,
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
                "Grey means no historic strength value is available."
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
        "Fixture sensitivity can tilt DEF and MID toward favourable fixtures; "
        "GK and FWD remain on canonical model xP."
    )

    st.markdown(
        "### Fixture strategy"
    )
    
    fixture_sensitivity_pct = st.slider(
        "Fixture sensitivity",
        min_value=0,
        max_value=125,
        value=10,
        step=5,
        help=(
            "0% = canonical model. "
            "100% = full fixture adjustment. "
            "Only defenders and midfielders are affected."
        ),
    )
    
    fixture_sensitivity = (
        fixture_sensitivity_pct
        / 100
    )
    
    st.caption(
        "Backtesting favoured a strong fixture adjustment. "
        "Only DEF and MID are affected; GK and FWD remain unchanged."
    )

    result = solve_wildcard(
        solver_pool,
        fixture_sensitivity=fixture_sensitivity,
    )

    squad = result["squad"]
    lineups = result["lineups"]
    scored_pool = result["scored_pool"]

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
        f"£{100 - result['total_cost']:.1f}m",
    )

    m3.metric(
        "8GW squad xP",
        f"{squad['scenario_xp_8gw'].sum():.1f}",
    )

    m4.metric(
        "Objective",
        f"{result['objective_value']:.1f}",
    )

    m5.metric(
        "Fixture sensitivity",
        f"{fixture_sensitivity_pct}%",
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

    matrix[
        "Total"
    ] = (
        starter_total
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

        styles.append(
            "font-weight: 700;"
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
                "Total": "{:.2f}"
            }
        )
    )

    st.dataframe(
        styled_matrix,
        use_container_width=True,
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
                            "scenario_fixture_uplift_8gw",
                        ]
                        if c
                        in pos_threats.columns
                    ]
                ]
            )

            st.dataframe(
                pos_threats,
                hide_index=True,
                use_container_width=True,
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
                    "scenario_fixture_uplift_8gw":
                        st.column_config.NumberColumn(
                            "Fixture Δ",
                            format="%+.2f",
                        ),
                },
            )

    st.caption(
        "Threat ranking uses the fixture-sensitive weighted xP, "
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
                "scenario_xp_8gw",
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
                        "scenario_fixture_uplift_8gw",
                        "starts",
                    ]
                    if c in pos.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
