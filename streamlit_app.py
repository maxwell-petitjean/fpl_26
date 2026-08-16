import numpy as np
import streamlit as st

from src.app.data import load_solver_pool
from src.app.wildcard import solve_wildcard


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

tab_pool, tab_wildcard = st.tabs(
    [
        "Player pool",
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
        value=90,
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
            "",
            "",
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
                            "scenario_weighted_xp_8gw",
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
                    "scenario_weighted_xp_8gw":
                        st.column_config.NumberColumn(
                            "Wtd xP",
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
                        "scenario_fixture_uplift_8gw",
                        "starts",
                    ]
                    if c in pos.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )