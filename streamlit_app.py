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

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Players",
        f"{len(solver_pool):,}",
    )

    col2.metric(
        "Teams",
        f"{solver_pool['team_name'].nunique():,}",
    )

    col3.metric(
        "Avg price",
        f"£{solver_pool['price'].mean():.1f}m",
    )

    col4.metric(
        "Eligible",
        f"{int(solver_pool['solver_eligible'].sum()):,}",
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
        "model_xp_8gw",
        "form_xp_8gw",
        "fixture_xp_8gw",
        "form_uplift_8gw",
        "fixture_uplift_8gw",
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
            "form_xp_8gw":
                st.column_config.NumberColumn(
                    "Form view",
                    format="%.2f",
                ),
            "fixture_xp_8gw":
                st.column_config.NumberColumn(
                    "Fixture view",
                    format="%.2f",
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
        "Uses the canonical model xP."
    )

    result = solve_wildcard(
        solver_pool
    )

    squad = result["squad"]
    lineups = result["lineups"]

    m1, m2, m3, m4 = st.columns(4)

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
        f"{squad['xp_8gw'].sum():.1f}",
    )

    m4.metric(
        "Objective",
        f"{result['objective_value']:.1f}",
    )

    for position in [
        "GK",
        "DEF",
        "MID",
        "FWD",
    ]:

        pos = (
            squad[
                squad["position"].eq(
                    position
                )
            ]
            .sort_values(
                "weighted_xp_8gw",
                ascending=False,
            )
        )

        st.markdown(
            f"### {position}"
        )

        st.dataframe(
            pos[
                [
                    c for c in [
                        "player_code",
                        "web_name",
                        "team_name",
                        "price",
                        "xp_next_gw",
                        "xp_8gw",
                        "weighted_xp_8gw",
                        "starts",
                    ]
                    if c in pos.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### Starting XI by gameweek"
    )

    gameweeks = sorted(
        lineups["gameweek"]
        .dropna()
        .unique()
        .tolist()
    )

    chosen_gw = st.selectbox(
        "Gameweek",
        gameweeks,
    )

    gw_lineup = (
        lineups[
            lineups["gameweek"].eq(
                chosen_gw
            )
        ]
        .sort_values(
            [
                "starting_xi",
                "position",
                "xp",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
    )

    st.dataframe(
        gw_lineup,
        use_container_width=True,
        hide_index=True,
    )
