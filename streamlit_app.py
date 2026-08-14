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

    # ========================================================
    # STARTING XI MATRIX
    # ========================================================

    st.markdown(
        "### Starting XI by gameweek"
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
                    "font-weight: 600;"
                )
            else:
                styles.append(
                    "opacity: 0.35; "
                    "background-color: #eeeeee;"
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
            cmap="YlGn",
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

    st.caption(
        "Heatmap shows predicted xP. "
        "Faded cells indicate the player is benched."
    )

    st.dataframe(
        styled_matrix,
        use_container_width=True,
    )
