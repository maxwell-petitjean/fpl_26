from pathlib import Path

import pandas as pd


PLAYER_FEATURES_PATH = Path(
    "data/features/fct_gw_features_historic.csv"
)

OPPONENT_FEATURES_PATH = Path(
    "data/features/fct_opponent_position_features_historic.csv"
)

OUTPUT_PATH = Path(
    "data/modelling/fct_xpp90_historic.csv"
)


POSITION_GROUP_MAP = {
    "GK": "GK",
    "DEF": "DEF",
    "MID": "ATT",
    "FWD": "ATT",
}


def build_xpp90_modelling_dataset(save=True):

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    players = pd.read_csv(
        PLAYER_FEATURES_PATH,
        low_memory=False,
    )

    opponents = pd.read_csv(
        OPPONENT_FEATURES_PATH,
        low_memory=False,
    )

    print("=== XPP90 MODELLING DATASET ===")
    print(f"Player rows: {len(players):,}")
    print(f"Opponent rows: {len(opponents):,}")

    # ---------------------------------------------------------
    # PLAYER POSITION GROUP
    # ---------------------------------------------------------

    players["position_group"] = (
        players["position"]
        .map(POSITION_GROUP_MAP)
    )

    if players["position_group"].isna().any():

        bad = (
            players.loc[
                players["position_group"].isna(),
                ["position"],
            ]
            .drop_duplicates()
        )

        raise ValueError(
            "Unmapped player positions:\n"
            + bad.to_string(index=False)
        )

    # ---------------------------------------------------------
    # SELECT OPPONENT FEATURES
    # ---------------------------------------------------------

    opponent_feature_cols = [
        c
        for c in opponents.columns
        if (
            c.startswith("opp_pos_")
            and (
                c.endswith("_filled")
                or c.endswith("_source")
            )
        )
    ]

    opponent_join_cols = [
        "season",
        "fixture_id",
        "opponent_team_code",
        "position_group",
    ]

    opponent_model = opponents[
        opponent_join_cols
        + opponent_feature_cols
    ].copy()

    # ---------------------------------------------------------
    # VALIDATE OPPONENT GRAIN
    # ---------------------------------------------------------

    duplicate_opponents = (
        opponent_model
        .duplicated(
            opponent_join_cols,
            keep=False,
        )
    )

    if duplicate_opponents.any():

        sample = opponent_model.loc[
            duplicate_opponents,
            opponent_join_cols,
        ].head(20)

        raise ValueError(
            "Duplicate opponent join grain:\n"
            + sample.to_string(index=False)
        )

    # ---------------------------------------------------------
    # JOIN
    # ---------------------------------------------------------

    out = players.merge(
        opponent_model,
        on=opponent_join_cols,
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    if len(out) != len(players):
        raise ValueError(
            "Join changed player row count."
        )

    filled_cols = [
        c
        for c in out.columns
        if (
            c.startswith("opp_pos_")
            and c.endswith("_filled")
        )
    ]

    source_cols = [
        c
        for c in out.columns
        if (
            c.startswith("opp_pos_")
            and c.endswith("_source")
        )
    ]

    print(f"Joined rows: {len(out):,}")
    print(
        f"Opponent filled features: "
        f"{len(filled_cols):,}"
    )
    print(
        f"Opponent source columns: "
        f"{len(source_cols):,}"
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    if save:

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        out.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print(f"Saved: {OUTPUT_PATH}")

    return out


if __name__ == "__main__":
    build_xpp90_modelling_dataset()
