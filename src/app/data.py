import pandas as pd
import streamlit as st

from google.cloud import bigquery
from google.oauth2 import service_account


PROJECT_ID = "mptestproject-489015"
DATASET_ID = "fpl"


def _get_client():

    # Streamlit Community Cloud:
    # read service-account credentials
    # from st.secrets when present.
    if "gcp_service_account" in st.secrets:

        credentials = (
            service_account
            .Credentials
            .from_service_account_info(
                dict(
                    st.secrets[
                        "gcp_service_account"
                    ]
                )
            )
        )

        return bigquery.Client(
            project=PROJECT_ID,
            credentials=credentials,
        )

    # Local development:
    # falls back to Application Default
    # Credentials / gcloud auth.
    return bigquery.Client(
        project=PROJECT_ID
    )


@st.cache_data(
    ttl=600,
    show_spinner="Loading solver pool...",
)
def load_solver_pool():

    client = _get_client()

    sql = f"""
    select *
    from `{PROJECT_ID}.{DATASET_ID}.solver`
    """

    df = (
        client.query(sql)
        .to_dataframe(
        create_bqstorage_client=False
    )
    )

    if "player_code" in df.columns:
        df["player_code"] = (
            pd.to_numeric(
                df["player_code"],
                errors="coerce",
            )
            .astype("Int64")
        )

    if "solver_eligible" in df.columns:
        df["solver_eligible"] = (
            df["solver_eligible"]
            .fillna(False)
            .astype(bool)
        )

    return df
