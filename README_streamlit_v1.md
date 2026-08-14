# Streamlit V1

Initial front end for the FPL model.

## V1 scope

- Read `mptestproject-489015.fpl.solver`
- Browse/filter the complete solver pool
- Surface model/form/fixture EDA columns when present
- Run the canonical optimal-wildcard solve
- Inspect the selected 15
- Inspect the optimal starting XI by forecast gameweek

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

Authenticate Google Cloud locally using Application Default Credentials,
then run:

```bash
streamlit run streamlit_app.py
```

## Later extensions

The optimisation function is intentionally dataframe-driven so future
controls can be added without changing the UI/data architecture:

- form / fixture weighting
- FPL team ID import
- include / exclude players
- include / exclude teams
- transfer-count limits
- hit costs
- free-transfer logic
- multiple alternative solves
- captaincy
- bench order
