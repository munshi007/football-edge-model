"""Store layer — load martj42 into a DuckDB warehouse as canonical tables.

This is the first real piece of the data platform: a single queryable store the feature
layer reads from. Start with the free historical backbone; other sources append later.
Run: .venv/bin/python store/build_store.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import duckdb
import pandas as pd

RAW = "https://raw.githubusercontent.com/martj42/international_results/master/"
DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.duckdb")


def build():
    res = pd.read_csv(RAW + "results.csv")
    gs = pd.read_csv(RAW + "goalscorers.csv")
    shootouts = pd.read_csv(RAW + "shootouts.csv")

    # canonical team dimension (one row per team, stable id)
    teams = sorted(set(res.home_team) | set(res.away_team))
    teams_df = pd.DataFrame({"team_id": range(len(teams)), "canonical_name": teams})

    con = duckdb.connect(DB)
    con.execute("CREATE OR REPLACE TABLE teams AS SELECT * FROM teams_df")
    con.execute("CREATE OR REPLACE TABLE matches AS SELECT * FROM res")
    con.execute("CREATE OR REPLACE TABLE goals AS SELECT * FROM gs")
    con.execute("CREATE OR REPLACE TABLE shootouts AS SELECT * FROM shootouts")
    # a source→canonical alias table, seeded empty; entity resolution fills it
    con.execute("CREATE OR REPLACE TABLE team_aliases (source VARCHAR, source_name VARCHAR, canonical_name VARCHAR)")

    print(f"store built → {os.path.abspath(DB)}")
    for t in ["teams", "matches", "goals", "shootouts", "team_aliases"]:
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t:<14} {n:>8,} rows")
    con.close()


if __name__ == "__main__":
    build()
