"""Phase 2a — build the model feature matrix (leak-free).

Joins the pre-match pi-ratings to each match and adds classic football signals computed in a
single chronological pass using only *past* matches: recent form, goals for/against, rest
days, and head-to-head. Writes a `features` table to the DuckDB store.

Run: .venv/bin/python features/build_features.py   (requires pi_ratings — run pi_ratings.py first)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb

DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.duckdb")


def importance(t):
    t = str(t)
    if "FIFA World Cup" in t and "qual" not in t.lower():
        return 60
    if any(k in t for k in ["UEFA Euro", "Copa América", "African Cup", "AFC Asian Cup", "Gold Cup"]) and "qual" not in t.lower():
        return 50
    if "qual" in t.lower():
        return 40
    if "Friendly" in t:
        return 20
    return 35


def run():
    con = duckdb.connect(DB, read_only=False)
    # base = played matches with scores + tournament, joined to pre-match pi-ratings
    df = con.execute("""
        SELECT m.date, m.home_team, m.away_team, m.home_score, m.away_score, m.tournament,
               p.neutral, p.home_pi_h, p.home_pi_a, p.away_pi_h, p.away_pi_a, p.exp_gd
        FROM matches m
        JOIN pi_ratings p
          ON m.date = p.date AND m.home_team = p.home_team AND m.away_team = p.away_team
        WHERE m.home_score IS NOT NULL
        ORDER BY m.date
    """).df()
    df["date"] = pd.to_datetime(df["date"])

    from collections import defaultdict, deque
    hist = defaultdict(lambda: deque(maxlen=10))   # per team: (points, gf, ga)
    last = {}                                        # per team: last match date
    h2h = defaultdict(lambda: deque(maxlen=10))      # (a,b) sorted: goal diff from a's view

    def feats(team):
        h = list(hist[team])
        if not h:
            return dict(ppg5=np.nan, ppg10=np.nan, gf5=np.nan, ga5=np.nan)
        last5, last10 = h[-5:], h
        return dict(
            ppg5=np.mean([x[0] for x in last5]),
            ppg10=np.mean([x[0] for x in last10]),
            gf5=np.mean([x[1] for x in last5]),
            ga5=np.mean([x[2] for x in last5]),
        )

    rows = []
    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        fh, fa = feats(h), feats(a)
        rest_h = (r.date - last[h]).days if h in last else np.nan
        rest_a = (r.date - last[a]).days if a in last else np.nan
        key = tuple(sorted([h, a]))
        hh = list(h2h[key])
        sign = 1 if key[0] == h else -1                 # orient h2h goal-diff to the home team
        h2h_gd = np.mean([x * sign for x in hh]) if hh else np.nan

        gd = r.home_score - r.away_score
        outcome = "home_win" if gd > 0 else ("away_win" if gd < 0 else "draw")
        rows.append({
            "date": r.date, "home_team": h, "away_team": a,
            "neutral": int(r.neutral), "importance": importance(r.tournament),
            "home_pi_h": r.home_pi_h, "home_pi_a": r.home_pi_a,
            "away_pi_h": r.away_pi_h, "away_pi_a": r.away_pi_a,
            "exp_gd": r.exp_gd,
            "pi_diff": (r.home_pi_h + r.home_pi_a) / 2 - (r.away_pi_h + r.away_pi_a) / 2,
            "home_ppg5": fh["ppg5"], "away_ppg5": fa["ppg5"],
            "home_ppg10": fh["ppg10"], "away_ppg10": fa["ppg10"],
            "form5_diff": (fh["ppg5"] - fa["ppg5"]) if not np.isnan(fh["ppg5"]) and not np.isnan(fa["ppg5"]) else np.nan,
            "home_gf5": fh["gf5"], "home_ga5": fh["ga5"],
            "away_gf5": fa["gf5"], "away_ga5": fa["ga5"],
            "home_rest": min(rest_h, 90) if rest_h == rest_h else np.nan,
            "away_rest": min(rest_a, 90) if rest_a == rest_a else np.nan,
            "h2h_gd": h2h_gd, "h2h_n": len(hh),
            "outcome": outcome,
        })

        # --- update histories (after using them) ---
        ph = 3 if gd > 0 else (1 if gd == 0 else 0)
        pa = 3 if gd < 0 else (1 if gd == 0 else 0)
        hist[h].append((ph, r.home_score, r.away_score))
        hist[a].append((pa, r.away_score, r.home_score))
        last[h] = last[a] = r.date
        h2h[key].append(gd if key[0] == h else -gd)

    feat = pd.DataFrame(rows)
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM feat")
    con.close()

    print(f"features built: {len(feat):,} matches x {feat.shape[1]-4} features")
    print("outcome mix:", feat.outcome.value_counts(normalize=True).round(3).to_dict())
    print("columns:", [c for c in feat.columns if c not in ("date", "home_team", "away_team", "outcome")])
    return feat


if __name__ == "__main__":
    run()
