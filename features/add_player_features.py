"""Phase 2c-1 — star-reliance features from goalscorers (the "Haaland effect"), trainable.

For each team, over its recent matches, how concentrated are its goals in one player?
A team whose goals funnel through a single star (Norway → Haaland) is a different, more
boom-or-bust proposition than one that scores by committee — a signal pi-ratings (which
see only scorelines) cannot represent. Computed leak-free from martj42 goalscorers.csv
(all history, already in the store) and joined onto the `features` table.

Run: .venv/bin/python features/add_player_features.py   (after build_features.py)
"""
import os
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd
import duckdb

DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.duckdb")
WINDOW = 12   # recent matches per team


def run():
    con = duckdb.connect(DB, read_only=False)
    feat = con.execute("SELECT * FROM features ORDER BY date").df()
    feat["date"] = pd.to_datetime(feat["date"])
    # per-match scorer lists for each team (exclude own goals — those aren't the team's players)
    goals = con.execute("""
        SELECT date, home_team, away_team, team, scorer
        FROM goals WHERE own_goal = FALSE AND scorer IS NOT NULL
    """).df()
    goals["date"] = pd.to_datetime(goals["date"])

    # index: (date, home, away, scoring_team) -> Counter(scorer -> goals)
    scorers = defaultdict(Counter)
    for r in goals.itertuples(index=False):
        scorers[(r.date, r.home_team, r.away_team, r.team)][r.scorer] += 1

    hist = defaultdict(lambda: deque(maxlen=WINDOW))   # per team: Counter per recent match

    def team_feats(team):
        agg = Counter()
        n = len(hist[team])
        for c in hist[team]:
            agg.update(c)
        total = sum(agg.values())
        if total == 0:
            return dict(star_share=np.nan, hhi=np.nan, depth=np.nan, gpg=(0.0 if n else np.nan))
        shares = np.array(list(agg.values())) / total
        return dict(
            star_share=float(shares.max()),               # top scorer's share of goals
            hhi=float((shares ** 2).sum()),                # concentration (1=one scorer)
            depth=int(len(agg)),                           # distinct scorers
            gpg=total / max(n, 1),                         # goals per game
        )

    rows = []
    for r in feat.itertuples(index=False):
        h, a = r.home_team, r.away_team
        fh, fa = team_feats(h), team_feats(a)
        rows.append({
            "date": r.date, "home_team": h, "away_team": a,
            "home_star_share": fh["star_share"], "away_star_share": fa["star_share"],
            "star_share_diff": (fh["star_share"] - fa["star_share"])
                if fh["star_share"] == fh["star_share"] and fa["star_share"] == fa["star_share"] else np.nan,
            "home_scorer_hhi": fh["hhi"], "away_scorer_hhi": fa["hhi"],
            "home_scorer_depth": fh["depth"], "away_scorer_depth": fa["depth"],
            "home_gpg": fh["gpg"], "away_gpg": fa["gpg"],
        })
        # update AFTER using (leak-free)
        hist[h].append(scorers.get((r.date, h, a, h), Counter()))
        hist[a].append(scorers.get((r.date, h, a, a), Counter()))

    pf = pd.DataFrame(rows).drop_duplicates(["date", "home_team", "away_team"])
    merged = (feat.drop_duplicates(["date", "home_team", "away_team"])
                  .merge(pf, on=["date", "home_team", "away_team"], how="left"))
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM merged")
    con.close()

    newcols = [c for c in pf.columns if c not in ("date", "home_team", "away_team")]
    print(f"added {len(newcols)} star-reliance features to {len(merged):,} matches:")
    print(" ", newcols)
    # quick sanity: recent star-reliant teams
    recent = pf[pf.date >= "2024-01-01"]
    top = (recent.groupby(pf["home_team"])["home_star_share"].mean()
           .dropna().sort_values(ascending=False).head(8))
    print("\n  most single-scorer-reliant teams (2024+, home_star_share):")
    for t, v in top.items():
        print(f"    {t:<18} {v:.2f}")
    return merged


if __name__ == "__main__":
    run()
