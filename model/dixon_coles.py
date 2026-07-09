"""Phase 3 (Option B) — Dixon-Coles bivariate-Poisson goals model.

Models each team's attack & defense strength and predicts a full SCORELINE distribution, from
which we read W/D/L, over/under 2.5, both-teams-to-score, correct score, and expected goals —
outputs TabPFN structurally cannot produce. Fit by weighted MLE with Dixon-Coles' low-score
correction (rho) and exponential time-decay so recent matches matter more.

    lambda (home goals) = exp(attack_home - defense_away + home_adv * [not neutral])
    mu     (away goals) = exp(attack_away - defense_home)

Run: .venv/bin/python model/dixon_coles.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import log_loss, accuracy_score

DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.duckdb")
CLASSES = ["away_win", "draw", "home_win"]
CUTOFF = "2024-01-01"
HALF_LIFE = 500.0     # days; recent matches weighted more
WINDOW_YEARS = 8      # only fit on matches within this window of the reference date
MAXG = 10             # scoreline grid 0..MAXG


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0); t[m00] = 1 - lam[m00] * mu[m00] * rho
    m01 = (x == 0) & (y == 1); t[m01] = 1 + lam[m01] * rho
    m10 = (x == 1) & (y == 0); t[m10] = 1 + mu[m10] * rho
    m11 = (x == 1) & (y == 1); t[m11] = 1 - rho
    return np.clip(t, 1e-6, None)


class DixonColes:
    def __init__(self, half_life=HALF_LIFE, reg=0.01):
        self.half_life, self.reg = half_life, reg

    def fit(self, df, ref_date):
        ref = pd.Timestamp(ref_date)
        df = df[(df.date < ref) & (df.date >= ref - pd.Timedelta(days=365 * WINDOW_YEARS))].copy()
        teams = sorted(set(df.home_team) | set(df.away_team))
        self.idx = {t: i for i, t in enumerate(teams)}
        self.teams = teams
        n = len(teams)
        hi = df.home_team.map(self.idx).values
        ai = df.away_team.map(self.idx).values
        x = df.home_score.values.astype(int)
        y = df.away_score.values.astype(int)
        neu = df.neutral.values.astype(float)
        w = 0.5 ** ((ref - df.date).dt.days.values / self.half_life)

        def nll(p):
            att, dfn = p[:n], p[n:2 * n]
            home_adv, rho = p[2 * n], p[2 * n + 1]
            lam = np.exp(att[hi] - dfn[ai] + home_adv * (1 - neu))
            mu = np.exp(att[ai] - dfn[hi])
            ll = w * (np.log(_tau(x, y, lam, mu, rho))
                      + poisson.logpmf(x, lam) + poisson.logpmf(y, mu))
            return -ll.sum() + self.reg * (att @ att + dfn @ dfn)

        p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=200, maxfun=40000))
        self.att = res.x[:n]; self.dfn = res.x[n:2 * n]
        self.home_adv = res.x[2 * n]; self.rho = res.x[2 * n + 1]
        return self

    def _rates(self, h, a, neutral):
        i, j = self.idx.get(h), self.idx.get(a)
        if i is None or j is None:
            return None
        lam = np.exp(self.att[i] - self.dfn[j] + self.home_adv * (0 if neutral else 1))
        mu = np.exp(self.att[j] - self.dfn[i])
        return lam, mu

    def predict_match(self, h, a, neutral=False):
        r = self._rates(h, a, neutral)
        if r is None:
            return None
        lam, mu = r
        gx = np.arange(MAXG + 1)
        px = poisson.pmf(gx, lam)[:, None]
        py = poisson.pmf(gx, mu)[None, :]
        M = px * py
        # Dixon-Coles low-score correction on the 2x2 corner
        M[0, 0] *= 1 - lam * mu * self.rho
        M[0, 1] *= 1 + lam * self.rho
        M[1, 0] *= 1 + mu * self.rho
        M[1, 1] *= 1 - self.rho
        M /= M.sum()
        xi, yi = np.indices(M.shape)
        p_home = M[xi > yi].sum(); p_draw = M[xi == yi].sum(); p_away = M[xi < yi].sum()
        top = np.unravel_index(M.argmax(), M.shape)
        return dict(
            p_home_win=float(p_home), p_draw=float(p_draw), p_away_win=float(p_away),
            over25=float(M[(xi + yi) >= 3].sum()),
            btts=float(M[(xi >= 1) & (yi >= 1)].sum()),
            top_score=f"{top[0]}-{top[1]}", xg_home=float(lam), xg_away=float(mu))


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""SELECT date, home_team, away_team, home_score, away_score,
                               CAST(neutral AS INT) AS neutral
                        FROM matches WHERE home_score IS NOT NULL ORDER BY date""").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])

    print(f"fitting Dixon-Coles (ref {CUTOFF}, half-life {HALF_LIFE:.0f}d)...")
    dc = DixonColes().fit(df, CUTOFF)
    print(f"  teams fitted: {len(dc.teams)} | home_adv={dc.home_adv:.3f} | rho={dc.rho:.3f}")
    rank = pd.DataFrame({"team": dc.teams, "attack": dc.att, "defense": dc.dfn})
    rank["strength"] = rank.attack + rank.defense
    print("  top 8 by attack+defense:",
          ", ".join(rank.sort_values("strength", ascending=False).team.head(8)))

    # --- W/D/L validation on the FRESH slice (first 6 months post-cutoff; ratings still current) ---
    te = df[(df.date >= CUTOFF) & (df.date < "2024-07-01")].copy()
    rows, yy = [], []
    for r in te.itertuples(index=False):
        pm = dc.predict_match(r.home_team, r.away_team, bool(r.neutral))
        if pm is None:
            continue
        rows.append([pm["p_away_win"], pm["p_draw"], pm["p_home_win"]])
        gd = r.home_score - r.away_score
        yy.append("home_win" if gd > 0 else ("away_win" if gd < 0 else "draw"))
    P = np.array(rows); yy = np.array(yy)
    pred = np.array(CLASSES)[P.argmax(1)]
    print(f"\nW/D/L validation ({len(yy)} matches, 2024 H1 fresh slice):")
    print(f"  Dixon-Coles: log-loss {log_loss(yy, P, labels=CLASSES):.4f}  accuracy {accuracy_score(yy, pred):.1%}")
    print("  (for reference: GBT/TabPFN ~0.86 log-loss on the full 2024+ holdout)")

    # --- capability demo: markets TabPFN can't produce, on live WC quarter-finals ---
    print("\nMarkets demo (Dixon-Coles) — 2026 World Cup quarter-finals:")
    qfs = [("France", "Morocco"), ("Spain", "Belgium"), ("Norway", "England"), ("Argentina", "Switzerland")]
    dc_now = DixonColes().fit(df, "2026-07-09")   # refit with current ratings
    print(f"  {'match':<26}{'W / D / L':>18}   {'xG':>9}  {'top':>4}  {'O2.5':>5}  {'BTTS':>5}")
    for h, a in qfs:
        pm = dc_now.predict_match(h, a, neutral=True)
        if pm is None:
            print(f"  {h} v {a}: (team not in fit window)"); continue
        wdl = f"{pm['p_home_win']:.0%} / {pm['p_draw']:.0%} / {pm['p_away_win']:.0%}"
        xg = f"{pm['xg_home']:.1f}-{pm['xg_away']:.1f}"
        print(f"  {h+' v '+a:<26}{wdl:>18}   {xg:>9}  {pm['top_score']:>4}  "
              f"{pm['over25']:>5.0%}  {pm['btts']:>5.0%}")


if __name__ == "__main__":
    main()
