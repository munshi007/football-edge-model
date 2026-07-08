"""Phase 1 — dynamic pi-ratings (Constantinou & Fenton, 2013).

Each team carries TWO ratings: home (R^H) and away (R^A). We sweep every played match in
date order (leak-free: the rating used for a match is the one held BEFORE it, updated only
after), so the per-match ratings are usable as model features with no lookahead.

Prediction:   expected goal difference ĝd = g(R_home^H) - g(R_away^A),  g(r)=sign(r)(10^(|r|/c)-1)
              (neutral venues: use each team's mean rating, so no home tilt)
Update:       error = observed_gd - ĝd ; weight ψ(e)=3·log10(1+|e|) (damps blowouts)
              home team's home rating moves by ψ·λ·sign(error); its away rating cross-updates by ·γ
              away team's away rating moves the opposite way; its home rating cross-updates by ·γ

Home advantage is NOT a global constant here — it's learned per team (a team's R^H drifts above
its R^A if it's genuinely better at home). That's the edge over plain Elo.

Run: .venv/bin/python features/pi_ratings.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb

DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.duckdb")

# literature-optimal defaults (Constantinou & Fenton 2013); tuned properly in Phase 2
LAMBDA = 0.035   # learning rate
GAMMA = 0.70     # home<->away cross-learning
C = 3.0          # rating -> goal-difference scale


def g(r):
    """Rating -> expected goal-difference contribution (diminishing)."""
    return np.sign(r) * (10 ** (np.abs(r) / C) - 1)


def psi(e):
    """Damping so a 5-0 doesn't move ratings 5x a 1-0."""
    return 3 * np.log10(1 + e)


def run(lam=LAMBDA, gamma=GAMMA):
    con = duckdb.connect(DB, read_only=False)
    m = con.execute("""
        SELECT date, home_team, away_team, home_score, away_score,
               CAST(neutral AS VARCHAR) AS neutral
        FROM matches
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date
    """).df()
    m["neutral"] = m["neutral"].str.upper().eq("TRUE")

    RH, RA = {}, {}          # home rating, away rating per team
    rows = []
    for r in m.itertuples(index=False):
        h, a = r.home_team, r.away_team
        rh_h, rh_a = RH.get(h, 0.0), RA.get(h, 0.0)   # home team's ratings
        ra_h, ra_a = RH.get(a, 0.0), RA.get(a, 0.0)   # away team's ratings

        if r.neutral:                                  # no home edge -> use mean ratings
            exp_gd = g((rh_h + rh_a) / 2) - g((ra_h + ra_a) / 2)
        else:
            exp_gd = g(rh_h) - g(ra_a)

        obs_gd = r.home_score - r.away_score
        rows.append((r.date, h, a, bool(r.neutral), rh_h, rh_a, ra_h, ra_a,
                     float(exp_gd), int(obs_gd)))

        # --- update ---
        err = obs_gd - exp_gd
        w = psi(abs(err)) * lam * np.sign(err)         # signed step for the home team
        if r.neutral:
            RH[h], RA[h] = rh_h + w, rh_a + w          # update both (no h/a role)
            RH[a], RA[a] = ra_h - w, ra_a - w
        else:
            RH[h] = rh_h + w                            # home team: primary = home rating
            RA[h] = rh_a + w * gamma                    # cross-update away rating
            RA[a] = ra_a - w                            # away team: primary = away rating
            RH[a] = ra_h - w * gamma                    # cross-update home rating

    feat = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "neutral",
        "home_pi_h", "home_pi_a", "away_pi_h", "away_pi_a", "exp_gd", "obs_gd"])

    # persist: per-match features + current rating per team
    con.execute("CREATE OR REPLACE TABLE pi_ratings AS SELECT * FROM feat")
    teams = sorted(set(RH) | set(RA))
    cur = pd.DataFrame({
        "team": teams,
        "pi_home": [RH.get(t, 0.0) for t in teams],
        "pi_away": [RA.get(t, 0.0) for t in teams],
    })
    cur["pi_mean"] = (cur.pi_home + cur.pi_away) / 2
    con.execute("CREATE OR REPLACE TABLE pi_current AS SELECT * FROM cur")
    con.close()
    return feat, cur


def validate(feat, cur):
    print("=" * 66 + "\nPhase 1 — pi-ratings built\n" + "=" * 66)
    print(f"  matches processed: {len(feat):,} | teams rated: {len(cur):,}")
    print("\n  TOP 15 teams by current rating (sanity check):")
    top = cur.sort_values("pi_mean", ascending=False).head(15)
    for i, r in enumerate(top.itertuples(index=False), 1):
        print(f"   {i:2d}. {r.team:<18} mean {r.pi_mean:+.3f}  (H {r.pi_home:+.3f} / A {r.pi_away:+.3f})")

    # predictive signal check on a recent holdout (naive threshold; real prob model = Phase 2)
    h = feat[feat.date >= "2022-01-01"].copy()
    corr = np.corrcoef(h.exp_gd, h.obs_gd)[0, 1]
    pred = np.where(h.exp_gd > 0.30, "H", np.where(h.exp_gd < -0.30, "A", "D"))
    actual = np.where(h.obs_gd > 0, "H", np.where(h.obs_gd < 0, "A", "D"))
    acc = (pred == actual).mean()
    print(f"\n  holdout 2022+ ({len(h):,} matches): corr(exp_gd, obs_gd)={corr:.3f} | "
          f"naive-threshold outcome accuracy={acc:.1%}")
    print("  (Phase 2 turns exp_gd + these ratings into calibrated W/D/L via gradient boosting)")


if __name__ == "__main__":
    feat, cur = run()
    validate(feat, cur)
