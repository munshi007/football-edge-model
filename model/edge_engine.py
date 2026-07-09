"""Phase 5 (Option A) — value / edge engine.

Different game from "beat TabPFN on log-loss": here we look for fixtures where our model
DISAGREES with the market, and size bets by expected value. A disagreement is only profitable
if WE are right and the market is wrong (rare) — so this surfaces *candidate* value, and the
only honest validation is forward ROI tracking (we have no historical odds to backtest).

Kelly for a bet at de-vigged market prob m when our model says p:  f = (p - m) / (1 - m),
expected value per unit staked = p/m - 1.  Flag when p - m >= EDGE and EV > 0.

Run: .venv/bin/python model/edge_engine.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb
from model.dixon_coles import DixonColes

DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.duckdb")
MARKET_CSV = "/Users/rohanmunshi/Desktop/FUN_PROJECT/tabpfn-football-predictions/predictions_20260708.csv"
EDGE = 0.05          # min model-vs-market disagreement to flag
KELLY_CAP = 0.05     # cap stake at 5% of bankroll (fractional Kelly, risk control)


def value(p, m):
    """Return (edge, expected-value-per-unit, kelly-fraction) for model p vs market m."""
    edge = p - m
    ev = p / m - 1 if m > 0 else 0.0
    kelly = max(0.0, edge / (1 - m)) if m < 1 else 0.0
    return edge, ev, min(kelly, KELLY_CAP)


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""SELECT date, home_team, away_team, home_score, away_score,
                               CAST(neutral AS INT) AS neutral
                        FROM matches WHERE home_score IS NOT NULL ORDER BY date""").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    dc = DixonColes().fit(df, "2026-07-09")

    market = pd.read_csv(MARKET_CSV)
    market = market.dropna(subset=["p_market_home"])
    print("Value scan — our Dixon-Coles model vs the de-vigged sharp market (2026 WC QFs)\n")
    print(f"{'fixture':<26}{'outcome':>8}{'model':>7}{'market':>8}{'edge':>7}{'EV':>7}{'stake':>7}")
    print("-" * 70)
    flagged = []
    for _, r in market.iterrows():
        pm = dc.predict_match(r.home_team, r.away_team, neutral=True)
        if pm is None:
            continue
        model = {"home": pm["p_home_win"], "draw": pm["p_draw"], "away": pm["p_away_win"]}
        mkt = {"home": r.p_market_home, "draw": r.p_market_draw, "away": r.p_market_away}
        fx = f"{r.home_team} v {r.away_team}"
        for side in ("home", "draw", "away"):
            edge, ev, kelly = value(model[side], mkt[side])
            flag = " <= VALUE" if (edge >= EDGE and ev > 0) else ""
            print(f"{fx:<26}{side:>8}{model[side]:>7.0%}{mkt[side]:>8.0%}"
                  f"{edge:>+7.0%}{ev:>+7.0%}{kelly:>6.1%}{flag}")
            if flag:
                flagged.append((fx, side, model[side], mkt[side], ev, kelly))
        print()

    print(f"Flagged {len(flagged)} candidate value bet(s) (model disagrees by >= {EDGE:.0%}, EV>0):")
    for fx, side, p, m, ev, k in flagged:
        print(f"  {fx:<26} bet {side.upper():<5} — model {p:.0%} vs market {m:.0%}, EV {ev:+.0%}, stake {k:.1%}")
    print("\n⚠️  Honest caveat: a disagreement is value ONLY if our model is right and the market is")
    print("    wrong — usually the market is right. These are candidates, not proven edges. With no")
    print("    historical odds we cannot backtest ROI; the only real test is tracking these forward.")


if __name__ == "__main__":
    main()
