# Phase 5 (Option A) — value / edge engine (results + honest read)

*Run: `.venv/bin/python -m model.edge_engine`. Date: 2026-07-09.*

The engine scans for fixtures where our model disagrees with the de-vigged sharp market,
computes expected value and fractional-Kelly stakes, and flags disagreements ≥5% with EV>0.

## What it flagged (Dixon-Coles vs market, 2026 WC QFs)
| Fixture | Bet | Model | Market | EV | Kelly |
|---|---|---|---|---|---|
| France v Morocco | DRAW | 35% | 25% | +42% | 5% |
| France v Morocco | MOROCCO | 28% | 14% | +109% | 5% |

## The honest read (this is the lesson, not a goldmine)
Those "value bets" are almost certainly **our model being wrong, not the market.** Dixon-Coles is
a *goals-only* model — in a low-scoring France–Morocco matchup it underrates France's overall
quality, so it disagrees with the market by a huge 25%. **A big divergence from a lone weak-ish
model is a red flag, not an edge.** The market at France 62% is very likely closer to correct.

The value-betting literature that actually profits uses models **competitive with the market**
finding **small, specific** disagreements — not a single model disagreeing wildly. So:

1. The engine framework is correct and working (EV, Kelly, flagging, risk cap).
2. Fed a lone goals model, it mostly surfaces that model's own errors.
3. **To be credible it must be fed the competitive ENSEMBLE** (GBT + Dixon-Coles + odds), look for
   *small* disagreements, and be **validated forward** — a divergence is value only if we're right.
4. **We cannot backtest ROI**: the free odds feed is live-only, so there is no historical odds
   series to test against. The only honest validation is tracking flagged bets going forward.

## Verdict
Option A is built and demonstrably functional, but it does **not** yet constitute proven edge —
and it honestly shows why beating the market is hard. Real progress here needs: (a) feed it the
ensemble not the lone goals model, (b) a source of historical odds to backtest, or (c) patient
forward tracking. This is the mature, non-hyped state of the value angle.
