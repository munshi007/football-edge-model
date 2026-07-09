# Phase 3 (Option B) — Dixon-Coles goals model (results)

*Run: `.venv/bin/python model/dixon_coles.py`. Date: 2026-07-09.*

Bivariate-Poisson goals model with Dixon-Coles low-score correction + exponential time-decay,
fit by weighted MLE. Produces a full **scoreline distribution** → W/D/L, over/under 2.5, BTTS,
correct score, and expected goals.

## Fit + validation
- 293 teams fitted; `home_adv = 0.259`, `rho = -0.022` (standard DC values).
- Top by attack+defense: Argentina, Spain, Portugal, England, Brazil, France, Belgium, Colombia — sane.
- **W/D/L log-loss 0.8577 / accuracy 60.4%** on the fresh 2024-H1 slice (588 matches) — **competitive with GBT/TabPFN (~0.86)**. A legitimate, independent model, not a weak sibling.

## Capabilities TabPFN cannot produce (2026 WC quarter-finals)
| Match | W / D / L | xG | top | O2.5 | BTTS |
|---|---|---|---|---|---|
| France v Morocco | 37 / 35 / 28 | 1.0–0.8 | 0-0 | 26% | 35% |
| Spain v Belgium | 54 / 26 / 20 | 1.7–0.9 | 1-1 | 48% | 50% |
| Norway v England | 26 / 27 / 47 | 1.1–1.6 | 1-1 | 50% | 54% |
| Argentina v Switzerland | 53 / 28 / 19 | 1.5–0.8 | 1-0 | 39% | 43% |

## The key finding for the edge engine
**Dixon-Coles disagrees with the market.** It rates France just **37%** vs Morocco (goals-based:
Morocco defends well, low xG game) where the sharp-odds blend had France at **59%**. That kind of
model-vs-market divergence is the raw material Option A (the value/edge engine) exploits — a
disagreement to bet on, not a calibration contest to lose.

## Role in the system
Dixon-Coles gives (a) scoreline **markets** the GBT/TabPFN can't, and (b) a structurally
**different** view of each match — valuable both for ensembling and for surfacing market edge.
