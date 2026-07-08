# Phase 1 — pi-ratings engine (results)

*Run: `.venv/bin/python features/pi_ratings.py`. Date: 2026-07-08.*

Dynamic pi-ratings (Constantinou & Fenton 2013) built leak-free over **49,501** played matches, **336** teams. Two ratings per team (home/away); home advantage is *learned per team*, not a global constant.

## Sanity check — top 15 by current rating
```
 1. Spain        +1.873   6. Portugal    +1.682   11. Japan        +1.621
 2. Argentina    +1.853   7. Belgium     +1.678   12. Morocco      +1.590
 3. France       +1.830   8. Colombia    +1.671   13. Switzerland  +1.582
 4. Brazil       +1.758   9. Netherlands +1.650   14. Norway       +1.546
 5. England      +1.701  10. Germany     +1.647   15. Ecuador      +1.545
```
These are the genuinely strong international sides — the ratings are meaningful, not noise. Note the learned home/away splits: **Morocco** (H +1.82 / A +1.36) and **Portugal** are much stronger at home; **England** slightly stronger away (tournament/neutral pedigree).

## Predictive signal (holdout 2022+, 4,664 matches)
- **corr(expected GD, actual GD) = 0.613** — strong for a high-variance sport.
- **Naive-threshold outcome accuracy = 57.6%** — using *only* the raw rating difference, no ML, no other features, no odds.

That last number matters: TabPFN's published baseline is ~59% *with* its full feature set. We're already at 57.6% from a single dynamic rating. Phase 2 (gradient boosting on pi-ratings + form/value/player features, trained on all 49k matches) is where this turns into calibrated W/D/L probabilities that go head-to-head with TabPFN.

## Note on the Haaland effect
Norway rates a solid #14, but Brazil is #4 — so pi-ratings alone would favor Brazil. That's *expected*: pi-ratings are results-based and can't see that one elite player (Haaland) can swing a knockout. This is exactly the gap the planned player-level features (star-concentration, availability) are meant to fill.

## Stored
`data/football.duckdb`: `pi_ratings` (per-match leak-free features) and `pi_current` (latest rating per team).
