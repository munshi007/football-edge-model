# Phase 2c-1 — star-reliance features (result: NEGATIVE)

*Run: `features/add_player_features.py` + `model/head_to_head.py`. Date: 2026-07-09.*

Added 9 trainable "Haaland-effect" features from `goalscorers.csv` — how concentrated a
team's recent goals are in one player (top-scorer share, HHI, scorer depth, goals/game),
home/away + diffs. Re-ran the identical 2024+ head-to-head.

## Result: no meaningful change

| Model | Phase 2 (no player feats) | Phase 2c (+ star-reliance) |
|---|---|---|
| Our GBT | 0.8617 | 0.8615 |
| TabPFN | 0.8622–0.8625 | 0.8622 |
| Ensemble | 0.8588 | 0.8584 |

Δ ≈ 0.0002 log-loss — noise. On the World-Cup-like subset, TabPFN remains ahead
(0.9127 vs our GBT 0.9215).

## Why it didn't help
Star-reliance affects a match's **variance** (boom-or-bust), not its **expected outcome** —
and log-loss rewards expected-outcome calibration. pi-ratings already encode team scoring
strength, so goal-concentration adds little independent signal for W/D/L.

## The bigger conclusion (now empirically confirmed twice)
On calibrated W/D/L **log-loss, we cannot beat TabPFN with available trainable information.**
Results-based features — including goalscorer concentration — tie it. This matches the
literature's ceiling and our own repeated negatives (symmetry, draw-recalibration, star-
reliance).

The only *untested* information is **live** (current squad values, confirmed lineups,
injuries). But that is (a) not trainable / not validatable on history, (b) already priced by
the market instantly, and (c) applies to too few remaining matches to prove anything.

**Verdict: stop optimizing log-loss vs TabPFN — it is a proven dead end.** The winnable,
higher-value directions are:
1. **Value / edge** — beat the *market* on ROI (not calibration). Different opponent, real
   headroom (Hubáček & Šír; Wilkens). Sidesteps the log-loss ceiling entirely.
2. **Capabilities** — Dixon-Coles scoreline distributions (over/under, BTTS, correct score,
   xG) that TabPFN structurally cannot produce. A better *system*, not a better decimal.
