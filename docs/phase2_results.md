# Phase 2 — GBT vs TabPFN head-to-head (results)

*Run: `.venv/bin/python model/head_to_head.py`. Date: 2026-07-09. Holdout: matches on/after 2024-01-01 (train 46,863 / test 2,642). Identical 21 features (pi-ratings + form + goals + rest + H2H). XGBoost trains on all history; TabPFN capped at its last 10k rows.*

## Result

**All 2,642 holdout internationals**
| Model | log-loss | RPS | accuracy |
|---|---|---|---|
| **Ensemble** (GBT + TabPFN, 50/50) | **0.8588** | 0.1656 | 60.6% |
| Our GBT (all 46,863 rows + pi-ratings) | 0.8617 | 0.1662 | **60.7%** |
| TabPFN (last 10k, same features) | 0.8625 | 0.1660 | 60.5% |
| baseline (base rates) | 1.0538 | 0.2274 | 47.5% |

**World-Cup-like subset (316 competitive neutral matches)**
| Model | log-loss | RPS | accuracy |
|---|---|---|---|
| **TabPFN** | **0.9145** | 0.1656 | 56.0% |
| Ensemble | 0.9159 | 0.1655 | 55.7% |
| Our GBT | 0.9203 | 0.1661 | 56.0% |

## Honest verdict

**We matched TabPFN; we did not decisively beat it.**
- Overall, our GBT edges TabPFN (0.8617 vs 0.8625 log-loss; 60.7% vs 60.5% acc) — but by **0.1%, which is within noise**. A statistical tie.
- The **ensemble** genuinely beats TabPFN-alone overall (0.8588, a ~0.4% log-loss gain) — a small but real result, and the strongest honest claim available.
- On the **matches that actually matter** (World-Cup-like), **TabPFN is slightly ahead** of our GBT, and the ensemble only ties it.

This is exactly what the literature predicted: *"the exact choice of features and the choice of model have only a minor influence."* Results-based features hit a ceiling, and both models sit on it.

## Two findings worth keeping

1. **Training on all 47k didn't give the edge we hoped.** TabPFN's 10k cap uses the *most recent* matches, which are the most predictive. The extra 37k *older* matches add little for forecasting 2024+ games — **recency beats volume**. The data-cap argument for beating TabPFN is weaker than it looked on paper.
2. **A from-scratch pi-ratings + GBT model matches a foundation model** on 2,642 held-out matches. Matching TabPFN with our own interpretable model, and beating it via a cheap ensemble, is a legitimate engineering outcome — just not the decisive win.

## What it would actually take to beat TabPFN (the real lever)

Not a better model — **information TabPFN can't see.** Both models here use the same results-based features. The decisive edge is the **player-level layer** we scoped but haven't integrated:
- **Star concentration** (top-1/top-3 squad value) — the Haaland effect that beat our WC predictions.
- **Player availability** (lineups/injuries) and **key-player club form**.
- **Match xG** (StatsBomb) for tournament matches.

This needs the data sources still pending keys: **Kaggle Transfermarkt** (player values) + **API-Football** (lineups/injuries) + **StatsBomb** (xG). That is Phase 2c, and it's the honest path from "ties TabPFN" to "beats TabPFN."

*Minor untested levers (CatBoost instead of XGBoost, post-hoc calibration) are unlikely to change the verdict given the model-choice-barely-matters finding.*
