# football-edge-model — Design & Plan

*A football-native prediction system for international match outcomes, built to (1) beat TabPFN as a standalone model and (2) do things TabPFN structurally cannot — with an honest shot at finding market edge.*

**Status:** planning · **Author:** munshi007 · **Date:** 2026-07-06

---

## 0. TL;DR — what this is and why

The existing project (`tabpfn-football-predictions`) already blends TabPFN with sharp odds and sits *at* the market ceiling on log-loss. This is a **separate, more ambitious project**: build our *own* model that beats TabPFN on this task, treat it as a **data-engineering project with a model on top**, and architect it so it could later become a product.

**One-line architecture:** a two-tier system — a **base model** trained on deep history (results + player values) using a football-native representation (dynamic pi-ratings), plus a **live adjustment layer** (lineups, injuries, current form) applied to the specific upcoming fixture — combined by a meta-learner and measured against the market.

---

## 1. The honest verdict (from research — read this first)

A multi-source evidence review (23 verified claims, peer-reviewed sources) produced a blunt reality check that shapes every goal below:

- **You cannot reliably beat sharp closing odds on log-loss. Essentially nobody does.** In the 2023 Soccer Prediction Challenge a bookmaker-consensus model beat the best deep-learning entry by **6.4% RPS**. [Springer *ML* 2024, 10.1007/s10994-024-06608-w]
- **The performance ceiling is narrow.** Published models cluster at **RPS 0.205–0.209, 51–54% accuracy**, and *"both the exact choice of features and the choice of model have only a minor influence."* Poisson ≈ NN ≈ gradient boosting. [Fischer & Heuer 2024, arXiv 2408.08331]
- **The competition-winning recipe is GBT + compact dynamic ratings.** CatBoost+pi-ratings was the single best model (RPS 0.2085), beating deep learning *and* a 205-feature set (0.2416). [Springer *ML* 2024]
- **Player-level ratings add real signal** over static Elo. [Holmes & McHale 2024, *IJF* 40(1)]
- **Market *edge* (ROI) is achievable even with a worse-calibrated model** — by decorrelating from market prices, not by out-predicting. [Hubáček & Šír 2023, *IJF* 39(2); Wilkens 2026 ~10–15% ROI]

**Implication:** "beat TabPFN on log-loss" is winnable but the margin will be *small*. The compelling win is broader — see §2.

**Biggest caveat:** almost all this evidence is **club** football. International football has fewer games/team, roster turnover, and neutral venues, so these numbers may not transfer. We validate on *our* data before believing any of it.

---

## 2. Goals & success criteria

| # | Goal | Metric | Honest expectation |
|---|---|---|---|
| **G1 (primary)** | Beat TabPFN **standalone** | Held-out log-loss + RPS, head-to-head on identical matches | Likely a **small** win (1–3%), driven by 4.7× more training data + pi-ratings |
| **G2** | Do what TabPFN **can't** | Produce full scoreline distributions → over/under, BTTS, correct score, xG | Undeniable capability gap — not a marginal-accuracy argument |
| **G3 (stretch)** | Find **market edge** | Backtested ROI on a disagreement-threshold subset | Real per literature, but fragile out-of-sample; claim only if it survives |
| **G4** | Product-ready **data platform** | Modular ingestion → store → features → serving | The actual moat; reusable asset |

**"Better than TabPFN" = G1 + G2 + G3 together**, not a single log-loss number. We report **both log-loss and RPS** (competitions use RPS) for comparability.

---

## 3. Architecture — data-platform-first

```
                        ┌─────────────── INGEST (one module per source) ───────────────┐
 martj42  Transfermarkt  StatsBomb   soccerdata(FBref/FotMob/Understat)  API-Football  the-odds-api
   │           │            │                    │                          │              │
   └───────────┴────────────┴─── timestamped RAW pulls (replayable) ────────┴──────────────┘
                                             │
                                             ▼
                          ┌──────────────── STORE (DuckDB → Postgres) ────────────────┐
                          │  canonical tables: matches, teams, players, ratings,       │
                          │  values, odds, lineups, injuries                           │
                          │  ENTITY RESOLUTION map: source_name → canonical_id         │
                          └───────────────────────────┬───────────────────────────────┘
                                                       ▼
                       ┌──────────────────────── FEATURES ────────────────────────┐
                       │  pi-ratings engine · star-concentration · goal-contrib    │
                       │  rate · rest/travel · (live layer: lineups/injuries/form) │
                       └───────────┬───────────────────────────┬──────────────────┘
                                   ▼                           ▼
                          ┌──────────────┐            ┌─────────────────┐
                          │ GBT (XGB/Cat)│            │ Dixon-Coles     │   + de-vigged odds
                          │ → W/D/L      │            │ → scorelines    │◄───────────────
                          └──────┬───────┘            └────────┬────────┘
                                 └────────────┬───────────────┘
                                              ▼
                                   ┌────────────────────┐
                                   │  META-LEARNER      │──► final probs ──► EDGE ENGINE (vs market)
                                   └────────────────────┘                    └─► SERVE (API/dashboard)
```

**Design principle:** clean seams between layers, so the personal project scales to a product by *swapping pieces* (DuckDB→Postgres, add a serving API, put ingestion on a scheduler) — not rewriting.

---

## 4. Data sources

### Historical (trainable across all ~47k matches)
| Data | Source | Access | Notes |
|---|---|---|---|
| Results, **goalscorers**, shootouts | [martj42/international_results](https://github.com/martj42/international_results) | git/CSV | Free, 1872→, international. Backbone. |
| **Player market values** | Transfermarkt → [Kaggle player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Kaggle download (weekly) | 400k+ valuations, incl. national-team stats |
| Team Elo (cross-check) | eloratings.net / ClubElo | scrape / soccerdata | We also compute pi-ratings ourselves |

### Live / recent (adjustment layer — NOT trainable across history)
| Data | Source | Access | Freshness |
|---|---|---|---|
| **Lineups, injuries**, player stats | [API-Football](https://www.api-football.com/) (api-sports.io) | REST + key (free tier) | Lineups ~20–40 min pre-kickoff |
| **xG / event data** | [StatsBomb Open Data](https://github.com/statsbomb/open-data) | `statsbombpy` | Major tournaments only |
| Club form of key players | [soccerdata](https://github.com/probberechts/soccerdata) (Understat/FotMob) | Python pkg | Top-5 club leagues, recent |
| Odds (closing) | the-odds-api | REST + key | Already integrated |

**Reality:** rich player data does not exist for old matches. So the **base model** learns on historical sources; the **live layer** only augments matches where that data exists — which, conveniently, is exactly the World Cup fixtures we predict.

**Product caveat:** free/scraped sources (Transfermarkt, Sofascore, StatsBomb open) are **non-commercial**. A paid product needs licensed feeds (API-Football paid, Opta/StatsBomb commercial). The ingestion layer is designed so licensed sources swap in trivially.

---

## 5. Feature design (discipline: ablate everything)

The research warns that **more features usually add noise** (a 205-feature set lost to compact pi-ratings). So we start small and *earn* every feature via held-out ablation.

### 5a. The engine — dynamic pi-ratings (Constantinou & Fenton 2013)
Per team, **separate home & away ratings**, updated chronologically after every match: rating difference → expected goal difference; the observed error nudges both ratings with **damping** (`ψ(e)=3·log₁₀(1+e)` — a 5-0 ≠ 5× a 1-0) and **home↔away cross-learning** (rate γ). Learning rates λ, γ tuned on held-out data. *This is the football-native representation TabPFN lacks.*

### 5b. Player / squad — targets the "Haaland blind spot"
| Factor | Feature | Trainable? |
|---|---|---|
| One elite player swings a knockout | **Star concentration** (top-1 / top-3 value, not squad sum) | ✅ |
| Is he actually playing? | Availability-adjusted XI value | ⚠️ live only |
| Is he in form at club? | Key-player recent club goals/xG | ⚠️ recent only |
| Delivers for country? | **International goal-contribution rate** (martj42 goalscorers) | ✅ |
| Attack-heavy vs balanced | Positional value split (attack € vs defense €) | ✅ |

### 5c. Context
Rest days, **travel distance** (WC 2026 spans US/Canada/Mexico), heat/altitude (Mexico City, summer), knockout stakes, host advantage, GK quality, penalty-shootout history.

**Trainable vs live split is explicit:** ✅ features feed the base model; ⚠️ features feed the live adjustment layer only.

---

## 6. The models

1. **pi-ratings engine** — leak-free chronological pass over 47k matches → home/away rating per team over time.
2. **Expert A — GBT (XGBoost/CatBoost):** predicts W/D/L from pi-rating diff + §5 features. **Trains on ALL 47k matches** (TabPFN's 10k cap is the weakness we exploit). This is the model that beats TabPFN standalone (G1).
3. **Expert B — Dixon-Coles bivariate Poisson:** team attack/defense + home adv + low-score correction (τ) + time decay, fit by MLE → **full scoreline matrix** → W/D/L + over/under + BTTS + correct score + xG (G2).
4. **Meta-learner (stacking):** learns per-situation weights over {Expert A, Expert B, de-vigged odds} on **out-of-fold** predictions (no leakage). Lands *at* the odds ceiling on log-loss.
5. **Edge engine:** flag fixtures where model disagrees with market ≥ threshold; backtest ROI with Kelly staking; report with brutal honesty about out-of-sample fragility (G3).

---

## 7. Validation methodology

- **Strict walk-forward time-split** — train on the past, test on the future only. Never leak.
- **Head-to-head vs TabPFN** — on the *identical* held-out matches, compare Expert A vs TabPFN on log-loss, RPS, accuracy. This single table *is* the G1 claim.
- **Report log-loss AND RPS.**
- **Ablations** — every feature added, measured, kept only if it earns its place.
- **International-specific checks** — verify the club-football literature actually transfers to national teams on our data before trusting any ceiling/ROI number.

---

## 8. Phased roadmap

| Phase | Deliverable | Success gate |
|---|---|---|
| **0 — Data foundation** | Verified access to every source (historical + live); entity-resolution map (canonical team/player IDs); DuckDB store; the feature dossier (source × coverage × cost) | Can pull a sample from each source and join it to martj42 by canonical ID |
| **1 — pi-ratings** | Leak-free pi-ratings engine over 47k matches | Ratings sane vs known team strength; back-tested predictive of results |
| **2 — GBT (beat TabPFN)** | XGBoost/CatBoost on all 47k + pi-ratings | **Beats TabPFN standalone** on walk-forward log-loss + RPS (G1) |
| **3 — Dixon-Coles** | Bivariate-Poisson goals model | Produces calibrated scoreline markets (G2) |
| **4 — Ensemble** | Stacked meta-learner + odds | Matches odds ceiling on log-loss |
| **5 — Edge engine** | Value/ROI backtest | Honest ROI report; claim edge only if out-of-sample survives (G3) |
| **6 — Productization** | Serving API + scheduler + monitoring notes | Optional; gated on data licensing |

**Recommended order: 0 → 1 → 2 first** (the concrete "beat TabPFN" win), then decide on 3–5.

---

## 9. Risks & honest caveats

- **Narrow ceiling** — the G1 log-loss win may be small; lean on G2 (capabilities) + G3 (edge) for the compelling story.
- **Club→international transfer** — most evidence is club football; may not hold. Validate on our data.
- **Live data can't be trained on historically** — it's an adjustment layer, and the market already prices lineups instantly, so it helps us *match* the market more than beat it.
- **ROI backtests are fragile** — often home-win-concentrated and vs soft odds, not sharp closing lines. Don't oversell.
- **Product = data licensing + cost** — free sources are non-commercial; a real product needs paid feeds. A business/data-eng decision, not a modeling one.
- **Over-engineering** — resist. Start with DuckDB + a few modules; add product infra only when a product is real.

---

## 10. Repo structure

```
football-edge-model/
├── PLAN.md                 # this document
├── README.md
├── ingest/                 # one module per source → timestamped raw pulls
├── store/                  # DuckDB schema + entity-resolution map
├── features/               # pi-ratings, star-concentration, live layer
├── model/                  # GBT, Dixon-Coles, meta-learner, edge engine
├── serve/                  # (later) API / dashboard
├── data/                   # local cache (gitignored)
└── docs/                   # feature dossier, results, notes
```

---

## 11. References (verified, peer-reviewed)

- Springer *Machine Learning* 2024 — 2023 Soccer Prediction Challenge (GBT+pi-ratings SOTA; bookmaker beat DL by 6.4%): `10.1007/s10994-024-06608-w`
- Fischer & Heuer 2024 — narrow ceiling, model/feature choice minor: arXiv 2408.08331
- Constantinou 2019 — *Dolores* (pi-ratings + Bayesian nets, 2nd place): `10.1007/s10994-018-5703-7`
- Constantinou & Fenton 2013 — pi-ratings method: *J. Quant. Analysis in Sports*
- Holmes & McHale 2024 — player-rating models: *Int. J. Forecasting* 40(1):302-312
- Hubáček & Šír 2023 — *Beating the Market with a Bad Predictive Model*: *IJF* 39(2); arXiv 2010.12508
- Wilkens 2026 — simple xG/Skellam, ~10–15% ROI: *J. Sports Analytics*
- Dixon & Coles 1997 — bivariate Poisson with low-score correction
