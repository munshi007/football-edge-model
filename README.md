# ⚽ football-edge-model

A football-native prediction system for **international match outcomes**, built to:

1. **Beat TabPFN** as a standalone model (train on all ~47k matches — no 10k cap — with dynamic *pi-ratings*),
2. Do what TabPFN **can't** — full scoreline distributions (over/under, both-teams-to-score, correct score, xG), and
3. Take an honest shot at finding **market edge** (value/ROI, not just calibration).

Treated as a **data-engineering project with a model on top** — architected so it can grow into a product.

> **Status: planning.** No model code yet. Start with the full design doc:

## 📄 [**Read PLAN.md →**](PLAN.md)

It covers the honest research verdict (you can't beat sharp odds on log-loss — nobody does), the winnable goals, the data-platform-first architecture, data sources (historical + live), feature design, the models (pi-ratings → GBT → Dixon-Coles → meta-learner → edge engine), validation, and the phased roadmap.

## Layout
```
ingest/    one module per source → timestamped raw pulls
store/     DuckDB schema + entity-resolution (canonical team/player IDs)
features/  pi-ratings, star-concentration, live layer
model/     GBT, Dixon-Coles, meta-learner, edge engine
serve/     (later) API / dashboard
data/      local cache (gitignored)
docs/      feature dossier, results
```

Sibling project: [`tabpfn-football-predictions`](https://github.com/munshi007/worldcup-2026-predictions) — the current TabPFN + sharp-odds World Cup entry this one aims to beat.
