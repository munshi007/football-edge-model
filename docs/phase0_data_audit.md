# Phase 0 — Data Audit (access verified)

*Run: `.venv/bin/python ingest/phase0_access_check.py` and `store/build_store.py`. Date: 2026-07-08.*

## What we proved works (free, no credentials)

| Source | Result | Coverage |
|---|---|---|
| **martj42** (results, goalscorers, shootouts) | ✅ **49,505 matches**, 47,886 goals, 15,386 scorers, **336 teams** (1872→2026) | Full history · the backbone |
| **StatsBomb Open Data** (`statsbombpy`) | ✅ 24 competitions; **WC 2018 & 2022** (64 matches, 32 teams) with real **player xG** (e.g. A. Johnston 0.64 xG vs Morocco) | Major tournaments only |
| **soccerdata** | ✅ imports; sources reachable (ClubElo, FBref, FotMob, Understat, Sofascore) | Mostly club — for key-player recent form |
| **DuckDB store** | ✅ built `data/football.duckdb`: `teams`, `matches`, `goals`, `shootouts`, `team_aliases` | The store layer works |

## Entity resolution — the key finding

**Team names: trivial.** All **32/32** StatsBomb WC teams match martj42 exactly. The core team join is clean; almost no alias map needed. ✅

**Player names: this is the real data-engineering work.** Only **49%** of StatsBomb lineup players exact-match martj42 scorers. Two distinct causes:
1. **Coverage gap** — martj42's `goalscorers.csv` lists *only players who scored*, not full squads. A goalkeeper (Aaron Ramsdale) or non-scorer simply isn't there. → For full-squad player data we **need a squad source** (StatsBomb lineups for tournaments; API-Football / Transfermarkt for the rest).
2. **Name-format mismatch** — "Achraf Hakimi Mouh" vs "Achraf Hakimi", "Alejandro Balde Martínez" vs "Alejandro Balde". → Needs a **normalization + fuzzy-match layer** (strip accents, drop middle/legal-name tokens, match on normalized key), with the resolved map stored in a `player_aliases` table.

**Implication for the plan:** the `store/` entity-resolution layer is where real effort goes, and it's **player-level, not team-level**. Team-level features (pi-ratings, results) are unblocked *today*. Player-level features need the squad source + normalization layer first.

## Blocked pending your credentials (free to obtain)

| Source | What it unlocks | You need to provide |
|---|---|---|
| **Transfermarkt** via [Kaggle player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Player market values → star-concentration, positional splits | **Kaggle API token** (`~/.kaggle/kaggle.json`) |
| **API-Football** (api-sports.io) | Live **lineups + injuries** → "is Haaland playing" layer | **API-Football key** (free tier, ~100 req/day) |
| the-odds-api | Closing odds | *Key already exists in sibling repo `.env`* |

## Verdict
The **free, historical, team-level foundation is fully verified and loaded** — enough to build Phases 1–2 (pi-ratings + GBT, the "beat TabPFN" milestone) *right now*. Player-value and live-lineup layers wait on two free API keys and the player entity-resolution layer.

## Next
- **Unblocked now:** Phase 1 — build the pi-ratings engine on the 49,505-match store.
- **Needs keys:** wire Kaggle (values) + API-Football (lineups/injuries), then build the player entity-resolution (`player_aliases`) layer.
