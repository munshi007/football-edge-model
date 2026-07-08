"""Phase 0 — access check.

Proves (or disproves) that each planned data source actually returns data, and surfaces
the entity-resolution problem: do the sources agree on team names, or must we build a
canonical-ID map? Run: .venv/bin/python ingest/phase0_access_check.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

RAW = "https://raw.githubusercontent.com/martj42/international_results/master/"


def check_martj42():
    print("\n" + "=" * 70 + "\n[1] martj42 — results + goalscorers (backbone, free, all history)\n" + "=" * 70)
    res = pd.read_csv(RAW + "results.csv")
    gs = pd.read_csv(RAW + "goalscorers.csv")
    teams = sorted(set(res.home_team) | set(res.away_team))
    print(f"  results:      {len(res):,} matches  ({res.date.min()} → {res.date.max()})")
    print(f"  goalscorers:  {len(gs):,} goals, {gs.scorer.nunique():,} unique scorers")
    print(f"  canonical teams: {len(teams)}")
    print(f"  sample scorer rows:\n{gs.tail(3).to_string(index=False)}")
    return set(teams)


def check_statsbomb():
    print("\n" + "=" * 70 + "\n[2] StatsBomb Open Data — xG / event data (tournaments only, free)\n" + "=" * 70)
    try:
        from statsbombpy import sb
        comps = sb.competitions()
        wc = comps[comps.competition_name.str.contains("World Cup", case=False, na=False)]
        wc = wc[~wc.competition_name.str.contains("Women", case=False, na=False)]
        print(f"  competitions available: {comps.competition_name.nunique()}")
        print(f"  World Cup seasons: {sorted(wc.season_name.unique())}")
        # pick the most recent men's WC season available
        row = wc.sort_values("season_name").iloc[-1]
        cid, sid = int(row.competition_id), int(row.season_id)
        matches = sb.matches(competition_id=cid, season_id=sid)
        sb_teams = sorted(set(matches.home_team) | set(matches.away_team))
        print(f"  {row.competition_name} {row.season_name}: {len(matches)} matches, {len(sb_teams)} teams")
        # sample player xG from one match
        m = matches.iloc[0]
        ev = sb.events(match_id=int(m.match_id))
        shots = ev[ev.type == "Shot"]
        if "shot_statsbomb_xg" in shots.columns:
            xg = shots.groupby("player")["shot_statsbomb_xg"].sum().sort_values(ascending=False).head(3)
            print(f"  sample player xG — {m.home_team} vs {m.away_team}:")
            for p, v in xg.items():
                print(f"      {p:<28} {v:.2f} xG")
        return set(sb_teams)
    except Exception as e:
        print(f"  ⚠️ StatsBomb check failed: {type(e).__name__}: {e}")
        return set()


def check_soccerdata():
    print("\n" + "=" * 70 + "\n[3] soccerdata — club form / Elo (mostly CLUB; import + reachability)\n" + "=" * 70)
    try:
        import soccerdata as sd
        print(f"  soccerdata {sd.__version__} import OK — sources: ClubElo, FBref, FotMob, Understat, Sofascore, WhoScored")
        print("  (club-focused; used for key-player recent club form — tested for reach only)")
        return True
    except Exception as e:
        print(f"  ⚠️ soccerdata import failed: {e}")
        return False


def entity_resolution(martj_teams, sb_teams):
    print("\n" + "=" * 70 + "\n[4] ENTITY RESOLUTION — do StatsBomb team names match martj42?\n" + "=" * 70)
    if not sb_teams:
        print("  (skipped — no StatsBomb teams pulled)")
        return
    matched = sb_teams & martj_teams
    mismatched = sorted(sb_teams - martj_teams)
    print(f"  StatsBomb WC teams: {len(sb_teams)} | exact-match to martj42: {len(matched)} | MISMATCH: {len(mismatched)}")
    if mismatched:
        print("  names that DON'T join (need alias map):")
        for t in mismatched:
            # nearest guess in martj42 by shared token
            guess = [m for m in martj_teams if t.split()[-1] in m or m.split()[-1] in t][:2]
            print(f"      StatsBomb '{t}'  →  martj42 candidates: {guess or '??'}")
    else:
        print("  ✅ all StatsBomb teams join cleanly — minimal alias map needed")


def credential_gated():
    print("\n" + "=" * 70 + "\n[5] CREDENTIAL-GATED sources (need keys from user — NOT tested)\n" + "=" * 70)
    print("  • Transfermarkt player values → Kaggle 'davidcariboo/player-scores'  (needs Kaggle API token)")
    print("  • API-Football (lineups + injuries, live)                            (needs API-Football key, free tier)")
    print("  • the-odds-api (closing odds)                                        (key exists in sibling repo .env)")


if __name__ == "__main__":
    mj = check_martj42()
    sb_teams = check_statsbomb()
    check_soccerdata()
    entity_resolution(mj, sb_teams)
    credential_gated()
    print("\nPhase 0 access check complete.")
