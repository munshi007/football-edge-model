"""Phase 2b — the head-to-head: our GBT vs TabPFN (the G1 "beat TabPFN" test).

Identical features, identical walk-forward holdout. The ONLY differences we're testing:
  • XGBoost trains on ALL history; TabPFN is capped at its last 10k rows (its design limit).
  • trees + dynamic pi-ratings vs a general foundation model.
Metrics: log-loss (our target), RPS (what the forecasting competitions use), accuracy.

Run: .venv/bin/python model/head_to_head.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import log_loss, accuracy_score

DB = os.path.join(os.path.dirname(__file__), "..", "data", "football.duckdb")
SIB_ENV = "/Users/rohanmunshi/Desktop/FUN_PROJECT/tabpfn-football-predictions/.env"
CLASSES = ["away_win", "draw", "home_win"]      # ordinal: away < draw < home
CUTOFF = "2024-01-01"
TABPFN_CAP = 10000

FEATURES = ["neutral", "importance", "home_pi_h", "home_pi_a", "away_pi_h", "away_pi_a",
            "exp_gd", "pi_diff", "home_ppg5", "away_ppg5", "home_ppg10", "away_ppg10",
            "form5_diff", "home_gf5", "home_ga5", "away_gf5", "away_ga5",
            "home_rest", "away_rest", "h2h_gd", "h2h_n"]


def rps(y_idx, proba):
    K = proba.shape[1]
    onehot = np.eye(K)[y_idx]
    return float(np.mean(np.sum((np.cumsum(proba, 1) - np.cumsum(onehot, 1)) ** 2, 1) / (K - 1)))


def report(name, y, proba):
    yi = np.array([CLASSES.index(v) for v in y])
    pred = np.array(CLASSES)[proba.argmax(1)]
    return dict(model=name, logloss=log_loss(y, proba, labels=CLASSES),
                rps=rps(yi, proba), acc=accuracy_score(y, pred))


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(f"SELECT date, {', '.join(FEATURES)}, outcome FROM features ORDER BY date").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])

    tr = df[df.date < CUTOFF]
    te = df[df.date >= CUTOFF]
    Xtr, ytr = tr[FEATURES], tr["outcome"].values
    Xte, yte = te[FEATURES], te["outcome"].values
    print(f"train {len(tr):,}  test {len(te):,}  (holdout {CUTOFF}+)\n")

    results = []

    # --- baseline: train base rates ---
    base = tr["outcome"].value_counts(normalize=True)
    p_base = np.tile([base[c] for c in CLASSES], (len(te), 1))
    results.append(report("baseline (base rates)", yte, p_base))

    # --- our model: XGBoost on ALL history ---
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(CLASSES)
    gbt = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=350, max_depth=4,
        learning_rate=0.04, subsample=0.85, colsample_bytree=0.8,
        reg_lambda=2.0, min_child_weight=5, tree_method="hist", n_jobs=-1, eval_metric="mlogloss")
    gbt.fit(Xtr, le.transform(ytr))
    # align xgb output (le.classes_ order) to CLASSES
    p_gbt = gbt.predict_proba(Xte)[:, [list(le.classes_).index(c) for c in CLASSES]]
    results.append(report(f"OUR GBT (XGBoost, all {len(tr):,} rows + pi-ratings)", yte, p_gbt))

    # --- TabPFN, capped at last 10k train rows (its design limit) ---
    try:
        from dotenv import load_dotenv
        load_dotenv(SIB_ENV)
        from tabpfn_client import TabPFNClassifier, set_access_token
        set_access_token(os.environ["TABPFN_API_KEY"])
        tr_recent = tr.tail(TABPFN_CAP)
        clf = TabPFNClassifier()
        clf.fit(tr_recent[FEATURES].values, tr_recent["outcome"].values)
        proba = clf.predict_proba(Xte.values)
        p_tab = proba[:, [list(clf.classes_).index(c) for c in CLASSES]]
        results.append(report(f"TabPFN (last {TABPFN_CAP:,} rows, same features)", yte, p_tab))
    except Exception as e:
        print(f"  ⚠️ TabPFN step skipped: {type(e).__name__}: {e}")

    # --- ensemble: GBT + TabPFN (they make different errors) ---
    probs = {"OUR GBT": p_gbt}
    if 'p_tab' in dir():
        probs["TabPFN"] = p_tab
        p_ens = (p_gbt + p_tab) / 2
        results.append(report("ENSEMBLE (GBT + TabPFN, 50/50)", yte, p_ens))
        probs["ENSEMBLE"] = p_ens
    probs["baseline"] = p_base

    def table(title, mask):
        yy = yte[mask]
        print(f"\n=== {title} ({mask.sum():,} matches) ===")
        print(f"{'model':<34} {'log-loss':>9} {'RPS':>8} {'acc':>7}")
        print("-" * 60)
        rr = [report(n, yy, p[mask]) for n, p in probs.items()]
        for r in sorted(rr, key=lambda z: z["logloss"]):
            print(f"{r['model']:<34} {r['logloss']:>9.4f} {r['rps']:>8.4f} {r['acc']:>6.1%}")

    table("ALL holdout", np.ones(len(te), bool))
    comp = ((te["neutral"] == 1) & (te["importance"] >= 50)).values
    if comp.sum() > 30:
        table("competitive neutral (World-Cup-like)", comp)


if __name__ == "__main__":
    main()
