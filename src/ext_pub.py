"""Convert public OOF / test prediction libraries (downloaded into ext/pub/, see README) into
ext/oof/<name>.npy and ext/preds/<name>.npy, aligned to train.csv / test.csv order and validated.

Only numeric arrays are read (csv / parquet / npz without pickles / npy); every member is checked for
shape, finiteness, and that its OOF AUC is plausible (a member whose OOF is not out-of-fold would show
an implausibly high AUC; anything above 0.9625 is rejected).

Usage: python src/ext_pub.py
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with python -P: only our src/
from common import ROOT, load  # noqa: E402

PUB = ROOT / "ext" / "pub"
OUT_O, OUT_P = ROOT / "ext" / "oof", ROOT / "ext" / "preds"
OUT_O.mkdir(parents=True, exist_ok=True)
OUT_P.mkdir(parents=True, exist_ok=True)
train, test, y = load()
tid, sid = train["id"].to_numpy(), test["id"].to_numpy()
kept, rejected = [], []


def add(name, o, t, src):
    o, t = np.asarray(o, dtype=np.float64), np.asarray(t, dtype=np.float64)
    if o.shape != (len(y),) or t.shape != (len(sid),):
        rejected.append((name, f"shape {o.shape} {t.shape}"))
        return
    if not (np.isfinite(o).all() and np.isfinite(t).all()):
        rejected.append((name, "non-finite"))
        return
    auc = roc_auc_score(y, o)
    if not 0.90 < auc < 0.9625:
        rejected.append((name, f"OOF AUC {auc:.5f}"))
        return
    np.save(OUT_O / f"pub_{name}.npy", o.astype(np.float32))
    np.save(OUT_P / f"pub_{name}.npy", t.astype(np.float32))
    kept.append((name, src, auc))


def by_id(df, ids, col):
    return df.set_index("id")[col].loc[ids].to_numpy()


def d(slug):
    return PUB / slug


# goodpjw2008 Part 8: 48 members
p = d("kn__s6e10-tabpfn-route-categories-lb-0-96160")
go, gt = pd.read_csv(p / "oof_members.csv"), pd.read_csv(p / "test_members.csv")
for c in go.columns[1:]:
    add("gp_" + c, by_id(go, tid, c), by_id(gt, sid, c), "goodpjw P8")

# megayak OOF library, RealMLP seeds averaged
p = d("ds__s6e10-oof-library")
mo, mt = pd.read_parquet(p / "oof_predictions.parquet"), pd.read_parquet(p / "test_predictions.parquet")
groups = {}
for c in mo.columns:
    if c in ("id", "fold", "satisfaction"):
        continue
    groups.setdefault(c.rsplit("_s", 1)[0] if c.startswith("realmlp") else c, []).append(c)
for g, cols in groups.items():
    add("mg_" + g, mo.set_index("id").loc[tid, cols].mean(1).to_numpy(),
        mt.set_index("id").loc[sid, cols].mean(1).to_numpy(), "megayak")

# dariushafshar golem library (row order of train/test)
p = d("ds__s6e10-golem-oof-library")
for k in "abcdefghijklmn":
    add("gl_" + k, np.load(p / f"oof_{k}.npy"), np.load(p / f"test_{k}.npy"), "golem")

# arhancanli12
p = d("ds__s6e10-oof-test-predictions")
ao, at = pd.read_parquet(p / "oof_predictions.parquet"), pd.read_parquet(p / "test_predictions.parquet")
for c in [c for c in ao.columns if c not in ("id", "fold", "satisfaction")]:
    add("ar_" + c, by_id(ao, tid, c), by_id(at, sid, c), "arhancanli12")

# sachith7 (npz with oof / test arrays in train/test order)
for f in sorted(glob.glob(str(d("ds__s6e10-stack-oof-predictions") / "*.npz"))):
    z = np.load(f, allow_pickle=False)
    add("sa_" + os.path.basename(f)[:-4], z["oof"], z["test"], "sachith7")

# najiama blends
p = d("ds__s6e10-oof")
for k in ("01", "02", "03"):
    add(f"nj_{k}", by_id(pd.read_csv(p / f"{k}_blend_oof.csv"), tid, "satisfaction"),
        by_id(pd.read_csv(p / f"{k}_submission.csv"), sid, "satisfaction"), "najiama")

# TabPFN-3.5 kernel outputs
for nm, slug in [("pfn_samanyu_route_og", "kn__s6e10-tabpfn-route-og"),
                 ("pfn_samanyu_seeds", "kn__s6e10-tabpfn-seeds"),
                 ("pfn_hemingweb_raw", "kn__s6e10-exp01-tabpfn35-full-context")]:
    p = d(slug)
    add(nm, by_id(pd.read_csv(p / "oof_exp01_tabpfn35.csv"), tid, "pred"),
        by_id(pd.read_csv(p / "test_exp01_tabpfn35.csv"), sid, "satisfaction"), "TabPFN")

# mitudru (7 members)
p = d("kn__s6e10-analysis-first-equality-blocks-stack")
mo2 = pd.read_parquet(p / "oof_predictions.parquet").set_index("id").loc[tid]
mt2 = pd.read_parquet(p / "test_predictions.parquet").set_index("id").loc[sid]
for c in mo2.columns:
    if c in ("fold", "satisfaction"):
        continue
    add("mi_" + c, mo2[c].to_numpy(), mt2[c].to_numpy(), "mitudru")

# wangxintong111
for slug, c in [("kn__s6e10-catboost-rating-crosses-cv-0-96080", "rating_segments"),
                ("kn__s6e10-lightgbm-context-stats-cv-0-96065", "context_numeric")]:
    p = d(slug)
    add("wx_" + c, by_id(pd.read_parquet(p / "oof_predictions.parquet"), tid, c),
        by_id(pd.read_parquet(p / "test_predictions.parquet"), sid, c), "wangxintong111")

# thisray (3) and megayak honest stack (4)
for slug, tag in [("kn__s6e10-xgb-realmlp-catboost-lb-0-96123", "th"),
                  ("kn__s6e10-honest-oof-stack", "mh")]:
    p = d(slug)
    o = pd.read_csv(p / "oof_members.csv").set_index("id").loc[tid]
    t = pd.read_csv(p / "test_members.csv").set_index("id").loc[sid]
    for c in t.columns:
        if c in o.columns:
            add(f"{tag}_{c}", o[c].to_numpy(), t[c].to_numpy(), tag)

# busyaprime's five models (their folds use seed 33; every row is still out of fold)
for f in sorted(glob.glob(str(d("ds__s6e10-busyaprime-five-members-oof") / "bp_*.npz"))):
    z = np.load(f, allow_pickle=False)
    add(os.path.basename(f)[:-4], z["oof"], z["test"], "busyaprime")

# sadamtorres artifacts (5 folds, seed 42): GBDT variants, kNN features, sparse LR, MLP, RealMLP seeds
for f in sorted(glob.glob(str(d("ds__ps-s6e10-airline-satisfaction-artifacts") / "preds_*.npz"))):
    z = np.load(f, allow_pickle=False)
    if "oof" in z.files and "test" in z.files:
        add("sd_" + os.path.basename(f)[6:-4], z["oof"], z["test"], "sadamtorres")

# amanatar: 7 base engines (5 folds, target encodings fitted inside each fold, original-data teacher);
# their three blends of these engines are skipped
z = np.load(d("kn__s6e10-realmlp-original-prior-gpu") / "oof_artifacts.npz", allow_pickle=False)
assert (z["y"] == y).all()
for c in ("lgbm_champion", "lgbm_te", "xgb_gpu", "catboost_gpu", "pytabkit_realmlp", "torch_prior",
          "torch_scratch"):
    add("am_" + c, z["oof_" + c], z["test_" + c], "amanatar")

# kratosyan "route ID + teacher" (10 folds seed 42, TE inside the fold, 5% inner split for early stopping,
# teacher fitted on the original data only, sachith7's label-free aux_ev features)
for f in sorted(glob.glob(str(d("kn__s6e10-xgboost-realmlp-route-id-teacher") / "members" / "*.npz"))):
    z = np.load(f, allow_pickle=False)
    add("kr_" + os.path.basename(f)[:-4], z["oof"], z["test"], "kratosyan")

# denpugovkin numeric-key CatBoost CTR (5 folds, 10% inner split for early stopping): per-fold files
p = d("kn__s6e10-numeric-keys-catboost-ctr") / "exp020"
fo, ft, nf = np.full(len(y), np.nan), np.zeros(len(sid)), 0
for f in sorted(glob.glob(str(p / "catboost_numeric_ctr_fold*.npz"))):
    z = np.load(f, allow_pickle=False)
    pos = pd.Series(np.arange(len(tid)), index=tid).loc[z["valid_ids"]].to_numpy()
    fo[pos] = z["valid_pred"]
    ft += pd.Series(z["test_pred"], index=z["test_ids"]).loc[sid].to_numpy()
    nf += 1
add("dp_catboost_numeric_ctr", fo, ft / max(nf, 1), "denpugovkin")

# amanatar's v5 run: only the engine that is new relative to v1 (row-level TE LightGBM)
z = np.load(d("kn__s6e10-the-original-prior-stack-v5") / "oof_artifacts.npz", allow_pickle=False)
assert (z["y"] == y).all()
add("am_lgbm_te_rows", z["oof_lgbm_te_rows"], z["test_lgbm_te_rows"], "amanatar")

# kagankoral "Encodings + CatBoost + RealMLP" (5 folds seed 42, TE inside the fold with an inner CV)
p = d("kn__s6e10-encodings-catboost-realmlp-lb-0-961")
o = pd.read_csv(p / "oof_members.csv").set_index("id").loc[tid]
t = pd.read_csv(p / "test_members.csv").set_index("id").loc[sid]
for c in t.columns:
    if c in o.columns:
        add("kg_" + c, o[c].to_numpy(), t[c].to_numpy(), "kagankoral")

# amanatar "Forge own pairs": rebuild of busyaprime's ladder (5 folds seed 33), near-copies of bp_* (corr >= 0.9993)
for f in sorted(glob.glob(str(d("ds__s6e10-forge-own-pairs") / "oof_forge_*.npy"))):
    nm = os.path.basename(f)[4:-4]
    add("fg_" + nm[6:], np.load(f), np.load(os.path.join(os.path.dirname(f), f"test_{nm}.npy")), "forge")

k = pd.DataFrame(kept, columns=["name", "source", "oof_auc"])
print(k.groupby("source").oof_auc.agg(["size", "max"]).sort_values("max", ascending=False).round(5))
print(f"kept {len(kept)} members; rejected {len(rejected)}: {rejected}")
k.to_csv(ROOT / "ext" / "pub_members.csv", index=False)
