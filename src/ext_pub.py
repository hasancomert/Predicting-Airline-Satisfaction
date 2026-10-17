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
    if min(o.min(), t.min()) < 0 or max(o.max(), t.max()) > 1:
        # stored as logits / scores, not probabilities (arhancanli12's *_5f): sigmoid, so that the stacker's
        # logit gives the values back instead of clipping them to +-13.8
        o, t = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))
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

# cdeotte "TFM OOF and test predictions": tabular foundation models (TabPFN-3.5, TabICL2, Mitra2, LimiX2,
# KumoRFM small/large, EXAONE, TabFM, Causilo) + RealMLP / XGBoost; train/test row order
for f in sorted(glob.glob(str(d("ds__s6e10-tfm-oof-and-test-predictions") / "oof_*.npy"))):
    nm = os.path.basename(f)[4:-4]
    add("cd_" + nm, np.load(f), np.load(os.path.join(os.path.dirname(f), f"test_{nm}.npy")), "cdeotte TFM")

# 09.10 refresh. golem's three new members, minus o_tabpfn35 (a copy of gp_tabpfn_catfd10, corr 0.99999)
p = d("ds__s6e10-golem-oof-library")
for k in ("p_lgbm_auxev", "q_lgbm_auxev_10fold"):
    add("gl_" + k, np.load(p / f"oof_{k}.npy"), np.load(p / f"test_{k}.npy"), "golem")
# 15.10: golem v6 members (same frozen folds; label-free aux EVs / counts / route profiles; trees early-stop on
# the validation fold, the RealMLPs take their 4 epochs with that fold as pytabkit validation set). The library
# itself was in a broken / processing state (404 for the full download); files fetched one by one
# tested on 15.10 with the s21 stack: 0.962285 -> 0.962285 / 0.962284 -> 0.962282 (meta seeds 7 / 1), no gain
# (as the author measured on the 247-pair pool), so they stay out of the pool
GOLEM_V6 = ("r_realmlp_auxev", "s_realmlp_raw_auxev_10fold", "t_xgb_d5_auxev", "u_catboost_ctr_auxev")

# abdullahsafwan333 "oof-dataset": goodpjw2008's recipes re-run on 15 folds (seed 42), TabPFN members with
# 3 seeds; the TabPFN code (aviator-tabpfn) uses the out-of-fold rows only as context. tabpfn_te_cnt has no
# public code (its AUC is in line with the larger context, not with a leak)
p = d("ds__oof-dataset")
for k in ("cat_v3", "xgb_v3", "tabpfn_catfd10", "tabpfn_raw", "tabpfn_te_cnt"):
    add("ab_" + k, by_id(pd.read_csv(p / f"{k}_oof.csv"), tid, "pred"),
        by_id(pd.read_csv(p / f"{k}_test.csv"), sid, "pred"), "abdullahsafwan333")

# hermengardo single CatBoost (5 folds, 10% inner split for early stopping, original-data lookup features)
p = d("kn__single-catboost-eda-ps6e10")
add("hg_catboost", np.load(p / "oof_train.npy"), np.load(p / "oof_test.npy"), "hermengardo")

# yekenot's public RealMLP (the recipe behind bp_realmlp and our yk view)
p = d("kn__ps-s6-e10-realmlp-pytabkit")
add("ye_realmlp", by_id(pd.read_csv(p / "oof_preds.csv"), tid, "satisfaction"),
    by_id(pd.read_csv(p / "submission.csv"), sid, "satisfaction"), "yekenot")

# 10.10: koumeimaki "s6e10-route-tfm-nn-oof": 7 members on the shared 5-fold split (fold column checked
# below), train/test row order: TabPFN-3.5 catfd10 (full context, 1 estimator), TabICL v2 (250k context),
# RealMLP (yekenot settings + aux columns), TabM, LightGBM / XGBoost / CatBoost on 109-144 engineered columns
p = d("ds__s6e10-route-tfm-nn-oof")
kf = pd.read_csv(p / "train_ids_folds.csv")
assert np.array_equal(kf["id"].to_numpy(), tid) and np.array_equal(pd.read_csv(p / "test_ids.csv")["id"].to_numpy(), sid)
from common import folds  # noqa: E402
f5 = np.full(len(y), -1)
for i, (_, va) in enumerate(folds(y, 5)):
    f5[va] = i
assert np.array_equal(kf["fold"].to_numpy(), f5), "koumeimaki folds differ from StratifiedKFold(5, True, 42)"
for k in pd.read_csv(p / "members.csv")["tag"]:
    add("km_" + k, np.load(p / f"oof_{k}.npy"), np.load(p / f"test_{k}.npy"), "koumeimaki")

# 11.10 refresh. goodpjw2008 "extra members": 17 unpublished members of the same series (outer folds
# StratifiedKFold(5, True, 42), _k10 = 10 folds; trees early-stop on the scored fold, as in their Part 8 members)
p = d("ds__s6e10-goodpjw2008-extra-members-oof")
for k in pd.read_csv(p / "members.csv")["member"]:
    add("gx_" + k, np.load(p / f"oof_{k}.npy"), np.load(p / f"test_{k}.npy"), "goodpjw extra")

# dmitriykolmikov CatBoost, every column categorical + frequencies + original-data means/teacher, original rows
# in each training fold, StratifiedKFold(5, True, 42), early stopping on the validation fold
p = d("ds__s6e10-catboost-all-categorical-oof-and-test")
add("dk_cb_cat", np.load(p / "oof_cb_cat.npy"), np.load(p / "test_cb_cat.npy"), "dmitriykolmikov")

# 13.10 refresh. najiama 07: single XGBoost (5 folds seed 42, sklearn TargetEncoder fitted on each training fold,
# label-free route profile / frequencies, early stopping on the validation fold); 04-06 are blends (not used)
p = d("ds__s6e10-oof")
add("nj_07_xgb", by_id(pd.read_csv(p / "07_XGBoost_oof.csv"), tid, "oof_pred"),
    by_id(pd.read_csv(p / "07_submission.csv"), sid, "satisfaction"), "najiama")

# kagankoral "ablation members": 17 members of the encodings pipeline of kagankoral's public notebook
# (StratifiedKFold(5 or 10, True, 42); te / te_pair with an inner CV inside each training fold, original-data
# model and rates, frequencies; trees early-stop on the validation fold, RealMLP fixed epochs)
p = d("ds__s6e10-ablation-members-oof")
for k in pd.read_csv(p / "members.csv")["member"]:
    add("ka_" + k, np.load(p / f"oof_{k}.npy"), np.load(p / f"test_{k}.npy"), "kagankoral ablation")

# 17.10 refresh. goodpjw2008 extra members, version 3: TabPFN-3.5 with its own 4-estimator ensemble
# (n_estimators=4, full 560k training-fold context, raw + catfd + catfd10, StratifiedKFold(5, True, 42); OOF
# 0.961491). Not in their members.csv; the other files of the new version are identical to the 11.10 copy.
p = d("ds__s6e10-goodpjw2008-extra-members-oof-v1017")
add("gx_tabpfn_ne4", np.load(p / "oof_tabpfn_ne4.npy"), np.load(p / "test_tabpfn_ne4.npy"), "goodpjw extra")

# jiatianzhang2 RealMLP (pytabkit RealMLP_TD, plr_sigma 32, n_ens 4, 20 epochs, best epoch on pytabkit's own
# inner validation split of the training fold; raw columns + Flight Distance categorical twin; StratifiedKFold(5,
# True, 42)); code read
p = d("kn__s6e10-realmlp-plr32-route-twin")
add("jz_realmlp_plr32", np.load(p / "oof_realmlp_plr32.npy"), np.load(p / "test_realmlp_plr32.npy"), "jiatianzhang2")

# goodpjw2008 Part 3 kernel, full re-run on 17.10 (26 members, the recipe of the gp_ copies from Part 8: inner
# 5-fold target encodings on the training rows, StratifiedKFold(5, True, 42), trees early-stop on the scored
# fold); a new GPU draw, logit correlation 0.9987-0.9999 with the gp_ copies
p = d("kn__s6e10-catboost-ctr-glm-margin-xgb-lb-0-96127")
go, gt = pd.read_csv(p / "oof_members.csv"), pd.read_csv(p / "test_members.csv")
for c in go.columns[1:]:
    add("g3_" + c, by_id(go, tid, c), by_id(gt, sid, c), "goodpjw P3 rerun")

k = pd.DataFrame(kept, columns=["name", "source", "oof_auc"])
print(k.groupby("source").oof_auc.agg(["size", "max"]).sort_values("max", ascending=False).round(5))
print(f"kept {len(kept)} members; rejected {len(rejected)}: {rejected}")
k.to_csv(ROOT / "ext" / "pub_members.csv", index=False)
