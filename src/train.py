"""K-fold CV for one model + feature-group set. Writes oof/<tag>.npy, preds/<tag>.npy and appends
a row to experiments.md.

Usage: python src/train.py <lgbm|xgb|cat> <groups> [key=value ...]
  groups  comma separated, see features.py (base,afill,delay,...) plus te1/te2/tecat for in-fold
          target encoding
  keys    lr, seed (model seed; the folds never change), folds, preset (default|std),
          orig=1 (original rows added to every training fold, with an is_orig column),
          name=<tag>, note="<text for experiments.md>", log=0 (no experiments.md row);
          anything else is passed to the model (e.g. num_leaves=127)
"""
import ast
import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

from common import FOLDS, ROOT, SEED, folds, load, save
from features import build, te_keys

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")

model_name, groups = sys.argv[1], sys.argv[2].split(",")
opts = {}
for a in sys.argv[3:]:
    k, v = a.split("=", 1)
    try:
        opts[k] = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        opts[k] = v
LR = float(opts.pop("lr", 0.1))
MSEED = int(opts.pop("seed", SEED))
K = int(opts.pop("folds", FOLDS))
PRESET = opts.pop("preset", "std")
USE_ORIG = int(opts.pop("orig", 0))
NOTE = opts.pop("note", "")
LOG = int(opts.pop("log", 1))
NAME = opts.pop("name", None)
THREADS = int(opts.pop("threads", 4))

need_orig = USE_ORIG or bool({"omean", "ofd"} & set(groups))
if need_orig:
    train, test, y, orig, yo = load(orig=True)
else:
    train, test, y = load()
    orig = yo = None
n, m = len(train), len(test)
F = build(groups, train, test, orig)
te_spec = [g for g in groups if g.startswith("te")]
KEYS = te_keys(te_spec, train, test, orig) if te_spec else None
if USE_ORIG:
    F["is_orig"] = np.r_[np.zeros(n + m), np.ones(len(orig))].astype(int)
CATCOLS = [c for c in F.columns if isinstance(F[c].dtype, pd.CategoricalDtype)]
X, X_test = F.iloc[:n].reset_index(drop=True), F.iloc[n:n + m].reset_index(drop=True)
X_orig = F.iloc[n + m:].reset_index(drop=True) if USE_ORIG else None
fold_idx = folds(y, K)


def fold_data(tr, va):
    Xtr, ytr, ktr = X.iloc[tr], y[tr], tr
    if USE_ORIG:
        Xtr = pd.concat([Xtr, X_orig], ignore_index=True)
        ytr = np.r_[ytr, yo]
        ktr = np.r_[tr, np.arange(n + m, n + m + len(orig))]
    Xva, Xte = X.iloc[va], X_test
    if KEYS is not None:
        enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
        cols = list(KEYS.columns)
        k = KEYS.to_numpy()
        add = lambda d, e: pd.concat([d.reset_index(drop=True), pd.DataFrame(e, columns=cols)],
                                     axis=1)
        Xtr = add(Xtr, enc.fit_transform(k[ktr], ytr))
        Xva = add(Xva, enc.transform(k[va]))
        Xte = add(Xte, enc.transform(k[n:n + m]))
    return Xtr, ytr, Xva, Xte


def lgbm(Xtr, ytr, Xva, yva, Xte):
    if PRESET == "default":  # library defaults (100 trees, lr 0.1, 31 leaves), no early stopping
        mdl = lgb.LGBMClassifier(random_state=MSEED, verbose=-1, n_jobs=THREADS, **opts)
        mdl.fit(Xtr, ytr)
        return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.n_estimators
    p = dict(n_estimators=20000, learning_rate=LR, num_leaves=63, min_child_samples=50,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
             max_bin=255, cat_smooth=10)
    p.update(opts)
    mdl = lgb.LGBMClassifier(random_state=MSEED, verbose=-1, n_jobs=THREADS, **p)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(max(50, int(20 / LR)), verbose=False)])
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.best_iteration_


def xgbm(Xtr, ytr, Xva, yva, Xte):
    p = dict(n_estimators=20000, learning_rate=LR, max_depth=6, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, max_bin=256)
    p.update(opts)
    mdl = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=4,
                            eval_metric="auc", early_stopping_rounds=max(50, int(20 / LR)),
                            random_state=MSEED, n_jobs=THREADS, **p)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.best_iteration


def cat(Xtr, ytr, Xva, yva, Xte):
    s = lambda d: d.astype({c: str for c in CATCOLS})
    p = dict(depth=6, border_count=254, l2_leaf_reg=3)
    p.update(opts)
    mdl = CatBoostClassifier(iterations=20000, learning_rate=LR, eval_metric="AUC",
                             od_type="Iter", od_wait=max(100, int(20 / LR)), cat_features=CATCOLS,
                             random_seed=MSEED, verbose=0, allow_writing_files=False,
                             thread_count=THREADS, **p)
    mdl.fit(s(Xtr), ytr, eval_set=(s(Xva), yva), use_best_model=True)
    return (mdl.predict_proba(s(Xva))[:, 1], mdl.predict_proba(s(Xte))[:, 1],
            mdl.get_best_iteration())


fit = {"lgbm": lgbm, "xgb": xgbm, "cat": cat}[model_name]
extra = "_".join(f"{k}{v}" for k, v in sorted(opts.items()))
tag = NAME or "_".join(x for x in [model_name, "+".join(groups), f"lr{LR}", PRESET if PRESET != "std" else "",
                                   "orig" if USE_ORIG else "", extra,
                                   f"s{MSEED}" if MSEED != SEED else "",
                                   f"k{K}" if K != FOLDS else ""] if x)
print(f"{tag}: {X.shape[1]} features, {len(KEYS.columns) if KEYS is not None else 0} TE keys",
      flush=True)
t0 = time.time()
oof, pred, aucs, its = np.zeros(n), np.zeros(m), [], []
for i, (tr, va) in enumerate(fold_idx):
    Xtr, ytr, Xva, Xte = fold_data(tr, va)
    oof[va], p, it = fit(Xtr, ytr, Xva, y[va], Xte)
    pred += p / K
    aucs.append(roc_auc_score(y[va], oof[va]))
    its.append(it)
    print(f"  fold {i}: {aucs[-1]:.5f} ({it} it, {time.time() - t0:.0f}s)", flush=True)
cv = roc_auc_score(y, oof)
mean, std = np.mean(aucs), np.std(aucs)
print(f"{tag} OOF AUC {cv:.5f} | fold mean {mean:.5f} ± {std:.5f} | "
      f"{time.time() - t0:.0f}s, {Xtr.shape[1]} cols", flush=True)
save(tag, oof, pred)
if LOG:
    with open(ROOT / "experiments.md", "a") as f:
        f.write(f"| {tag} | {NOTE} | {mean:.5f} ± {std:.5f} | {cv:.5f} | "
                f"{int(np.mean(its))} | {time.time() - t0:.0f}s | |\n")
