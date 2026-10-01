"""Budgeted Optuna search for LightGBM / XGBoost / CatBoost on a fixed feature set.

Usage: python src/tune.py <lgbm|xgb|cat> <groups> trials=40 lr=0.1 nfold=2 [timeout=sec] [orig=0]
Each trial trains on `nfold` of the 5 fixed folds (same split as train.py) with early stopping and
returns the mean validation AUC. The study is stored in optuna/<model>_<groups>.db so a stopped run
resumes where it left off. Best params are printed as key=value to paste into train.py.
"""
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

from common import ROOT, SEED, folds, load
from features import build, te_keys

warnings.filterwarnings("ignore")
model_name, groups = sys.argv[1], sys.argv[2].split(",")
opts = dict(a.split("=", 1) for a in sys.argv[3:])
TRIALS, LR = int(opts.get("trials", 40)), float(opts.get("lr", 0.1))
NF, TIMEOUT = int(opts.get("nfold", 2)), opts.get("timeout")
USE_ORIG = int(opts.get("orig", 0))
THREADS = int(opts.get("threads", 4))

if USE_ORIG or {"omean", "ofd"} & set(groups):
    train, test, y, orig, yo = load(orig=True)
else:
    train, test, y = load()
    orig = yo = None
n, m = len(train), len(test)
F = build(groups, train, test, orig)
te_spec = [g for g in groups if g.startswith("te")]
KEYS = te_keys(te_spec, train, test, orig).to_numpy() if te_spec else None
if USE_ORIG:
    F["is_orig"] = np.r_[np.zeros(n + m), np.ones(len(orig))].astype(int)
CATCOLS = [c for c in F.columns if isinstance(F[c].dtype, pd.CategoricalDtype)]
X = F.iloc[:n].reset_index(drop=True)
X_orig = F.iloc[n + m:].reset_index(drop=True) if USE_ORIG else None

DATA = []  # cached fold matrices (TE fitted once per fold)
for tr, va in folds(y)[:NF]:
    Xtr, ytr, ktr = X.iloc[tr], y[tr], tr
    if USE_ORIG:
        Xtr = pd.concat([Xtr, X_orig], ignore_index=True)
        ytr = np.r_[ytr, yo]
        ktr = np.r_[tr, np.arange(n + m, n + m + len(orig))]
    Xva = X.iloc[va]
    if KEYS is not None:
        enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
        cols = [f"te{i}" for i in range(KEYS.shape[1])]
        Xtr = pd.concat([Xtr.reset_index(drop=True),
                         pd.DataFrame(enc.fit_transform(KEYS[ktr], ytr), columns=cols)], axis=1)
        Xva = pd.concat([Xva.reset_index(drop=True),
                         pd.DataFrame(enc.transform(KEYS[va]), columns=cols)], axis=1)
    if model_name == "cat":
        Xtr, Xva = (d.astype({c: str for c in CATCOLS}) for d in (Xtr, Xva))
    DATA.append((Xtr, ytr, Xva, y[va]))


def objective(trial):
    aucs = []
    for Xtr, ytr, Xva, yva in DATA:
        if model_name == "lgbm":
            p = dict(num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                     max_depth=trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10]),
                     min_child_samples=trial.suggest_int("min_child_samples", 5, 400, log=True),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                     reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                     min_split_gain=trial.suggest_float("min_split_gain", 1e-4, 1, log=True),
                     max_bin=trial.suggest_categorical("max_bin", [63, 255, 511, 1023]),
                     cat_smooth=trial.suggest_float("cat_smooth", 1, 100, log=True))
            mdl = lgb.LGBMClassifier(n_estimators=20000, learning_rate=LR, subsample_freq=1,
                                     random_state=SEED, verbose=-1, n_jobs=THREADS, **p)
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                    callbacks=[lgb.early_stopping(max(50, int(20 / LR)), verbose=False)])
            pv = mdl.predict_proba(Xva)[:, 1]
        elif model_name == "xgb":
            p = dict(max_depth=trial.suggest_int("max_depth", 3, 10),
                     min_child_weight=trial.suggest_float("min_child_weight", 0.5, 100, log=True),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                     reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                     gamma=trial.suggest_float("gamma", 1e-4, 1, log=True),
                     max_bin=trial.suggest_categorical("max_bin", [64, 256, 512, 1024]))
            mdl = xgb.XGBClassifier(n_estimators=20000, learning_rate=LR, tree_method="hist",
                                    enable_categorical=True, max_cat_to_onehot=4,
                                    eval_metric="auc", early_stopping_rounds=max(50, int(20 / LR)),
                                    random_state=SEED, n_jobs=THREADS, **p)
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pv = mdl.predict_proba(Xva)[:, 1]
        else:
            p = dict(depth=trial.suggest_int("depth", 4, 10),
                     l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
                     random_strength=trial.suggest_float("random_strength", 0.01, 10, log=True),
                     bagging_temperature=trial.suggest_float("bagging_temperature", 0, 2),
                     min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 200, log=True))
            mdl = CatBoostClassifier(iterations=20000, learning_rate=LR, eval_metric="AUC",
                                     od_type="Iter", od_wait=max(100, int(20 / LR)),
                                     cat_features=CATCOLS, border_count=254, random_seed=SEED,
                                     verbose=0, allow_writing_files=False, thread_count=THREADS,
                                     **p)
            mdl.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
            pv = mdl.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(yva, pv))
        trial.report(float(np.mean(aucs)), len(aucs) - 1)
    return float(np.mean(aucs))


(ROOT / "optuna").mkdir(exist_ok=True)
name = f"{model_name}_{'+'.join(groups)}_lr{LR}{'_orig' if USE_ORIG else ''}"
study = optuna.create_study(direction="maximize", study_name=name, load_if_exists=True,
                            storage=f"sqlite:///{ROOT / 'optuna' / (name + '.db')}",
                            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10))
t0 = time.time()
done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])


def cb(st, tr):
    print(f"trial {tr.number}: {tr.value:.5f} | best {st.best_value:.5f} (#{st.best_trial.number}) "
          f"| {time.time() - t0:.0f}s | {tr.params}", flush=True)


study.optimize(objective, n_trials=max(0, TRIALS - done),
               timeout=float(TIMEOUT) if TIMEOUT else None, callbacks=[cb])
print("BEST", study.best_value)
print("PARAMS", " ".join(f"{k}={v}" for k, v in study.best_params.items()))
