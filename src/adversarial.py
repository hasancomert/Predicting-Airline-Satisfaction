"""Adversarial validation: can a model tell train rows from test rows (and from original rows)?"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import COLS, CATS, load


def adv(a, b, name):
    X = pd.concat([a[COLS], b[COLS]], ignore_index=True)
    for c in CATS:
        X[c] = X[c].astype("category")
    t = np.r_[np.zeros(len(a)), np.ones(len(b))]
    oof = np.zeros(len(t))
    imp = np.zeros(len(COLS))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, t):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1,
                               colsample_bytree=0.8)
        m.fit(X.iloc[tr], t[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        imp += m.booster_.feature_importance("gain")
    print(f"{name}: adversarial AUC {roc_auc_score(t, oof):.5f}")
    top = pd.Series(imp / imp.sum(), index=COLS).sort_values(ascending=False).head(6)
    print("  top gain:", {k: round(v, 3) for k, v in top.items()})


train, test, y, orig, yo = load(orig=True)
adv(train, test, "train vs test")
adv(train.sample(len(orig), random_state=0), orig, "train vs original")
