"""Compare ensembles of saved OOF predictions and write the chosen one's test predictions.

Usage: python src/blend.py <out_name> <tag> [<tag> ...] [method=hill|rank|mean|logit|all]
  tags        files in oof/ and preds/ (without .npy); "a,b,c" averages several tags (seeds) first
  method      hill  : repeated-selection greedy hill climbing on ranks (weights = counts)
              rank  : equal-weight rank average
              mean  : equal-weight probability average
              logit : logistic regression stacking on logit(probabilities)
              all   : report all four, write the best by nested CV
Every method is scored two ways: on the full OOF (in-sample for the weights) and nested: weights are
fitted on 4/5 of the OOF rows and scored on the held-out 1/5 (5 splits, seed 42); the final weights
are fitted on all OOF rows. Writes submissions/<out_name>.csv when out_name != "-".
"""
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import OOF_DIR, PRED_DIR, load  # noqa: E402
from submit_check import write_submission  # noqa: E402

args = [a for a in sys.argv[1:] if "=" not in a]
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
out, tags = args[0], args[1:]
method = opts.get("method", "all")
_, _, y = load()


def ld(d, t):
    # tags starting with "ext_" are public OOF/test predictions kept in ext/oof and ext/preds
    path = lambda x: (d.parent / "ext" / d.name if x.startswith("ext_") else d) / f"{x}.npy"
    return np.mean([np.load(path(x)).astype(np.float64) for x in t.split(",")], axis=0)


O = np.column_stack([ld(OOF_DIR, t) for t in tags])
P = np.column_stack([ld(PRED_DIR, t) for t in tags])
rk = lambda a: np.column_stack([rankdata(c) / len(c) for c in a.T])
lg = lambda a: logit(np.clip(a, 1e-6, 1 - 1e-6))
OR, PR, OL, PL = rk(O), rk(P), lg(O), lg(P)
for t, o in zip(tags, O.T):
    print(f"  {roc_auc_score(y, o):.5f}  {t}")
c = np.corrcoef(OR.T)
if len(tags) > 1:
    print("  rank corr min/max off-diag:",
          f"{c[~np.eye(len(tags), dtype=bool)].min():.4f} / {c[~np.eye(len(tags), dtype=bool)].max():.4f}")


def fit_hill(idx):
    single = [roc_auc_score(y[idx], OR[idx, j]) for j in range(len(tags))]
    cnt = np.zeros(len(tags))
    cnt[int(np.argmax(single))] = 1
    best = max(single)
    s = OR[idx] @ cnt
    for _ in range(200):
        score, j = max((roc_auc_score(y[idx], s + OR[idx, j]), j) for j in range(len(tags)))
        if score <= best + 1e-7:
            break
        best, cnt[j], s = score, cnt[j] + 1, s + OR[idx, j]
    w = cnt / cnt.sum()
    return lambda A_rank, A_logit: A_rank @ w, w


def fit_rank(idx):
    w = np.ones(len(tags)) / len(tags)
    return lambda A_rank, A_logit: A_rank @ w, w


def fit_mean(idx):
    w = np.ones(len(tags)) / len(tags)
    return lambda A_rank, A_logit: (1 / (1 + np.exp(-A_logit))) @ w, w


def fit_logit(idx):
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(OL[idx], y[idx])
    return lambda A_rank, A_logit: lr.decision_function(A_logit), lr.coef_.ravel()


FITS = {"hill": fit_hill, "rank": fit_rank, "mean": fit_mean, "logit": fit_logit}
methods = list(FITS) if method == "all" else [method]
res = {}
for mth in methods:
    nested = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(O, y):
        f, _ = FITS[mth](tr)
        nested[va] = f(OR[va], OL[va])
    # score scales differ between splits, so report the mean of the per-split AUCs
    per = [roc_auc_score(y[va], nested[va]) for _, va in
           StratifiedKFold(5, shuffle=True, random_state=42).split(O, y)]
    f, w = FITS[mth](np.arange(len(y)))
    full = roc_auc_score(y, f(OR, OL))
    res[mth] = (np.mean(per), full, f, w)
    print(f"{mth:6s} full-OOF {full:.5f} | nested mean {np.mean(per):.5f} ± {np.std(per):.5f} | "
          f"w = {np.round(w, 3).tolist()}")

best = max(res, key=lambda k: res[k][0])
print(f"best by nested CV: {best}")
if out != "-":
    f = res[best][2]
    pred = f(PR, PL)
    if best in ("logit",):
        pred = 1 / (1 + np.exp(-pred))
    write_submission(out, pred)
