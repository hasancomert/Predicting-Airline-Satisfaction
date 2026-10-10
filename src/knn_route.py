"""Route-local k-nearest neighbours: for every row, the k most similar training passengers on the same
route (same Flight Distance), by ratings and profile; score = smoothed share of satisfied neighbours.

Flight Distance behaves like a route id. GBDTs see the route through target encodings of the route and of
route x single columns; this member instead compares whole rating vectors inside the route. The neighbour
pool is the competition training fold only (no original rows), so no row is ever matched to external labels.

Usage: python src/knn_route.py [k=30] [m=10] [k_folds=10] [only=0,..] [name=<tag>] [note=...]
  m: smoothing toward the route's mean (pseudo-count); routes with < 2 training rows fall back to the
     segment (Type of Travel x Class) mean
Writes oof/<tag>.npy and preds/<tag>.npy (several k as separate members: knn_route_k<k>_m<m>_k10).
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RATINGS, load, folds, save

opts = dict(a.split("=", 1) for a in sys.argv[1:])
KS = [int(k) for k in opts.get("k", "10,30,100").split(",")]
M = float(opts.get("m", 10))
K = int(opts.get("k_folds", 10))
T0 = time.time()
train, test, y = load()
n, m = len(train), len(test)
A = pd.concat([train, test], ignore_index=True)


def emb(D):
    """Distance space: ratings (0-5), age / 10, log delays, profile mismatches weighted by 3."""
    cols = [D[r].astype(float).to_numpy() for r in RATINGS]
    cols.append(D["Age"].astype(float).to_numpy() / 10)
    for c in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        cols.append(np.log1p(D[c].fillna(0).astype(float).to_numpy()))
    for c in ["Type of Travel", "Class", "Customer Type", "Gender"]:
        w = 1.5 if c == "Gender" else 3.0
        for v in sorted(D[c].unique()):
            cols.append(w * (D[c] == v).to_numpy(float))
    return np.column_stack(cols).astype(np.float32)


E = emb(A)
route = A["Flight Distance"].to_numpy()
seg = (A["Type of Travel"].astype(str) + "|" + A["Class"].astype(str)).to_numpy()
order = np.argsort(route, kind="stable")
bounds = np.flatnonzero(np.diff(route[order])) + 1
groups = np.split(order, bounds)                       # row indices (train+test) per route


def score_fold(tr_mask, q_mask):
    """Scores for the query rows (q_mask) from neighbours among the training rows (tr_mask)."""
    out = {k: np.full(n + m, np.nan) for k in KS}
    yy = np.zeros(n + m)
    yy[:n] = y
    prior_seg = pd.Series(yy[tr_mask]).groupby(seg[tr_mask]).mean()
    for g in groups:
        q = g[q_mask[g]]
        if len(q) == 0:
            continue
        t = g[tr_mask[g]]
        base = prior_seg.reindex(seg[q]).to_numpy()
        if len(t) < 2:
            for k in KS:
                out[k][q] = base
            continue
        rm = yy[t].mean()
        d = ((E[q, None, :] - E[None, t, :]) ** 2).sum(-1) if len(q) * len(t) < 4_000_000 else None
        if d is None:                                      # very large route: chunk the queries
            d = np.concatenate([((E[q[i:i + 500], None, :] - E[None, t, :]) ** 2).sum(-1)
                                for i in range(0, len(q), 500)])
        idx = np.argsort(d, axis=1)
        for k in KS:
            kk = min(k, len(t))
            s = yy[t][idx[:, :kk]].sum(1)
            # smoothing toward the route mean, itself shrunk toward the segment mean
            prior = (rm * len(t) + base * 20) / (len(t) + 20)
            out[k][q] = (s + M * prior) / (kk + M)
    return out


F = folds(y, K)
ONLY = [int(f) for f in opts.get("only", ",".join(map(str, range(K)))).split(",")]
oof = {k: np.zeros(n) for k in KS}
pred = {k: np.zeros(m) for k in KS}
for i, (tr, va) in enumerate(F):
    if i not in ONLY:
        continue
    tr_mask = np.zeros(n + m, bool)
    tr_mask[tr] = True
    q_mask = np.zeros(n + m, bool)
    q_mask[va] = True
    q_mask[n:] = True
    s = score_fold(tr_mask, q_mask)
    msg = []
    for k in KS:
        oof[k][va] = s[k][va]
        pred[k] += s[k][n:] / len(ONLY)
        msg.append(f"k{k} {roc_auc_score(y[va], s[k][va]):.5f}")
    print(f"fold {i}: " + ", ".join(msg) + f" ({time.time() - T0:.0f}s)", flush=True)
if len(ONLY) == K:
    for k in KS:
        tag = f"knn_route_k{k}_m{M:g}_k{K}"
        print(f"{tag}: OOF AUC {roc_auc_score(y, oof[k]):.5f}")
        save(tag, oof[k], pred[k])
