"""Sparse logistic regression on one-hot encoded values (every column treated as categorical) and
one-hot encoded column pairs. A different model family for the ensemble.

Usage: python src/glm.py [pairs=all|low|none] [C=0.1] [fdpairs=1] [name=<tag>] [note=...]
  pairs    all : every pair of the 20 low-cardinality columns; low : only pairs among the 4
           categoricals and the 13 ratings; none : main effects only
  fdpairs  Flight Distance crossed with Class / Type of Travel / Customer Type
Rare levels (< 5 rows over train+test) are merged into one level per column.
"""
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import CATS, COLS, RATINGS, ROOT, folds, load, save
from features import clean

warnings.filterwarnings("ignore")
opts = dict(a.split("=", 1) for a in sys.argv[1:])
PAIRS = opts.get("pairs", "all")
C = float(opts.get("C", 0.1))
FDP = int(opts.get("fdpairs", 1))
NOTE = opts.get("note", "")

train, test, y = load()
n, m = len(train), len(test)
A = pd.concat([train[COLS], test[COLS]], ignore_index=True)
A["Arrival Delay in Minutes"] = A["Arrival Delay in Minutes"].fillna(-1)
codes = {}
for c in COLS:
    v = A[c].astype(str)
    vc = v.value_counts()
    v = v.where(v.map(vc) >= 5, "__rare__")
    codes[c] = pd.factorize(v)[0]
low = [c for c in COLS if c != "Flight Distance"]
pair_cols = {"all": low, "low": CATS + RATINGS, "none": []}[PAIRS]
blocks = [codes[c] for c in COLS]
for a, b in combinations(pair_cols, 2):
    blocks.append(codes[a] * (codes[b].max() + 1) + codes[b])
if FDP:
    for c in ["Class", "Type of Travel", "Customer Type"]:
        blocks.append(codes["Flight Distance"] * 10 + codes[c])
# one nonzero per block in every row: build the CSR arrays directly (int32, low memory)
idx = np.empty((n + m, len(blocks)), dtype=np.int32)
off = 0
for j, b in enumerate(blocks):
    b = pd.factorize(b)[0]
    idx[:, j] = b + off
    off += b.max() + 1
X = sp.csr_matrix((np.ones(idx.size, dtype=np.float32), idx.ravel(),
                   np.arange(0, idx.size + 1, len(blocks), dtype=np.int64)), shape=(n + m, off))
del idx
X_tr, X_te = X[:n], X[n:]
tag = opts.get("name", f"glm_pairs{PAIRS}_fd{FDP}_C{C}")
print(f"{tag}: {X.shape[1]} one-hot columns, {len(blocks)} blocks", flush=True)

t0 = time.time()
oof, pred, aucs = np.zeros(n), np.zeros(m), []
for i, (tr, va) in enumerate(folds(y)):
    mdl = LogisticRegression(C=C, max_iter=300, tol=1e-4)
    mdl.fit(X_tr[tr], y[tr])
    oof[va] = mdl.decision_function(X_tr[va])
    pred += mdl.decision_function(X_te) / 5
    aucs.append(roc_auc_score(y[va], oof[va]))
    print(f"  fold {i}: {aucs[-1]:.5f} ({time.time() - t0:.0f}s)", flush=True)
oof, pred = 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-pred))
cv = roc_auc_score(y, oof)
print(f"{tag} OOF AUC {cv:.5f} | fold mean {np.mean(aucs):.5f} ± {np.std(aucs):.5f}", flush=True)
save(tag, oof, pred)
with open(ROOT / "experiments.md", "a") as f:
    f.write(f"| {tag} | {NOTE} | {np.mean(aucs):.5f} ± {np.std(aucs):.5f} | {cv:.5f} | - | "
            f"{time.time() - t0:.0f}s | |\n")
