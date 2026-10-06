"""Probe: LightGBM meta-learner on member logits (+ optional raw columns), same nested meta folds as
stack.py. Compares with the LR stack on the same folds; also LR-stack + GBDT residual blend."""
import sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OOF_DIR, PRED_DIR, ROOT, load, COLS, CATS

opts = dict(a.split("=", 1) for a in sys.argv[1:])
train, test, y = load()
def path(d, x):
    return (ROOT / "ext" / d.name if x.startswith(("ext_", "pub_")) else d) / f"{x}.npy"
def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
names, Z = [], []
for line in open(opts["own"]):
    if line.strip():
        tags = line.strip().split(",")
        Z.append(lg(np.mean([np.load(path(OOF_DIR, t)).astype(np.float64) for t in tags], axis=0)))
        names.append("own:" + tags[0][:40])
excl = [e for e in opts.get("exclude", "").split(",") if e]
for p in sorted((ROOT / "ext" / "oof").glob("pub_*.npy")):
    if any(p.stem.startswith("pub_" + e) for e in excl):
        continue
    Z.append(lg(np.load(p).astype(np.float64))); names.append(p.stem)
Z = np.column_stack(Z).astype(np.float32)
print(Z.shape, flush=True)
RAW = int(opts.get("raw", 0))
if RAW:
    R = train[COLS].copy()
    for c in CATS:
        R[c] = R[c].astype("category").cat.codes
    Z = np.c_[Z, R.to_numpy(np.float32)]
META = list(StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y))
s_lr, s_gb = np.zeros(len(y)), np.zeros(len(y))
params = dict(objective="binary", learning_rate=float(opts.get("lr", 0.03)), num_leaves=int(opts.get("leaves", 15)),
              min_child_samples=int(opts.get("mcs", 2000)), feature_fraction=float(opts.get("ff", 0.3)),
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=int(opts.get("threads", 2)),
              verbose=-1, seed=0)
ncols = Z.shape[1] - (len(COLS) if RAW else 0)
for k, (a, b) in enumerate(META):
    lr = LogisticRegression(C=0.001, max_iter=3000).fit(Z[a][:, :ncols], y[a])
    s_lr[b] = lr.decision_function(Z[b][:, :ncols])
    # inner split of a for early stopping
    rng = np.random.RandomState(k); m = rng.rand(len(a)) < 0.85
    dtr = lgb.Dataset(Z[a][m], y[a][m]); dva = lgb.Dataset(Z[a][~m], y[a][~m])
    g = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
    s_gb[b] = g.predict(Z[b], raw_score=True)
    print(f"meta fold {k}: lr {roc_auc_score(y[b], s_lr[b]):.6f} gbdt {roc_auc_score(y[b], s_gb[b]):.6f} "
          f"(it {g.best_iteration})", flush=True)
print(f"LR {roc_auc_score(y, s_lr):.6f}  GBDT {roc_auc_score(y, s_gb):.6f}")
from scipy.stats import rankdata
for w in (0.2, 0.35, 0.5):
    print(f"  rank blend w_gbdt={w}: {roc_auc_score(y, (1 - w) * rankdata(s_lr) + w * rankdata(s_gb)):.6f}")
