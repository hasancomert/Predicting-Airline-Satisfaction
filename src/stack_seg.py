"""Segment-aware variants of the logistic stack (src/stack.py), nested CV on the same meta folds.

The global stack has one weight per member and one intercept. Member quality and calibration may differ
between passenger segments, which a single weight vector cannot express. Variants (segments from the raw
columns, label-free):
  dummies   global stack + one-hot segment intercepts
  split     a separate stack per segment
  interact  member logits + member logits x segment indicator (segments: 2 levels only)

Usage: python src/stack_seg.py own=<file> pub=all [exclude=...] [C=0.001] seg=tt|tc|ttc|ttcc
                               modes=dummies,split,interact [out=<submission name> mode=<mode>]
  tt = Type of Travel, tc = Class, ttc = Type of Travel x Class, ttcc = Type of Travel x Class x Customer Type
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OOF_DIR, PRED_DIR, ROOT, load  # noqa: E402

opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
C = float(opts.get("C", 0.001))
train, test, y = load()


def path(d, x):
    return (ROOT / "ext" / d.name if x.startswith(("ext_", "pub_")) else d) / f"{x}.npy"


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


Z, ZT = [], []
for line in open(opts["own"]):
    if line.strip():
        tags = line.strip().split(",")
        Z.append(lg(np.mean([np.load(path(OOF_DIR, t)).astype(np.float64) for t in tags], axis=0)))
        ZT.append(lg(np.mean([np.load(path(PRED_DIR, t)).astype(np.float64) for t in tags], axis=0)))
excl = [e for e in opts.get("exclude", "").split(",") if e]
for t in sorted(p.stem for p in (ROOT / "ext" / "oof").glob("pub_*.npy")):
    if not any(t.startswith("pub_" + e) for e in excl):
        Z.append(lg(np.load(path(OOF_DIR, t)).astype(np.float64)))
        ZT.append(lg(np.load(path(PRED_DIR, t)).astype(np.float64)))
Z, ZT = np.column_stack(Z), np.column_stack(ZT)
META = list(StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y))
print(f"{Z.shape[1]} members", flush=True)

SEG = {"tt": ["Type of Travel"], "tc": ["Class"], "ttc": ["Type of Travel", "Class"],
       "ttcc": ["Type of Travel", "Class", "Customer Type"]}[opts.get("seg", "tt")]
key = train[SEG].astype(str).agg("|".join, axis=1)
keyT = test[SEG].astype(str).agg("|".join, axis=1)
levels = sorted(key.unique())
g, gT = key.map({v: i for i, v in enumerate(levels)}).to_numpy(), keyT.map({v: i for i, v in enumerate(levels)}).to_numpy()
print("segments:", {v: int((g == i).sum()) for i, v in enumerate(levels)}, flush=True)
D = np.eye(len(levels))[g][:, 1:]
DT = np.eye(len(levels))[gT][:, 1:]


def fit(A, t):
    m = LogisticRegression(C=C, max_iter=3000).fit(A, t)
    return m.coef_.ravel(), m.intercept_[0]


def design(mode, Zx, Dx, gx):
    if mode == "dummies":
        return np.c_[Zx, Dx]
    if mode == "interact":
        assert len(levels) == 2, "interact needs a 2-level segment"
        return np.c_[Zx, Zx * (gx == 1)[:, None], Dx]
    return Zx


def predict(mode, a_rows, Za, ya, ga, Da, Zb, gb, Db):
    """Fit on (Za, ya) and score Zb; split mode fits one stack per segment."""
    if mode == "split":
        s = np.zeros(len(Zb))
        for i in range(len(levels)):
            w, b0 = fit(Za[ga == i], ya[ga == i])
            s[gb == i] = Zb[gb == i] @ w + b0
        return s
    w, b0 = fit(design(mode, Za, Da, ga), ya)
    return design(mode, Zb, Db, gb) @ w + b0


res = {}
for mode in ["global"] + opts.get("modes", "dummies,split").split(","):
    s = np.zeros(len(y))
    for a, b in META:
        s[b] = predict(mode, a, Z[a], y[a], g[a], D[a], Z[b], g[b], D[b])
    res[mode] = roc_auc_score(y, s)
    print(f"{mode:9s} nested CV {res[mode]:.6f} ({res[mode] - res['global']:+.6f})", flush=True)

if opts.get("out"):
    from submit_check import write_submission
    mode = opts["mode"]
    pt = predict(mode, None, Z, y, g, D, ZT, gT, DT)
    write_submission(opts["out"], 1 / (1 + np.exp(-pt)))
