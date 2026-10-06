"""Logistic-regression stack over many members (own groups + public OOF libraries), nested CV.

Usage: python src/stack.py <out_name|-> own=<file with one own group per line> pub=<all|none|file>
                           [exclude=prefix1,prefix2] [C=1.0] [ablate=1] [tf=logit|probit] [nonneg=1]
                           [clip=<max |logit|>] [scale=1]
  own     each line is one member: tags joined by ',' are averaged (seeds); "ext_..." tags work too
  pub     all: every ext/oof/pub_*.npy (minus exclude prefixes); none; or a file of names
  ablate  leave-one-source-out deltas (source = own / prefix of the public name)
  clip    clip member logits to [-clip, clip] (some public members have logit std ~12)
  scale   standardise every member column (OOF mean / std, same transform on test) before the
          penalised fit, so the L2 penalty treats members alike
  nonneg  member weights constrained to >= 0 (same penalised objective as sklearn, L-BFGS-B); a stack
          that cannot lean on small differences between near-duplicate members
Stacker: LogisticRegression on clipped logits; nested score = each row predicted by a stacker fitted on
the other 4/5 of the rows (StratifiedKFold(5, shuffle, seed 7)). The submission uses a stacker fitted on
all rows. Writes submissions/<out_name>.csv.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OOF_DIR, PRED_DIR, ROOT, load  # noqa: E402
from submit_check import write_submission  # noqa: E402

args = [a for a in sys.argv[1:] if "=" not in a]
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
out = args[0] if args else "-"
C = float(opts.get("C", 1.0))
TF = opts.get("tf", "logit")  # logit | probit (probit of the normalized rank, as in S6E9)
NONNEG = int(opts.get("nonneg", 0))
CLIP = float(opts.get("clip", 0))
SCALE = int(opts.get("scale", 0))
_, _, y = load()


def path(d, x):
    return (ROOT / "ext" / d.name if x.startswith(("ext_", "pub_")) else d) / f"{x}.npy"


def lg(p):
    if TF == "probit":
        from scipy.special import ndtri
        from scipy.stats import rankdata
        return ndtri((rankdata(p) - 0.5) / len(p))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


names, src, Z, ZT = [], [], [], []


def add(name, tags, source):
    o = np.mean([np.load(path(OOF_DIR, t)).astype(np.float64) for t in tags], axis=0)
    p = np.mean([np.load(path(PRED_DIR, t)).astype(np.float64) for t in tags], axis=0)
    names.append(name)
    src.append(source)
    Z.append(lg(o))
    ZT.append(lg(p))


if opts.get("own"):
    for line in open(opts["own"]):
        if line.strip():
            tags = line.strip().split(",")
            add("own:" + tags[0][:60], tags, "own")
pub = opts.get("pub", "all")
excl = [e for e in opts.get("exclude", "").split(",") if e]
if pub == "all":
    pubs = sorted(p.stem for p in (ROOT / "ext" / "oof").glob("pub_*.npy"))
elif pub == "none":
    pubs = []
else:
    pubs = [l.strip() for l in open(pub) if l.strip()]
for t in pubs:
    if any(t.startswith("pub_" + e) for e in excl):
        continue
    add(t, [t], t[4:].split("_")[0])
Z, ZT = np.column_stack(Z), np.column_stack(ZT)
if CLIP:
    Z, ZT = np.clip(Z, -CLIP, CLIP), np.clip(ZT, -CLIP, CLIP)
if SCALE:
    mu, sd = Z.mean(0), Z.std(0)
    Z, ZT = (Z - mu) / sd, (ZT - mu) / sd
print(f"{len(names)} members ({src.count('own')} own, {len(names) - src.count('own')} public)", flush=True)
META = list(StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y))


def fit_lr(A, t):
    """(weights, intercept) of the L2 logistic regression C * sum(logloss) + |w|^2 / 2."""
    if not NONNEG:
        m = LogisticRegression(C=C, max_iter=3000).fit(A, t)
        return m.coef_.ravel(), m.intercept_[0]
    from scipy.optimize import minimize

    def f(x):
        z = A @ x[:-1] + x[-1]
        p = 1 / (1 + np.exp(-z))
        loss = C * (np.logaddexp(0, z) - t * z).sum() + 0.5 * x[:-1] @ x[:-1]
        return loss, np.r_[C * (A.T @ (p - t)) + x[:-1], C * (p - t).sum()]
    r = minimize(f, np.zeros(A.shape[1] + 1), jac=True, method="L-BFGS-B",
                 bounds=[(0, None)] * A.shape[1] + [(None, None)], options=dict(maxiter=5000))
    return r.x[:-1], r.x[-1]


def nested(cols):
    s = np.zeros(len(y))
    for a, b in META:
        w, b0 = fit_lr(Z[a][:, cols], y[a])
        s[b] = Z[b][:, cols] @ w + b0
    return roc_auc_score(y, s)


allc = list(range(len(names)))
full = nested(allc)
print(f"nested CV, all members: {full:.6f}", flush=True)
if int(opts.get("ablate", 0)):
    for s_ in sorted(set(src)):
        cols = [i for i in allc if src[i] != s_]
        print(f"  without {s_:15s} ({src.count(s_):3d}): {nested(cols) - full:+.6f}", flush=True)
coef, b0 = fit_lr(Z, y)
w = sorted(zip(coef, names), key=lambda t: -abs(t[0]))[:12]
print("largest |weights|:", [(n[:40], round(float(c), 3)) for c, n in w])
print(f"non-zero weights: {(np.abs(coef) > 1e-6).sum()} / {len(coef)}, negative: {(coef < -1e-6).sum()}")
if out != "-":
    write_submission(out, 1 / (1 + np.exp(-(ZT @ coef + b0))))
