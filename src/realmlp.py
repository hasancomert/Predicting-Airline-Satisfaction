"""RealMLP (pytabkit) on the fixed 5 folds. Every numeric column also gets a categorical "twin" so
the net learns one embedding per distinct value (per route for Flight Distance). Hyper-parameters
follow the public PS-S6E10 RealMLP notebook (yekenot); feature views are ours.

Usage: python src/realmlp.py [feats=pub|v4] [epochs=3] [n_ens=8] [seed=42] [te=te1,tefd]
                             [k=5] [only=0,1,..] [device=cpu] [threads=4] [name=<tag>] [note=...]
  k     number of folds (StratifiedKFold(k, shuffle, seed 42)); only: run a subset of folds
  pub  raw columns + categorical twins
  v4   pub + label-free route profile (fdprof) + value counts (cnt) + original-model logit (opred)
       + in-fold target encodings (te=...; same cache as train.py, original rows not used)
Per-fold checkpoints in cache/partial, so a restarted run resumes.
"""
import sys
import time
import warnings
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from common import CATS, COLS, NUMS, ROOT, folds, load, save
from features import build, te_keys
from te_cache import te_block

warnings.filterwarnings("ignore")
for name in ("lightning", "lightning.pytorch", "pytorch_lightning"):
    logging.getLogger(name).setLevel(logging.ERROR)
opts = dict(a.split("=", 1) for a in sys.argv[1:])
FEATS = opts.get("feats", "pub")
EPOCHS, N_ENS, SEED = int(opts.get("epochs", 3)), int(opts.get("n_ens", 8)), int(opts.get("seed", 42))
TE = [t for t in opts.get("te", "te1" if FEATS == "v4" else "").split(",") if t]
DEVICE, NOTE = opts.get("device", "cpu"), opts.get("note", "")
K = int(opts.get("k", 5))
ONLY = [int(f) for f in opts.get("only", ",".join(map(str, range(K)))).split(",")]
torch.set_num_threads(int(opts.get("threads", 4)))

REALMLP = dict(
    n_ens=N_ENS, n_epochs=EPOCHS, batch_size=256, use_early_stopping=False,
    lr=0.053, wd=0.0236, sq_mom=0.988, lr_sched="lin_cos_log_15", wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18,
    hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="expm4t",
    plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
    ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False,
    bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
)
from pytabkit import RealMLP_TD_Classifier  # noqa: E402  (slow import)

need_orig = FEATS == "v4"
if need_orig:
    train, test, y, orig, yo = load(orig=True)
else:
    train, test, y = load()
    orig = None
n, m = len(train), len(test)
full = pd.concat([train[COLS], test[COLS]], ignore_index=True)
X = full.copy()
X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(0.0)
cat_cols = list(CATS)
for c in NUMS:
    X[c + "_cat_"] = X[c].astype(int).astype(str)
    cat_cols.append(c + "_cat_")
for c in cat_cols:
    X[c] = X[c].astype(str).astype("category")
if FEATS == "v4":
    F = build(["base", "fdprof", "cnt", "opred"], train, test, orig).iloc[:n + m]
    extra = [c for c in F.columns if c.startswith(("fdp_", "opred")) or c.endswith("_cnt")]
    for c in extra:
        v = F[c].to_numpy(np.float64)
        if c == "opred":
            v = np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
        X[c] = v.astype(np.float32)
X.columns = [c.replace("/", "_") for c in X.columns]
cat_cols = [c.replace("/", "_") for c in cat_cols]
KEYS = te_keys(TE, train, test) if TE else None

tag = opts.get("name", f"realmlp_{FEATS}_e{EPOCHS}_ens{N_ENS}"
                       f"{'_' + '+'.join(TE) if TE and FEATS != 'v4' else ''}"
                       f"{f'_s{SEED}' if SEED != 42 else ''}{f'_k{K}' if K != 5 else ''}")
if (ROOT / "oof" / f"{tag}.npy").exists():
    print(f"{tag}: already done, skipping")
    sys.exit(0)
print(f"{tag}: {X.shape[1]} columns ({len(cat_cols)} categorical), TE keys "
      f"{0 if KEYS is None else KEYS.shape[1]}, device {DEVICE}", flush=True)
PART = ROOT / "cache" / "partial"
PART.mkdir(parents=True, exist_ok=True)
t0 = time.time()
oof, pred, aucs = np.zeros(n), np.zeros(m), []
for i, (tr, va) in enumerate(folds(y, K)):
    if i not in ONLY:
        continue
    ck = PART / f"{tag}__f{i}.npz"
    if ck.exists():
        d = np.load(ck)
        oof[va], p = d["oof"], d["pred"]
        print(f"  fold {i}: resumed from checkpoint", flush=True)
    else:
        Xa, Xb, Xt = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), \
            X.iloc[n:].reset_index(drop=True)
        if KEYS is not None:
            cols, etr, eva, ete = te_block(KEYS, i, tr, y[tr], va, n, m, 0, K)
            for j, c in enumerate(cols):
                Xa[c], Xb[c], Xt[c] = etr[:, j], eva[:, j], ete[:, j]
        mdl = RealMLP_TD_Classifier(**REALMLP, device=DEVICE, random_state=SEED + i,
                                    val_metric_name="1-auc_ovr", verbosity=0)
        mdl.fit(Xa, y[tr], X_val=Xb, y_val=y[va], cat_col_names=cat_cols)
        oof[va] = mdl.predict_proba(Xb)[:, 1]
        p = mdl.predict_proba(Xt)[:, 1]
        np.savez(ck, oof=oof[va], pred=p)
    pred += p / len(ONLY)
    aucs.append(roc_auc_score(y[va], oof[va]))
    print(f"  fold {i}: {aucs[-1]:.5f} ({time.time() - t0:.0f}s)", flush=True)
if len(ONLY) < K:
    print(f"{tag}: folds {ONLY} mean {np.mean(aucs):.5f} (partial run, nothing saved)")
    sys.exit(0)
cv = roc_auc_score(y, oof)
print(f"{tag} OOF AUC {cv:.5f} | fold mean {np.mean(aucs):.5f} ± {np.std(aucs):.5f} | "
      f"{time.time() - t0:.0f}s", flush=True)
save(tag, oof, pred)
for i in range(K):
    (PART / f"{tag}__f{i}.npz").unlink(missing_ok=True)
with open(ROOT / "experiments.md", "a") as f:
    f.write(f"| {tag} | {NOTE} | {np.mean(aucs):.5f} ± {np.std(aucs):.5f} | {cv:.5f} | - | "
            f"{time.time() - t0:.0f}s | |\n")
