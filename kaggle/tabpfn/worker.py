# TabPFN-3.5 full-context worker: one (fold, seed) unit on one GPU.
# Adapted from hemingweb's public notebook "S6E10 | EXP01 TabPFN-3.5 full context" (per-fold data files).
# Context = ALL labelled rows outside the fold (no subsampling). Predicts the
# validation fold (OOF) and the full test set with the same KV cache.
import os, sys, time, json, argparse, gc
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--work", required=True)        # dir with data.npz
ap.add_argument("--out", required=True)         # dir for unit outputs
ap.add_argument("--ckpt", required=True)
ap.add_argument("--batch", type=int, default=32768)
ap.add_argument("--deadline", type=float, required=True)  # unix time: abort test pred after this
args = ap.parse_args()

import torch
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
import lean_patch; lean_patch.apply()   # chunked KV-cache build -> full context fits a T4

TAG = f"[f{args.fold} s{args.seed} gpu{os.environ.get('CUDA_VISIBLE_DEVICES','?')}]"
DEV = os.environ.get("TABPFN_DEVICE", "cuda")
CUDA = DEV.startswith("cuda")
def empty():
    if CUDA: torch.cuda.empty_cache()
def peak():
    return torch.cuda.max_memory_allocated() / 2**30 if CUDA else 0.0
def log(*a):
    print(TAG, time.strftime("%H:%M:%S"), *a, flush=True)

# per-fold arrays (fold-specific in-fold target encodings), written by the prep step
d = np.load(os.path.join(args.work, f"data_f{args.fold}.npz"))
Xtr, ytr, Xva, yva, XT = d["Xctx"], d["yctx"], d["Xva"], d["yva"], d["XT"]
cat_idx = list(d["cat_idx"])
log(f"context={len(Xtr):,} val={len(Xva):,} test={len(XT):,} feats={X.shape[1]}")

# Fallback chain if the full-context fit does not fit in T4 memory.
CONFIGS = [
    dict(memory_saving_mode="auto", keep_cache_on_device=True),
    dict(memory_saving_mode=True, keep_cache_on_device=True),
    dict(memory_saving_mode=True, keep_cache_on_device=False),
]

def is_oom(e):
    s = str(e).lower()
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in s or "cuda error: out of memory" in s

def make(cfg):
    return TabPFNClassifier(
        model_path=args.ckpt,
        n_estimators=1,                       # one estimator per unit; seeds = extra ensemble members
        random_state=args.seed,
        device=DEV,
        ignore_pretraining_limits=True,
        fit_mode="fit_with_cache",            # KV cache built once, reused for every batch
        kv_cache_precision="int8",            # needed on a 16 GB T4 at ~560k context
        inference_precision="autocast",
        categorical_features_indices=cat_idx,
        **cfg,
    )

clf, used = None, None
t0 = time.time()
for i, cfg in enumerate(CONFIGS):
    try:
        empty()
        if CUDA: torch.cuda.reset_peak_memory_stats()
        clf = make(cfg)
        clf.fit(Xtr, ytr)
        used = cfg
        break
    except Exception as e:
        if is_oom(e) and i + 1 < len(CONFIGS):
            log(f"OOM in fit with {cfg} -> next fallback"); clf = None; gc.collect(); empty()
            continue
        raise
t_fit = time.time() - t0
log(f"fit done in {t_fit/60:.1f} min, cfg={used}, peak={peak():.2f} GB")

def predict(Xq, name, deadline=None):
    out = np.empty(len(Xq), dtype=np.float32)
    bs, i, t = args.batch, 0, time.time()
    while i < len(Xq):
        if deadline and time.time() > deadline:
            log(f"{name}: deadline reached at {i:,}/{len(Xq):,} -> abort"); return None
        j = min(i + bs, len(Xq))
        try:
            out[i:j] = clf.predict_proba(Xq[i:j])[:, 1]
        except Exception as e:
            if is_oom(e) and bs > 1024:
                bs //= 2; empty(); log(f"{name}: OOM -> batch {bs}"); continue
            raise
        i = j
        el = time.time() - t
        log(f"{name}: {i:,}/{len(Xq):,} rows  {el/60:.1f} min  ETA {el/i*(len(Xq)-i)/60:.1f} min")
    return out

tag = f"f{args.fold}_s{args.seed}"
t1 = time.time()
pv = predict(Xva, "val")
t_val = time.time() - t1
auc = roc_auc_score(yva, pv)
np.save(os.path.join(args.out, f"{tag}_val.npy"), pv)
log(f"VAL AUC = {auc:.6f}  (val pred {t_val/60:.1f} min)")

t2 = time.time()
pt = predict(XT, "test", deadline=args.deadline)
t_test = time.time() - t2
if pt is not None:
    np.save(os.path.join(args.out, f"{tag}_test.npy"), pt)

meta = dict(fold=args.fold, seed=args.seed, auc=float(auc), n_ctx=int(len(Xtr)),
            t_fit=t_fit, t_val=t_val, t_test=t_test, total=time.time() - t0,
            test_done=pt is not None, cfg=str(used),
            peak_gb=peak())
json.dump(meta, open(os.path.join(args.out, f"{tag}_meta.json"), "w"), indent=1)
log("DONE", json.dumps(meta))
