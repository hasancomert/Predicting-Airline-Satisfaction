"""NVIDIA Kumo-Tabular (in-context tabular foundation model) on our folds, run on a Kaggle GPU.

Context = the training rows of the fold (optionally a class-stratified subsample), raw 22 columns, target as a
categorical string; query = validation fold + test. Held-out labels are never passed to the model. Recipe after
Chris Deotte's public "Kumo-Tabular Starter" (small model, 100k context, 2 estimators); here the model size,
the context size and the number of estimators are options.

Usage (inside a Kaggle kernel, see kaggle/gpu_kernel.py):
  python src/kumo.py [size=large] [ctx=100000|0 (0 = full fold)] [est=2] [k=5] [only=0,1,..] [qchunk=150000]
                     [name=<tag>] [note=...]
  python src/kumo.py probe=1 [budget_h=1.8]   # fold 0 of 5, several configs, stops before the time budget
Per-fold checkpoints in cache/partial; a full run writes oof/<tag>.npy and preds/<tag>.npy.
"""
import gc
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import COLS, ROOT, TARGET, folds, load, save

opts = dict(a.split("=", 1) for a in sys.argv[1:])
try:
    import sdm  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/NVIDIA/structured-data-models.git"], check=True)
import sdm  # noqa: E402
import torch  # noqa: E402

T0 = time.time()
device = torch.device("cuda:0")
train, test, y = load()
n, m = len(train), len(test)
stypes = sdm.infer_stypes(train[COLS + [TARGET]].assign(**{TARGET: y.astype(str)}),
                          overrides={TARGET: "categorical"})
MODELS = {}


def model_of(size):
    if size not in MODELS:
        MODELS[size] = sdm.models.KumoTabular(task="classification", size=size, device=device)
    return MODELS[size]


def predict_fold(tr, va, size, ctx, est, qchunk, seed):
    """Probabilities for the validation rows and the test rows of one fold."""
    torch.manual_seed(seed)
    c_idx = tr
    if ctx and len(tr) > ctx:
        c_idx, _ = train_test_split(tr, train_size=ctx, stratify=y[tr], random_state=seed)
    context = train.iloc[c_idx][COLS].copy()
    context[TARGET] = y[c_idx].astype(str)
    query = pd.concat([train.iloc[va][COLS], test[COLS]], ignore_index=True)
    out = np.empty(len(query))
    mdl = model_of(size)
    s = 0
    while s < len(query):
        q = query.iloc[s:s + qchunk].copy()
        q[TARGET] = "0"  # placeholder, dropped below
        df = pd.concat([context, q], ignore_index=True)
        try:
            table = sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=device)
            with torch.amp.autocast(device.type, dtype=torch.float16):
                probs = mdl(x_context=table[:len(c_idx)].drop_columns(TARGET),
                            y_context=table[:len(c_idx), TARGET],
                            x_query=table[len(c_idx):].drop_columns(TARGET), num_estimators=est)
        except torch.cuda.OutOfMemoryError:
            del df
            gc.collect()
            torch.cuda.empty_cache()
            if qchunk <= 10000:
                raise
            qchunk //= 2
            print(f"    OOM -> query chunk {qchunk:,}", flush=True)
            continue
        cols = probs.columns
        names = list(cols[sdm.Stype.numerical] if isinstance(cols, dict) else cols)
        out[s:s + len(q)] = probs.numerical[:, names.index("1")].float().cpu().numpy()
        s += len(q)
        del table, probs, df
        gc.collect()
        torch.cuda.empty_cache()
    return out[:len(va)], out[len(va):], len(c_idx), qchunk


if int(opts.get("probe", 0)):
    budget = float(opts.get("budget_h", 1.8)) * 3600
    tr, va = folds(y, 5)[0]
    configs = [("large", 100_000, 2), ("large", 300_000, 2), ("large", 0, 2), ("large", 100_000, 8),
               ("small", 0, 2)]
    durations = []
    for size, ctx, est in configs:
        if durations and time.time() - T0 + 1.3 * max(durations) > budget:
            print(f"skip {size} ctx={ctx} est={est}: time budget", flush=True)
            continue
        t = time.time()
        try:
            pv, _, nc, qc = predict_fold(tr, va, size, ctx, est, int(opts.get("qchunk", 150000)), 42)
            durations.append(time.time() - t)
            print(f"PROBE size={size} ctx={nc:,} est={est}: fold-0 AUC {roc_auc_score(y[va], pv):.6f} "
                  f"({durations[-1] / 60:.1f} min, query chunk {qc:,}, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
        except Exception as e:  # OOM at the smallest chunk or an API problem: report and go on
            durations.append(time.time() - t)
            print(f"PROBE size={size} ctx={ctx} est={est}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch.cuda.reset_peak_memory_stats()
    sys.exit(0)

SIZE, CTX, EST = opts.get("size", "large"), int(opts.get("ctx", 100000)), int(opts.get("est", 2))
K = int(opts.get("k", 5))
ONLY = [int(f) for f in opts.get("only", ",".join(map(str, range(K)))).split(",")]
QCHUNK = int(opts.get("qchunk", 150000))
tag = opts.get("name", f"kumo_{SIZE}_ctx{CTX // 1000 if CTX else 'full'}_e{EST}_k{K}")
PART = ROOT / "cache" / "partial"
PART.mkdir(parents=True, exist_ok=True)
oof, pred, aucs = np.zeros(n), np.zeros(m), []
for i, (tr, va) in enumerate(folds(y, K)):
    if i not in ONLY:
        continue
    ck = PART / f"{tag}__f{i}.npz"
    if ck.exists():
        d = np.load(ck)
        oof[va], p = d["oof"], d["pred"]
    else:
        oof[va], p, nc, QCHUNK = predict_fold(tr, va, SIZE, CTX, EST, QCHUNK, 42 + i)
        np.savez(ck, oof=oof[va], pred=p)
    pred += p / len(ONLY)
    aucs.append(roc_auc_score(y[va], oof[va]))
    print(f"  fold {i}: {aucs[-1]:.5f} ({time.time() - T0:.0f}s)", flush=True)
if len(ONLY) < K:
    print(f"{tag}: folds {ONLY} mean {np.mean(aucs):.5f} (partial run, nothing saved)")
    sys.exit(0)
cv = roc_auc_score(y, oof)
print(f"{tag} OOF AUC {cv:.5f} | fold mean {np.mean(aucs):.5f} ± {np.std(aucs):.5f} | {time.time() - T0:.0f}s")
save(tag, oof, pred)
with open(ROOT / "experiments.md", "a") as f:
    f.write(f"| {tag} | {opts.get('note', '')} | {np.mean(aucs):.5f} ± {np.std(aucs):.5f} | {cv:.5f} | - | "
            f"{time.time() - T0:.0f}s | |\n")
