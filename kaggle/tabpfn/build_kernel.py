"""Build + push a private Kaggle kernel that runs TabPFN-3.5 (full training-fold context) on our own
feature view, on a 2xT4 machine, and writes oof/<tag>.npy + preds/<tag>.npy for the stack.

Feature view "te4op" (22 columns, same count as the raw data so the T4 memory budget of the public
full-context recipe still holds): Age, Flight Distance, 12 ratings (Food and drink dropped),
Customer Type / Type of Travel / Class (categorical), in-fold target encodings of Flight Distance and
of Flight Distance x Class / Type of Travel / Customer Type, and the logit of a LightGBM trained on
the original data only (opred). Gender and the two delays are dropped (lowest LightGBM gain).

Folds: StratifiedKFold(5, shuffle=True, random_state=42) (= the public libraries' split).
Units (fold, seed) run one per GPU; seed 0 for every fold first, then seed 1 while time allows.

Usage: python kaggle/tabpfn/build_kernel.py push <slug> [tag] [limit_hours] [n_seeds]
       python kaggle/gpu_kernel.py fetch <slug>     (same fetch as the other kernels)
The worker and the memory-lean KV-cache patch are adapted from hemingweb's public notebook
"S6E10 | EXP01 TabPFN-3.5 full context".
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / "kaggle" / "build"
USER = "hasancmert"

RUN = r'''
import base64, glob, json, os, subprocess, sys, threading, time, collections
T0 = time.time()
TAG = __TAG__
LIMIT_H = __LIMIT__
SRC = __SRC__
EXTRA = __EXTRA__
proj, work, out = "/tmp/proj", "/tmp/pfn", "/kaggle/working"
for d in (proj + "/src", proj + "/data/orig", work, out + "/units", out + "/oof", out + "/preds"):
    os.makedirs(d, exist_ok=True)
for name, b64 in SRC.items():
    open(f"{proj}/src/{name}", "wb").write(base64.b64decode(b64))
for name, b64 in EXTRA.items():
    open(f"{work}/{name}", "wb").write(base64.b64decode(b64))
def find(name, hint):
    hits = sorted(p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p)
    assert hits, (name, hint)
    return hits[0]
for f in ("train.csv", "test.csv", "sample_submission.csv"):
    os.symlink(find(f, "playground-series-s6e10"), f"{proj}/data/{f}")
os.symlink(find("data.csv", "aviation"), f"{proj}/data/orig/data.csv")
def say(*a): print(time.strftime("%H:%M:%S"), f"[+{(time.time()-T0)/3600:.2f}h]", *a, flush=True)
say(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g",
                   shell=True, capture_output=True, text=True).stdout)
subprocess.run("pip install -q tabpfn==9.0.0", shell=True, check=True)
from huggingface_hub import hf_hub_download
CKPT = hf_hub_download("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-20260909.safetensors", local_dir="/tmp/ckpt")
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
GPUS = [str(i) for i in range(max(1, torch.cuda.device_count()))]
say("GPUs", GPUS, "ckpt", CKPT)

# ---------------- prep: our feature view, in-fold TE per fold ----------------
os.environ.update(TE_CACHE="0", DATA_DIR=proj + "/data")
sys.path.insert(0, proj + "/src")
from common import load, folds, NUMS, CATS
from features import build, te_keys, clean
from te_cache import te_block
train, test, y, orig, yo = load(orig=True)
n, m = len(train), len(test)
F = build(["base", "opred"], train, test, orig).iloc[:n + m].reset_index(drop=True)
drop = [clean(c) for c in ("Gender", "Departure Delay in Minutes", "Arrival Delay in Minutes", "Food and drink")]
keep = [c for c in F.columns if c not in drop]
B = F[keep].copy()
cat_cols = [clean(c) for c in CATS if clean(c) in keep]
for c in cat_cols:
    B[c] = B[c].cat.codes.astype(np.float32)
op = np.clip(B["opred"].to_numpy(np.float64), 1e-6, 1 - 1e-6)
B["opred"] = np.log(op / (1 - op))
B = B.astype(np.float32)
KEYS = te_keys(["te1", "tefd"], train, test)
KEYS = KEYS[["te_Flight_Distance", "te_fd__Class", "te_fd__Type_of_Travel", "te_fd__Customer_Type"]]
FOLDS = folds(y, 5)
for k, (tr, va) in enumerate(FOLDS):
    _, etr, eva, ete = te_block(KEYS, k, tr, y[tr], va, n, m, 0, 5)
    Xb = B.to_numpy()
    cols = list(B.columns) + list(KEYS.columns)
    np.savez(f"{work}/data_f{k}.npz",
             Xctx=np.c_[Xb[tr], etr].astype(np.float32), yctx=y[tr].astype(np.int64),
             Xva=np.c_[Xb[va], eva].astype(np.float32), yva=y[va].astype(np.int64),
             XT=np.c_[Xb[n:n + m], ete].astype(np.float32),
             cat_idx=np.array([cols.index(c) for c in cat_cols]))
say("prep done:", len(cols), "features:", cols)

# ---------------- scheduler: (seed, fold) units, one GPU each ----------------
HARD = T0 + LIMIT_H * 3600
queue = collections.deque((s, f) for s in range(__NSEEDS__) for f in range(5))
lock = threading.Lock()
durations = []
def gpu_loop(g):
    while True:
        with lock:
            if not queue:
                return
            est = float(np.median(durations)) if durations else 1.9 * 3600
            if time.time() + 1.05 * est > HARD:
                say(f"gpu{g}: not enough time for another unit (est {est/60:.0f} min) -> stop")
                return
            s, f = queue.popleft()
        say(f"gpu{g}: START fold {f} seed {s}")
        t = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   PYTHONPATH=work)
        r = subprocess.run([sys.executable, f"{work}/worker.py", "--fold", str(f), "--seed", str(s),
                            "--work", work, "--out", out + "/units", "--ckpt", CKPT,
                            "--deadline", str(HARD - 600)], env=env)
        with lock:
            if r.returncode == 0:
                durations.append(time.time() - t)
        say(f"gpu{g}: END fold {f} seed {s} rc={r.returncode} ({(time.time()-t)/60:.0f} min)")
th = [threading.Thread(target=gpu_loop, args=(g,)) for g in GPUS]
for t in th: t.start()
for t in th: t.join()

# ---------------- aggregate ----------------
oof, ok = np.zeros(n), True
tests = []
for k, (tr, va) in enumerate(FOLDS):
    vals = [np.load(p) for p in sorted(glob.glob(f"{out}/units/f{k}_s*_val.npy"))]
    if not vals:
        ok = False; say(f"fold {k}: no unit finished"); continue
    oof[va] = np.mean(vals, axis=0)
    say(f"fold {k}: {len(vals)} seed(s), AUC {roc_auc_score(y[va], oof[va]):.5f}")
    tests += [np.load(p) for p in sorted(glob.glob(f"{out}/units/f{k}_s*_test.npy"))]
if ok and tests:
    np.save(f"{out}/oof/{TAG}.npy", oof.astype(np.float32))
    np.save(f"{out}/preds/{TAG}.npy", np.mean(tests, axis=0).astype(np.float32))
    say(f"{TAG} OOF AUC {roc_auc_score(y, oof):.5f} | test from {len(tests)} units")
'''


def push(slug, tag, limit_h, n_seeds):
    d = BUILD / slug
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    enc = lambda p: base64.b64encode(p.read_bytes()).decode()
    src = {p.name: enc(p) for p in (ROOT / "src").glob("*.py")}
    extra = {p.name: enc(p) for p in (HERE / "worker.py", HERE / "lean_patch.py")}
    code = (RUN.replace("__TAG__", repr(tag)).replace("__LIMIT__", repr(float(limit_h)))
            .replace("__SRC__", repr(src)).replace("__EXTRA__", repr(extra))
            .replace("__NSEEDS__", str(int(n_seeds))))
    (d / "run.py").write_text(code)
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "competition_sources": ["playground-series-s6e10"],
            "dataset_sources": ["arseniyshutko/binary-aviation-satisfaction-129k"], "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=True)


if __name__ == "__main__":
    if sys.argv[1] == "push":
        push(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "tabpfn35_te4op_k5",
             sys.argv[4] if len(sys.argv) > 4 else 9.0, sys.argv[5] if len(sys.argv) > 5 else 1)
