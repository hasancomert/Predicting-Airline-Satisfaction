"""Build + push a private Kaggle kernel that runs TabPFN-3.5 (full training-fold context) on one or more
feature views, on a 2xT4 machine, and writes per-unit predictions (+ oof/preds when every fold is done).

Views (VARIANTS, each {"name", "view", "orig"}):
  te4op   22 columns: Age, Flight Distance, 12 ratings (Food and drink dropped), Customer Type / Type of
          Travel / Class (categorical), in-fold target encodings of Flight Distance and of Flight Distance x
          Class / Type of Travel / Customer Type, and the logit of a LightGBM trained on the original data only.
  catfd10 abdullahsafwan333's recipe (goodpjw2008's catfd10): the 22 raw columns in a fixed order + Flight
          Distance // 10; the categoricals, Flight Distance and its //10 declared categorical.
  orig=1  the original rows are added to every context (with an is_orig column, 0 for competition rows).
Folds: StratifiedKFold(K, shuffle=True, random_state=42) (K=5: the public libraries' split; K=15: the split of
abdullahsafwan333's 15-fold TabPFN members). Units (variant, seed, fold) run one per GPU.

Usage: python kaggle/tabpfn/build_kernel.py push <slug> [tag] [limit_hours] [n_seeds]          (te4op, 5 folds)
       python kaggle/tabpfn/build_kernel.py pushx <slug> <tag> <limit_hours> <n_seeds> <K> <only|all> <variants>
         variants: comma list of name:view:orig[:drop], e.g. "c10:catfd10:0,c10og:catfd10:1:Gender|Food and drink";
         only: e.g. "0" or "0,1,2". Memory on a 16 GB T4 is set by the context rows (lean_patch v1: 560 k fit,
         653 k OOM in the decoder keys, 783 k OOM in the ICL attention; v2 removes both peaks)
       python kaggle/gpu_kernel.py fetch <slug>     (oof/preds when complete; units stay in the build output dir)
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

# ---------------- prep: per-variant, per-fold context / validation / test arrays ----------------
os.environ.update(TE_CACHE="0", DATA_DIR=proj + "/data")
sys.path.insert(0, proj + "/src")
from common import load, folds, NUMS, CATS
from features import build, te_keys, clean
from te_cache import te_block
K, ONLY, VARIANTS = __K__, __ONLY__, __VARIANTS__
train, test, y, orig, yo = load(orig=True)
n, m, n_o = len(train), len(test), len(orig)
FOLDS = folds(y, K)
ONLY = list(range(K)) if ONLY is None else ONLY
for v in VARIANTS:
    vdir = f"{work}/{v['name']}"
    os.makedirs(vdir, exist_ok=True)
    if v["view"] == "te4op":
        assert not v["orig"], "te4op with original context is not implemented"
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
        Xb = B.to_numpy()
        cols = list(B.columns) + list(KEYS.columns)
        for k in ONLY:
            tr, va = FOLDS[k]
            _, etr, eva, ete = te_block(KEYS, k, tr, y[tr], va, n, m, 0, K)
            np.savez(f"{vdir}/data_f{k}.npz",
                     Xctx=np.c_[Xb[tr], etr].astype(np.float32), yctx=y[tr].astype(np.int64),
                     Xva=np.c_[Xb[va], eva].astype(np.float32), yva=y[va].astype(np.int64),
                     XT=np.c_[Xb[n:n + m], ete].astype(np.float32),
                     cat_idx=np.array([cols.index(c) for c in cat_cols]))
    elif v["view"] == "catfd10":
        NUM4 = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
        RAT = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
               "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
               "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
        CAT4 = ["Gender", "Customer Type", "Type of Travel", "Class"]
        FE = [c for c in NUM4 + RAT + CAT4 if c not in v.get("drop", [])]   # recipe order, minus dropped columns
        CAT4 = [c for c in CAT4 if c in FE]
        parts = [train[FE], test[FE]] + ([orig[FE]] if v["orig"] else [])
        full = pd.concat(parts, ignore_index=True)
        tt = pd.concat([train[FE], test[FE]], ignore_index=True)
        for c in CAT4:                               # codes from train+test, as in the recipe
            full[c] = full[c].map({u: i for i, u in enumerate(sorted(tt[c].dropna().unique()))})
        full = full.astype(np.float32)
        full["fd_div10"] = (full["Flight Distance"] // 10).astype(np.float32)
        if v["orig"]:
            full["is_orig"] = np.r_[np.zeros(n + m), np.ones(n_o)].astype(np.float32)
        cols = list(full.columns)
        cat_idx = np.array([FE.index(c) for c in CAT4] + [FE.index("Flight Distance"), cols.index("fd_div10")])
        X = full.to_numpy(np.float32)
        for k in ONLY:
            tr, va = FOLDS[k]
            ctx = np.r_[tr, np.arange(n + m, n + m + n_o)] if v["orig"] else tr
            yctx = np.r_[y[tr], yo] if v["orig"] else y[tr]
            np.savez(f"{vdir}/data_f{k}.npz", Xctx=X[ctx], yctx=yctx.astype(np.int64),
                     Xva=X[va], yva=y[va].astype(np.int64), XT=X[n:n + m], cat_idx=cat_idx)
    else:
        raise ValueError(v["view"])
    say(f"prep {v['name']} ({v['view']}, orig={v['orig']}): {len(cols)} features {cols}")

# ---------------- scheduler: (seed, fold) units, one GPU each ----------------
HARD = T0 + LIMIT_H * 3600
queue = collections.deque((v["name"], s, f) for s in range(__NSEEDS__) for f in ONLY for v in VARIANTS)
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
            vn, s, f = queue.popleft()
        say(f"gpu{g}: START {vn} fold {f} seed {s}")
        t = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   PYTHONPATH=work, LEAN_LOG="1")
        os.makedirs(f"{out}/units/{vn}", exist_ok=True)
        r = subprocess.run([sys.executable, f"{work}/worker.py", "--fold", str(f), "--seed", str(s),
                            "--work", f"{work}/{vn}", "--out", f"{out}/units/{vn}", "--ckpt", CKPT,
                            "--deadline", str(HARD - 600)], env=env)
        with lock:
            if r.returncode == 0:
                durations.append(time.time() - t)
        say(f"gpu{g}: END {vn} fold {f} seed {s} rc={r.returncode} ({(time.time()-t)/60:.0f} min)")
th = [threading.Thread(target=gpu_loop, args=(g,)) for g in GPUS]
for t in th: t.start()
for t in th: t.join()

# ---------------- aggregate (per variant, when every fold has at least one unit) ----------------
for v in VARIANTS:
    vn, ok, tests = v["name"], True, []
    oof = np.zeros(n)
    for k, (tr, va) in enumerate(FOLDS):
        vals = [np.load(p) for p in sorted(glob.glob(f"{out}/units/{vn}/f{k}_s*_val.npy"))]
        if not vals:
            ok = False
            continue
        oof[va] = np.mean(vals, axis=0)
        say(f"{vn} fold {k}: {len(vals)} seed(s), AUC {roc_auc_score(y[va], oof[va]):.6f}")
        tests += [np.load(p) for p in sorted(glob.glob(f"{out}/units/{vn}/f{k}_s*_test.npy"))]
    if ok and tests:
        np.save(f"{out}/oof/{TAG}_{vn}.npy", oof.astype(np.float32))
        np.save(f"{out}/preds/{TAG}_{vn}.npy", np.mean(tests, axis=0).astype(np.float32))
        say(f"{TAG}_{vn} OOF AUC {roc_auc_score(y, oof):.6f} | test from {len(tests)} units")
'''


def push(slug, tag, limit_h, n_seeds, k=5, only=None, variants=None):
    variants = variants or [{"name": "te4op", "view": "te4op", "orig": 0}]
    d = BUILD / slug
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    enc = lambda p: base64.b64encode(p.read_bytes()).decode()
    src = {p.name: enc(p) for p in (ROOT / "src").glob("*.py")}
    extra = {p.name: enc(p) for p in (HERE / "worker.py", HERE / "lean_patch.py")}
    code = (RUN.replace("__TAG__", repr(tag)).replace("__LIMIT__", repr(float(limit_h)))
            .replace("__SRC__", repr(src)).replace("__EXTRA__", repr(extra))
            .replace("__NSEEDS__", str(int(n_seeds))).replace("__K__", str(int(k)))
            .replace("__ONLY__", repr(only)).replace("__VARIANTS__", repr(variants)))
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
    elif sys.argv[1] == "pushx":
        slug, tag, limit_h, n_seeds, k, only, vs = sys.argv[2:9]
        only = None if only == "all" else [int(f) for f in only.split(",")]
        variants = []
        for x in vs.split(","):                      # name:view:orig[:col|col|...] (columns to drop)
            f = x.split(":")
            variants.append({"name": f[0], "view": f[1], "orig": int(f[2]),
                             "drop": f[3].split("|") if len(f) > 3 and f[3] else []})
        push(slug, tag, limit_h, n_seeds, int(k), only, variants)
