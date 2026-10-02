"""MLP with an embedding per column (every column treated as categorical) plus standardized numeric
copies. Same 5 folds; the epoch with the best validation AUC is kept (like early stopping).

Usage: python src/nn.py [epochs=8] [bs=2048] [lr=2e-3] [emb=12] [hidden=512,256,128]
                        [drop=0.2] [orig=0] [seed=42] [te=te1,tefd] [name=<tag>] [note=...]
  te  in-fold target encodings (same cache as train.py) added as standardized logit inputs
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from common import COLS, DELAYS, RATINGS, ROOT, folds, load, save
from features import te_keys
from te_cache import te_block

opts = dict(a.split("=", 1) for a in sys.argv[1:])
EPOCHS, BS = int(opts.get("epochs", 8)), int(opts.get("bs", 2048))
LR, EMB = float(opts.get("lr", 2e-3)), int(opts.get("emb", 12))
HID = [int(h) for h in opts.get("hidden", "512,256,128").split(",")]
DROP, USE_ORIG = float(opts.get("drop", 0.2)), int(opts.get("orig", 0))
SEED, NOTE = int(opts.get("seed", 42)), opts.get("note", "")
TE = [t for t in opts.get("te", "").split(",") if t]
K = int(opts.get("k", 5))                       # number of folds
DEV = torch.device(opts.get("device", "cpu"))   # cuda on a Kaggle GPU
torch.set_num_threads(int(opts.get("threads", 4)))

if USE_ORIG:
    train, test, y, orig, yo = load(orig=True)
    parts = [train[COLS], test[COLS], orig[COLS]]
else:
    train, test, y = load()
    parts = [train[COLS], test[COLS]]
if int(opts.get("debug", 0)):  # smoke test on a small slice
    train, test, y = train.iloc[:20000], test.iloc[:5000], y[:20000]
    parts = [train[COLS], test[COLS]] + parts[2:]
n, m = len(train), len(test)
A = pd.concat(parts, ignore_index=True)
A["Arrival Delay in Minutes"] = A["Arrival Delay in Minutes"].fillna(-1)
cat = np.zeros((len(A), len(COLS)), dtype=np.int64)
cards = []
for j, c in enumerate(COLS):
    v = A[c].astype(str)
    vc = v.iloc[:n + m].value_counts()
    v = v.where(v.map(vc).fillna(0) >= 10, "__rare__")
    cat[:, j] = pd.factorize(v)[0]
    cards.append(int(cat[:, j].max()) + 1)
num = pd.DataFrame({"age": A["Age"], "fd": np.log1p(A["Flight Distance"])})
for c in DELAYS:
    num[c] = np.log1p(A[c].clip(lower=0))
for c in RATINGS:
    num[c] = A[c]
num["arr_nan"] = (A["Arrival Delay in Minutes"] < 0).astype(float)
num = ((num - num.iloc[:n].mean()) / num.iloc[:n].std()).to_numpy(np.float32)
cat_t, num_t = torch.from_numpy(cat), torch.from_numpy(num)
KEYS = te_keys(TE, train, test, orig if USE_ORIG else None) if TE else None
N_TE = KEYS.shape[1] if KEYS is not None else 0


def fold_num(i, tr, va):
    """Numeric inputs for fold i: base columns plus the fold's target encodings (as logits)."""
    if KEYS is None:
        return num_t
    ktr = np.r_[tr, np.arange(n + m, len(A))] if USE_ORIG else tr
    yk = np.r_[y[tr], yo] if USE_ORIG else y[tr]
    _, etr, eva, ete = te_block(KEYS, i, ktr, yk, va, n, m, USE_ORIG, K)
    E = np.zeros((len(A), N_TE), dtype=np.float32)
    E[ktr], E[va], E[n:n + m] = etr, eva, ete
    E = np.log(np.clip(E, 1e-4, 1 - 1e-4) / (1 - np.clip(E, 1e-4, 1 - 1e-4)))
    mu, sd = E[ktr].mean(0), E[ktr].std(0) + 1e-6
    return torch.from_numpy(np.concatenate([num, (E - mu) / sd], axis=1).astype(np.float32))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(k, min(EMB, k + 1)) for k in cards])
        d = sum(e.embedding_dim for e in self.embs) + num.shape[1] + N_TE
        layers = []
        for h in HID:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(DROP)]
            d = h
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, xc, xn):
        e = [emb(xc[:, j]) for j, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [xn], dim=1)).squeeze(1)


def predict(model, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 16384):
            b = idx[s:s + 16384]
            out.append(model(cat_d[b], num_d[b]).cpu().numpy())
    return np.concatenate(out)


tag = opts.get("name", f"nn_e{EPOCHS}_emb{EMB}_h{'-'.join(map(str, HID))}_d{DROP}"
                       f"{'_' + '+'.join(TE) if TE else ''}"
                       f"{'_orig' if USE_ORIG else ''}{f'_s{SEED}' if SEED != 42 else ''}"
                       f"{f'_k{K}' if K != 5 else ''}")
print(tag, flush=True)
t0 = time.time()
oof, pred, aucs = np.zeros(n), np.zeros(m), []
test_idx = torch.arange(n, n + m).to(DEV)
cat_d = cat_t.to(DEV)
for i, (tr, va) in enumerate(folds(y, K)):
    num_t = fold_num(i, tr, va)
    num_d = num_t.to(DEV)
    torch.manual_seed(SEED + i)
    rng = np.random.default_rng(SEED + i)
    tr_idx = np.r_[tr, np.arange(n + m, len(A))] if USE_ORIG else tr
    ytr = torch.from_numpy(np.r_[y[tr], yo] if USE_ORIG else y[tr]).float().to(DEV)
    model = Net().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * int(np.ceil(len(tr_idx) / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()
    best, best_va, best_te = -1, None, None
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(tr_idx))
        for s in range(0, len(perm), BS):
            p = perm[s:s + BS]
            b = torch.from_numpy(tr_idx[p]).to(DEV)
            opt.zero_grad()
            loss = lossf(model(cat_d[b], num_d[b]), ytr[torch.from_numpy(p).to(DEV)])
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(model, torch.from_numpy(va).to(DEV))
        auc = roc_auc_score(y[va], pv)
        if auc > best:
            best, best_va, best_te = auc, pv, predict(model, test_idx)
        print(f"  fold {i} epoch {ep}: {auc:.5f} ({time.time() - t0:.0f}s)", flush=True)
    oof[va] = best_va
    pred += 1 / (1 + np.exp(-best_te)) / K
    aucs.append(best)
    print(f"  fold {i}: {best:.5f}", flush=True)
oof = 1 / (1 + np.exp(-oof))
cv = roc_auc_score(y, oof)
print(f"{tag} OOF AUC {cv:.5f} | fold mean {np.mean(aucs):.5f} ± {np.std(aucs):.5f} | "
      f"{time.time() - t0:.0f}s", flush=True)
save(tag, oof, pred)
with open(ROOT / "experiments.md", "a") as f:
    f.write(f"| {tag} | {NOTE} | {np.mean(aucs):.5f} ± {np.std(aucs):.5f} | {cv:.5f} | - | "
            f"{time.time() - t0:.0f}s | |\n")
