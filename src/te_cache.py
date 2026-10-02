"""In-fold target encodings, cached per key column and fold on disk. TargetEncoder encodes every
column independently, so a cached column is reused by any feature-group set and any model."""
import os

import numpy as np
from sklearn.preprocessing import TargetEncoder

from common import ROOT, SEED

CACHE = ROOT / "cache"
CHUNK = 40  # key columns encoded per TargetEncoder fit


def te_block(keys, i, ktr, ytr, va, n, m, use_orig, k_folds):
    """keys: DataFrame of integer keys over [train, test, orig?] rows. Returns
    (column names, train-fold block, validation block, test block) as float32."""
    cols = list(keys.columns)
    if os.environ.get("TE_CACHE", "1") == "0":  # no disk cache (e.g. on a Kaggle kernel)
        out = [[], [], []]
        for s in range(0, len(cols), CHUNK):
            k = keys[cols[s:s + CHUNK]].to_numpy()
            enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
            out[0].append(enc.fit_transform(k[ktr], ytr).astype(np.float32))
            out[1].append(enc.transform(k[va]).astype(np.float32))
            out[2].append(enc.transform(k[n:n + m]).astype(np.float32))
        return (cols, *(np.concatenate(o, axis=1) for o in out))
    CACHE.mkdir(exist_ok=True)
    tag = f"o{int(use_orig)}_k{k_folds}_f{i}"
    path = lambda c: CACHE / f"{c}__{tag}.npy"
    missing = [c for c in cols if not path(c).exists()]
    for s in range(0, len(missing), CHUNK):  # chunks keep TargetEncoder's memory bounded
        chunk = missing[s:s + CHUNK]
        enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
        k = keys[chunk].to_numpy()
        parts = (enc.fit_transform(k[ktr], ytr), enc.transform(k[va]), enc.transform(k[n:n + m]))
        for j, c in enumerate(chunk):
            tmp = CACHE / f"{c}__{tag}.{np.random.randint(1 << 30)}.tmp.npy"
            np.save(tmp, np.concatenate([p[:, j] for p in parts]).astype(np.float32))
            tmp.rename(path(c))
    M = np.column_stack([np.load(path(c)) for c in cols])
    a, b = len(ktr), len(ktr) + len(va)
    return cols, M[:a], M[a:b], M[b:]
