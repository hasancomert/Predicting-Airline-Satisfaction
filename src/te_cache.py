"""In-fold target encodings, cached per key column and fold on disk. TargetEncoder encodes every
column independently, so a cached column is reused by any feature-group set and any model."""
import os

import numpy as np
from sklearn.preprocessing import TargetEncoder

from common import ROOT, SEED

CACHE = ROOT / "cache"


def te_block(keys, i, ktr, ytr, va, n, m, use_orig, k_folds):
    """keys: DataFrame of integer keys over [train, test, orig?] rows. Returns
    (column names, train-fold block, validation block, test block) as float32."""
    cols = list(keys.columns)
    if os.environ.get("TE_CACHE", "1") == "0":  # no disk cache (e.g. on a Kaggle kernel)
        enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
        k = keys.to_numpy()
        return (cols, enc.fit_transform(k[ktr], ytr).astype(np.float32),
                enc.transform(k[va]).astype(np.float32), enc.transform(k[n:n + m]).astype(np.float32))
    CACHE.mkdir(exist_ok=True)
    tag = f"o{int(use_orig)}_k{k_folds}_f{i}"
    path = lambda c: CACHE / f"{c}__{tag}.npy"
    missing = [c for c in cols if not path(c).exists()]
    if missing:
        enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
        k = keys[missing].to_numpy()
        parts = (enc.fit_transform(k[ktr], ytr), enc.transform(k[va]), enc.transform(k[n:n + m]))
        for j, c in enumerate(missing):
            tmp = CACHE / f"{c}__{tag}.{np.random.randint(1 << 30)}.tmp.npy"
            np.save(tmp, np.concatenate([p[:, j] for p in parts]).astype(np.float32))
            tmp.rename(path(c))
    M = np.column_stack([np.load(path(c)) for c in cols])
    a, b = len(ktr), len(ktr) + len(va)
    return cols, M[:a], M[a:b], M[b:]
