"""Assemble a TabPFN variant from per-unit outputs fetched from one or more Kaggle kernels.

A long run is split over kernels (e.g. folds 0-7 and 8-14); each kernel only writes its units
(kaggle/build/<slug>/output/units/<variant>/f<k>_s<seed>_{val,test}.npy). This collects the units of every
given slug, checks that each fold of StratifiedKFold(K, shuffle=True, random_state=42) has a validation
prediction and a test prediction, averages seeds per fold (validation) and all units (test), and writes
oof/<tag>.npy and preds/<tag>.npy.

Usage: python kaggle/tabpfn/aggregate.py <variant> <K> <tag> <slug> [<slug> ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import folds, load, save  # noqa: E402

vn, K, tag, slugs = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
train, test, y = load()
vals, tests, metas = defaultdict(dict), {}, []
for slug in slugs:
    d = ROOT / "kaggle" / "build" / slug / "output" / "units"
    d = d / vn if (d / vn).is_dir() else d                 # the first te4op kernel had no variant dirs
    for p in sorted(d.glob("f*_s*_val.npy")):
        unit = p.name[:-len("_val.npy")]                     # f<k>_s<seed>
        k = int(unit.split("_")[0][1:])
        vals[k][unit] = np.load(p)
        t = d / f"{unit}_test.npy"
        if t.exists():
            tests[unit] = np.load(t)
        m = d / f"{unit}_meta.json"
        if m.exists():
            metas.append(json.loads(m.read_text()))
missing = [k for k in range(K) if k not in vals]
no_test = sorted({u for k in vals for u in vals[k]} - set(tests))
assert not missing, f"folds without a validation unit: {missing}"
assert not no_test, f"units without a test prediction: {no_test}"
oof = np.zeros(len(y))
for k, (tr, va) in enumerate(folds(y, K)):
    v = np.mean(list(vals[k].values()), axis=0)
    assert v.shape == (len(va),), (k, v.shape, len(va))
    oof[va] = v
    print(f"fold {k:2d}: {len(vals[k])} unit(s), AUC {roc_auc_score(y[va], v):.6f}")
pred = np.mean(list(tests.values()), axis=0)
assert pred.shape == (len(test),)
auc = roc_auc_score(y, oof)
if metas:
    t = np.array([m["total"] for m in metas]) / 60
    pk = max(m.get("peak_gb", 0) for m in metas)
    print(f"{len(metas)} units: {t.mean():.0f} min mean ({t.min():.0f}-{t.max():.0f}), peak {pk:.2f} GB")
print(f"{tag}: OOF AUC {auc:.6f} | test from {len(tests)} units")
save(tag, oof, pred)
