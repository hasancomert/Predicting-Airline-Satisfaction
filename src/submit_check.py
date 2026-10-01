"""Write a submission in sample_submission order and check its format."""
import numpy as np
import pandas as pd

from common import DATA, ID, SUB_DIR, TARGET


def write_submission(name, pred):
    ss = pd.read_csv(DATA / "sample_submission.csv")
    pred = np.asarray(pred, dtype=np.float64)
    assert len(pred) == len(ss), (len(pred), len(ss))
    assert np.isfinite(pred).all(), "non-finite predictions"
    if pred.min() < 0 or pred.max() > 1:  # ranks / margins -> [0, 1], AUC unchanged
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    sub = pd.DataFrame({ID: ss[ID], TARGET: pred})
    SUB_DIR.mkdir(exist_ok=True)
    path = SUB_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    check(path)
    return path


def check(path):
    ss = pd.read_csv(DATA / "sample_submission.csv")
    sub = pd.read_csv(path)
    assert list(sub.columns) == list(ss.columns), sub.columns
    assert len(sub) == len(ss) and (sub[ID].values == ss[ID].values).all(), "id mismatch"
    assert sub[TARGET].between(0, 1).all() and sub[TARGET].notna().all(), "bad values"
    assert sub[TARGET].nunique() > 1000, "suspiciously few distinct values"
    print(f"{path.name}: format OK ({len(sub)} rows, mean {sub[TARGET].mean():.4f})")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    check(Path(sys.argv[1]))
