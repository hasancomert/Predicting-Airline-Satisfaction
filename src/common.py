"""Shared constants, data loading and the fixed CV split used by every model."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OOF_DIR, PRED_DIR, SUB_DIR = ROOT / "oof", ROOT / "preds", ROOT / "submissions"

SEED, FOLDS = 42, 5
TARGET, ID = "satisfaction", "id"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Gate location", "Food and drink", "Online boarding", "Seat comfort",
           "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
           "Checkin service", "Cleanliness"]
DELAYS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
NUMS = ["Age", "Flight Distance"] + RATINGS + DELAYS
COLS = NUMS + CATS


def load(orig=False):
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).to_numpy()
    if not orig:
        return train, test, y
    o = pd.read_csv(DATA / "orig" / "data.csv")
    return train, test, y, o, o[TARGET].astype(int).to_numpy()


def folds(y, k=FOLDS, seed=SEED):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def save(tag, oof, pred):
    OOF_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    np.save(OOF_DIR / f"{tag}.npy", oof.astype(np.float32))
    np.save(PRED_DIR / f"{tag}.npy", pred.astype(np.float32))
