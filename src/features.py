"""Feature groups. Everything here is label-free and built on train+test(+original) together;
target-dependent encodings are done inside the folds in train.py.

Groups (comma separated on the command line):
  base   raw columns, the 4 string columns as pandas categoricals
  afill  missing Arrival Delay filled with Departure Delay
  delay  delay sum / diff / max / log / ratio, arrival-missing flag, any-delay flag
  rnan   rating 0 ("not applicable") -> NaN in the rating columns, plus a zero-count column
  zflag  one 0/1 flag per rating column that has a 0 level
  agg    mean / min / max / std / sum of the ratings (0 excluded), counts of 1s and 5s
  grp    group means: digital (wifi, online booking, online boarding), cabin (seat comfort,
         leg room, entertainment), service (on-board, check-in, baggage, cleanliness) and diffs
  inter  Customer Type x Type of Travel x Class (and pairs) as categoricals
  cnt    value counts of Flight Distance, Age and the delays over train+test
  rcat   copies of the rating columns as categoricals
  omean  original-data satisfaction rate per value of each column (smoothed)
  fdprof label-free profile of each Flight Distance value over train+test: share of Business
         class / business travel / loyal customers, mean age and mean of every rating
  ofd    the same profile computed on the original data, plus the original row count per value
  allcat categorical copies of every numeric column (meant for CatBoost's own CTR encodings)
"""
import numpy as np
import pandas as pd

from common import CATS, COLS, DELAYS, NUMS, RATINGS, TARGET

DIGITAL = ["Inflight wifi service", "Ease of Online booking", "Online boarding"]
CABIN = ["Seat comfort", "Leg room service", "Inflight entertainment"]
SERVICE = ["On-board service", "Checkin service", "Baggage handling", "Cleanliness"]
DEP, ARR = DELAYS


def clean(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def build(groups, train, test, orig=None):
    """Return a feature frame with rows [train, test, orig?] and the TE key frame."""
    parts = [train[COLS], test[COLS]] + ([orig[COLS]] if orig is not None else [])
    A = pd.concat(parts, ignore_index=True)
    n_tt = len(train) + len(test)
    F = pd.DataFrame(index=A.index)
    for c in NUMS:
        F[c] = A[c].astype(float)
    for c in CATS:
        F[c] = pd.Categorical(A[c])

    dep, arr = A[DEP].astype(float), A[ARR].astype(float)
    if "afill" in groups:
        arr = arr.fillna(dep)
        F[ARR] = arr
    if "delay" in groups:
        F["arr_missing"] = A[ARR].isna().astype(int)
        F["delay_sum"] = dep + arr
        F["delay_diff"] = arr - dep
        F["delay_max"] = np.fmax(dep, arr)
        F["log_dep"] = np.log1p(dep)
        F["log_arr"] = np.log1p(arr)
        F["delay_ratio"] = (arr + 1) / (dep + 1)
        F["any_delay"] = ((dep > 0) | (arr > 0)).astype(int)
    R = A[RATINGS].astype(float)
    Rn = R.replace(0, np.nan)
    if "rnan" in groups:
        for c in RATINGS:
            F[c] = Rn[c]
        F["n_zero"] = (R == 0).sum(axis=1)
    if "zflag" in groups:
        for c in RATINGS:
            if (R[c] == 0).any():
                F[c + "_zero"] = (R[c] == 0).astype(int)
    if "agg" in groups:
        F["r_mean"] = Rn.mean(axis=1)
        F["r_min"] = Rn.min(axis=1)
        F["r_max"] = Rn.max(axis=1)
        F["r_std"] = Rn.std(axis=1)
        F["r_sum"] = R.sum(axis=1)
        F["r_n5"] = (R == 5).sum(axis=1)
        F["r_n1"] = (R == 1).sum(axis=1)
    if "grp" in groups:
        g = {k: Rn[v].mean(axis=1) for k, v in
             (("g_digital", DIGITAL), ("g_cabin", CABIN), ("g_service", SERVICE))}
        for k, v in g.items():
            F[k] = v
        F["g_dig_minus_cab"] = g["g_digital"] - g["g_cabin"]
        F["g_dig_minus_srv"] = g["g_digital"] - g["g_service"]
        F["g_cab_minus_srv"] = g["g_cabin"] - g["g_service"]
    if "inter" in groups:
        ct, tt, cl = (A[c].astype(str) for c in ("Customer Type", "Type of Travel", "Class"))
        F["ct_tt_cl"] = pd.Categorical(ct + "|" + tt + "|" + cl)
        F["ct_tt"] = pd.Categorical(ct + "|" + tt)
        F["tt_cl"] = pd.Categorical(tt + "|" + cl)
        F["ct_cl"] = pd.Categorical(ct + "|" + cl)
    if "cnt" in groups:
        base = A.iloc[:n_tt]
        for c in ["Flight Distance", "Age", DEP, ARR]:
            F[c + "_cnt"] = A[c].map(base[c].value_counts()).fillna(0)
    if "allcat" in groups:
        for c in NUMS:
            F[c + "_c"] = pd.Categorical(A[c].fillna(-1).astype(int))
    if "rcat" in groups:
        for c in RATINGS:
            F[c + "_cat"] = pd.Categorical(A[c])
    if "omean" in groups:
        if orig is None:
            raise ValueError("omean needs the original data")
        yo = orig[TARGET].astype(float)
        prior, k = yo.mean(), 20
        for c in COLS:
            s = pd.DataFrame({"v": orig[c].astype(str), "y": yo}).groupby("v").y.agg(["sum", "count"])
            rate = (s["sum"] + prior * k) / (s["count"] + k)
            F[c + "_omean"] = A[c].astype(str).map(rate).fillna(prior).astype(float)

    if "fdprof" in groups or "ofd" in groups:
        def profile(D):
            P = pd.DataFrame({"biz": (D["Class"] == "Business").astype(float),
                              "btrav": (D["Type of Travel"] == "Business travel").astype(float),
                              "loyal": (D["Customer Type"] == "Loyal Customer").astype(float),
                              "age": D["Age"].astype(float)})
            for c in RATINGS:
                P[clean(c)] = D[c].astype(float)
            P["fd"] = D["Flight Distance"].values
            return P.groupby("fd").mean()
        fd = A["Flight Distance"]
        if "fdprof" in groups:
            prof = profile(A.iloc[:n_tt])
            for c in prof.columns:
                F[f"fdp_{c}"] = fd.map(prof[c]).astype(float)
        if "ofd" in groups:
            if orig is None:
                raise ValueError("ofd needs the original data")
            prof = profile(orig)
            for c in prof.columns:
                F[f"ofd_{c}"] = fd.map(prof[c]).astype(float)
            F["ofd_cnt"] = fd.map(orig["Flight Distance"].value_counts()).fillna(0)

    F.columns = [clean(c) for c in F.columns]
    return F


def te_keys(spec, train, test, orig=None):
    """Integer keys for in-fold target encoding.
    spec: 'te1' every column alone; 'te2' all pairs of the low-cardinality columns plus
    Flight Distance alone; 'tecat' the categorical interactions."""
    from itertools import combinations
    parts = [train[COLS], test[COLS]] + ([orig[COLS]] if orig is not None else [])
    A = pd.concat(parts, ignore_index=True)
    codes = {c: pd.factorize(A[c].astype(str))[0].astype(np.int64) for c in COLS}
    keys = {}
    if "te1" in spec:
        keys.update({f"te_{clean(c)}": v for c, v in codes.items()})
    if "te2" in spec:
        low = [c for c in COLS if c != "Flight Distance"]
        for a, b in combinations(low, 2):
            keys[f"te_{clean(a)}__{clean(b)}"] = codes[a] * (codes[b].max() + 1) + codes[b]
    if "tefd" in spec:  # Flight Distance (a proxy for the route) crossed with every other column
        fd = codes["Flight Distance"]
        for c in COLS:
            if c != "Flight Distance":
                keys[f"te_fd__{clean(c)}"] = fd * (codes[c].max() + 1) + codes[c]
    if "tecat" in spec:
        cc = codes["Customer Type"] * 100 + codes["Type of Travel"] * 10 + codes["Class"]
        keys["te_ct_tt_cl"] = cc
        for c in RATINGS + ["Age"]:
            keys[f"te_ctc_{clean(c)}"] = cc * 1000 + codes[c]
    return pd.DataFrame(keys)
