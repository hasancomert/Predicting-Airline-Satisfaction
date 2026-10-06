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
  opred  prediction of a LightGBM trained only on the original data (raw columns); the
         competition labels are never used, so it is leak-free for every fold
  cnt2   label-free counts (train+test) of every pair of the 20 low-cardinality columns and of
         Flight Distance x each categorical
  digit  decimal digits (units .. thousands) of Age, Flight Distance and the two delays
         (generator artefacts; busyaprime's ladder: +0.0002 for a LightGBM)
  fdd    each row's deviation from its route profile (fdprof): Business / business travel / loyal
         flags, age and every rating minus the train+test mean of its Flight Distance value
  aux    expected value of every rating given the other 20 columns (LightGBM regression, 5-fold
         cross-predicted over train+test(+original) rows; label-free, idea from sachith7's aux_ev)
         and the rating minus that expectation; cached in cache/aux_<n rows>.npy
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
    if "opred" in groups:
        if orig is None:
            raise ValueError("opred needs the original data")
        import lightgbm as lgb
        O = pd.DataFrame({c: orig[c].astype(float) for c in NUMS})
        Ab = pd.DataFrame({c: A[c].astype(float) for c in NUMS})
        for c in CATS:
            cats = sorted(A[c].astype(str).unique())
            O[c] = pd.Categorical(orig[c].astype(str), categories=cats)
            Ab[c] = pd.Categorical(A[c].astype(str), categories=cats)
        O.columns = Ab.columns = [clean(c) for c in O.columns]
        mdl = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63,
                                 min_child_samples=20, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.7, random_state=0, verbose=-1)
        mdl.fit(O, orig[TARGET].astype(int))
        F["opred"] = mdl.predict_proba(Ab)[:, 1]
    if "cnt2" in groups:
        from itertools import combinations
        base = A.iloc[:n_tt]
        low = [c for c in COLS if c != "Flight Distance"]
        code = {c: pd.factorize(A[c].astype(str))[0].astype(np.int64) for c in COLS}
        pairs = list(combinations(low, 2)) + [("Flight Distance", c) for c in CATS]
        cnt = {}
        for a, b in pairs:
            k = pd.Series(code[a] * (code[b].max() + 1) + code[b])
            cnt[f"cnt_{clean(a)}__{clean(b)}"] = k.map(k.iloc[:n_tt].value_counts()).fillna(0).values
        F = pd.concat([F, pd.DataFrame(cnt, index=F.index)], axis=1)
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

    if "fdprof" in groups or "ofd" in groups or "fdd" in groups:
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
        if "fdd" in groups:
            prof = profile(A.iloc[:n_tt])
            row = pd.DataFrame({"biz": (A["Class"] == "Business").astype(float),
                                "btrav": (A["Type of Travel"] == "Business travel").astype(float),
                                "loyal": (A["Customer Type"] == "Loyal Customer").astype(float),
                                "age": A["Age"].astype(float)})
            for c in RATINGS:
                row[clean(c)] = A[c].astype(float)
            for c in prof.columns:
                F[f"fdd_{c}"] = row[c] - fd.map(prof[c]).astype(float)
        if "ofd" in groups:
            if orig is None:
                raise ValueError("ofd needs the original data")
            prof = profile(orig)
            for c in prof.columns:
                F[f"ofd_{c}"] = fd.map(prof[c]).astype(float)
            F["ofd_cnt"] = fd.map(orig["Flight Distance"].value_counts()).fillna(0)

    if "aux" in groups:
        F = pd.concat([F, aux_block(A)], axis=1)
    if "digit" in groups:
        dig = {}
        for c in ["Age", "Flight Distance", DEP, ARR]:
            v = A[c].fillna(0).astype(float).round().astype(np.int64)
            for k in range(2 if c == "Age" else 4):
                dig[f"dig_{clean(c)}_{k}"] = (v // 10 ** k % 10).astype(np.int8)
        F = pd.concat([F, pd.DataFrame(dig, index=F.index)], axis=1)
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
    if "te3" in spec:  # triples of the strongest columns (by single-key TE AUC of the pairs)
        top = ["Online boarding", "Class", "Type of Travel", "Inflight wifi service",
               "Inflight entertainment", "Customer Type", "Leg room service", "Seat comfort"]
        trip = list(combinations(top, 3)) + [("Flight Distance", "Class", "Type of Travel"),
                                             ("Flight Distance", "Customer Type", "Type of Travel"),
                                             ("Flight Distance", "Online boarding", "Class")]
        for a, b, c in trip:
            k = (codes[a] * (codes[b].max() + 1) + codes[b]) * (codes[c].max() + 1) + codes[c]
            keys[f"te3_{clean(a)}__{clean(b)}__{clean(c)}"] = k
    if "tecat" in spec:
        cc = codes["Customer Type"] * 100 + codes["Type of Travel"] * 10 + codes["Class"]
        keys["te_ct_tt_cl"] = cc
        for c in RATINGS + ["Age"]:
            keys[f"te_ctc_{clean(c)}"] = cc * 1000 + codes[c]
    return pd.DataFrame(keys)


def aux_block(A):
    """Label-free rating expectations: each rating regressed on the other 20 columns, cross-predicted."""
    from pathlib import Path
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    path = Path(__file__).resolve().parent.parent / "cache" / f"aux_{len(A)}.npy"
    if path.exists():
        E = np.load(path)
    else:
        X = pd.DataFrame({c: (pd.Categorical(A[c]) if c in CATS else A[c].astype(float)) for c in COLS})
        X.columns = [clean(c) for c in X.columns]
        E = np.zeros((len(A), len(RATINGS)), dtype=np.float32)
        for j, r in enumerate(RATINGS):
            cols = [c for c in X.columns if c != clean(r)]
            t = A[r].astype(float).to_numpy()
            for a, b in KFold(5, shuffle=True, random_state=0).split(X):
                m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.08, num_leaves=127,
                                      min_child_samples=50, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, verbose=-1, n_jobs=3, random_state=j)
                m.fit(X.iloc[a][cols], t[a])
                E[b, j] = m.predict(X.iloc[b][cols])
            print(f"  aux: {r} done", flush=True)
        path.parent.mkdir(exist_ok=True)
        np.save(path, E)
    out = {}
    for j, r in enumerate(RATINGS):
        out[f"aux_ev_{clean(r)}"] = E[:, j]
        out[f"aux_res_{clean(r)}"] = A[r].astype(float).to_numpy() - E[:, j]
    return pd.DataFrame(out, index=A.index)
