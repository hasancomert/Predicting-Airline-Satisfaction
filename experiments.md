# Experiments

All runs: `StratifiedKFold(5, shuffle=True, random_state=42)` on the 699,635 train rows, identical folds for
every model. "CV" = mean ± std of the 5 per-fold ROC AUCs; "OOF" = AUC of the pooled out-of-fold predictions.
Rows are appended by `src/train.py`; the notes column says what changed. Decisions are made on CV, not LB.

| Experiment (tag) | Change | CV mean ± std | OOF AUC | avg trees | time | Public LB |
|---|---|---|---|---|---|---|
| lgbm_base_lr0.1_default | Baseline: LightGBM library defaults (100 trees), native categorical | 0.95780 ± 0.00059 | 0.95779 | 100 | 16s | |
| lgbm_base_lr0.1 | LGBM std preset (63 leaves, ES) on raw columns: reference for ablations | 0.95879 ± 0.00061 | 0.95878 | 301 | 78s | |
| lgbm_base+afill_lr0.1 | + arrival delay NaN filled from departure delay | 0.95876 ± 0.00057 | 0.95875 | 322 | 82s | |
| lgbm_base+delay_lr0.1 | + delay sum/diff/max/log/ratio/flags | 0.95870 ± 0.00059 | 0.95867 | 347 | 103s | |
| lgbm_base+rnan_lr0.1 | + rating 0 -> NaN, zero count | 0.95880 ± 0.00069 | 0.95879 | 315 | 77s | |
| lgbm_base+zflag_lr0.1 | + per-rating zero flags | 0.95867 ± 0.00067 | 0.95865 | 354 | 88s | |
| lgbm_base+agg_lr0.1 | + rating mean/min/max/std/sum/n5/n1 | 0.95843 ± 0.00069 | 0.95842 | 234 | 70s | |
| lgbm_base+grp_lr0.1 | + digital/cabin/service group means and diffs | 0.95846 ± 0.00067 | 0.95845 | 227 | 69s | |
| lgbm_base+inter_lr0.1 | + CustomerType x TravelType x Class interactions (categorical) | 0.95892 ± 0.00070 | 0.95890 | 252 | 87s | |
| lgbm_base+cnt_lr0.1 | + value counts (distance, age, delays) | 0.95942 ± 0.00071 | 0.95942 | 317 | 96s | |
| lgbm_base+rcat_lr0.1 | + ratings duplicated as categoricals | 0.95866 ± 0.00065 | 0.95865 | 214 | 76s | |
| lgbm_base+te1_lr0.1 | + in-fold target encoding of every column | 0.95997 ± 0.00061 | 0.95995 | 203 | 91s | |
| lgbm_base+tecat_lr0.1 | + in-fold TE of CT x TT x Class and its pairs with each rating/age | 0.95876 ± 0.00061 | 0.95874 | 169 | 78s | |
| lgbm_base+omean_lr0.1 | + original-data satisfaction rate per value | 0.95979 ± 0.00062 | 0.95978 | 207 | 78s | |
| lgbm_base_lr0.1_orig | + original rows in every training fold (is_orig flag) | 0.95888 ± 0.00072 | 0.95886 | 375 | 98s | |
