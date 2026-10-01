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
| glm_pairsall_fd1_C0.1 | Sparse LR: one-hot of all values + all pairs of low-card columns + FD x cats | 0.95802 ± 0.00066 | 0.95802 | - | 327s | |
| lgbm_base+te1+cnt_lr0.1 | te1 + cnt | 0.96005 ± 0.00061 | 0.96004 | 210 | 532s | |
| lgbm_base+te1+omean_lr0.1 | te1 + omean | 0.95997 ± 0.00056 | 0.95997 | 219 | 105s | |
| lgbm_base+te1+cnt+omean_lr0.1 | te1 + cnt + omean | 0.96000 ± 0.00057 | 0.95999 | 182 | 101s | |
| lgbm_base+te1+inter_lr0.1 | te1 + inter | 0.96004 ± 0.00062 | 0.96004 | 177 | 87s | |
| lgbm_base+te1+tefd_lr0.1 | te1 + TE of FlightDistance x every other column | 0.96018 ± 0.00058 | 0.96017 | 131 | 130s | |
| lgbm_base+te1+fdprof_lr0.1 | te1 + label-free per-FlightDistance profile (train+test) | 0.96012 ± 0.00060 | 0.96011 | 197 | 100s | |
| lgbm_base+te1_lr0.1_orig | te1 + original rows in training folds | 0.96011 ± 0.00051 | 0.96010 | 254 | 114s | |
| lgbm_base+te1_lr0.1_s1 | te1, model seed 1 (noise check) | 0.95987 ± 0.00064 | 0.95986 | 200 | 92s | |
| lgbm_base+te1_lr0.1_s2 | te1, model seed 2 (noise check) | 0.95994 ± 0.00065 | 0.95993 | 199 | 92s | |
| lgbm_base+te1+ofd_lr0.1 | te1 + per-FlightDistance profile and count from original data | 0.96015 ± 0.00061 | 0.96013 | 224 | 110s | |
| lgbm_base+te1+tefd+fdprof_lr0.1 | te1 + tefd + fdprof | 0.96020 ± 0.00054 | 0.96020 | 157 | 174s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter_lr0.1 | te1 + tefd + fdprof + cnt + inter | 0.96021 ± 0.00054 | 0.96018 | 157 | 163s | |
| lgbm_base+te1+tefd+fdprof_lr0.1_orig | te1 + tefd + fdprof + original rows | 0.96031 ± 0.00056 | 0.96031 | 212 | 223s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter_lr0.1_orig | te1 + tefd + fdprof + cnt + inter + original rows | 0.96052 ± 0.00059 | 0.96052 | 207 | 217s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+ofd_lr0.1_orig | FS1 + ofd (original per-FD profile) | 0.96048 ± 0.00054 | 0.96047 | 188 | 351s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+omean_lr0.1_orig | FS1 + omean | 0.96050 ± 0.00060 | 0.96049 | 176 | 275s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2_lr0.1_orig | FS1 + TE of all low-card column pairs | 0.96082 ± 0.00055 | 0.96082 | 116 | 591s | |
