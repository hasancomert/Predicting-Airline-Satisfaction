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
| xgb_base+te1+tefd+fdprof+cnt+inter_lr0.1_orig | XGBoost std preset (depth 6) on FS1 + original rows | 0.96028 ± 0.00063 | 0.96028 | 398 | 478s | |
| cat_base+te1+tefd+fdprof+cnt+inter_lr0.1_orig | CatBoost depth 6 on FS1 + original rows | 0.96054 ± 0.00059 | 0.96054 | 911 | 1718s | |

## Submissions

| # | File | Content | CV (OOF) | Nested CV | Public LB |
|---|---|---|---|---|---|
| 1 | s01_blend_lr0.1_lgbmFS1-catFS1-lgbmFS2.csv | rank hill-climb: LGBM FS1 0.25, CatBoost FS1 0.25, LGBM FS2 0.5 (all lr 0.1) | 0.96113 | 0.96113 | 0.96059 |
| 2 | s02_logitstack_lgbmFS3-xgbFS2-lr0.03_lgbm-xgb-cat-lr0.1_glm_nn.csv | logistic stacking on logit(p) of 7 models: LGBM FS3 t1 lr0.03 (0.567), XGB FS2 t1 lr0.03 (0.182), CatBoost FS1 lr0.1 (0.161), MLP (0.061), GLM (0.038), LGBM FS2 t1 lr0.1 (0.009), XGB FS1 t1 lr0.1 (-0.027) | 0.96147 | 0.96146 | 0.96087 |
| 3 (hazır, gönderilmedi) | s03_final.csv | logistic stacking, 8 aile: LGBM FS3 t1 lr0.03 (2 tohum) 0.273, no-TE LGBM (3 koşu) 0.234, XGB FS3 0.183, MLP (7 koşu) 0.140, CatBoost FS2 lr0.08 0.087, XGB FS2 0.076, LGBM FS2 (2 koşu) -0.007, GLM 0.000 | 0.96159 | 0.96160 | |
| 3-alt (hazır, gönderilmedi) | s03alt_rankavg_8families.csv | aynı 8 ailenin eşit ağırlıklı sıra ortalaması | 0.96153 | 0.96153 | |

## Model runs (continued)

| Experiment (tag) | Change | CV mean ± std | OOF AUC | avg trees | time | Public LB |
|---|---|---|---|---|---|---|
| nn_e8_emb12_h512-256-128_d0.2_orig | MLP: embedding per column + numeric copies, original rows, 8 epochs | 0.95846 ± 0.00054 | 0.95844 | - | 597s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+opred_lr0.1_orig | FS1 + opred (LightGBM trained on original data only) | 0.96075 ± 0.00049 | 0.96073 | 151 | 116s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+cnt2_lr0.1_orig | FS1 + label-free pair counts | 0.96084 ± 0.00049 | 0.96083 | 147 | 250s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2_lr0.1_orig | FS2 (FS1 + te2), cached TE: rerun for reference | 0.96085 ± 0.00051 | 0.96084 | 132 | 608s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+opred_lr0.1_orig | FS2 + opred | 0.96083 ± 0.00052 | 0.96082 | 114 | 273s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.1_orig | FS3 = FS2 + cnt2 + opred | 0.96089 ± 0.00052 | 0.96089 | 84 | 372s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2_lr0.1_t1_orig | FS2, tuned LGBM preset t1 | 0.96098 ± 0.00057 | 0.96098 | 96 | 529s | |
| xgb_base+te1+tefd+fdprof+cnt+inter_lr0.1_t1_orig | FS1, tuned XGB preset t1 | 0.96055 ± 0.00053 | 0.96053 | 164 | 281s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig | FINAL: FS3, LGBM t1, lr 0.03 | 0.96131 ± 0.00053 | 0.96130 | 402 | 1026s | |
| xgb_base+te1+tefd+fdprof+cnt+inter+te2_lr0.03_t1_orig | FINAL: FS2, XGB t1, lr 0.03 | 0.96118 ± 0.00056 | 0.96117 | 536 | 2226s | |
| cat_base+te1+tefd+fdprof+cnt+inter_lr0.05_orig | FINAL: FS1, CatBoost depth 6, lr 0.05 | 0.96065 ± 0.00057 | 0.96065 | 2576 | 3707s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2_lr0.03_t1_orig_s7 | FINAL: FS2, LGBM t1, lr 0.03, seed 7 | 0.96122 ± 0.00050 | 0.96122 | 536 | 765s | |
| xgb_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig | FS3, XGB t1, lr 0.03 | 0.96130 ± 0.00057 | 0.96129 | 383 | 2973s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+te3_lr0.1_t1_orig | FS2 + te3 (56+3 triples of the strongest columns), LGBM t1 | 0.96106 ± 0.00058 | 0.96105 | 107 | 413s | |
| nn_e12_emb16_h512-256-128_d0.2_orig | MLP, 12 epochs, emb 16 | 0.95910 ± 0.00058 | 0.95907 | - | 713s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig_s7 | FS3, LGBM t1, lr 0.03, seed 7 | 0.96131 ± 0.00051 | 0.96130 | 424 | 1241s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter_lr0.1_t1_orig | FS1, tuned LGBM preset t1 | 0.96074 ± 0.00065 | 0.96074 | 137 | 120s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter_lr0.1_t1_orig_ow0.5 | FS1 t1, original rows weight 0.5 | 0.96063 ± 0.00056 | 0.96063 | 150 | 122s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter_lr0.1_t1_orig_ow2 | FS1 t1, original rows weight 2 | 0.96065 ± 0.00057 | 0.96065 | 141 | 118s | |
| nn_e12_emb16_h512-256-128_d0.2_te1+tefd_orig | MLP 12 ep + TE inputs (te1, tefd) | 0.95961 ± 0.00054 | 0.95957 | - | 702s | |
| nn_e12_emb16_h512-256-128_d0.2_te1+tefd+te2_orig | MLP 12 ep + TE inputs (te1, tefd, te2) | 0.95984 ± 0.00056 | 0.95980 | - | 882s | |
| nn_e16_emb24_h1024-512-256_d0.25_orig | MLP 16 ep, emb 24, 1024-512-256 | 0.95917 ± 0.00047 | 0.95910 | - | 1938s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s7 | MLP 12 ep, emb 16, seed 7 | 0.95902 ± 0.00047 | 0.95897 | - | 712s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s11 | MLP 12 ep, emb 16, seed 11 | 0.95889 ± 0.00063 | 0.95888 | - | 1148s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s23 | MLP 12 ep, emb 16, seed 23 | 0.95896 ± 0.00070 | 0.95891 | - | 1187s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s37 | MLP 12 ep, emb 16, seed 37 | 0.95911 ± 0.00057 | 0.95907 | - | 1189s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s51 | MLP 12 ep, emb 16, seed 51 | 0.95904 ± 0.00061 | 0.95901 | - | 1213s | |
| cat_base+te1+tefd+fdprof+cnt+inter+te2_lr0.08_orig | FS2, CatBoost depth 6, lr 0.08 | 0.96096 ± 0.00049 | 0.96096 | 1401 | 4971s | |
| lgbm_base+cnt+inter+fdprof_lr0.03_t1_orig | no-TE LGBM (raw + counts + interactions + FD profile), t1, lr 0.03 | 0.96098 ± 0.00050 | 0.96098 | 1005 | 798s | |
| glm_pairsall_fd1_C0.03 | Sparse LR, C 0.03 | 0.95805 ± 0.00066 | 0.95805 | - | 202s | |
| xgb_base+cnt+inter+fdprof_lr0.03_t1_orig | no-TE XGB, t1, lr 0.03 | 0.96084 ± 0.00055 | 0.96084 | 1081 | 1370s | |
| lgbm_base+cnt+inter+fdprof_lr0.03_t2_orig | no-TE LGBM, preset t2 (tuned on this set), lr 0.03 | 0.96095 ± 0.00051 | 0.96095 | 727 | 418s | |
| lgbm_base+cnt+inter+fdprof_lr0.03_t2_orig_s7 | no-TE LGBM, t2, lr 0.03, seed 7 | 0.96096 ± 0.00050 | 0.96095 | 815 | 473s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2_lr0.03_t2_orig | FS2 LGBM with preset t2, lr 0.03 | 0.96118 ± 0.00050 | 0.96118 | 407 | 934s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s61 | MLP 12 ep, emb 16, seed 61 | 0.95911 ± 0.00065 | 0.95910 | - | 774s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s73 | MLP 12 ep, emb 16, seed 73 | 0.95890 ± 0.00059 | 0.95886 | - | 765s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s89 | MLP 12 ep, emb 16, seed 89 | 0.95911 ± 0.00072 | 0.95903 | - | 781s | |

## Blend notes

- s03 aday listesi, MLP 7 koşu ortalaması (0.96017) ile: lojistik istifleme iç içe 0.96160 → `s03_final.csv`.
- Aynı liste, MLP 10 koşu ortalaması (0.96022) ile: iç içe 0.96159, değişmedi; `s03_final.csv` korundu.
- TE girdili MLP (0.95957 / 0.95980) harmanda ağırlık almadı (0.003); hedef kodlamasız XGBoost (0.96084), hedef kodlamasız LightGBM varken 0.009.
- CatBoost FS1 lr 0.05 (0.96065), CatBoost FS2 lr 0.08 (0.96096) eklenince ≈0.
