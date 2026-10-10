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
| 3 | s03_final.csv | logistic stacking, 8 aile: LGBM FS3 t1 lr0.03 (2 tohum) 0.273, no-TE LGBM (3 koşu) 0.234, XGB FS3 0.183, MLP (7 koşu) 0.140, CatBoost FS2 lr0.08 0.087, XGB FS2 0.076, LGBM FS2 (2 koşu) -0.007, GLM 0.000 | 0.96159 | 0.96160 | 0.96106 |
| 3-alt | s03alt_rankavg_8families.csv | aynı 8 ailenin eşit ağırlıklı sıra ortalaması (kullanıcı gönderdi) | 0.96153 | 0.96153 | 0.96092 |
| 4 | s04_hill_6fam_realmlp.csv | sıra uzayında hill climbing; 10 aileden 6'sı seçildi, her biri 1/6: LGBM FS3 (2 tohum), XGB FS3, CatBoost FS2, hedef kodlamasız LGBM (3 koşu), RealMLP pub (3 ve 6 epoch), RealMLP v4 (2 tohum). Lojistik istifleme aynı listede iç içe 0.96164 | 0.96165 | 0.96165 | 0.96107 |
| 5a (hazır, gönderilmedi) | s05a_hill_own5fold_plus_public.csv | s04 listesi + goodpjw2008 açık OOF grupları (aynı 5 kat, hizalama doğrulandı); hill climbing 7 üye seçti, her biri 1/7: LGBM FS3, XGB FS3, CatBoost FS2, hedef kodlamasız LGBM, RealMLP pub (bizim), açık XGB grubu, açık RealMLP v4 grubu. Lojistik istifleme 0.96169 ama ±0.59 ağırlıklı (iki aynı tarifli RealMLP arasında fark alıyor), sağlam değil | 0.96168 | 0.96167 | |
| 5 | s05_k10_hill_8fam.csv | **10 kat** (StratifiedKFold(10, seed 42)), 8 aile; hill climbing ağırlıkları: XGB FS3 (2 tohum) 3/13, RealMLP v4+te2 (3 tohum) 3/13, LGBM FS3 (3 tohum) 2/13, hedef kodlamasız LGBM (t1+t2) 2/13, CatBoost FS2 1/13, RealMLP pub (3) 1/13, MLP (15 tohum) 1/13, RealMLP v4 (4 tohum) 0. Eşit sıra ortalaması 0.96180, lojistik istifleme 0.96180 | 0.96182 | 0.96181 | 0.96114 |
| 6 | s06_k10_logit_9fam.csv | 10 kat, 9 aile; lojistik istifleme (hepsi pozitif ağırlık): RealMLP v4+te2/v5 (5 koşu) 0.203, RealMLP pub (3) 0.190, hedef kodlamasız LGBM 0.144, XGB FS3 (3 tohum) 0.095, LGBM FS3 (3 tohum) 0.092, LGBM FS2+te3+opred (2 tohum) 0.080, XGB FS3+te3 0.078, CatBoost FS2 (2 tohum) 0.068, MLP (15) 0.046. Hill 0.96180, eşit sıra ortalaması 0.96181; s05 ile sıra korelasyonu 0.99967 | 0.96181 | 0.96181 | 0.96116 |
| 7 | s07_k10_10fam_tabm.csv | s06 listesi + TabM (3 tohum, ortalaması 0.96142); lojistik istifleme: TabM 0.171, RealMLP v4+te2/v5 0.171, RealMLP pub 0.161, hedef kodlamasız LGBM 0.135, CatBoost 0.078, LGBM FS3 0.070, XGB FS3+te3 0.059, LGBM FS2+te3 0.058, XGB FS3 0.053, MLP 0.042. Hill / sıra / ortalama 0.96183; s06 ile sıra korelasyonu 0.99974 | 0.96184 | 0.96184 | 0.96119 |
| 8 | s08_stack_own10fam_pub124_C0.001.csv | lojistik istifleme (C=0.001, logit) — kendi 10 katlı 10 ailemiz + 124 açık OOF üyesi (aynı train id'leri; çoğu StratifiedKFold(5, seed 42)). Yalnız açık üyeler 0.962046; kendi üyelerimiz +0.00006–0.00008 ekliyor, kaynakları tek tek çıkarınca en büyük kayıp bizim ailelerimizde (−0.00007). C taraması: 1 → 0.962103, 0.1 → 0.962106, 0.01 → 0.962121, 0.001 → 0.962127, 0.0003 → 0.962102 | 0.962127 | 0.962127 | 0.96165 |
| 9 | s09_stack_own15_pub169_C0.001.csv | lojistik istifleme (C=0.001, logit) — kendi 15 üyemiz (s08 + RealMLP pub/v4te2 orijinal satırlarla, hedef kodlamasız XGB, kendi TabPFN-3.5'imiz, ham TabM) + 169 açık OOF üyesi (+sadamtorres 38, amanatar 7). Kaynak ablasyonu: kendi üyelerimiz −0.000087, goodpjw −0.000031, busyaprime −0.000025, kalanlar ≤ 0.000006. C taraması: 0.0005 → 0.962136, 0.001 → 0.962149, 0.002 → 0.962150 | 0.962149 | 0.962149 | 0.96171 (24./948) |
| 10 | s10_stack_own15seeds_pub169_C0.001.csv | s09 listesi, istifte büyük ağırlıklı kendi üyelerimize ek tohumlar: RealMLP pub + orijinal ×4, RealMLP v4te2 + orijinal ×2 (ortalama 0.96151/0.96152), hedef kodlamasız XGB ×2 (0.96109). s09 ile Spearman 0.999996. Negatif olmayan istif denendi (0.96200, reddedildi). C=0.002 → 0.962153 | 0.962150 | 0.962150 | 0.96171 |
| 11 | s11_stack_own15_pub178_C0.001.csv | s10 kendi listesi + yenilenen açık havuz (178): kratosyan route-ID + öğretmen (2), denpugovkin sayısal anahtar CatBoost CTR (1), amanatar satır TE (1), kagankoral (5). Ablasyon: kendi −0.000088, goodpjw −0.000032, busyaprime −0.000022, denpugovkin −0.000005, kagankoral −0.000005, kratosyan +0.000001 | 0.962159 | 0.962159 | 0.96173 (16/961) |
| 12 | s12_stack_own17_pub178_C0.001.csv | s11 + kendi iki yeni RealMLP üyemiz: yekenot görünümü + orijinal satırlar, son epoch, 3 tohum (ortalama 0.96147; ağırlık 0.102), aynı görünüm + v4 eklentileri + fold içi TE (0.96142). s11 ile Spearman 0.99994 | 0.962167 | 0.962167 | 0.96174 (14/965) |
| 13 | s13_stack_own19_pub178_C0.001.csv | s12 + RealMLP yk+v4 tohum 1/2 (ortalama), yk+v4+sütun çifti TE (0.96150), yk 5 epoch (0.96138, ağırlık 0.100). İç içe CV değişmedi (varyantlar 0.962164–0.962167); ek tohumlar test gürültüsünü azaltmak için. s12 ile Spearman 0.99995 | 0.962167 | 0.962167 | 0.96175 (17/968) |
| 14 | s14_stack_own19seeds_pub184_C0.001.csv | s13 + Kaggle CPU tohumları (RealMLP yk tohum 3/4: 0.96144/0.96143; yk+v4+te2 tohum 1: 0.96148) + açık havuzda Forge (184). s13 ile Spearman 0.999994 | 0.962170 | 0.962170 | 0.96175 (90/1086) |
| 15 | s15_stack_own19_pub195tfm_C0.001.csv | s14 + cdeotte'nin tablo temel modeli OOF'ları (11: TabPFN-3.5, TabICL2, Mitra2, LimiX2, KumoRFM büyük/küçük, EXAONE, TabFM, Causilo, RealMLP, XGBoost; tek başına 0.957–0.9611). En büyük ağırlıklar TabFM 0.163, LimiX2 0.123. s14 ile Spearman 0.99966 | 0.962200 | 0.962200 | 0.96180 (13/1087) |
| 16 | s16_stack_own19_pub204_C0.001.csv | s15 + 09.10 açık havuz yenilemesi (204): abdullahsafwan333'ün goodpjw2008 tariflerini 15 katta yeniden koşusu (TabPFN catfd10 0.96160, ağırlık 0.14; raw; XGB v3; Cat v3), golem lgbm_auxev ×2, sachith7 xgb_A_auxall, hermengardo CatBoost, yekenot RealMLP; kodu yayınlanmamış ab_tabpfn_te_cnt hariç (dahil 0.962240). arhancanli12 logit düzeltmesi tek başına 0.962198 (etkisiz). s15 ile Spearman 0.99981 | 0.962237 | 0.962237 | 0.96183 (21/1245) |

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
- s09 açık havuz yenilemesi (06.10): goodpjw2008 "TabPFN member" veri seti P8'dekilerle aynı (korelasyon 1.0),
  busyaprime'ın yeni sürümü eskisiyle aynı (≥0.9998); yeni olan amanatar'ın 7 motoru: iç içe 0.962130 → 0.962135 (gürültü).
- İstif girdisi probit(sıra) (S6E9 hilesi): C=0.01 0.961015, C=0.001 0.961265 — logit (0.962127) çok daha iyi.
- RealMLP pub + orijinal satırlar (0.96127) ayrı üye olarak: 0.962135 → 0.962137 (ağırlığı 0.095, en büyük 2.).
- LightGBM meta-öğrenici (180 logit, 15 yaprak, erken durdurma): 0.961997 < LR 0.962137; sıra harmanı
  (%20 GBDT) 0.962145 (+0.000008, gürültü) → lojistik istif kaldı.
- Hedef kodlamasız XGB t1 (0.96102) ayrı üye: 0.962137 → 0.962145 (ağırlığı 0.074).
- Hedef kodlamasız CatBoost lr 0.08 (0.96068) ayrı üye: 0.962145 → 0.962145 (katkı yok, listeye alınmadı).
- Hedef kodlamasız LightGBM extra_trees (0.95625): 0.962145 → 0.962145 (katkı yok).
- s10: negatif olmayan ağırlıklı istif (nonneg=1, aynı cezalı amaç, L-BFGS-B): C=0.001 0.962006, 0.01 0.961989, 0.1 0.961988 (55/184 üye sıfırdan farklı) — serbest LR 0.962149 (81/184 negatif ağırlık). Benzer üyeler arasındaki farklar (negatif ağırlıklar) iç içe CV'de gerçek bilgi taşıyor → serbest LR kaldı.
- s10 kontrol: test tahminleri kat ortalaması olduğu için OOF'tan yumuşak mı? Üye logit std oranı test/OOF 0.991–1.002; üyelerin ortak eğilim etrafındaki sapması 0.88–1.00 (medyan 0.95); istif skorunun ortak kısmı 0.998, fark (kontrast) kısmı 0.967. Yani test'te kontrast ağırlıkları fiilen ~%3 zayıf; C eğrisi bu bölgede düz, düzeltme gereksiz.
- s11 özellik denemeleri (hedef kodlamasız LightGBM t2 lr0.1, 5 katın 0–1. katları, temel 0.96040): basamaklar (digit) 0.96041, rota ortalamasından sapma (fdd) 0.96015, ikisi 0.96025, etiketsiz puan beklentisi + artığı (aux, sachith7 fikri) 0.96035 → hiçbiri alınmadı. İstif girdisi kırpma (|logit| ≤ 5/8) 0.962150/0.962151, sütun standardizasyonu (C 0.001–0.03) ≤ 0.962143 → değişiklik yok. Train ile test arasında birebir aynı satır yok.
- s11 dürüstlük kontrolü: pytabkit en iyi epoch'u verdiğimiz doğrulama katında seçiyor (puanlanan satırlar). Son epoch ile (stop_epoch) aynı RealMLP pub + orijinal: 0.96126, en iyi epoch ile 0.96127/0.96128 → yanlılık ~0.00001, ihmal edilebilir.
- 07.10 açık havuz kontrolü: amanatar "Forge own pairs" (6, busyaprime üyelerinin yeniden üretimi, korelasyon 0.9993–0.99997) eklendi: s13 listesiyle 0.962167 → 0.962168 (gürültü, yeni gönderim yok). cdeotte'nin 124'lük derlemesi mevcut üyelerin kopyası; garv3068 AutoGluon OOF'u pickle (okunmadı).
- 09.10 açık havuz yenilemesi: en iyi açık istiflerin girdileri tarandı (karttikjangid05, taisei7, goodpjw2008 meta stack, matterhorn3838). Yeni: abdullahsafwan333 oof-dataset (15 kat), golem +3 (o_tabpfn35 = gp_tabpfn_catfd10 kopyası, alınmadı), sachith7 +1, hermengardo, yekenot; najiama 04 harman (alınmadı). Bulunan hata: arhancanli12 *_5f üyeleri olasılık değil [−2, 2.4] skor; istif bunları [1e-6, 1−1e-6]'ya kırpıyordu → ext_pub'da sigmoid. Etkisi yok (0.962198), ama doğru. cdeotte'nin notu TFM üyelerinin StratifiedKFold(5, seed 42) olduğunu doğruluyor.
- 09.10 08:46 rutin kontrolü: sabahki yenilemeden sonra yeni OOF kaynağı yok (alexanderwang2001 ve anhadmahajan06 veri setleri yalnız gönderim dosyası; hitarthjain0 signal-stack-master elimizdeki kaynakları kullanıyor). Kaggle CPU kernel'larının hepsi alınmış.
- TFM uygulanabilirlik (09.10): TabFM 16 GB T4'te ~8.000 bağlam satırıyla sınırlı (paiky1995'in ölçümü), daha fazla kat bağlamı büyütmez. Kumo-Tabular denemesi (`src/kumo.py probe=1`, 5 katın 0. katı): büyük model 100k / 300k / tam bağlamda, 2 ve 8 tahminciyle T4'te bellek yetersiz (100k bağlamda 55.9 GB istiyor, 9.375 satırlık sorgu parçasında bile); küçük model tam bağlam (559.708) 2 tahminci 0.958366, 130 dk (Deotte'nin 100k bağlamlı küçük modeli bu katta 0.958421; havuzdaki kumolarge 0.959946). Sonuç: Kaggle donanımında daha güçlü bir Kumo/TabFM üyesi üretilemiyor; bu yol kapandı. GPU kotası bu deneyle tükendi (30.17 / 30 saat).
- 15 katlı TabPFN-3.5 denemesi (10.10, `kaggle/tabpfn/build_kernel.py pushx`, 15 katın 0. katı): catfd10 görünümü 652.992 × 22 bağlamla ve orijinal satırlı 782.872 × 23 bağlamla, üç bellek modunda da T4'te bellek yetersiz (abdullahsafwan333'ün notu: tam bağlam 24+ GB GPU ister). Ölçü: açık route_og TabPFN'i 559.708 × 23 = 12.9 M hücrede 13.2 GB kullanıyor. Zayıf sütunlar atılmış 'lite' varyantlarla yeniden deneniyor (653 k × 18 = 11.8 M; orijinalli 783 k × 17 = 13.3 M).
| realmlp_pub_e3_ens8 | RealMLP public recipe (raw + categorical twins), 3 epochs, n_ens 8 | 0.96077 ± 0.00054 | 0.96076 | - | 1126s | |
| realmlp_v4_e6_ens8 | RealMLP v4: + route profile, counts, original-model logit, in-fold TE (te1, tefd), 6 epochs | 0.96116 ± 0.00054 | 0.96114 | - | 3817s | |
| realmlp_pub_e6_ens8_s1 | RealMLP public recipe, 6 epochs, seed 1 | 0.96101 ± 0.00054 | 0.96100 | - | 2487s | |
| realmlp_v4_e6_ens8_s1 | RealMLP v4, 6 epochs, seed 1 | 0.96111 ± 0.00058 | 0.96107 | - | 3592s | |
| lgbm_base+cnt+inter+fdprof_lr0.03_t1_orig_k10 | 10-fold: no-TE LGBM t1 lr0.03 | 0.96107 ± 0.00064 | 0.96106 | 1097 | 908s | |
| lgbm_base+cnt+inter+fdprof_lr0.03_t2_orig_k10 | 10-fold: no-TE LGBM t2 lr0.03 | 0.96110 ± 0.00061 | 0.96109 | 769 | 858s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig_k10 | 10-fold: FS3 LGBM t1 lr0.03 | 0.96145 ± 0.00067 | 0.96144 | 483 | 2763s | |
| xgb_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig_k10 | 10-fold: FS3 XGB t1 lr0.03 (Kaggle GPU) | 0.96149 ± 0.00065 | 0.96148 | 512 | 2718s | |
| xgb_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig_s7_k10 | 10-fold: FS3 XGB t1 lr0.03 seed 7 (Kaggle GPU) | 0.96145 ± 0.00067 | 0.96145 | 479 | 2686s | |
| realmlp_pub_e3_ens8_k10 | 10-fold: RealMLP pub e3 (Kaggle GPU) | 0.96089 ± 0.00066 | 0.96088 | - | 1034s | |
| realmlp_v4_e6_ens8_k10 | 10-fold: RealMLP v4 e6 (Kaggle GPU) | 0.96129 ± 0.00061 | 0.96124 | - | 2025s | |
| realmlp_pub_e6_ens8_s1_k10 | 10-fold: RealMLP pub e6 s1 (Kaggle GPU) | 0.96111 ± 0.00070 | 0.96110 | - | 1758s | |
| realmlp_v4_e6_ens8_s1_k10 | 10-fold: RealMLP v4 e6 s1 (Kaggle GPU) | 0.96129 ± 0.00061 | 0.96125 | - | 1985s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig_s7_k10 | 10-fold: FS3 LGBM t1 lr0.03 seed 7 | 0.96147 ± 0.00065 | 0.96146 | 428 | 2567s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_k10 | 10-fold: MLP 12 ep emb 16 seed 42 (Kaggle GPU) | 0.95940 ± 0.00050 | 0.95938 | - | 258s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s7_k10 | 10-fold: MLP 12 ep emb 16 seed 7 (Kaggle GPU) | 0.95926 ± 0.00066 | 0.95920 | - | 251s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s11_k10 | 10-fold: MLP 12 ep emb 16 seed 11 (Kaggle GPU) | 0.95935 ± 0.00067 | 0.95929 | - | 251s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s23_k10 | 10-fold: MLP 12 ep emb 16 seed 23 (Kaggle GPU) | 0.95933 ± 0.00075 | 0.95931 | - | 252s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s37_k10 | 10-fold: MLP 12 ep emb 16 seed 37 (Kaggle GPU) | 0.95945 ± 0.00054 | 0.95943 | - | 250s | |
| realmlp_v4_e6_ens8_s2_k10 | 10-fold: RealMLP v4 e6 s2 (Kaggle GPU) | 0.96125 ± 0.00063 | 0.96123 | - | 1645s | |
| realmlp_pub_e6_ens8_s2_k10 | 10-fold: RealMLP pub e6 s2 (Kaggle GPU) | 0.96112 ± 0.00067 | 0.96112 | - | 1428s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s51_k10 | 10-fold: MLP 12 ep emb 16 seed 51 (Kaggle GPU) | 0.95935 ± 0.00066 | 0.95932 | - | 251s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s61_k10 | 10-fold: MLP 12 ep emb 16 seed 61 (Kaggle GPU) | 0.95938 ± 0.00065 | 0.95935 | - | 245s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s73_k10 | 10-fold: MLP 12 ep emb 16 seed 73 (Kaggle GPU) | 0.95920 ± 0.00066 | 0.95916 | - | 247s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s89_k10 | 10-fold: MLP 12 ep emb 16 seed 89 (Kaggle GPU) | 0.95929 ± 0.00063 | 0.95924 | - | 249s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s101_k10 | 10-fold: MLP 12 ep emb 16 seed 101 (Kaggle GPU) | 0.95934 ± 0.00079 | 0.95931 | - | 250s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s113_k10 | 10-fold: MLP 12 ep emb 16 seed 113 (Kaggle GPU) | 0.95937 ± 0.00072 | 0.95932 | - | 246s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s127_k10 | 10-fold: MLP 12 ep emb 16 seed 127 (Kaggle GPU) | 0.95935 ± 0.00075 | 0.95932 | - | 248s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s131_k10 | 10-fold: MLP 12 ep emb 16 seed 131 (Kaggle GPU) | 0.95931 ± 0.00063 | 0.95928 | - | 248s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s149_k10 | 10-fold: MLP 12 ep emb 16 seed 149 (Kaggle GPU) | 0.95931 ± 0.00065 | 0.95921 | - | 249s | |
| nn_e12_emb16_h512-256-128_d0.2_orig_s163_k10 | 10-fold: MLP 12 ep emb 16 seed 163 (Kaggle GPU) | 0.95924 ± 0.00068 | 0.95921 | - | 248s | |
| realmlp_v4te2_e6_ens8_k10 | 10-fold: RealMLP v4 + te2 TE inputs, e6 (Kaggle GPU) | 0.96147 ± 0.00060 | 0.96143 | - | 3736s | |
| realmlp_v4_e6_ens8_s3_k10 | 10-fold: RealMLP v4 e6 s3 (Kaggle GPU) | 0.96131 ± 0.00063 | 0.96126 | - | 1711s | |
| cat_base+te1+tefd+fdprof+cnt+inter+te2_lr0.08_orig_k10 | 10-fold: FS2 CatBoost lr0.08 | 0.96109 ± 0.00069 | 0.96109 | 1511 | 7326s | |
| realmlp_v4te2_e6_ens8_s1_k10 | 10-fold: RealMLP v4 + te2, e6, seed 1 (Kaggle GPU) | 0.96148 ± 0.00059 | 0.96144 | - | 3489s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig_s3_k10 | 10-fold: FS3 LGBM t1 lr0.03 seed 3 | 0.96146 ± 0.00066 | 0.96145 | 473 | 2695s | |
| realmlp_v4te2_e6_ens8_s2_k10 | 10-fold: RealMLP v4 + te2, e6, seed 2 (Kaggle GPU) | 0.96148 ± 0.00061 | 0.96145 | - | 3625s | |
| realmlp_v5_e6_ens8_te1+tefd+te2_s1_k10 | 10-fold: RealMLP v5, e6, seed 1 (Kaggle GPU) | 0.96147 ± 0.00060 | 0.96143 | - | 3579s | |
| realmlp_v5_e6_ens8_te1+tefd+te2_k10 | 10-fold: RealMLP v5 (v4+te2 + original-data RealMLP logit), e6 (Kaggle GPU) | 0.96149 ± 0.00060 | 0.96145 | - | 3692s | |
| cat_base+te1+tefd+fdprof+cnt+inter+te2_lr0.08_orig_s2_k10 | 10-fold: FS2 CatBoost lr0.08 seed 2 | 0.96105 ± 0.00070 | 0.96105 | 1404 | 6676s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+te3+opred_lr0.03_t1_orig_k10 | 10-fold: FS2 + te3 + opred LGBM t1 lr0.03 | 0.96146 ± 0.00073 | 0.96146 | 449 | 1881s | |
| lgbm_base+te1+tefd+fdprof+cnt+inter+te2+te3+opred_lr0.03_t1_orig_s7_k10 | 10-fold: FS2 + te3 + opred LGBM t1 lr0.03 seed 7 | 0.96143 ± 0.00072 | 0.96142 | 407 | 1689s | |
| xgb_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred+te3_lr0.03_t1_orig_k10 | 10-fold: FS3 + te3 XGB t1 lr0.03 (Kaggle GPU) | 0.96145 ± 0.00071 | 0.96144 | 423 | 3004s | |
| xgb_base+te1+tefd+fdprof+cnt+inter+te2+cnt2+opred_lr0.03_t1_orig_s3_k10 | 10-fold: FS3 XGB t1 lr0.03 seed 3 (Kaggle GPU) | 0.96143 ± 0.00067 | 0.96142 | 502 | 2598s | |
| tabm_v4te2_k10 | 10-fold: TabM (pytabkit TabM_D) on v4 + te2 features, no twins (Kaggle GPU) | 0.96139 ± 0.00062 | 0.96137 | - | 11371s | |
| tabm_v4te2_bs2048_s1_k10 | 10-fold: TabM v4+te2, batch 2048, seed 1 (Kaggle GPU) | 0.96139 ± 0.00061 | 0.96137 | - | 9872s | |
| tabm_v4te2_s2_k10 | 10-fold: TabM v4+te2, seed 2 (Kaggle GPU) | 0.96137 ± 0.00060 | 0.96133 | - | 10903s | |
| realmlp_pub_e6_ens8_orig_k10 | 10-fold: RealMLP pub e6 + original rows (Kaggle GPU) | 0.96128 ± 0.00069 | 0.96127 | - | 1851s | |
| xgb_base+cnt+inter+fdprof_lr0.03_t1_orig_k10 | 10-fold: no-TE XGB t1 lr0.03 | 0.96103 ± 0.00063 | 0.96102 | 1258 | 3180s | |
| cat_base+cnt+inter+fdprof_lr0.08_orig_k10 | 10-fold: no-TE CatBoost lr0.08 | 0.96069 ± 0.00074 | 0.96068 | 2162 | 7228s | |
| lgbm_base+cnt+inter+fdprof_lr0.03_t2_orig_extra_treesTrue_k10 | 10-fold: no-TE LGBM t2 lr0.03 extra_trees | 0.95626 ± 0.00078 | 0.95625 | 5632 | 4131s | |
| realmlp_v4te2_e6_ens8_orig_k10 | 10-fold: RealMLP v4+te2 e6 + original rows (Kaggle GPU) | 0.96152 ± 0.00061 | 0.96149 | - | 4181s | |
| realmlp_pub_e6_ens8_orig_s1_k10 | 10-fold: RealMLP pub e6 + original rows, seed 1 (Kaggle GPU) | 0.96129 ± 0.00072 | 0.96128 | - | 1842s | |
| tabm_pub_k10 | 10-fold: TabM on raw columns + categorical twins (Kaggle GPU) | 0.96011 ± 0.00070 | 0.96001 | - | 7269s | |
| tabpfn35_te4op_k5 | TabPFN-3.5 full fold context (5 folds, seed 42) on 22 cols: raw minus Gender/delays/Food, in-fold TE of route and route x Class/Travel/Customer, opred logit (Kaggle 2xT4) | 0.96121 ± 0.00053 | 0.96120 | - | 15720s | |
| xgb_base+cnt+inter+fdprof_lr0.03_t1_orig_s7_k10 | 10-fold: no-TE XGB t1 lr0.03 seed 7 | 0.96105 ± 0.00065 | 0.96104 | 1303 | 3081s | |
| realmlp_v4te2_e6_ens8_orig_s1_k10 | 10-fold: RealMLP v4+te2 e6 + original rows, seed 1 (Kaggle GPU) | 0.96152 ± 0.00059 | 0.96151 | - | 4161s | |
| realmlp_pub_e6_ens8_orig_s2_k10 | 10-fold: RealMLP pub e6 + original rows, seed 2 (Kaggle GPU) | 0.96128 ± 0.00069 | 0.96127 | - | 1814s | |
| realmlp_pub_e6_ens8_orig_s3_k10 | 10-fold: RealMLP pub e6 + original rows, seed 3 (Kaggle GPU) | 0.96129 ± 0.00070 | 0.96128 | - | 1809s | |
| realmlp_pub_e6_ens8_orig_last_k10 | 10-fold: RealMLP pub e6 + original rows, last epoch (no selection on the scored fold) (Kaggle GPU) | 0.96127 ± 0.00071 | 0.96126 | - | 1724s | |
| realmlp_yk_e3_ens8_orig_last_k10 | 10-fold: RealMLP yekenot view (counts + 3 combos, TE in fold) e3 + original rows, last epoch (Kaggle GPU) | 0.96145 ± 0.00070 | 0.96145 | - | 961s | |
| realmlp_yk_e3_ens8_orig_last_s1_k10 | 10-fold: RealMLP yekenot view + orig, last epoch, seed 1 (Kaggle GPU) | 0.96143 ± 0.00070 | 0.96142 | - | 1130s | |
| realmlp_yk_e3_ens8_orig_last_s2_k10 | 10-fold: RealMLP yekenot view + orig, last epoch, seed 2 (Kaggle GPU) | 0.96142 ± 0.00068 | 0.96142 | - | 1147s | |
| realmlp_ykv4_e3_ens8_te1+tefd_orig_last_k10 | 10-fold: RealMLP yekenot view + v4 extras (route profile, counts, opred) + TE te1/tefd + orig, last epoch (Kaggle GPU) | 0.96141 ± 0.00059 | 0.96142 | - | 1257s | |
| realmlp_yk_e5_ens8_orig_last_k10 | 10-fold: RealMLP yekenot view + orig, 5 epochs, last epoch (Kaggle GPU) | 0.96138 ± 0.00072 | 0.96138 | - | 1460s | |
| realmlp_ykv4_e3_ens8_te1+tefd_orig_last_s1_k10 | 10-fold: RealMLP yk + v4 extras + TE te1/tefd + orig, last epoch, seed 1 (Kaggle GPU) | 0.96137 ± 0.00059 | 0.96137 | - | 1149s | |
| realmlp_ykv4_e3_ens8_te1+tefd_orig_last_s2_k10 | 10-fold: RealMLP yk + v4 extras + TE te1/tefd + orig, last epoch, seed 2 (Kaggle GPU) | 0.96138 ± 0.00063 | 0.96138 | - | 1141s | |
| realmlp_ykv4_e3_ens8_te1+tefd+te2_orig_last_k10 | 10-fold: RealMLP yk + v4 extras + TE te1/tefd/te2 (pairs) + orig, last epoch (Kaggle GPU) | 0.96151 ± 0.00061 | 0.96150 | - | 3013s | |
| realmlp_ykv4_e3_ens8_te1+tefd+te2_orig_last_s1_k10 | 10-fold: RealMLP yk + v4 extras + TE te1/tefd/te2 + orig, last epoch, seed 1 (Kaggle CPU) | 0.96149 ± 0.00061 | 0.96148 | - | 27805s | |
| realmlp_yk_e3_ens8_orig_last_s3_k10 | 10-fold: RealMLP yekenot view + orig, last epoch, seed 3 (Kaggle CPU) | 0.96144 ± 0.00072 | 0.96144 | - | 8989s | |
| realmlp_yk_e3_ens8_orig_last_s4_k10 | 10-fold: RealMLP yekenot view + orig, last epoch, seed 4 (Kaggle CPU) | 0.96144 ± 0.00072 | 0.96143 | - | 8997s | |
| realmlp_ykv4_e3_ens8_te1+tefd+te2_orig_last_s2_k10 | 10-fold: RealMLP yk + v4 extras + TE te1/tefd/te2 + orig, last epoch, seed 2 (Kaggle CPU) | 0.96151 ± 0.00061 | 0.96150 | - | 38447s | |
