#!/bin/bash
# Restart-safe (finished runs skipped, folds checkpointed). Tuned checks, then lower-lr finals.
cd "$(dirname "$0")/.."
FS1=base,te1,tefd,fdprof,cnt,inter
FS2=$FS1,te2
FS3=$FS1,te2,cnt2,opred
python src/train.py lgbm $FS2 orig=1 preset=t1 note="FS2, tuned LGBM preset t1"
python src/train.py xgb $FS1 orig=1 preset=t1 note="FS1, tuned XGB preset t1"
python src/train.py lgbm $FS3 orig=1 preset=t1 lr=0.03 note="FINAL: FS3, LGBM t1, lr 0.03"
python src/train.py xgb $FS2 orig=1 preset=t1 lr=0.03 note="FINAL: FS2, XGB t1, lr 0.03"
python src/train.py cat $FS1 orig=1 lr=0.05 note="FINAL: FS1, CatBoost depth 6, lr 0.05"
python src/train.py lgbm $FS2 orig=1 preset=t1 lr=0.03 seed=7 note="FINAL: FS2, LGBM t1, lr 0.03, seed 7"
