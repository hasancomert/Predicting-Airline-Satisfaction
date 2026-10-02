#!/bin/bash
# s06 CPU members (10-fold). Restart-safe.
cd "$(dirname "$0")/.."
FS2=base,te1,tefd,fdprof,cnt,inter,te2
python src/train.py cat $FS2 orig=1 lr=0.08 folds=10 seed=2 note="10-fold: FS2 CatBoost lr0.08 seed 2"
python src/train.py lgbm $FS2,te3,opred orig=1 preset=t1 lr=0.03 folds=10 note="10-fold: FS2 + te3 + opred LGBM t1 lr0.03"
