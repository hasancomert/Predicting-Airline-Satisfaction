#!/bin/bash
cd "$(dirname "$0")/.."
FS3=base,te1,tefd,fdprof,cnt,inter,te2,cnt2,opred
python src/train.py lgbm $FS3 orig=1 preset=t1 lr=0.03 folds=10 seed=3 note="10-fold: FS3 LGBM t1 lr0.03 seed 3"
