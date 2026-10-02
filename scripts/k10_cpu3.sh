#!/bin/bash
# 10-fold CPU queue, reordered: strongest members first. Restart-safe.
cd "$(dirname "$0")/.."
while pgrep -f "^python src/train.py" >/dev/null; do sleep 30; done
FS3=base,te1,tefd,fdprof,cnt,inter,te2,cnt2,opred
FS2=base,te1,tefd,fdprof,cnt,inter,te2
NT=base,cnt,inter,fdprof
python src/train.py lgbm $NT orig=1 preset=t2 lr=0.03 folds=10 note="10-fold: no-TE LGBM t2 lr0.03"
python src/train.py lgbm $FS3 orig=1 preset=t1 lr=0.03 folds=10 note="10-fold: FS3 LGBM t1 lr0.03"
python src/train.py lgbm $FS3 orig=1 preset=t1 lr=0.03 folds=10 seed=7 note="10-fold: FS3 LGBM t1 lr0.03 seed 7"
python src/train.py cat $FS2 orig=1 lr=0.08 folds=10 note="10-fold: FS2 CatBoost lr0.08"
