#!/bin/bash
# Restart-safe: finished train.py runs are skipped, Optuna resumes from its sqlite study.
cd "$(dirname "$0")/.."
FS1=base,te1,tefd,fdprof,cnt,inter
python src/tune.py xgb $FS1 orig=1 trials=18 lr=0.1 nfold=2
python src/train.py lgbm $FS1 orig=1 preset=t1 note="FS1, tuned LGBM preset t1"
python src/train.py lgbm $FS1,te2 orig=1 preset=t1 note="FS2, tuned LGBM preset t1"
python src/train.py lgbm $FS1,te2,cnt2,opred orig=1 preset=t1 note="FS3, tuned LGBM preset t1"
