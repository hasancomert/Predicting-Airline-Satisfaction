#!/bin/bash
# Optuna (2 of the 5 fixed folds, lr 0.1) on FS1 (88 cols, fast); params then re-checked on FS3.
cd "$(dirname "$0")/.."
while kill -0 8820 2>/dev/null; do sleep 20; done   # fe6.sh
FS1=base,te1,tefd,fdprof,cnt,inter
python src/tune.py lgbm $FS1 orig=1 trials=30 lr=0.1 nfold=2
python src/tune.py xgb $FS1 orig=1 trials=25 lr=0.1 nfold=2
