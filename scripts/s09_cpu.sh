#!/bin/bash
# s09 CPU members: target-encoding-free views (the no-TE LightGBM was a top stack member).
cd "$(dirname "$0")/.."
NT=base,cnt,inter,fdprof
python src/train.py xgb $NT orig=1 preset=t1 lr=0.03 folds=10 note="10-fold: no-TE XGB t1 lr0.03"
python src/train.py cat $NT orig=1 lr=0.08 folds=10 note="10-fold: no-TE CatBoost lr0.08"
