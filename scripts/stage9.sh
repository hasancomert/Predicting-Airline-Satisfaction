#!/bin/bash
# More seeds / te3 for the strongest GBDTs.
cd "$(dirname "$0")/.."
while [ -n "8517" ] && kill -0 8517 2>/dev/null; do sleep 30; done
FS3=base,te1,tefd,fdprof,cnt,inter,te2,cnt2,opred
python src/train.py lgbm $FS3,te3 orig=1 preset=t1 lr=0.03 note="FS3 + te3, LGBM t1, lr 0.03"
python src/train.py xgb $FS3 orig=1 preset=t1 lr=0.03 seed=7 note="FS3, XGB t1, lr 0.03, seed 7"
