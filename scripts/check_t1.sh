#!/bin/bash
# 5-fold check of the tuned LightGBM preset t1 on FS1 / FS2 / FS3 at lr 0.1.
cd "$(dirname "$0")/.."
while kill -0 8837 2>/dev/null; do sleep 20; done   # tune1.sh
FS1=base,te1,tefd,fdprof,cnt,inter
python src/train.py lgbm $FS1 orig=1 preset=t1 note="FS1, tuned LGBM preset t1"
python src/train.py lgbm $FS1,te2 orig=1 preset=t1 note="FS2, tuned LGBM preset t1"
python src/train.py lgbm $FS1,te2,cnt2,opred orig=1 preset=t1 note="FS3, tuned LGBM preset t1"
