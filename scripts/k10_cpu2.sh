#!/bin/bash
# Re-run the 10-fold FS3 LightGBM (OOM-killed while a blend ran alongside); resumes from fold checkpoints.
cd "$(dirname "$0")/.."
while [ -n "563" ] && kill -0 563 2>/dev/null; do sleep 30; done
FS3=base,te1,tefd,fdprof,cnt,inter,te2,cnt2,opred
python src/train.py lgbm $FS3 orig=1 preset=t1 lr=0.03 folds=10 note="10-fold: FS3 LGBM t1 lr0.03"
