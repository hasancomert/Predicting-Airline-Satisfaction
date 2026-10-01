#!/bin/bash
# After models1: original-model prediction and pair counts on top of FS1, then FS2 reference run.
cd "$(dirname "$0")/.."
while kill -0 4643 2>/dev/null; do sleep 20; done   # models1.sh
FS1=base,te1,tefd,fdprof,cnt,inter
python src/train.py lgbm $FS1,opred orig=1 note="FS1 + opred (LightGBM trained on original data only)"
python src/train.py lgbm $FS1,cnt2 orig=1 note="FS1 + label-free pair counts"
python src/train.py lgbm $FS1,te2 orig=1 note="FS2 (FS1 + te2), cached TE: rerun for reference"
