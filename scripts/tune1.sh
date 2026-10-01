#!/bin/bash
# After models1: FS2 LightGBM at lr 0.1 (fills the TE cache), then Optuna for LightGBM on FS2.
cd "$(dirname "$0")/.."
while kill -0 4643 2>/dev/null; do sleep 20; done   # models1.sh
FS2=base,te1,tefd,fdprof,cnt,inter,te2
python src/train.py lgbm $FS2 orig=1 note="FS2 (FS1 + te2), cached TE: rerun for reference"
python src/tune.py lgbm $FS2 orig=1 trials=30 lr=0.1 nfold=2
