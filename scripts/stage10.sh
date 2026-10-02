#!/bin/bash
# No-target-encoding branch: tune LightGBM for it, then final members.
cd "$(dirname "$0")/.."
NT=base,cnt,inter,fdprof
python src/tune.py lgbm $NT orig=1 trials=25 lr=0.1 nfold=2 threads=3
