#!/bin/bash
# Combine the round-5 winners: FS3 = FS1 + te2 + cnt2 + opred.
cd "$(dirname "$0")/.."
while kill -0 7705 2>/dev/null; do sleep 20; done   # fe5.sh
FS1=base,te1,tefd,fdprof,cnt,inter
python src/train.py lgbm $FS1,te2,opred orig=1 note="FS2 + opred"
python src/train.py lgbm $FS1,te2,cnt2,opred orig=1 note="FS3 = FS2 + cnt2 + opred"
