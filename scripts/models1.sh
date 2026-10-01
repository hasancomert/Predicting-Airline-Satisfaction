#!/bin/bash
# XGBoost / CatBoost / MLP on the current best feature set (lr 0.1 for comparison with LightGBM).
cd "$(dirname "$0")/.."
while pgrep -f scripts/fe4.sh >/dev/null; do sleep 20; done
FS1=base,te1,tefd,fdprof,cnt,inter
python src/train.py xgb $FS1 orig=1 note="XGBoost std preset (depth 6) on FS1 + original rows"
python src/train.py cat $FS1 orig=1 note="CatBoost depth 6 on FS1 + original rows"
python src/train.py cat base,allcat orig=1 note="CatBoost depth 6, raw columns + every column also categorical (CatBoost CTRs), original rows"
python src/nn.py orig=1 note="MLP: embedding per column + numeric copies, original rows, 8 epochs"
