#!/bin/bash
# Original-row weight check on FS1 (fast), tuned LGBM, lr 0.1.
cd "$(dirname "$0")/.."
while [ -n "5022" ] && kill -0 5022 2>/dev/null; do sleep 30; done
FS1=base,te1,tefd,fdprof,cnt,inter
python src/train.py lgbm $FS1 orig=1 preset=t1 note="FS1, tuned LGBM preset t1"
python src/train.py lgbm $FS1 orig=1 ow=0.5 preset=t1 note="FS1 t1, original rows weight 0.5"
python src/train.py lgbm $FS1 orig=1 ow=2 preset=t1 note="FS1 t1, original rows weight 2"
