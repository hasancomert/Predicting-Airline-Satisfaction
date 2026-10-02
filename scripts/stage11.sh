#!/bin/bash
# Finals with the no-TE tuned preset t2.
cd "$(dirname "$0")/.."
NT=base,cnt,inter,fdprof
python src/train.py lgbm $NT orig=1 preset=t2 lr=0.03 note="no-TE LGBM, preset t2 (tuned on this set), lr 0.03"
python src/train.py lgbm $NT orig=1 preset=t2 lr=0.03 seed=7 note="no-TE LGBM, t2, lr 0.03, seed 7"
python src/train.py lgbm base,te1,tefd,fdprof,cnt,inter,te2 orig=1 preset=t2 lr=0.03 note="FS2 LGBM with preset t2, lr 0.03"
