#!/bin/bash
# Overnight, restart-safe. Waits for the running XGB FS3 job (if any), then te3 check + members.
cd "$(dirname "$0")/.."
while pgrep -f "^python src/train.py" >/dev/null; do sleep 30; done
FS1=base,te1,tefd,fdprof,cnt,inter
FS2=$FS1,te2
FS3=$FS1,te2,cnt2,opred
python src/train.py xgb $FS3 orig=1 preset=t1 lr=0.03 note="FS3, XGB t1, lr 0.03"
python src/train.py lgbm $FS2,te3 orig=1 preset=t1 note="FS2 + te3 (56+3 triples of the strongest columns), LGBM t1"
[ -f oof/nn_e12_emb16_h512-256-128_d0.2_orig.npy ] || python src/nn.py orig=1 epochs=12 emb=16 note="MLP, 12 epochs, emb 16"
python src/train.py lgbm $FS3 orig=1 preset=t1 lr=0.03 seed=7 note="FS3, LGBM t1, lr 0.03, seed 7"
