#!/bin/bash
# Overnight, restart-safe: more diverse / seeded members for submission 3.
cd "$(dirname "$0")/.."
while [ -n "431" ] && kill -0 431 2>/dev/null; do sleep 30; done   # stage3.sh
FS1=base,te1,tefd,fdprof,cnt,inter
FS2=$FS1,te2
FS3=$FS1,te2,cnt2,opred
python src/train.py xgb $FS3 orig=1 preset=t1 lr=0.03 note="FS3, XGB t1, lr 0.03"
python src/train.py lgbm $FS3 orig=1 preset=t1 lr=0.03 seed=7 note="FS3, LGBM t1, lr 0.03, seed 7"
[ -f oof/nn_e12_emb16_h512-256-128_d0.2_orig.npy ] || python src/nn.py orig=1 epochs=12 emb=16 note="MLP, 12 epochs, emb 16"
[ -f oof/nn_e12_emb16_h512-256-128_d0.2_orig_s7.npy ] || python src/nn.py orig=1 epochs=12 emb=16 seed=7 note="MLP, 12 epochs, emb 16, seed 7"
python src/train.py cat $FS2 orig=1 lr=0.08 note="FS2, CatBoost depth 6, lr 0.08"
