#!/bin/bash
# Diversity members: plain MLPs (no TE inputs got the larger blend weight) and CatBoost variants.
cd "$(dirname "$0")/.."
while [ -n "7376" ] && kill -0 7376 2>/dev/null; do sleep 30; done
FS1=base,te1,tefd,fdprof,cnt,inter
[ -f oof/nn_e16_emb24_h1024-512-256_d0.25_orig.npy ] || python src/nn.py orig=1 epochs=16 emb=24 hidden=1024,512,256 drop=0.25 note="MLP 16 ep, emb 24, 1024-512-256"
[ -f oof/nn_e12_emb16_h512-256-128_d0.2_orig_s7.npy ] || python src/nn.py orig=1 epochs=12 emb=16 seed=7 note="MLP 12 ep, emb 16, seed 7"
python src/train.py cat $FS1,te2 orig=1 lr=0.08 note="FS2, CatBoost depth 6, lr 0.08"
python src/train.py cat $FS1 orig=1 lr=0.08 depth=8 note="FS1, CatBoost depth 8, lr 0.08"
