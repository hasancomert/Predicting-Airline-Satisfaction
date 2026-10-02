#!/bin/bash
# MLP with in-fold target encodings as extra inputs.
cd "$(dirname "$0")/.."
while [ -n "7014" ] && kill -0 7014 2>/dev/null; do sleep 30; done
[ -f oof/nn_e12_emb16_h512-256-128_d0.2_te1+tefd_orig.npy ] || python src/nn.py orig=1 epochs=12 emb=16 te=te1,tefd note="MLP 12 ep + TE inputs (te1, tefd)"
[ -f oof/nn_e12_emb16_h512-256-128_d0.2_te1+tefd+te2_orig.npy ] || python src/nn.py orig=1 epochs=12 emb=16 te=te1,tefd,te2 note="MLP 12 ep + TE inputs (te1, tefd, te2)"
