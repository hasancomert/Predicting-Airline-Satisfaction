#!/bin/bash
cd "$(dirname "$0")/.."
for s in 61 73 89; do
  [ -f oof/nn_e12_emb16_h512-256-128_d0.2_orig_s$s.npy ] || python src/nn.py orig=1 epochs=12 emb=16 seed=$s threads=3 note="MLP 12 ep, emb 16, seed $s"
done
