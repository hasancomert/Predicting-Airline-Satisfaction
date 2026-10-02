#!/bin/bash
# Extra plain-MLP seeds (seed averaging gave +0.0008 for the MLP). Runs alongside the GBDT queue.
cd "$(dirname "$0")/.."
for s in 11 23 37 51; do
  [ -f oof/nn_e12_emb16_h512-256-128_d0.2_orig_s$s.npy ] || python src/nn.py orig=1 epochs=12 emb=16 seed=$s threads=2 note="MLP 12 ep, emb 16, seed $s"
done
