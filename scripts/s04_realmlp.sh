#!/bin/bash
# Submission 4 work: RealMLP (pytabkit) members. Restart-safe (per-fold checkpoints, done runs skipped).
cd "$(dirname "$0")/.."
python src/realmlp.py feats=pub epochs=3 note="RealMLP public recipe (raw + categorical twins), 3 epochs, n_ens 8"
python src/realmlp.py feats=v4 epochs=6 te=te1,tefd note="RealMLP v4: + route profile, counts, original-model logit, in-fold TE (te1, tefd), 6 epochs"
python src/realmlp.py feats=pub epochs=6 seed=1 note="RealMLP public recipe, 6 epochs, seed 1"
python src/realmlp.py feats=v4 epochs=6 te=te1,tefd seed=1 note="RealMLP v4, 6 epochs, seed 1"
