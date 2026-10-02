#!/bin/bash
# Cheap diverse members (no target encoding) run alongside the CatBoost queue.
cd "$(dirname "$0")/.."
python src/train.py lgbm base,cnt,inter,fdprof orig=1 preset=t1 lr=0.03 threads=2 note="no-TE LGBM (raw + counts + interactions + FD profile), t1, lr 0.03"
[ -f oof/glm_pairsall_fd1_C0.03.npy ] || OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/glm.py pairs=all C=0.03 note="Sparse LR, C 0.03"
python src/train.py xgb base,cnt,inter,fdprof orig=1 preset=t1 lr=0.03 threads=2 note="no-TE XGB, t1, lr 0.03"
