#!/bin/bash
# Push the XGB kernel as soon as one of the two GPU sessions is free (Kaggle allows 2 at a time).
cd "$(dirname "$0")/.."
while true; do
  busy=0
  for k in s6e10-s06-realmlp-v5a s6e10-s06-realmlp-v5b; do
    kaggle kernels status hasancmert/$k 2>&1 | grep -qiE "running|queued" && busy=$((busy+1))
  done
  if [ $busy -lt 2 ]; then
    python kaggle/gpu_kernel.py push s6e10-s06-xgb \
      "python src/train.py xgb base,te1,tefd,fdprof,cnt,inter,te2,cnt2,opred,te3 orig=1 preset=t1 lr=0.03 folds=10 device=cuda note='10-fold: FS3 + te3 XGB t1 lr0.03 (Kaggle GPU)'" \
      "python src/train.py xgb base,te1,tefd,fdprof,cnt,inter,te2,cnt2,opred orig=1 preset=t1 lr=0.03 folds=10 seed=3 device=cuda note='10-fold: FS3 XGB t1 lr0.03 seed 3 (Kaggle GPU)'" && break
  fi
  sleep 120
done
echo "pushed at $(date -u)"
