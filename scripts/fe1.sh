#!/bin/bash
# Feature-group ablations, LightGBM std preset, lr 0.1, early stopping. Each adds one group to base.
cd "$(dirname "$0")/.."
run() { python src/train.py lgbm "$1" note="$2"; }
run base "LGBM std preset (63 leaves, ES) on raw columns: reference for ablations"
run base,afill "+ arrival delay NaN filled from departure delay"
run base,delay "+ delay sum/diff/max/log/ratio/flags"
run base,rnan "+ rating 0 -> NaN, zero count"
run base,zflag "+ per-rating zero flags"
run base,agg "+ rating mean/min/max/std/sum/n5/n1"
run base,grp "+ digital/cabin/service group means and diffs"
run base,inter "+ CustomerType x TravelType x Class interactions (categorical)"
run base,cnt "+ value counts (distance, age, delays)"
run base,rcat "+ ratings duplicated as categoricals"
run base,te1 "+ in-fold target encoding of every column"
run base,tecat "+ in-fold TE of CT x TT x Class and its pairs with each rating/age"
run base,omean "+ original-data satisfaction rate per value"
python src/train.py lgbm base orig=1 note="+ original rows in every training fold (is_orig flag)"
