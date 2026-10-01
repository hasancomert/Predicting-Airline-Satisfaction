#!/bin/bash
# Round 2: combine the round-1 winners (te1, cnt, omean, inter, orig) and Flight-Distance ideas.
cd "$(dirname "$0")/.."
run() { python src/train.py lgbm "$@"; }
run base,te1,cnt note="te1 + cnt"
run base,te1,omean note="te1 + omean"
run base,te1,cnt,omean note="te1 + cnt + omean"
run base,te1,inter note="te1 + inter"
run base,te1,tefd note="te1 + TE of FlightDistance x every other column"
run base,te1,fdprof note="te1 + label-free per-FlightDistance profile (train+test)"
run base,te1,ofd note="te1 + per-FlightDistance profile and count from original data"
run base,te1 orig=1 note="te1 + original rows in training folds"
