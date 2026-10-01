#!/bin/bash
# Round 3: noise check (model seeds) and combinations of the round-2 winners.
cd "$(dirname "$0")/.."
run() { python src/train.py lgbm "$@"; }
run base,te1 seed=1 note="te1, model seed 1 (noise check)"
run base,te1 seed=2 note="te1, model seed 2 (noise check)"
run base,te1,ofd note="te1 + per-FlightDistance profile and count from original data"
run base,te1,tefd,fdprof note="te1 + tefd + fdprof"
run base,te1,tefd,fdprof,cnt,inter note="te1 + tefd + fdprof + cnt + inter"
run base,te1,tefd,fdprof orig=1 note="te1 + tefd + fdprof + original rows"
run base,te1,tefd,fdprof,cnt,inter orig=1 note="te1 + tefd + fdprof + cnt + inter + original rows"
