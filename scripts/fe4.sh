#!/bin/bash
# Round 4: extras on top of the current best set FS1 = base,te1,tefd,fdprof,cnt,inter + original rows.
cd "$(dirname "$0")/.."
FS1=base,te1,tefd,fdprof,cnt,inter
python src/train.py lgbm $FS1,ofd orig=1 note="FS1 + ofd (original per-FD profile)"
python src/train.py lgbm $FS1,omean orig=1 note="FS1 + omean"
python src/train.py lgbm $FS1,te2 orig=1 note="FS1 + TE of all low-card column pairs"
