#!/usr/bin/env bash
# Experiment A: does the conventional lead survive a methane-only reward?
#
# The conventional envelope selects on methane alone, while the reported
# reward also charges a feed-rate and a flow-smoothness term.  This retrains
# a representative set with those two terms zeroed (ADM1_METHANE_ONLY=1) and
# the safety channel untouched: same cost signal, same cost limits, same
# seeds, so the only change is the part of the objective the envelope does
# not share.
#
#   bash scripts/run_methane_only.sh <outdir>
set -u
OUT="${1:?usage: run_methane_only.sh <outdir>}"
PY="${PY_SAFE:-python3}"
cd "$(dirname "$0")/.."

# Two on-policy and two off-policy families, spanning the dual rules.
ALGOS=(CPO PPOLag SACLag TD3Lag)
LIMITS=(0.6 1.8 3.0 6.0)
SEEDS=(1 2 3)
JOBS=16          # 24 cores; the rollout is single-threaded ADM1 integration

mkdir -p "$OUT"
for a in "${ALGOS[@]}"; do
  for c in "${LIMITS[@]}"; do
    for s in "${SEEDS[@]}"; do
      echo "$a $c $s"
    done
  done
done | xargs -P "$JOBS" -n 3 bash -c '
  a=$0; c=$1; s=$2
  mkdir -p '"$OUT"'/"$a"_c"$c"_s"$s"
  ADM1_METHANE_ONLY=1 '"$PY"' training/train_omnisafe.py \
    --algo "$a" --cost-limit "$c" --seed "$s" \
    --output-dir '"$OUT"'/"$a"_c"$c"_s"$s" > '"$OUT"'/"$a"_c"$c"_s"$s".log 2>&1
  echo "done $a c=$c seed=$s"
'
echo "ALL DONE"
