#!/bin/bash
# 011 GPU queue: waits for the 009 reader-free chain (QUEUE_DONE), then AR vectors (v1), then the swaps (s1) if the AR
# validation accepted a convention.
set -eo pipefail
cd /workspace/causal-concordance
PY=/venv/main/bin/python
LOG=outputs/011_olens_swaps/queue.log
mkdir -p outputs/011_olens_swaps
echo "[$(date -u +%H:%M:%S)] waiting for 009 chain" >> $LOG
until [ -f outputs/009_brewcode/QUEUE_DONE ]; do sleep 20; done
sleep 20
echo "[$(date -u +%H:%M:%S)] AR vectors" >> $LOG
$PY scripts/011_olens_swaps/ar_vectors.py --run-id v1 >> outputs/011_olens_swaps/ar_v1.out 2>&1
ACC=$($PY -c "import json;print(json.load(open('results/011_olens_swaps/ar_validation.json'))['accepted'])")
echo "[$(date -u +%H:%M:%S)] AR accepted: $ACC" >> $LOG
[ "$ACC" = "True" ] || { echo "no accepted convention; stop" >> $LOG; exit 1; }
echo "[$(date -u +%H:%M:%S)] swaps" >> $LOG
$PY scripts/011_olens_swaps/olens_swaps.py --vec-run v1 --run-id s1 >> outputs/011_olens_swaps/swaps_s1.out 2>&1
echo "[$(date -u +%H:%M:%S)] 011 done" >> $LOG
touch outputs/011_olens_swaps/DONE
