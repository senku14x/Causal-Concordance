#!/bin/bash
# 009 §E-009 early-stop rule: after >= 30 depth-2 tables, if the Wilson 95% upper bound of the pair pass share < 0.25,
# stop the depth-2 gate (and the queue waiting on it) and run the depth-1 chain. Otherwise do nothing.
set -eo pipefail
cd /workspace/causal-concordance
PY=/venv/main/bin/python
LOG=outputs/009_brewcode/gpu_queue.log
until [ "$(ls outputs/009_brewcode/gate/g2/*.jsonl 2>/dev/null | wc -l)" -ge 30 ]; do sleep 15; done
$PY scripts/009_brewcode/gate.py --depth 2 --run-id g2 --summarize-only > /dev/null 2>&1
UB=$($PY - <<'PY'
import json, math
s = json.load(open("results/009_brewcode/analysis/gate_g2.json"))
n, k = s["pairs"]["candidate"], s["pairs"]["in"]
z = 1.959964
p = k / n
den = 1 + z * z / n
ub = (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
print(f"{ub:.4f} {k} {n} {s['n_tables_gated']}")
PY
)
echo "[$(date -u +%H:%M:%S)] early-stop check: wilson_ub k n tables = $UB" >> $LOG
read ub k n t <<< "$UB"
if $PY -c "import sys; sys.exit(0 if float('$ub') < 0.25 else 1)"; then
  echo "[$(date -u +%H:%M:%S)] early stop: depth 2 below the bar; stopping g2 and the queue" >> $LOG
  QPID=$(ps -eo pid,args | awk '/bash scripts\/009_brewcode\/[g]pu_queue.sh/ {print $1}')
  GPID=$(ps -eo pid,args | awk '/[g]ate.py --depth 2 --run-id g2$/ {print $1}')
  [ -n "$QPID" ] && kill $QPID || true
  [ -n "$GPID" ] && kill $GPID || true
  sleep 20
  echo "[$(date -u +%H:%M:%S)] depth-1 gate (g1, max 4 partners)" >> $LOG
  $PY scripts/009_brewcode/gate.py --depth 1 --run-id g1 --max-partners 4 >> outputs/009_brewcode/gate_g1.out 2>&1
  DEC1=$($PY -c "import json;print(json.load(open('results/009_brewcode/analysis/gate_g1.json'))['decision'])")
  echo "[$(date -u +%H:%M:%S)] g1 decision: $DEC1" >> $LOG
  [ "$DEC1" = "ok" ] || { echo "depth 1 below the bar; stop" >> $LOG; exit 1; }
  echo "[$(date -u +%H:%M:%S)] outfacing on g1" >> $LOG
  $PY scripts/009_brewcode/derisk.py outfacing --gate-run g1 --run-id o1 >> outputs/009_brewcode/outfacing_o1.out 2>&1
  echo "[$(date -u +%H:%M:%S)] captures" >> $LOG
  $PY scripts/009_brewcode/derisk.py captures --gate-run g1 --run-id c1 >> outputs/009_brewcode/captures_c1.out 2>&1
  echo "[$(date -u +%H:%M:%S)] probes (CPU, bg) + profile" >> $LOG
  (CUDA_VISIBLE_DEVICES="" $PY scripts/009_brewcode/derisk.py probes --gate-run g1 --run-id c1 --n-jobs 32 \
     >> outputs/009_brewcode/probes_c1.out 2>&1; echo "[$(date -u +%H:%M:%S)] probes done" >> $LOG) &
  $PY scripts/009_brewcode/derisk.py profile --gate-run g1 --run-id p1 --limit-pairs 48 >> outputs/009_brewcode/profile_p1.out 2>&1
  wait
  echo "[$(date -u +%H:%M:%S)] depth-1 chain done" >> $LOG
  touch outputs/009_brewcode/QUEUE_DONE
else
  echo "[$(date -u +%H:%M:%S)] no early stop; g2 continues" >> $LOG
fi
