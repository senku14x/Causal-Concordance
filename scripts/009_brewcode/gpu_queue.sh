#!/bin/bash
# 009 reader-free GPU queue (design 008/009 §2): waits for 008 Track B run b1 to finish (its final state.json carries
# "G_RA_pass"), then gate smoke -> full depth-2 gate -> §2.4 depth rule (depth-1 fallback) -> outfacing -> captures ->
# probes (CPU) -> profile. No readers (design §2.6). Stops on the first failure.
set -eo pipefail
cd /workspace/causal-concordance
PY=/venv/main/bin/python
LOG=outputs/009_brewcode/gpu_queue.log
mkdir -p outputs/009_brewcode
echo "[$(date -u +%H:%M:%S)] queue start; waiting for 008 b1" >> $LOG
until grep -q '"G_RA_pass"' outputs/008_trackb/b1/state.json 2>/dev/null; do
  if grep -q "Traceback" outputs/008_trackb/b1_stdout.txt 2>/dev/null; then echo "008 b1 crashed; queue stops" >> $LOG; exit 1; fi
  sleep 30
done
sleep 30
echo "[$(date -u +%H:%M:%S)] 008 done; gate smoke" >> $LOG
$PY scripts/009_brewcode/gate.py --depth 2 --run-id g2smoke --limit-tables 3 >> outputs/009_brewcode/gate_g2smoke.out 2>&1
echo "[$(date -u +%H:%M:%S)] full depth-2 gate" >> $LOG
$PY scripts/009_brewcode/gate.py --depth 2 --run-id g2 >> outputs/009_brewcode/gate_g2.out 2>&1
DEC=$($PY -c "import json;print(json.load(open('results/009_brewcode/analysis/gate_g2.json'))['decision'])")
echo "[$(date -u +%H:%M:%S)] g2 decision: $DEC" >> $LOG
GR=g2
if [ "$DEC" != "depth 2 ok" ]; then
  echo "[$(date -u +%H:%M:%S)] depth-1 fallback gate" >> $LOG
  $PY scripts/009_brewcode/gate.py --depth 1 --run-id g1 --max-partners 4 >> outputs/009_brewcode/gate_g1.out 2>&1
  DEC1=$($PY -c "import json;print(json.load(open('results/009_brewcode/analysis/gate_g1.json'))['decision'])")
  echo "[$(date -u +%H:%M:%S)] g1 decision: $DEC1" >> $LOG
  [ "$DEC1" = "ok" ] || { echo "depth 1 below the bar; queue stops" >> $LOG; exit 1; }
  GR=g1
fi
echo "[$(date -u +%H:%M:%S)] outfacing on $GR" >> $LOG
$PY scripts/009_brewcode/derisk.py outfacing --gate-run $GR --run-id o1 >> outputs/009_brewcode/outfacing_o1.out 2>&1
echo "[$(date -u +%H:%M:%S)] captures" >> $LOG
$PY scripts/009_brewcode/derisk.py captures --gate-run $GR --run-id c1 >> outputs/009_brewcode/captures_c1.out 2>&1
echo "[$(date -u +%H:%M:%S)] probes (CPU, in background) + profile (GPU)" >> $LOG
(CUDA_VISIBLE_DEVICES="" $PY scripts/009_brewcode/derisk.py probes --gate-run $GR --run-id c1 --n-jobs 32 \
   >> outputs/009_brewcode/probes_c1.out 2>&1; echo "[$(date -u +%H:%M:%S)] probes done" >> $LOG) &
$PY scripts/009_brewcode/derisk.py profile --gate-run $GR --run-id p1 --limit-pairs 48 >> outputs/009_brewcode/profile_p1.out 2>&1
wait
echo "[$(date -u +%H:%M:%S)] queue done" >> $LOG
touch outputs/009_brewcode/QUEUE_DONE
