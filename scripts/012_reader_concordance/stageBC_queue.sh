#!/bin/bash
# 012 Stage B-lite -> gates -> Stage C pilot (Oracle + token lenses) -> subject freed -> NLA. EXPLORATORY (design §12).
#   bash stageBC_queue.sh smoke   # 4 items end to end (outputs under stageB/smoke, stageC/smoke; *_smoke gates)
#   bash stageBC_queue.sh full    # 87 covered items of causal_st_multihop
# Each stage is its own process (the subject and the NLA never share the GPU). Done-files make reruns resume.
set -eo pipefail
MODE=${1:?smoke|full}
cd /workspace/causal-concordance
export HF_HOME=/workspace/.hf_home PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src
PY=/venv/main/bin/python
S=scripts/012_reader_concordance
L=outputs/012_reader_concordance/logs
mkdir -p $L
case $MODE in smoke) ARGS="--limit 4 --tag smoke"; CARGS="--tag smoke"; G=stageB_gates_smoke.json;;
              full)  ARGS="--tag full"; CARGS="--tag full"; G=stageB_gates.json;;
              *) echo "smoke|full"; exit 2;; esac
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L/BC_queue_$MODE.log; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "GPU busy ($used MiB used); abort"; exit 1; }

step() {  # name cmd...
  local n=$1; shift
  local done=$L/BC_${MODE}_$n.done
  [ -f $done ] && { log "skip $n (done)"; return; }
  log "start $n"; local t=$(date +%s)
  "$@" > $L/BC_${MODE}_$n.log 2>&1
  log "done $n ($(( $(date +%s) - t ))s)"; touch $done
}

step stageB $PY $S/stageB_capture.py $ARGS
$PY - <<EOF | tee -a $L/BC_queue_$MODE.log
import json, sys
g = json.load(open("results/012_reader_concordance/analysis/$G"))
ok = {k: g[k]["pass"] for k in g if k.startswith("G")}
print("gates:", ok, "read-position mismatches:", len(g["read_position"]["mismatches"]))
sys.exit(0 if all(ok.values()) else 1)
EOF
step stageC_readers $PY $S/stageC_readers.py $CARGS
step stageC_nla $PY $S/stageC_nla.py $CARGS
(cd $S && $PY stageBC_summary.py $CARGS) | tee -a $L/BC_queue_$MODE.log
log "BC $MODE DONE"; touch $L/BC_${MODE}_ALL.done
