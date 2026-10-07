#!/bin/bash
# 012 Stage I (design §18, EXPLORATORY): stageI_capture (model side, gates) -> readers (subject) -> NLA.
# Gates: G0, write masks, prefix tokenization (hold-state identity is recorded, not gated).
#   bash stageI_queue.sh smoke|full
set -eo pipefail
MODE=${1:?smoke|full}
cd /workspace/causal-concordance
export HF_HOME=/workspace/.hf_home PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src
PY=/venv/main/bin/python
S=scripts/012_reader_concordance
L=outputs/012_reader_concordance/logs
case $MODE in smoke) BA="--clues 2 --tag smoke"; CA="--tag smoke"; G=stageI_gates_smoke.json;;
              full)  BA="--tag full"; CA="--tag full"; G=stageI_gates.json;;
              *) echo "smoke|full"; exit 2;; esac
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L/I_queue_$MODE.log; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "GPU busy ($used MiB used); abort"; exit 1; }
step() {
  local n=$1; shift
  local done=$L/I_${MODE}_$n.done
  [ -f $done ] && { log "skip $n (done)"; return; }
  log "start $n"; local t=$(date +%s)
  "$@" > $L/I_${MODE}_$n.log 2>&1
  log "done $n ($(( $(date +%s) - t ))s)"; touch $done
}
step capture $PY $S/stageI_capture.py $BA
$PY -c "import json,sys; g=json.load(open('results/012_reader_concordance/analysis/$G')); print('gates', {k: v for k, v in g.items()}); sys.exit(0 if g['pass'] else 1)" \
  | tee -a $L/I_queue_$MODE.log
step readers_subject $PY $S/stageI_readers.py --part subject $CA
step readers_nla $PY $S/stageI_readers.py --part nla $CA
log "I $MODE DONE"
