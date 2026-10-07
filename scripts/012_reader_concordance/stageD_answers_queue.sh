#!/bin/bash
# 012 add-on (EXPLORATORY): model's own read-position output (model side) -> reader answer LLs (subject) -> NLA.
# On Stage B's saved captures; needs stageBC_queue.sh <mode> done first.
#   bash stageD_answers_queue.sh smoke|full
set -eo pipefail
MODE=${1:?smoke|full}
cd /workspace/causal-concordance
export HF_HOME=/workspace/.hf_home PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src
PY=/venv/main/bin/python
S=scripts/012_reader_concordance
L=outputs/012_reader_concordance/logs
case $MODE in smoke) BA="--limit 4 --tag smoke"; CA="--tag smoke"; G=model_output_parity_smoke.json;;
              full)  BA="--tag full"; CA="--tag full"; G=model_output_parity.json;;
              *) echo "smoke|full"; exit 2;; esac
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L/D_answers_queue_$MODE.log; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "GPU busy ($used MiB used); abort"; exit 1; }
step() {
  local n=$1; shift
  local done=$L/D_${MODE}_$n.done
  [ -f $done ] && { log "skip $n (done)"; return; }
  log "start $n"; local t=$(date +%s)
  "$@" > $L/D_${MODE}_$n.log 2>&1
  log "done $n ($(( $(date +%s) - t ))s)"; touch $done
}
step model_output $PY $S/stageB_model_output.py $BA
$PY -c "import json,sys; g=json.load(open('results/012_reader_concordance/analysis/$G')); print('parity', g); sys.exit(0 if g['pass'] else 1)" \
  | tee -a $L/D_answers_queue_$MODE.log
step answers_subject $PY $S/stageC_answers.py --part subject $CA
step answers_nla $PY $S/stageC_answers.py --part nla $CA
log "D answers $MODE DONE"
