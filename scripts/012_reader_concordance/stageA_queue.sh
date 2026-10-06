#!/bin/bash
# 012 Stage A: reference lever with Jan's phrase-lens code, UNMODIFIED (read-only clone at ea85583), on our instance.
# Model-side only; no reader is involved. EXPLORATORY.
#   bash stageA_queue.sh smoke   # build J/logit token banks + a 4-item J swap at L24-47; then STOP
#   bash stageA_queue.sh full    # every condition below (PI go after the smoke)
set -eo pipefail
MODE=${1:?smoke|full}
cd /workspace/causal-concordance
export HF_HOME=/workspace/.hf_home PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src
PY="/venv/main/bin/python -m phrase_lens"
COMMON="--model Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --local-files-only --device cuda --dtype bfloat16"
JNP=/workspace/.hf_home/hub/models--neuronpedia--jacobian-lens/snapshots/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt
D=outputs/012_reader_concordance/stageA
mkdir -p $D/runs $D/logs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $D/logs/queue.log; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a $D/logs/queue.log

[ -d $D/bank_j ] || { log "build J bank"; $PY build-token $COMMON --vocabulary $D/words_st.json --jacobians $JNP --out $D/bank_j > $D/logs/build_j.log 2>&1; }
[ -d $D/bank_logit ] || { log "build logit bank"; $PY build-token $COMMON --vocabulary $D/words_st.json --out $D/bank_logit > $D/logs/build_logit.log 2>&1; }

run() {  # name bank arm benchmark condition first last [extra]
  local f=$D/runs/$1.jsonl
  [ -f $f ] && { log "skip $1 (exists)"; return; }
  log "start $1"; local t=$(date +%s)
  $PY causal $COMMON --benchmark $4 --bank $D/$2 --arm $3 --condition $5 --layers $6 $7 --alpha 1 --cohort all ${8:-} \
      --out $f > $D/logs/$1.log 2>&1
  log "done $1 ($(( $(date +%s) - t ))s)"
}

if [ "$MODE" = smoke ]; then
  run smoke_j_swap_L24-47 bank_j j causal_st_multihop swap 24 47 "--limit 4"
  log "SMOKE DONE"; touch $D/SMOKE_DONE; exit 0
fi

# full: the lever, its band dependence (for downstream reader reads), controls and broadcast
run j_swap_L24-47      bank_j     j     causal_st_multihop swap     24 47
run j_ablation_L24-47  bank_j     j     causal_st_multihop ablation 24 47
run logit_swap_L24-47  bank_logit logit causal_st_multihop swap     24 47
run j_swap_L24-39      bank_j     j     causal_st_multihop swap     24 39
run j_swap_L24-35      bank_j     j     causal_st_multihop swap     24 35
run j_swap_L36-47      bank_j     j     causal_st_multihop swap     36 47
run j_swap_L40-48      bank_j     j     causal_st_multihop swap     40 48
run j_bcast_L24-47     bank_j     j     causal_st_broadcast swap    24 47
log "FULL DONE"; touch $D/FULL_DONE
