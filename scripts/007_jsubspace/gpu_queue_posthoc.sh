#!/bin/bash
# 007 GPU queue (execution note §20: no idle GPU between stages). Runs the commands in gpu_queue.txt one after another;
# each finished command is appended to gpu_queue_done.txt with its exit code and seconds. A failing command does not
# stop the queue (rule 12: the next independent stage runs; dependencies are checked inside each stage's script).
cd /workspace/causal-concordance
Q=outputs/007_jsubspace/gpu_queue_posthoc.txt; DONE=outputs/007_jsubspace/gpu_queue_posthoc_done.txt
touch $Q $DONE
while true; do
  n_done=$(wc -l < $DONE); total=$(wc -l < $Q)
  if [ "$n_done" -lt "$total" ]; then
    cmd=$(sed -n "$((n_done+1))p" $Q)
    echo "[$(date -u +%H:%M:%S)] start: $cmd"
    start=$(date +%s); bash -c "$cmd"; rc=$?
    echo "$cmd|rc=$rc|$(( $(date +%s)-start ))s|end=$(date -u +%H:%M:%S)" >> $DONE
    echo "[$(date -u +%H:%M:%S)] done rc=$rc: $cmd"
  else
    sleep 15
  fi
done
