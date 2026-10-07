#!/bin/bash
# 014 full run (EXPLORATORY; PI go 2026-10-07). Parts A/B on all 100 typo items, then Part C up to scoring.
# Part C analysis runs only after the hand check of the judge output (013 rule G2), so it is not chained here.
set -eo pipefail
ROOT=/workspace/causal-concordance
PY=/venv/main/bin/python
L=$ROOT/outputs/014_typo
mkdir -p $L
cd $ROOT/scripts/014_typo
$PY typo.py subject > $L/subject.log 2>&1
$PY typo.py nla > $L/nla.log 2>&1
$PY typo.py analyze > $L/analyze.log 2>&1
echo "[run] parts A/B done"
cd $ROOT/scripts/013_nla_hallucination
export NLAH_FAMILY=typo NLAH_TYPO_TAG=full
set -a; . /workspace/.env; set +a
$PY hallucination.py gate > $L/gate.log 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$ROOT/results/014_typo/analysis/nlah_gate_full.json'))['R1_pass'] else 1)"; then
  echo "[run] gate R1 FAILED: Part C stops (spec §5)"; exit 0
fi
echo "[run] gate R1 passed"
$PY hallucination.py judge --subset all > $L/judge.log 2>&1
$PY hallucination.py handset --subset all > /dev/null
$PY hallucination.py score --subset all > $L/score.log 2>&1
$PY rewrite.py judge --subset all > $L/rewrite_judge.log 2>&1
$PY rewrite.py handset --subset all > /dev/null
$PY rewrite.py score --subset all > $L/rewrite_score.log 2>&1
echo "[run] part C scored; analysis waits for the hand check"
