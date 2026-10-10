#!/bin/bash
# Rebuild this project's environment on a fresh instance (/workspace is NOT a persistent volume here).
# Sequential on purpose: packages -> jlens -> source clones -> artifacts (download + hash verify) -> CPU tests.
#   bash scripts/000_setup/rebuild_env.sh            # label + reader stack (/venv/main)
#   bash scripts/000_setup/rebuild_env.sh --vllm     # also the separate vLLM venv (/venv/vllm) for reader generation
# Private GitHub sources (RHUT, SEC) need `gh auth login` first; clone with `env -u GITHUB_TOKEN` if a stale token is set.
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "packages (/venv/main)"
uv pip install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "transformers==5.15.0" "peft==0.21.2" "accelerate==1.15.0" "safetensors==0.8.0" "tokenizers==0.22.2" \
    "huggingface_hub==1.30.0" "numpy==2.5.3" "scipy==1.18.1" "statsmodels==0.15.0" "scikit-learn==1.9.1" \
    "pandas==3.0.6" "pyarrow==25.0.1" "matplotlib==3.11.2" "PyYAML==6.0.3" "pytest==9.1.1"

log "jacobian-lens (jlens) @ 581d398"
[ -d /workspace/jacobian-lens ] || git clone -q https://github.com/anthropics/jacobian-lens /workspace/jacobian-lens
(cd /workspace/jacobian-lens && git checkout -q 581d398 && uv pip install -q -e . --no-deps)

log "source clones (pinned commits)"
mkdir -p /workspace/sources/repos
clone() {  # url dir commit
  [ -d "$2" ] || git clone -q "$1" "$2"
  git -C "$2" fetch -q origin "$3" 2>/dev/null || true
  git -C "$2" checkout -q "$3"
}
clone https://github.com/camilablank/workspace-bench          /workspace/sources/repos/workspace-bench 92d763e722377f5bfae045e829a308ab5919a820
clone https://github.com/senku14x/Readable-Here-Used-There     /workspace/sources/repos/RHUT            d4197b00150954dcf96fc4977898f6f3b8eb902a
clone https://github.com/senku14x/jacobian-lens                /workspace/sources/repos/jacobian-lens   a3720ab424d70fdc579ab74e5c8b8cd10fe421a9
clone https://github.com/asherps/EasyNLA                       /workspace/sources/repos/EasyNLA         4d728477960c18cdfa36dc04ec738d7f55af9f0b
clone https://github.com/ceselder/vllm-metamodels              /workspace/sources/repos/vllm-metamodels 6a0a0e56a81d4c234346575238ee5fe8f580f332
# Jan Chen's phrase-lens (012's J-row swap and random-target construction; read-only, never installed). Added
# 2026-10-10 (claude_ovarida 40, T4). It lives under sources/ext/, the path CLAUDE.md and 012 use.
mkdir -p /workspace/sources/ext
clone https://github.com/chen-jan/phrase-lens                  /workspace/sources/ext/phrase-lens-src   ea85583341b2daf74247d4f60a93f75ec6d05156

log "artifacts: model, lenses, reader checkpoints (pinned revisions), then sha256 verification"
python "$PROJ/scripts/common/artifacts.py" download --what model,lenses,readers
python "$PROJ/scripts/common/artifacts.py" verify --out "$PROJ/results/000_setup/analysis/artifact_hashes.json"

if [ "${1:-}" = "--vllm" ]; then
  log "vLLM venv (/venv/vllm)"
  bash "$PROJ/scripts/000_setup/install_vllm_venv.sh"
fi

log "CPU tests"
python -m pytest "$PROJ/scripts/common/tests" -m "not gpu" -q -p no:cacheprovider
log "done; GPU tests: python -m pytest scripts/common/tests -m gpu -q"
