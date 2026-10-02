#!/bin/bash
# Separate venv for vllm-metamodels: READER GENERATION ONLY, never the label stack (its torch build differs).
# vLLM 0.21.0 + transformers 5.5.4 is the stack the nla-L42 NLA was RL-trained with (model card).
set -euo pipefail
export UV_CACHE_DIR=/workspace/.uv_cache
uv venv /venv/vllm --python 3.12
uv pip install --python /venv/vllm/bin/python "vllm==0.21.0"
uv pip install --python /venv/vllm/bin/python --no-deps "git+https://github.com/ceselder/vllm-metamodels@6a0a0e56a81d4c234346575238ee5fe8f580f332"
uv pip install --python /venv/vllm/bin/python "zstandard>=0.23.0" "pydantic>=2.0" "datasets>=4.0.0" pyyaml pandas pyarrow
OV=$(mktemp); echo "transformers==5.5.4" > "$OV"
uv pip install --python /venv/vllm/bin/python --override "$OV" "transformers==5.5.4"
/venv/vllm/bin/python -c "import torch, vllm, transformers, vllm_lens; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__)"
