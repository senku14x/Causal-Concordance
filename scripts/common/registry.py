"""Model and tokenizer loading and the project pins, from registry.json (ported from RHUT common/scripts/registry.py).

Every scientific script loads through this module so the revision, dtype, attention implementation and thinking flag
are asserted in one place. Lens files are loaded by readers.py (reader code), never here: label code imports this
module.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
REGISTRY = json.loads((HERE / "registry.json").read_text())

os.environ.setdefault("HF_HOME", REGISTRY["paths"]["hf_home"])
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL = REGISTRY["model"]["repo"]
REV = REGISTRY["model"]["revision"]
N_BLOCKS = REGISTRY["model"]["n_blocks"]
D_MODEL = REGISTRY["model"]["d_model"]
N_SRC = N_BLOCKS - 1  # lens source layers 0..62


def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()


def make_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=REV)


def load_model(attn_impl=None, device="cuda"):
    """bf16 model, eager attention (the RHUT-validated regime), eval mode."""
    import torch
    from transformers import AutoModelForCausalLM
    attn_impl = attn_impl or REGISTRY["model"]["attn_implementation"]
    m = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map=device,
                                             attn_implementation=attn_impl)
    m.eval().requires_grad_(False)
    assert len(blocks(m)) == N_BLOCKS, f"expected {N_BLOCKS} blocks, got {len(blocks(m))}"
    assert m.config.get_text_config().hidden_size == D_MODEL
    return m


def blocks(model):
    """The decoder block list, also through a PEFT wrapper."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model.layers


def final_norm(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model.norm


def layer_types(model):
    tc = model.config.get_text_config()
    lt = list(getattr(tc, "layer_types", None) or ["full_attention"] * N_BLOCKS)
    assert len(lt) == N_BLOCKS
    return lt


def full_attention_layers(model):
    return [i for i, t in enumerate(layer_types(model)) if t == "full_attention"]


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty():
    try:
        return bool(subprocess.check_output(["git", "-C", str(PROJECT), "status", "--porcelain"], text=True).strip())
    except Exception:
        return None


def manifest_block():
    """Immutable pins for run manifests."""
    import torch
    import transformers
    try:                                     # version string only; importing peft here would pull reader code
        from importlib.metadata import version
        peft_v = version("peft")
    except Exception:
        peft_v = None
    return {"model": REGISTRY["model"], "lenses": REGISTRY["lenses"], "readers": REGISTRY["readers"],
            "wsb": REGISTRY["wsb"],
            "software": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_v,
                         "jlens_commit": REGISTRY["software"]["jlens_commit"],
                         "cuda": torch.version.cuda},
            "git": {"commit": git_commit(), "dirty": git_dirty()}}


def out_dir(*parts):
    d = PROJECT.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
