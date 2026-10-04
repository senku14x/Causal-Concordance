"""nla-rl AV reader (readers.NLArl): WSB's injection arithmetic, registry pins, hash verification, the GPU-room guard and
the prompt. CPU tests need no weights (the prompt test needs the av_base tokenizer files and skips without them)."""
import os
import re

import pytest
import torch

import readers as RD
import registry as R


def _wsb_hook(h, v, mp):
    """WSB produce/methods.py NLA.bind hook body, verbatim except for the closure (v from self._vec)."""
    if v is None or h.shape[1] <= mp:
        return h
    hp = h[:, mp, :]
    vu = v.to(h.dtype).to(h.device) / (v.norm() + 1e-9)
    h[:, mp, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
    return h


def _h(B=3, T=10, d=64, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, T, d, generator=g) * 3).to(dtype)


def test_add_matches_wsb_bitwise():
    h = _h()
    v = torch.randn(64, generator=torch.Generator().manual_seed(1))
    ours, theirs = h.clone(), h.clone()
    assert RD.karvonen_add_(ours, v, 5)
    _wsb_hook(theirs, v, 5)
    assert torch.equal(ours, theirs)
    assert torch.equal(ours[:, :5], h[:, :5]) and torch.equal(ours[:, 6:], h[:, 6:])   # only the marker moves


def test_add_geometry_float32():
    h = _h(dtype=torch.float32)
    v = torch.randn(64, generator=torch.Generator().manual_seed(2))
    out = h.clone()
    RD.karvonen_add_(out, v, 4)
    d = out[:, 4] - h[:, 4]
    torch.testing.assert_close(d.norm(dim=-1), h[:, 4].norm(dim=-1), rtol=1e-5, atol=0)   # ||added|| = ||h||
    cos = torch.nn.functional.cosine_similarity(d, v.expand_as(d), dim=-1)
    assert float(cos.min()) > 1 - 1e-6


def test_per_row_and_no_activation_rows():
    h = _h()
    g = torch.Generator().manual_seed(3)
    V = torch.randn(3, 64, generator=g)
    V[1] = 0.0                                                    # no-activation row
    out = h.clone()
    RD.karvonen_add_(out, V, 5)
    assert torch.equal(out[1], h[1])
    for b in (0, 2):
        ref = h[b:b + 1].clone()
        _wsb_hook(ref, V[b], 5)
        assert torch.equal(out[b], ref[0])


def test_decode_steps_and_none_are_noops():
    h = _h(T=1)
    assert not RD.karvonen_add_(h, torch.ones(64), 5)
    h2 = _h()
    before = h2.clone()
    assert not RD.karvonen_add_(h2, None, 5)
    assert torch.equal(h2, before)


def test_wsb_postprocess():
    assert RD.wsb_explanation("<explanation> a b </explanation> tail") == "a b"
    assert RD.wsb_explanation("no tags at all ") == "no tags at all"


def test_registry_pins():
    n = R.REGISTRY["readers"]["nla_rl"]
    assert n["hf_repo"] == "ceselder/qwen3.6-27b-nla-rl" and re.fullmatch(r"[0-9a-f]{40}", n["hf_revision"])
    assert n["adapter"] == "av_rl_adapters/iter_000400" and n["layer"] == 42
    assert all(re.fullmatch(r"[0-9a-f]{64}", s) for s in n["sha256"].values())
    assert n["sha256"]["av_rl_adapters/iter_000400/adapter_model.safetensors"].startswith("b54e23e4")
    shards = [r for r in n["sha256"] if r.startswith("av_base/model-") and r.endswith(".safetensors")]
    assert len(shards) == 2 and n["meta"] in n["sha256"]
    assert n["contract"]["marker_token_id"] == 158983 and n["contract"]["marker_neighbors"] == [29, 510]


def test_verify_detects_mismatch_and_missing(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    entry = {"sha256": {"a.bin": R.sha256(tmp_path / "a.bin")}}
    assert RD.verify_nla_rl(str(tmp_path), entry) == entry["sha256"]
    with pytest.raises(RuntimeError, match="mismatch"):
        RD.verify_nla_rl(str(tmp_path), {"sha256": {"a.bin": "0" * 64}})
    with pytest.raises(FileNotFoundError):
        RD.verify_nla_rl(str(tmp_path), {"sha256": {"b.bin": "0" * 64}})


def test_gpu_room_guard(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (40e9, 96e9))   # subject still resident
    with pytest.raises(RuntimeError, match="capture subject activations first"):
        RD.check_gpu_room("cuda", int(55.7e9))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (95e9, 96e9))
    RD.check_gpu_room("cuda", int(55.7e9))
    RD.check_gpu_room("cpu", int(1e15))                                               # no check off-GPU


def test_prompt_ids_match_contract_and_subject_tokenizer():
    import yaml
    from transformers import AutoTokenizer
    root, n = RD.nla_rl_paths()
    base = os.path.join(root, n["av_base"])
    if not os.path.exists(os.path.join(base, "tokenizer.json")):
        pytest.skip("av_base tokenizer not downloaded")
    with open(os.path.join(root, n["meta"])) as f:
        meta = yaml.safe_load(f)
    ids, marker = RD.nla_rl_prompt_ids(AutoTokenizer.from_pretrained(base), meta)
    assert ids[marker] == 158983 and ids[marker - 1] == 29 and ids[marker + 1] == 510
    assert RD.nla_rl_prompt_ids(R.make_tokenizer(), meta) == (ids, marker)            # same ids either tokenizer


@pytest.mark.gpu
def test_reader_loads_marks_and_injects():
    """Needs the weights and ~58 GB free on the GPU with no subject loaded."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    nla = RD.NLArl()
    try:
        assert nla.scaling == [2.0]                                                   # rsLoRA: 16 / sqrt(64)
        with pytest.raises(RuntimeError, match="modified in place"):
            R.assert_subject_model_intact(nla.model)
        acts, _ = RD.nla_reference_activations()
        a = nla.read(acts[0], k=2, seed=0)
        b = nla.read(acts[0], k=2, seed=0)
        assert a == b and nla.injections == 2
        assert nla.last_write["rows"] == 2 and nla.last_write["cos_min"] > 0.99
        assert 0.97 < nla.last_write["rel_min"] <= nla.last_write["rel_max"] < 1.03
        none = nla.read(None, k=2, seed=0)
        assert nla.injections == 2 and none != a
    finally:
        nla.close()
