"""Block c addendum: can vllm-metamodels serve the NLA-L42 verbalizer equivalently to the HF reference, and how much
faster? Reader engineering only (no label, no outcome). Test vectors: layer-42 block outputs (stock bf16 capture) at
WSB brew cells of the two items already used in block 000 (brew-s2-r10d2-0000 / -0001: emit_asst, emit_think,
emit_colon, last question token).

    /venv/main/bin/python scripts/000_setup/vllm_nla_check.py hf        # HF reference: vectors, residual, logprobs, samples
    /venv/vllm/bin/python scripts/000_setup/vllm_nla_check.py vllm      # vLLM 0.21 + vllm-metamodels, same inputs
    /venv/main/bin/python scripts/000_setup/vllm_nla_check.py compare   # AR FVE of both sample sets + agreement table

vLLM path: the SFT and RL rsLoRA adapters are stacked into one rank-256 adapter (B scaled by alpha/sqrt(r) each) and merged
into the served weights once (`merge_lora`, one bf16 rounding of W0 + dW_sft + dW_rl); the HF reference merges the SFT
adapter (bf16) and runs the RL adapter unmerged. Injection: SteeringVector(norm_match=True, scale=1.0) at layer 1 output
at the marker = h + ||h|| v/||v|| (the contract). Writes results/000_setup/analysis/vllm_nla_check.json.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import torch

import registry as R

CELLS = [("brew-s2-r10d2-0000", p) for p in (153, 158, 162, 164)] + [("brew-s2-r10d2-0001", p) for p in (153, 158, 162, 164)]
LAYER, K, SEED, TOPK = 42, 8, 20261002, 20
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=64, max_new_tokens=256)
OUT = R.out_dir("outputs", "000_setup", "vllm_nla_check")


def hf():
    import readers as RD
    import wsb
    from transformers import AutoTokenizer
    tok = R.make_tokenizer()
    model = R.load_model()
    _, bank = wsb.load_bank("brew")
    vecs = []
    for name, pos in CELLS:
        it = next(x for x in bank if x["name"] == name)
        r = wsb.render_item(tok, "brew", it)
        vecs.append(RD.capture_stock(model, r.ids, [LAYER], [pos])[LAYER][0])
    V = torch.stack(vecs)
    _, p, _ = RD.nla_paths()
    av = RD.NLAVerbalizer(model, AutoTokenizer.from_pretrained(p["av_sft_lora"]))
    (block1,) = [m for n, m in av.model.named_modules() if n.endswith("model.layers.1")]
    post = {}
    def grab(m, i, o):
        hh = o[0] if isinstance(o, tuple) else o
        if hh.shape[1] > av.marker:                     # prefill only; decode steps have length 1
            post["h"] = hh[0, av.marker].detach().float().cpu()
    hk = block1.register_forward_hook(grab)            # registered after the injection hook: sees the injected row
    res, samples, t_gen = [], [], 0.0
    x = torch.tensor([av.ids], device=model.device)
    for v in V:
        av._v["v"] = v.float()
        with torch.no_grad():
            lp = torch.log_softmax(av.model(x).logits[0, -1].float(), -1)
        av._v["v"] = None
        top = lp.topk(TOPK)
        res.append({"post_injection_resid": post["h"], "top_ids": top.indices.cpu(), "top_lp": top.values.cpu()})
        t0 = time.time()
        samples.append(av.read(v, k=K, seed=SEED, sampling=SAMPLING))
        t_gen += time.time() - t0
    hk.remove()
    torch.save({"cells": CELLS, "V": V, "ids": av.ids, "marker": av.marker, "res": res, "samples": samples,
                "gen_s": t_gen, "n_seqs": len(V) * K}, OUT / "hf.pt")
    print("hf done", t_gen, "s for", len(V) * K, "sequences")


def stacked_lora():
    """PEFT-style tensors for dW = s_sft B_sft A_sft + s_rl B_rl A_rl as one rank-256 adapter (scaling 1)."""
    from safetensors.torch import load_file
    import readers as RD
    _, p, _ = RD.nla_paths()
    out = {}
    parts = []
    for key in ("av_sft_lora", "av_rl_lora"):
        cfg = json.load(open(os.path.join(p[key], "adapter_config.json")))
        s = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
        parts.append((load_file(os.path.join(p[key], "adapter_model.safetensors")), s))
    mods = sorted({k.rsplit(".lora_", 1)[0] for sd, _ in parts for k in sd})
    for m in mods:
        A = [sd[f"{m}.lora_A.weight"].float() for sd, _ in parts if f"{m}.lora_A.weight" in sd]
        B = [sd[f"{m}.lora_B.weight"].float() * s for sd, s in parts if f"{m}.lora_B.weight" in sd]
        out[f"{m}.lora_A.weight"] = torch.cat(A, 0)
        out[f"{m}.lora_B.weight"] = torch.cat(B, 1)
    return out, len(mods)


def vllm():
    os.environ.setdefault("VLLM_LENS_CUDA_GRAPHS", "1")
    from vllm import LLM, SamplingParams
    from vllm_lens import SteeringVector
    from vllm_lens.metamodel import merge_lora
    hfres = torch.load(OUT / "hf.pt", weights_only=False)
    ids, marker, V = hfres["ids"], hfres["marker"], hfres["V"]
    t0 = time.time()
    llm = LLM(model=R.MODEL, revision=R.REV, dtype="bfloat16", language_model_only=True, max_model_len=512,
              max_num_seqs=64, gpu_memory_utilization=0.92, enable_prefix_caching=False, seed=SEED)
    t_load = time.time() - t0
    tensors, n_mod = stacked_lora()
    t0 = time.time()
    mstat = merge_lora(llm, tensors=tensors, scaling=1.0, keep_base="none")
    t_merge = time.time() - t0

    def sv(v):
        return SteeringVector(activations=v.float().view(1, 1, -1), layer_indices=[1], position_indices=[marker],
                              norm_match=True, scale=1.0)
    prompt = {"prompt_token_ids": list(ids)}
    # first-token logprobs + post-injection residual at layer 1 (marker)
    p1 = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOPK,
                         extra_args={"apply_steering_vectors": [sv(v)], "output_residual_stream": [1],
                                     "capture_positions": [marker]}) for v in V]
    o1 = llm.generate([prompt] * len(V), p1, use_tqdm=False)
    res = []
    for o in o1:
        lp = o.outputs[0].logprobs[0]
        tops = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:TOPK]
        act = getattr(o, "activations", None) or {}
        rs = act.get("residual_stream") if isinstance(act, dict) else None
        res.append({"top_ids": torch.tensor([t for t, _ in tops]), "top_lp": torch.tensor([x.logprob for _, x in tops]),
                    "post_injection_resid": (rs.reshape(-1, rs.shape[-1])[0].float() if rs is not None else None)})
    # samples: K per vector, one generate() call
    sp = [SamplingParams(n=K, temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"],
                         max_tokens=SAMPLING["max_new_tokens"], seed=SEED + i,
                         extra_args={"apply_steering_vectors": [sv(v)]}) for i, v in enumerate(V)]
    t0 = time.time()
    og = llm.generate([prompt] * len(V), sp, use_tqdm=False)
    t_gen = time.time() - t0
    samples = [[c.text for c in o.outputs] for o in og]
    n_tok = sum(len(c.token_ids) for o in og for c in o.outputs)
    # throughput at a larger batch (4x the vectors, K samples each)
    sp4 = [SamplingParams(n=K, temperature=1.0, top_p=0.95, top_k=64, max_tokens=256, seed=SEED + 100 + i,
                          extra_args={"apply_steering_vectors": [sv(V[i % len(V)])]}) for i in range(4 * len(V))]
    t0 = time.time()
    og4 = llm.generate([prompt] * len(sp4), sp4, use_tqdm=False)
    t_gen4 = time.time() - t0
    n_tok4 = sum(len(c.token_ids) for o in og4 for c in o.outputs)
    try:
        stats = llm.collective_rpc("steering_stats")[0]
    except Exception as e:
        stats = {"error": str(e)}
    torch.save({"res": res, "samples": samples, "load_s": t_load, "merge_s": t_merge, "merge": mstat, "n_lora_modules": n_mod,
                "gen_s": t_gen, "n_seqs": len(V) * K, "n_tokens": n_tok, "gen4_s": t_gen4, "n_seqs4": len(sp4) * K,
                "n_tokens4": n_tok4, "steering_stats": stats}, OUT / "vllm.pt")
    print("vllm done load", t_load, "merge", t_merge, "gen", t_gen, "gen4", t_gen4, "stats", stats)


def compare():
    import readers as RD
    h = torch.load(OUT / "hf.pt", weights_only=False)
    v = torch.load(OUT / "vllm.pt", weights_only=False)
    tok = R.make_tokenizer()
    ar = RD.NLAReconstructor()
    ref, _ = RD.nla_reference_activations()
    rows = []
    for i, cell in enumerate(h["cells"]):
        gold = h["V"][i]
        r = {"cell": cell}
        a, b = h["res"][i], v["res"][i]
        if b["post_injection_resid"] is not None:
            x, y = a["post_injection_resid"], b["post_injection_resid"]
            r["post_injection_cos"] = float(torch.nn.functional.cosine_similarity(x, y, 0))
            r["post_injection_norm_ratio"] = float(y.norm() / x.norm())
        common = set(a["top_ids"].tolist()) & set(b["top_ids"].tolist())
        la = dict(zip(a["top_ids"].tolist(), a["top_lp"].tolist()))
        lb = dict(zip(b["top_ids"].tolist(), b["top_lp"].tolist()))
        r["argmax_equal"] = int(a["top_ids"][0]) == int(b["top_ids"][0])
        r["top20_overlap"] = len(common)
        r["max_abs_dlogprob_common"] = max(abs(la[t] - lb[t]) for t in common) if common else None
        for eng, S in (("hf", h["samples"][i]), ("vllm", v["samples"][i])):
            ex = [RD.extract_explanation(s) for s in S]
            f = [ar.fve(ar.predict(tok, e), gold, ref)[0] for e in ex if e is not None]
            r[f"{eng}_n_parsed"] = len(f)
            r[f"{eng}_fve_mean"] = sum(f) / len(f) if f else None
            r[f"{eng}_colours"] = [RD.colours_named(e or "") for e in ex]
        rows.append(r)
    out = {"rows": rows,
           "hf_gen_s": h["gen_s"], "hf_n_seqs": h["n_seqs"],
           "vllm_load_s": v["load_s"], "vllm_merge_s": v["merge_s"], "vllm_gen_s": v["gen_s"], "vllm_n_seqs": v["n_seqs"],
           "vllm_n_tokens": v["n_tokens"], "vllm_gen4_s": v["gen4_s"], "vllm_n_seqs4": v["n_seqs4"], "vllm_n_tokens4": v["n_tokens4"],
           "n_lora_modules": v["n_lora_modules"], "merge": v["merge"], "steering_stats": v["steering_stats"]}
    fh = [x["hf_fve_mean"] for x in rows if x["hf_fve_mean"] is not None]
    fv = [x["vllm_fve_mean"] for x in rows if x["vllm_fve_mean"] is not None]
    out["fve_mean_hf"] = sum(fh) / len(fh)
    out["fve_mean_vllm"] = sum(fv) / len(fv)
    out["speedup_same_batch"] = h["gen_s"] / v["gen_s"]
    od = R.out_dir("results", "000_setup", "analysis")
    (od / "vllm_nla_check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: x for k, x in out.items() if k != "rows"}, indent=1, default=str))
    for x in rows:
        print(x["cell"], {k: (round(y, 4) if isinstance(y, float) else y) for k, y in x.items() if k not in ("cell", "hf_colours", "vllm_colours")})


if __name__ == "__main__":
    {"hf": hf, "vllm": vllm, "compare": compare}[sys.argv[1]]()
