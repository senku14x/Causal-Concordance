"""Block b done-criterion: the pinned model loads, and one forward on a brew prompt gives the gated answer.

Writes results/000_setup/analysis/env_check.json (versions, GPU, load time/memory, greedy decode, top-5 next tokens,
the answer margin vs the runner-up colour, and the bf16 vs float32-residual agreement on this prompt).
"""
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import torch

import engine as E
import margin as MG
import registry as R
import wsb

ITEM = "brew-s2-r10d2-0000"


def main():
    out = {"item": ITEM, "manifest": R.manifest_block(), "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)),
           "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}
    tok = R.make_tokenizer()
    t0 = time.time()
    model = R.load_model()
    out["load_s"] = time.time() - t0
    out["gpu_alloc_gb_after_load"] = torch.cuda.memory_allocated() / 2**30
    out["layer_types_full_attention"] = R.full_attention_layers(model)
    out["model_class"] = type(model).__name__
    _, bank = wsb.load_bank("brew")
    it = next(x for x in bank if x["name"] == ITEM)
    r = wsb.render_item(tok, "brew", it)
    wsb.assert_brew_positions(r, it)
    x = torch.tensor([r.ids], device=model.device)
    sup = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    t0 = time.time()
    with torch.no_grad():
        g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=16,
                           suppress_tokens=sup, pad_token_id=tok.eos_token_id)
    out["greedy_s"] = time.time() - t0
    out["greedy"] = tok.decode(g[0, len(r.ids):], skip_special_tokens=True)
    out["gold"] = it["answer"]
    out["greedy_correct"] = out["greedy"].strip().lower().startswith(it["answer"])
    with torch.no_grad():
        lg = model(x).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    top = lp.topk(5)
    out["top5_stock_bf16"] = [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]
    for regime, f in (("bf16", None), ("fp32_residual_from0", 0)):
        eng = E.Engine(model, fp32_from=f)
        t0 = time.time()
        sc = MG.score(eng, r.ids, MG.spellings(tok, "brew", it["answer"]), MG.spellings(tok, "brew", "brown"))
        out[f"forward_s_{regime}"] = time.time() - t0
        out[f"score_{regime}"] = {"lse_gold": sc["lse_a"], "lse_brown": sc["lse_b"], "m_brown_minus_gold": sc["m"],
                                  "per_spelling": {tok.decode(list(k)): v for k, v in sc["spell"].items()}}
        del eng
        torch.cuda.empty_cache()
    out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    od = R.out_dir("results", "000_setup", "analysis")
    (od / "env_check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "manifest"}, indent=1, default=str))


if __name__ == "__main__":
    main()
