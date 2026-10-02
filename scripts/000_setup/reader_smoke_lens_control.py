"""Positive control for the vector-lens readout code (block c addendum): at the same brew cell the logit lens at the
last block (63) is the model's own next-token distribution, so its top-1 must equal the model's greedy answer token;
the J lenses (source layers 0..62) are read at late layers where the answer should be readable. Appends
"vector_lenses_positive_control" to results/000_setup/analysis/reader_smoke.json."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import torch

import readers as RD
import registry as R
import wsb

ITEM, POS = "brew-s2-r10d2-0000", 164
LL_LAYERS, J_LAYERS = [48, 52, 56, 60, 62, 63], [56, 60, 62]


def main():
    tok = R.make_tokenizer()
    model = R.load_model()
    _, bank = wsb.load_bank("brew")
    it = next(x for x in bank if x["name"] == ITEM)
    r = wsb.render_item(tok, "brew", it)
    acts = RD.capture_stock(model, r.ids, LL_LAYERS, [POS])
    with torch.no_grad():
        model_top = tok.decode([int(model(torch.tensor([r.ids], device=model.device)).logits[0, -1].argmax())])
    vl = RD.VectorLenses(model, tok, J_LAYERS)
    out = {"model_greedy_next_token": model_top, "logit_lens": {}, "J": {}}
    for L in LL_LAYERS:
        v = vl.logit_lens(acts[L][0])
        out["logit_lens"][f"L{L}"] = {"top10": v, "colours_named": RD.colours_named(" ".join(t for t, _ in v))}
    for L in J_LAYERS:
        for nm in ("J_NP", "J_CB", "R_CB"):
            for kind, fn in (("standard", vl.j_standard), ("cosine", vl.j_cosine)):
                v = fn(nm, acts[L][0], L)
                out["J"][f"L{L}_{nm}_{kind}"] = {"top10": v, "colours_named": RD.colours_named(" ".join(t for t, _ in v))}
    out["pass"] = out["logit_lens"]["L63"]["top10"][0][0] == model_top
    p = R.out_dir("results", "000_setup", "analysis") / "reader_smoke.json"
    d = json.loads(p.read_text())
    d["vector_lenses_positive_control"] = out
    p.write_text(json.dumps(d, indent=1, default=str))
    print("model greedy:", repr(model_top), "pass:", out["pass"])
    for k, v in {**out["logit_lens"], **out["J"]}.items():
        print(f"{k:22s} colours={v['colours_named']} top5={[t for t, _ in v['top10'][:5]]}")


if __name__ == "__main__":
    main()
