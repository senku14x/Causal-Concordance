"""Timing-only smoke run for the design-001 budget. Runs the 001 primitives on one brew pair and one chain pair and
records wall-clock times ONLY: margins, shares and probe results are computed (the forwards must happen) but are
neither saved nor printed, so the design is not informed by outcomes.

Writes results/000_setup/analysis/timing_smoke.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import torch

import engine as E
import margin as MG
import partners as P
import registry as R
import wsb

LAYER = 42


def timed(fn, n=1):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def profile_pair(eng, tok, model, fam, pair, it, cells):
    r_m, r_p, diff = P.token_align(tok, pair, it)
    Sa, Sb = MG.spellings(tok, fam, pair.m.a), MG.spellings(tok, fam, pair.p.a)
    multi = [s for s in Sa + Sb if len(s) > 1]
    out = {"pair_id": pair.pair_id, "T": len(r_m), "n_spellings": len(Sa + Sb), "n_multitoken_spellings": len(multi),
           "n_cells_per_layer": len(cells)}
    t0 = time.time()
    g = P.gate_pair(model, tok, pair, r_m, r_p)
    out["gate_s"] = time.time() - t0
    out["gate_pass"] = g["pass"]
    out["clean_score_s"] = timed(lambda: MG.score(eng, r_m.ids, Sa, Sb))
    # partner sources for every continuation the scorer will use (matching sequence length), all three layers
    conts = [()] + multi
    srcs = {}

    def capture():
        for c in conts:
            srcs[c] = eng.forward(r_p.ids + list(c), record=[40, 42, 44]).acts
    out["partner_capture_s"] = timed(capture)
    t0 = time.time()
    for pos in cells:
        MG.score(eng, r_m.ids, Sa, Sb,
                 writes_fn=lambda c, pos=pos: [E.CellPatch(eng.blocks, LAYER, [pos], srcs[c][LAYER][[pos]])])
    torch.cuda.synchronize()
    out["full_cell_patch_s_per_cell"] = (time.time() - t0) / len(cells)
    span = sorted(cells)
    out["span_patch_s"] = timed(lambda: MG.score(eng, r_m.ids, Sa, Sb, writes_fn=lambda c: [
        E.SpanWriter(eng.blocks, [40, 42, 44], span, {l: srcs[c][l][span] for l in (40, 42, 44)})]))
    out["probe_capture_s_per_render"] = timed(lambda: eng.forward(r_m.ids, record=[40, 42, 44]))
    return out


def main():
    tok = R.make_tokenizer()
    model = R.load_model()
    eng = E.Engine(model, fp32_from=0)
    res = {"layer_profiled": LAYER, "regime": "rhut_fp32res_from0", "manifest": R.manifest_block()}
    _, brew = wsb.load_bank("brew")
    it = next(x for x in brew if x["name"] == "brew-s2-r10d2-0001")
    pair = {p.p.inp: p for p in P.brew_candidates(it)}["blue"]
    cells = sorted(int(p) for p in it["eval_positions"])
    res["brew"] = profile_pair(eng, tok, model, "brew", pair, it, cells)
    print("brew", {k: v for k, v in res["brew"].items()}, flush=True)
    _, chain = wsb.load_bank("chain")
    res["chain_gate_attempts"] = []
    for ci in (x for x in chain if P.chain_single_token_item(x)):
        cands = P.chain_candidates(ci)
        if not cands:
            continue
        cp = cands[0]
        r_m, r_p, _ = P.token_align(tok, cp, ci)
        t0 = time.time()
        g = P.gate_pair(model, tok, cp, r_m, r_p)
        res["chain_gate_attempts"].append({"pair_id": cp.pair_id, "pass": g["pass"], "s": time.time() - t0})
        if g["pass"]:
            res["chain"] = profile_pair(eng, tok, model, "chain", cp, ci, r_m.read_positions)
            break
    print("chain", res.get("chain"), flush=True)
    od = R.out_dir("results", "000_setup", "analysis")
    (od / "timing_smoke.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
