"""Regime check (AMENDMENT_01 §3, precision): the same 5 edits under RHUT's float32-residual regime (the default) and
SEC's true-float32 replay of blocks 40..63; switch the default to SEC if any edit's dm differs by > 5% of G.

Mechanical check on one gated brew pair (dev material, hand-checked partner); no label is computed or kept.
Also reported: the stock bf16 forward and RHUT's own from_layer=35 variant as context columns, and each regime's
numerical noise floor (|dm| under 8 random perturbations of 1e-4 relative norm at one cell, which should be ~0).

Writes results/000_setup/analysis/regime_check.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import torch

import engine as E
import margin as MG
import partners as P
import regime as RG
import registry as R
import wsb

L0 = 39
EDITS = [  # (name, kind, position, layer)
    ("patch_emit_colon_L44", "patch", 164, 44),
    ("patch_emit_colon_L40", "patch", 164, 40),
    ("patch_start_token_L40", "patch", 142, 40),
    ("swap1d_emit_colon_L42_along_emit_think_diff", "swap_diff", 164, 42),
    ("swap1d_emit_colon_L42_random_dir", "swap_rand", 164, 42),
]
NOISE = {"pos": 164, "layer": 42, "rel": 1e-4, "k": 8, "seed": 300}
SWITCH_FRAC = 0.05


def unit_random(seed, d, device):
    v = torch.randn(d, generator=torch.Generator().manual_seed(seed)).to(device)
    return v / v.norm()


def lse_m(lp, Sa, Sb):
    a = torch.logsumexp(lp[[s[0] for s in Sa]].double(), 0)
    b = torch.logsumexp(lp[[s[0] for s in Sb]].double(), 0)
    return float(b - a)


def rhut_regime(model, fp32_from, r_m, r_p, Sa, Sb):
    eng = E.Engine(model, fp32_from=fp32_from)
    layers = sorted({e[3] for e in EDITS} | {NOISE["layer"]})
    t0 = time.time()
    cm = MG.score(eng, r_m.ids, Sa, Sb)
    t_fwd = time.time() - t0
    cp = MG.score(eng, r_p.ids, Sa, Sb)
    am = eng.forward(r_m.ids, record=layers).acts
    ap = eng.forward(r_p.ids, record=layers).acts
    G = cp["m"] - cm["m"]
    out = {"fp32_from": fp32_from, "m_member": cm["m"], "m_partner": cp["m"], "G": G, "forward_s": t_fwd, "edits": {}}
    for name, kind, pos, L in EDITS:
        if kind == "patch":
            mk = lambda c, pos=pos, L=L: [E.CellPatch(eng.blocks, L, [pos], ap[L][[pos]])]
        else:
            if kind == "swap_diff":
                d = ap[L][162] - am[L][162]
                u = d / d.norm()
            else:
                u = unit_random(7, am[L].shape[1], am[L].device)
            tgt = ap[L][[pos]] @ u
            mk = lambda c, pos=pos, L=L, u=u, tgt=tgt: [E.Swap1D(eng.blocks, L, [pos], u, tgt)]
        sc = MG.score(eng, r_m.ids, Sa, Sb, writes_fn=mk)
        st = E.realized_write_stats(sc["writes"][0])
        out["edits"][name] = {"dm": sc["m"] - cm["m"], "share": (sc["m"] - cm["m"]) / G, "rho": st["rho"],
                              "kappa": st["kappa"], "req_norm": st["req_norm"], "write_dtype": st["dtype"]}
    pos, L = NOISE["pos"], NOISE["layer"]
    h = am[L][pos]
    dms = []
    for k in range(NOISE["k"]):
        src = (h + NOISE["rel"] * h.norm() * unit_random(NOISE["seed"] + k, h.shape[0], h.device))[None]
        sc = MG.score(eng, r_m.ids, Sa, Sb, writes_fn=lambda c, src=src: [E.CellPatch(eng.blocks, L, [pos], src)])
        dms.append(sc["m"] - cm["m"])
    out["noise_floor"] = {**NOISE, "abs_dm": [abs(x) for x in dms], "max_abs_dm": float(np.max(np.abs(dms))),
                          "max_abs_dm_over_G": float(np.max(np.abs(dms)) / abs(G))}
    del eng, am, ap
    torch.cuda.empty_cache()
    return out


def sec_regime(model, r_m, r_p, Sa, Sb):
    sec = RG.SecReplay(model, L0)
    t0 = time.time()
    sec.build()
    out = {"l0": L0, "build_s": time.time() - t0, "gpu_alloc_gb_resident": torch.cuda.memory_allocated() / 2**30}
    layers = sorted({e[3] for e in EDITS} | {NOISE["layer"]})
    hm, kwm = sec.capture(r_m.ids)
    hp, kwp = sec.capture(r_p.ids)
    out["rotary_maxdiff_cos_sin"] = sec.rotary_check(kwm, len(r_m))
    T = len(r_m)
    t0 = time.time()
    fm, am = sec.replay(hm[L0], kwm, record=layers)
    out["replay_s"] = time.time() - t0
    fp, ap = sec.replay(hp[L0], kwp, record=layers)
    m_m = lse_m(sec.logprobs(fm, [T - 1])[0], Sa, Sb)
    m_p = lse_m(sec.logprobs(fp, [T - 1])[0], Sa, Sb)
    G = m_p - m_m
    out.update({"m_member": m_m, "m_partner": m_p, "G": G, "edits": {}})
    for name, kind, pos, L in EDITS:
        if kind == "patch":
            fn = RG.patch_fn([pos], ap[L][[pos]])
        else:
            if kind == "swap_diff":
                d = ap[L][162] - am[L][162]
                u = d / d.norm()
            else:
                u = unit_random(7, am[L].shape[1], am[L].device)
            fn = RG.swap_fn([pos], u, ap[L][[pos]] @ u)
        fe, _ = sec.replay(hm[L0], kwm, edits={L: fn})
        dm = lse_m(sec.logprobs(fe, [T - 1])[0], Sa, Sb) - m_m
        out["edits"][name] = {"dm": dm, "share": dm / G}
    pos, L = NOISE["pos"], NOISE["layer"]
    h = am[L][pos]
    dms = []
    for k in range(NOISE["k"]):
        src = (h + NOISE["rel"] * h.norm() * unit_random(NOISE["seed"] + k, h.shape[0], h.device))[None]
        fe, _ = sec.replay(hm[L0], kwm, edits={L: RG.patch_fn([pos], src)})
        dms.append(lse_m(sec.logprobs(fe, [T - 1])[0], Sa, Sb) - m_m)
    out["noise_floor"] = {**NOISE, "abs_dm": [abs(x) for x in dms], "max_abs_dm": float(np.max(np.abs(dms))),
                          "max_abs_dm_over_G": float(np.max(np.abs(dms)) / abs(G))}
    out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    sec.restore()
    return out


def main():
    tok = R.make_tokenizer()
    model = R.load_model()
    _, bank = wsb.load_bank("brew")
    # first pair in bank order (items, then partner starts alphabetically) that passes the gate with single-token
    # spellings; every attempt is kept. Items touched here are engineering material and go to the dev split.
    attempts, chosen = [], None
    for it in bank:
        for pair in P.brew_candidates(it):
            al = P.token_align(tok, pair, it)
            Sa, Sb = MG.spellings(tok, "brew", pair.m.a), MG.spellings(tok, "brew", pair.p.a)
            if al is None or not all(len(s) == 1 for s in Sa + Sb):
                continue
            r_m, r_p, diff = al
            t0 = time.time()
            g = P.gate_pair(model, tok, pair, r_m, r_p)
            attempts.append({"pair_id": pair.pair_id, "gate": g, "gate_s": time.time() - t0})
            print("gate", pair.pair_id, g["pass"], g["m"]["n_correct"], g["p"]["n_correct"],
                  repr(g["p"]["greedy"][:12]), [x[:8] for x in g["p"]["samples"]], flush=True)
            if g["pass"]:
                chosen = (it, pair, r_m, r_p, diff, Sa, Sb)
                break
        if chosen:
            break
    it, pair, r_m, r_p, diff, Sa, Sb = chosen
    res = {"item": it["name"], "pair_id": pair.pair_id, "member": {"start": pair.m.inp, "z": pair.m.z, "a": pair.m.a},
           "partner": {"start": pair.p.inp, "z": pair.p.z, "a": pair.p.a}, "diff_positions": diff,
           "edits": [dict(zip(("name", "kind", "position", "layer"), e)) for e in EDITS], "switch_frac": SWITCH_FRAC,
           "gate_attempts": attempts}
    assert diff == [142] and r_m.regions[162] == "emit_think" and r_m.regions[164] == "emit_colon", "positions moved"
    res["regimes"] = {}
    for name, f in (("rhut_fp32res_from0", 0), ("rhut_fp32res_from35", 35), ("stock_bf16", None)):
        res["regimes"][name] = rhut_regime(model, f, r_m, r_p, Sa, Sb)
        print(name, json.dumps({k: v for k, v in res["regimes"][name].items() if k != "edits"}, default=str), flush=True)
    res["regimes"]["sec_true_fp32_from40"] = sec_regime(model, r_m, r_p, Sa, Sb)
    print("sec", {k: v for k, v in res["regimes"]["sec_true_fp32_from40"].items() if k != "edits"}, flush=True)
    sec = res["regimes"]["sec_true_fp32_from40"]
    cmp = {}
    for base in ("rhut_fp32res_from0", "rhut_fp32res_from35", "stock_bf16"):
        rows = {}
        for name, *_ in EDITS:
            a, b = res["regimes"][base]["edits"][name]["dm"], sec["edits"][name]["dm"]
            rows[name] = {"dm_regime": a, "dm_sec": b, "abs_diff": abs(a - b), "abs_diff_over_G_sec": abs(a - b) / abs(sec["G"])}
        mx = max(v["abs_diff_over_G_sec"] for v in rows.values())
        cmp[base] = {"per_edit": rows, "max_abs_diff_over_G": mx, "G_regime": res["regimes"][base]["G"], "G_sec": sec["G"],
                     "switch_to_sec": bool(mx > SWITCH_FRAC)}
    res["comparison_vs_sec"] = cmp
    res["decision"] = ("switch default to SEC true-fp32" if cmp["rhut_fp32res_from0"]["switch_to_sec"]
                       else "keep RHUT fp32-residual (from block 0) as the default")
    res["manifest"] = R.manifest_block()
    od = R.out_dir("results", "000_setup", "analysis")
    (od / "regime_check.json").write_text(json.dumps(res, indent=1, default=str))
    for base, c in cmp.items():
        print(f"{base}: max |dm - dm_sec| / G = {c['max_abs_diff_over_G']:.4f} -> switch={c['switch_to_sec']}")
        for name, v in c["per_edit"].items():
            print(f"   {name:45s} dm={v['dm_regime']:+.4f} sec={v['dm_sec']:+.4f} |diff|/G={v['abs_diff_over_G_sec']:.4f}")
    print("decision:", res["decision"])


if __name__ == "__main__":
    main()
