"""012 Stage F readers (design §13; EXPLORATORY; reader side). Reads stageF_capture.py's saved captures only.

Same primitives and layout as Stage C: Oracle = reader_split.oracle_fast after PREFIX0["oracle"]; NLA = 006's exact
NLArl.score_seq after PREFIX0["nla"]; spellings = multihop.string_spellings, LSE per candidate; every batch is one
[clean, edited] pair (acts_per_batch 2). Token lenses (J_NP, J_CB standard; logit): log_softmax at the resolved
single-token bridge forms; answers by tokforms012's first-token rule (marked first_token).
F1 (read cell): states J/rand x readonly/allbutread at Oracle L40/L44 + lenses L40/L44 (--part subject) or NLA L42
(--part nla); kind = bridge (src vs tgt -> d_r) and answer (y_a vs y_b -> d_ans).
F2 (positions 0..T-2): J_L24-35 / rand_L24-35 vs clean, bridge only, same readers and layers -> d_r per position.

  python stageF_readers.py --part subject|nla [--tag smoke] [--stage G]

--stage G (design §14) reuses the F1 path unchanged on Stage G's whitened-swap captures (g_captures.pt, g_items.jsonl):
states J_white / rand_white at the read cell, bridge and answer; no F2. Outputs g_ll_{oracle_lenses,nla}.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("common", "007_jsubspace", "005_planted_concepts"):
    sys.path.insert(0, str(ROOT / "scripts" / p))
import readers as RD                                # noqa: E402
import reader_split as RS                           # noqa: E402
from multihop import string_spellings               # noqa: E402
from c012 import d_r, lse_owner                     # noqa: E402
from stageC_answers import lse_sets                 # noqa: E402
from tokforms012 import first_token_sets, lse_ids   # noqa: E402

F1_STATES = ["J_readonly_L24-35", "rand_readonly_L24-35", "J_allbutread_L24-35", "rand_allbutread_L24-35"]
F2_STATES = ["J_L24-35", "rand_L24-35"]
G_STATES = ["J_white_L24-35", "rand_white_L24-35"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["subject", "nla"], required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--stage", choices=["F", "G"], default="F")
    a = ap.parse_args()
    D = ROOT / f"outputs/012_reader_concordance/stage{a.stage}" / ("" if a.tag == "full" else a.tag)
    t0 = time.time()
    pre1 = "f1" if a.stage == "F" else "g"
    STATES = F1_STATES if a.stage == "F" else G_STATES
    c1 = torch.load(D / f"{pre1}_captures.pt")
    c2 = torch.load(D / "f2_captures.pt") if a.stage == "F" else None
    items = {r["name"]: r for r in map(json.loads, open(D / f"{pre1}_items.jsonl"))}
    names = c1["items"]
    h1 = lambda i, s, L: c1["X"][i, c1["arms"].index(s), c1["layers"].index(L)]
    h2 = lambda nm, s, L: c2["X"][nm][c2["states"].index(s), c2["layers"].index(L)]      # [P, D]
    f1, f2, timing, checks = [], [], {}, {}
    rec = lambda it, kind, ed, cl, **kw: {"kind": kind, "LL_a": ed[0], "LL_b": ed[1], "LL_a_clean": cl[0],
                                          "LL_b_clean": cl[1], "d": None if None in (*ed, *cl) else (ed[1] - ed[0]) - (cl[1] - cl[0]), **kw}

    def pair_reader(reader, layers, tok, prefix, score):
        """score(hs, L, base, seqs) -> [len(hs), S] numpy; hs interleaved [clean, edited] pairs."""
        t1 = time.time()
        for L in layers:
            for i, nm in enumerate(names):
                it = items[nm]
                for kind, cands in (("bridge", ([it["src"]], [it["tgt"]])), ("answer", (it["y_a"], it["y_b"]))):
                    base, seqs, owner = string_spellings(tok, prefix, cands[0] + cands[1])
                    hs = [x for s in STATES for x in (h1(i, "clean", L), h1(i, s, L))]
                    out = score(hs, L, base, seqs)
                    for k, s in enumerate(STATES):
                        e, c = (lse_sets(out[2 * k + j], owner, *cands) for j in (1, 0))
                        f1.append({"item": nm, "reader": reader, "layer": L, "state": s, "first_token": False,
                                   **rec(it, kind, (e["ya"], e["yb"]), (c["ya"], c["yb"]))})
                if c2 is None:
                    continue
                base, seqs, owner = string_spellings(tok, prefix, [it["src"], it["tgt"]])     # F2: bridge per position
                for s in F2_STATES:
                    C_, E_ = h2(nm, "clean", L), h2(nm, s, L)
                    out = score([x for p in range(C_.shape[0]) for x in (C_[p], E_[p])], L, base, seqs)
                    for p in range(C_.shape[0]):
                        e, c = lse_owner(out[2 * p + 1], owner, it), lse_owner(out[2 * p], owner, it)
                        f2.append({"item": nm, "reader": reader, "layer": L, "pos": p, "state": s, "LL_src": e["src"],
                                   "LL_tgt": e["tgt"], "LL_src_clean": c["src"], "LL_tgt_clean": c["tgt"], "d_r": d_r(e, c)})
        timing[f"{reader}_s"] = round(time.time() - t1, 1)

    if a.part == "subject":
        import registry as R
        tok, model = R.make_tokenizer(), R.load_model()
        forms = json.loads((ROOT / "outputs/012_reader_concordance/stageA/words_st_report.json").read_text())["resolved_form"]
        fid = {w: tok.encode(f, add_special_tokens=False)[0] for w, f in forms.items()}
        t1 = time.time()
        VL = RD.VectorLenses(model, tok, [40, 44], lens_names=("J_NP", "J_CB"))
        with torch.no_grad():
            def lps(H, L):
                H = H.to(VL.W.device).float()
                o = {"logit": torch.log_softmax(VL._n(H) @ VL.W.T, -1).cpu()}
                for ln in ("J_NP", "J_CB"):
                    o[ln] = torch.log_softmax(VL._n(VL.lenses[ln].transport(H, L)) @ VL.W.T, -1).cpu()
                return o
            for L in (40, 44):
                for i, nm in enumerate(names):
                    it = items[nm]
                    ida, idb, _ = first_token_sets(tok, it["y_a"], it["y_b"])
                    ts, tt = fid[it["src"]], fid[it["tgt"]]
                    st = ["clean", *STATES]
                    for rd, M in lps(torch.stack([h1(i, s, L) for s in st]), L).items():
                        for k, s in enumerate(st[1:], 1):
                            f1.append({"item": nm, "reader": rd, "layer": L, "state": s, "first_token": False,
                                       **rec(it, "bridge", (float(M[k, ts]), float(M[k, tt])), (float(M[0, ts]), float(M[0, tt])))})
                            f1.append({"item": nm, "reader": rd, "layer": L, "state": s, "first_token": True,
                                       **rec(it, "answer", (lse_ids(M[k], ida), lse_ids(M[k], idb)),
                                             (lse_ids(M[0], ida), lse_ids(M[0], idb)))})
                    if c2 is None:
                        continue
                    C_ = h2(nm, "clean", L)
                    P = C_.shape[0]
                    Ms = lps(torch.cat([C_] + [h2(nm, s, L) for s in F2_STATES]), L)
                    for rd, M in Ms.items():
                        for j, s in enumerate(F2_STATES, 1):
                            for p in range(P):
                                e, c = M[j * P + p], M[p]
                                ed, cl = {"src": float(e[ts]), "tgt": float(e[tt])}, {"src": float(c[ts]), "tgt": float(c[tt])}
                                f2.append({"item": nm, "reader": rd, "layer": L, "pos": p, "state": s, "LL_src": ed["src"],
                                           "LL_tgt": ed["tgt"], "LL_src_clean": cl["src"], "LL_tgt_clean": cl["tgt"],
                                           "d_r": d_r(ed, cl)})
        del VL
        torch.cuda.empty_cache()
        timing["token_lenses_s"] = round(time.time() - t1, 1)
        W32 = model.get_output_embeddings().weight.detach().float()
        ol = RD.OracleLens(model, tok)
        try:
            def score(hs, L, base, seqs):
                first = L not in checks
                out = RS.oracle_fast(ol, hs, L, base, seqs, W32, acts_per_batch=2, check=first).numpy()
                if first:
                    checks[L] = ol.last_check
                return out
            pair_reader("oracle", (40, 44), tok, RS.PREFIX0["oracle"], score)
        finally:
            ol.unload()
        name = "oracle_lenses"
    else:
        assert RD.seq_logprob_from_hidden is RS._ORIG_SEQ_LP
        nla = RD.NLArl()
        try:
            def score(hs, L, base, seqs):
                first = L not in checks
                out = nla.score_seq(hs, base, seqs, acts_per_batch=2, check=first).numpy()
                if first:
                    checks[L] = {"own_logit_max_abs": nla.last_check, "last_write": nla.last_write}
                return out
            pair_reader("nla", (42,), nla.tok, RS.PREFIX0["nla"], score)
        finally:
            nla.close()
        name = "nla"
    for fn, rows in (((f"f1_ll_{name}.jsonl", f1), (f"f2_ll_{name}.jsonl", f2)) if a.stage == "F" else ((f"g_ll_{name}.jsonl", f1),)):
        with open(D / fn, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (D / f"summary_readers_{'' if a.stage == 'F' else 'g_'}{name}.json").write_text(json.dumps({"timing": timing, "own_logit_check": checks,
                                                                "n_f1_rows": len(f1), "n_f2_rows": len(f2),
                                                                "wall_s": round(time.time() - t0, 1)}, indent=1, default=str))
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
