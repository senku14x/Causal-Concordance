"""012 Stage I readers (design §18; EXPLORATORY; reader side) on the saved hold states (never recomputed).

Hold states: one per (clue, carrier), positions H1 / H2, L40/42/44, states clean / J_pre / rand_pre.
Bridge LL (src vs tgt names) with the Stage C/F primitives and layouts: Oracle (reader_split.oracle_fast after
PREFIX0["oracle"], [clean, edited] pairs, acts_per_batch 2) at L40/L44; NLA (006's NLArl.score_seq, acts_per_batch 2)
at L42; token lenses J_NP / J_CB standard and logit at L40/L44 (resolved single-token forms). d_r per state.
Readouts: 2 samples per state for Oracle L44 and NLA L42 at H2 (read_batch, k 2, seed p007.SEED, chunks of 32).

  python stageI_readers.py --part subject|nla [--tag smoke] [--stage J]

--stage J (design §19, J2): the same path on stageJ/j2_hold_captures.pt (clean / J_clue / rand_clue); sampled readouts
at H1 instead of H2; outputs j2_hold_ll_*.jsonl, j2_readouts_*.jsonl.

--stage L (design §22): stageL/l_hold_captures.pt (Stage J's states at more layers); token lenses at L48/52/56/60, Oracle
at L52/56/60, sampled Oracle readouts at H2, L56; subject part only (the NLA was trained on L42). Outputs l_hold_ll_*.jsonl.
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
import p007 as C                                    # noqa: E402
from multihop import string_spellings               # noqa: E402
from c012 import category, d_r, lse_owner           # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["subject", "nla"], required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--stage", choices=["I", "J", "L"], default="I")
    a = ap.parse_args()
    assert not (a.stage == "L" and a.part == "nla"), "Stage L runs the subject-side readers only"
    D = ROOT / f"outputs/012_reader_concordance/stage{a.stage}" / ("" if a.tag == "full" else a.tag)
    t0 = time.time()
    pre, RPOS = {"I": ("i_", "H2"), "J": ("j2_", "H1"), "L": ("l_", "H2")}[a.stage]
    cap = torch.load(D / {"I": "hold_captures.pt", "J": "j2_hold_captures.pt", "L": "l_hold_captures.pt"}[a.stage])
    LENS_L, ORACLE_L, RO_L = ((48, 52, 56, 60), (52, 56, 60), 56) if a.stage == "L" else ((40, 44), (40, 44), 44)
    holds, arms, poss, layers, X, meta = cap["holds"], cap["arms"], cap["positions"], cap["layers"], cap["X"], cap["meta"]
    h = lambda i, s, P, L: X[i, arms.index(s), poss.index(P), layers.index(L)]
    EDITED = arms[1:]
    rows, timing, checks = [], {}, {}

    def pair_reader(reader, Ls, tok, prefix, score):
        for L in Ls:
            for i, hk in enumerate(holds):
                mt = meta[hk]
                base, seqs, owner = string_spellings(tok, prefix, [mt["src"], mt["tgt"]])
                hs = [x for P in poss for s in EDITED for x in (h(i, "clean", P, L), h(i, s, P, L))]
                out = score(hs, L, base, seqs)
                k = 0
                for P in poss:
                    for s in EDITED:
                        e, c = lse_owner(out[2 * k + 1], owner, mt), lse_owner(out[2 * k], owner, mt)
                        rows.append({"hold": hk, **mt, "reader": reader, "layer": L, "pos": P, "state": s, "LL_src": e["src"],
                                     "LL_tgt": e["tgt"], "LL_src_clean": c["src"], "LL_tgt_clean": c["tgt"], "d_r": d_r(e, c)})
                        k += 1

    def readouts(reader, L, read):
        order = [(i, s) for i in range(len(holds)) for s in arms]
        outs = []
        for j in range(0, len(order), 32):
            outs += read([h(i, s, RPOS, L) for i, s in order[j:j + 32]])
        with open(D / f"{pre}readouts_{reader}.jsonl", "w") as f:
            for (i, s), smp in zip(order, outs):
                mt = meta[holds[i]]
                f.write(json.dumps({"hold": holds[i], **mt, "reader": reader, "layer": L, "pos": RPOS, "state": s, "samples": smp,
                                    "category": [category(x, [mt["src"]], [mt["tgt"]]) for x in smp]}, ensure_ascii=False) + "\n")

    if a.part == "subject":
        import registry as R
        tok, model = R.make_tokenizer(), R.load_model()
        forms = json.loads((ROOT / "outputs/012_reader_concordance/stageA/words_st_report.json").read_text())["resolved_form"]
        fid = {w: tok.encode(f, add_special_tokens=False)[0] for w, f in forms.items()}
        t1 = time.time()
        VL = RD.VectorLenses(model, tok, list(LENS_L), lens_names=("J_NP", "J_CB"))
        with torch.no_grad():
            for L in LENS_L:
                for i, hk in enumerate(holds):
                    mt = meta[hk]
                    ts, tt = fid[mt["src"]], fid[mt["tgt"]]
                    for P in poss:
                        H = torch.stack([h(i, s, P, L) for s in arms]).to(VL.W.device).float()
                        lp = {"logit": torch.log_softmax(VL._n(H) @ VL.W.T, -1)}
                        for ln in ("J_NP", "J_CB"):
                            lp[ln] = torch.log_softmax(VL._n(VL.lenses[ln].transport(H, L)) @ VL.W.T, -1)
                        for rd, M in lp.items():
                            c = {"src": float(M[0, ts]), "tgt": float(M[0, tt])}
                            for k, s in enumerate(arms[1:], 1):
                                e = {"src": float(M[k, ts]), "tgt": float(M[k, tt])}
                                rows.append({"hold": hk, **mt, "reader": rd, "layer": L, "pos": P, "state": s, "LL_src": e["src"],
                                             "LL_tgt": e["tgt"], "LL_src_clean": c["src"], "LL_tgt_clean": c["tgt"], "d_r": d_r(e, c)})
        del VL
        torch.cuda.empty_cache()
        timing["token_lenses_s"] = round(time.time() - t1, 1)
        t1 = time.time()
        W32 = model.get_output_embeddings().weight.detach().float()
        ol = RD.OracleLens(model, tok)
        try:
            def score(hs, L, base, seqs):
                first = L not in checks
                o = RS.oracle_fast(ol, hs, L, base, seqs, W32, acts_per_batch=2, check=first).numpy()
                if first:
                    checks[L] = ol.last_check
                return o
            pair_reader("oracle", ORACLE_L, tok, RS.PREFIX0["oracle"], score)
            timing["oracle_ll_s"] = round(time.time() - t1, 1)
            t1 = time.time()
            readouts("oracle", RO_L, lambda hs: ol.read_batch(hs, RO_L, k=2, seed=C.SEED))
            timing["oracle_readouts_s"] = round(time.time() - t1, 1)
        finally:
            ol.unload()
        name = "oracle_lenses"
    else:
        assert RD.seq_logprob_from_hidden is RS._ORIG_SEQ_LP
        nla = RD.NLArl()
        t1 = time.time()
        try:
            def score(hs, L, base, seqs):
                first = L not in checks
                o = nla.score_seq(hs, base, seqs, acts_per_batch=2, check=first).numpy()
                if first:
                    checks[L] = {"own_logit_max_abs": nla.last_check, "last_write": nla.last_write}
                return o
            pair_reader("nla", (42,), nla.tok, RS.PREFIX0["nla"], score)
            timing["nla_ll_s"] = round(time.time() - t1, 1)
            t1 = time.time()
            readouts("nla", 42, lambda hs: nla.read_batch(hs, k=2, seed=C.SEED))
            timing["nla_readouts_s"] = round(time.time() - t1, 1)
        finally:
            nla.close()
        name = "nla"
    with open(D / f"{pre}hold_ll_{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (D / f"summary_readers_{pre}{name}.json").write_text(json.dumps({"timing": timing, "own_logit_check": checks, "n_rows": len(rows),
                                                                  "wall_s": round(time.time() - t0, 1)}, indent=1, default=str))
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
