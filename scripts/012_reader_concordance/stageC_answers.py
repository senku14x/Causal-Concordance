"""012 Stage C add-on (reader side): answer likelihoods y_a vs y_b on Stage B's saved captures (never recomputed).
EXPLORATORY (lead's request after Stage D: are the readers reading answer-facing content rather than the bridge?).

Same readers x layers x states, prefixes, spellings convention and fixed [clean, edited] layout as the bridge scoring
(stageC_readers.py / stageC_nla.py), with candidates y_a (+alts) vs y_b (+alts) instead of src vs tgt:
  --part subject: token lenses (J_NP, J_CB standard; logit) at L40/44/52/56, FIRST-TOKEN log-prob (tokforms012 rule,
                  LSE over ids; marked first_token), and the Oracle (full sequence, oracle_fast, LSE over spellings);
  --part nla:     the NLA at L42 / L44 (off-training), 006's exact scorer (score_seq, acts_per_batch 2).
d_ans = [LL(y_b|edited) - LL(y_a|edited)] - [LL(y_b|clean) - LL(y_a|clean)]. No readouts (they are saved already).

  python stageC_answers.py --part subject|nla [--tag smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("common", "007_jsubspace", "005_planted_concepts"):
    sys.path.insert(0, str(ROOT / "scripts" / p))
import readers as RD                                # noqa: E402
import reader_split as RS                           # noqa: E402
from multihop import string_spellings               # noqa: E402
from c012 import paths                              # noqa: E402
from tokforms012 import first_token_sets, lse_ids   # noqa: E402

SUBJECT = {40: "L24-35", 44: "L24-35", 52: "L24-47", 56: "L24-47"}
NLA = {42: "L24-35", 44: "L24-35"}


def lse_sets(arr, owner, ya, yb):
    arr, owner = np.asarray(arr, dtype=np.float64), np.asarray(owner)
    out = {}
    for k, names in (("ya", ya), ("yb", yb)):
        cols = arr[np.isin(owner, names)]
        assert cols.size >= 1, names
        m = cols.max()
        out[k] = float(m + np.log(np.exp(cols - m).sum()))
    return out


def row(nm, reader, L, b, s, ed, cl, **kw):
    d = None if None in (ed["ya"], ed["yb"], cl["ya"], cl["yb"]) else (ed["yb"] - ed["ya"]) - (cl["yb"] - cl["ya"])
    return {"item": nm, "reader": reader, "layer": L, "band": b, "state": s, "LL_ya": ed["ya"], "LL_yb": ed["yb"],
            "LL_ya_clean": cl["ya"], "LL_yb_clean": cl["yb"], "d_ans": d, **kw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["subject", "nla"], required=True)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    P = paths(a.tag)
    t0 = time.time()
    cap = torch.load(P["captures"])
    items = {r["name"]: r for r in map(json.loads, open(P["items"])) if r["status"] == "evaluated"}
    names, arms, layers, X = cap["items"], cap["arms"], cap["layers"], cap["X"]
    h = lambda i, arm, L: X[i, arms.index(arm), layers.index(L)]
    ans = lambda it: ([it["y_a"], *it["y_a_alts"]], [it["y_b"], *it["y_b_alts"]])
    rows, timing, checks = [], {}, {}

    def score_pairs(reader, reads, tok, prefix, fn):
        for L, b in reads.items():
            for i, nm in enumerate(names):
                ya, yb = ans(items[nm])
                base, seqs, owner = string_spellings(tok, prefix, ya + yb)
                clean_seen = []
                for s in (f"J_{b}", f"rand_{b}"):
                    first = L not in checks
                    out, chk = fn([h(i, "clean", L), h(i, s, L)], L, base, seqs, first)
                    if first:
                        checks[L] = chk
                    clean_seen.append(out[0])
                    rows.append(row(nm, reader, L, b, s, lse_sets(out[1], owner, ya, yb), lse_sets(out[0], owner, ya, yb),
                                    first_token=False, off_training=(reader == "nla" and L != 42)))
                rows[-1]["clean_repeat_max_abs"] = float(abs(clean_seen[0] - clean_seen[1]).max())

    if a.part == "subject":
        import registry as R
        tok, model = R.make_tokenizer(), R.load_model()
        timing["load_s"] = round(time.time() - t0, 1)
        t1 = time.time()
        VL = RD.VectorLenses(model, tok, list(SUBJECT), lens_names=("J_NP", "J_CB"))
        fts = {nm: first_token_sets(tok, *ans(items[nm])) for nm in names}
        with torch.no_grad():
            for L, b in SUBJECT.items():
                for i, nm in enumerate(names):
                    ida, idb, _ = fts[nm]
                    st = ["clean", f"J_{b}", f"rand_{b}"]
                    H = torch.stack([h(i, s, L) for s in st]).to(VL.W.device).float()
                    lp = {"logit": torch.log_softmax(VL._n(H) @ VL.W.T, -1)}
                    for ln in ("J_NP", "J_CB"):
                        lp[ln] = torch.log_softmax(VL._n(VL.lenses[ln].transport(H, L)) @ VL.W.T, -1)
                    for rd, M in lp.items():
                        M = M.cpu()
                        ll = [{"ya": lse_ids(M[k], ida), "yb": lse_ids(M[k], idb)} for k in range(3)]
                        for k in (1, 2):
                            rows.append(row(nm, rd, L, b, st[k], ll[k], ll[0], first_token=True,
                                            first_token_ids={"y_a": ida, "y_b": idb}))
        del VL
        torch.cuda.empty_cache()
        timing["token_lenses_s"] = round(time.time() - t1, 1)
        t1 = time.time()
        W32 = model.get_output_embeddings().weight.detach().float()
        ol = RD.OracleLens(model, tok)
        try:
            def fn(hs, L, base, seqs, check):
                out = RS.oracle_fast(ol, hs, L, base, seqs, W32, acts_per_batch=2, check=check).numpy()
                return out, ({"own_logit_max_abs": ol.last_check} if check else None)
            score_pairs("oracle", SUBJECT, tok, RS.PREFIX0["oracle"], fn)
        finally:
            ol.unload()
        timing["oracle_ll_s"] = round(time.time() - t1, 1)
        name = "oracle_lenses"
    else:
        assert RD.seq_logprob_from_hidden is RS._ORIG_SEQ_LP                 # 006's exact arithmetic (007 A2)
        nla = RD.NLArl()
        timing["load_s"] = round(time.time() - t0, 1)
        t1 = time.time()
        try:
            def fn(hs, L, base, seqs, check):
                out = nla.score_seq(hs, base, seqs, acts_per_batch=2, check=check).numpy()
                return out, ({"own_logit_max_abs": nla.last_check, "last_write": nla.last_write} if check else None)
            score_pairs("nla", NLA, nla.tok, RS.PREFIX0["nla"], fn)
        finally:
            nla.close()
        timing["nla_ll_s"] = round(time.time() - t1, 1)
        name = "nla"
    c = P["ll"]("x").parent
    with open(c / f"ll_answers_{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (c / f"summary_answers_{name}.json").write_text(json.dumps(
        {"timing": timing, "own_logit_check": checks, "n_items": len(names),
         "clean_repeat_max_abs": max([r.get("clean_repeat_max_abs", 0.0) for r in rows] or [0.0]),
         "wall_s": round(time.time() - t0, 1)}, indent=1, default=str))
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
