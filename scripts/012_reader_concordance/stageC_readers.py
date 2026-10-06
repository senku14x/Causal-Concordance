"""012 Stage C pilot, subject-resident readers (design §12): Oracle Lens + token lenses on Stage B's saved captures.
Reader-side (imports scripts/common/readers.py); never recomputes captures. EXPLORATORY.

States per band: band L24-35 -> clean / J_L24-35 / rand_L24-35, read at L40, L44 (Oracle and token lenses);
                 band L24-47 -> clean / J_L24-47 / rand_L24-47, read at L52, L56.
Oracle LL: 007 Stage 3's path (multihop.string_spellings after reader_split.PREFIX0["oracle"]; reader_split.oracle_fast;
LSE over a string's spellings). Fixed layout: every call scores [clean, edited] (2 activations) with the item's
[src, tgt] spellings. Readouts: OracleLens.read_batch, k = 2, seed p007.SEED, chunks of 32 activations, as 007.
Token lenses (J_NP, J_CB standard readouts; logit lens): log_softmax(W_U final_norm(J_L h)) at the concept's resolved
single-token form (Stage A's words_st_report.json); NA when unresolved.
d_r = [LL(tgt|edited) - LL(src|edited)] - [LL(tgt|clean) - LL(src|clean)].

  python stageC_readers.py [--tag smoke]
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
import registry as R                                # noqa: E402
import reader_split as RS                           # noqa: E402
import p007 as C                                    # noqa: E402
from multihop import string_spellings               # noqa: E402
from c012 import answer_category, category, concept_names, d_r, lse_owner, paths  # noqa: E402

READS = {40: "L24-35", 44: "L24-35", 52: "L24-47", 56: "L24-47"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    P = paths(a.tag)
    t0 = time.time()
    cap = torch.load(P["captures"])
    items = {r["name"]: r for r in map(json.loads, open(P["items"])) if r["status"] == "evaluated"}
    names, arms, layers, X = cap["items"], cap["arms"], cap["layers"], cap["X"]
    forms = json.loads((ROOT / "outputs/012_reader_concordance/stageA/words_st_report.json").read_text())["resolved_form"]
    tok = R.make_tokenizer()
    model = R.load_model()
    print(f"loaded {len(names)} items in {time.time() - t0:.0f}s", flush=True)
    h = lambda i, arm, L: X[i, arms.index(arm), layers.index(L)]
    rows, timing = [], {}

    # ---- token lenses (before any adapter is attached)
    t1 = time.time()
    VL = RD.VectorLenses(model, tok, list(READS), lens_names=("J_NP", "J_CB"))
    tid = {w: tok.encode(f, add_special_tokens=False) for w, f in forms.items()}
    assert all(len(v) == 1 for v in tid.values())
    with torch.no_grad():
        for L, b in READS.items():
            for i, nm in enumerate(names):
                it = items[nm]
                st = ["clean", f"J_{b}", f"rand_{b}"]
                H = torch.stack([h(i, s, L) for s in st]).to(VL.W.device).float()
                lp = {"logit": torch.log_softmax(VL._n(H) @ VL.W.T, -1)}
                for ln in ("J_NP", "J_CB"):
                    lp[ln] = torch.log_softmax(VL._n(VL.lenses[ln].transport(H, L)) @ VL.W.T, -1)
                for rd, M in lp.items():
                    ll = {s: {c: (float(M[k, tid[it[c]][0]]) if it[c] in tid else None) for c in ("src", "tgt")}
                          for k, s in enumerate(st)}
                    for s in st[1:]:
                        rows.append({"item": nm, "reader": rd, "layer": L, "band": b, "state": s,
                                     "LL_src": ll[s]["src"], "LL_tgt": ll[s]["tgt"],
                                     "LL_src_clean": ll["clean"]["src"], "LL_tgt_clean": ll["clean"]["tgt"],
                                     "d_r": d_r(ll[s], ll["clean"])})
    del VL
    torch.cuda.empty_cache()
    timing["token_lenses_s"] = round(time.time() - t1, 1)

    # ---- Oracle Lens
    t1 = time.time()
    W32 = model.get_output_embeddings().weight.detach().float()
    ol = RD.OracleLens(model, tok)
    pre = RS.PREFIX0["oracle"]
    checks = {}
    try:
        for L, b in READS.items():
            for i, nm in enumerate(names):
                it = items[nm]
                base, seqs, owner = string_spellings(tok, pre, [it["src"], it["tgt"]])
                clean_seen = []
                for s in (f"J_{b}", f"rand_{b}"):
                    first = not checks.get(L)
                    out = RS.oracle_fast(ol, [h(i, "clean", L), h(i, s, L)], L, base, seqs, W32, acts_per_batch=2,
                                         check=first).numpy()
                    if first:
                        checks[L] = {"own_logit_max_abs": ol.last_check, "item": nm}
                    llc, lle = lse_owner(out[0], owner, it), lse_owner(out[1], owner, it)
                    clean_seen.append(out[0])
                    rows.append({"item": nm, "reader": "oracle", "layer": L, "band": b, "state": s,
                                 "LL_src": lle["src"], "LL_tgt": lle["tgt"], "LL_src_clean": llc["src"],
                                 "LL_tgt_clean": llc["tgt"], "d_r": d_r(lle, llc),
                                 "spellings": {"owner": owner, "clean": out[0].tolist(), "edited": out[1].tolist()}})
                rows[-1]["clean_repeat_max_abs"] = float(abs(clean_seen[0] - clean_seen[1]).max())
        timing["oracle_ll_s"] = round(time.time() - t1, 1)
        t1 = time.time()
        with open(P["readouts"]("oracle"), "w") as f:
            for L, b in READS.items():
                st = ["clean", f"J_{b}", f"rand_{b}"]
                order = [(i, s) for i in range(len(names)) for s in st]
                outs = []
                for j in range(0, len(order), 32):
                    outs += ol.read_batch([h(i, s, L) for i, s in order[j:j + 32]], L, k=2, seed=C.SEED)
                for (i, s), smp in zip(order, outs):
                    it = items[names[i]]
                    f.write(json.dumps({"item": names[i], "reader": "oracle", "layer": L, "band": b, "state": s,
                                        "src": it["src"], "tgt": it["tgt"], "samples": smp,
                                        "category": [category(x, *concept_names(it)) for x in smp],
                                        "answer_category": [answer_category(x, it) for x in smp]},
                                       ensure_ascii=False) + "\n")
        timing["oracle_readouts_s"] = round(time.time() - t1, 1)
    finally:
        ol.unload()
    with open(P["ll"]("oracle_lenses"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    P["summary"]("oracle_lenses").write_text(json.dumps({"timing": timing, "oracle_own_logit_check": checks,
                                                          "n_items": len(names), "wall_s": round(time.time() - t0, 1)},
                                                         indent=1))
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
