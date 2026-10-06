"""012 Stage C pilot, NLA (design §12): nla-rl av_base + iter_000400 (readers.NLArl) on Stage B's saved captures, after
the subject model is freed (separate process). Reader-side. EXPLORATORY.

States: clean / J_L24-35 / rand_L24-35 at L42 (the NLA's training layer) and L44 (off-training, labelled so).
LL path = 007's NLA post hoc path (A2: 006's exact scorer, NLArl.score_seq, acts_per_batch = 2, the original
readers.seq_logprob_from_hidden; prefix reader_split.PREFIX0["nla"]; multihop.string_spellings). One fixed layout:
every call scores [clean, edited] (2 activations, one batch) with the item's [src, tgt] spellings.
Readouts: NLArl.read_batch, k = 2, seed p007.SEED, chunks of 32 activations, as 007.

  python stageC_nla.py [--tag smoke]
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
from c012 import answer_category, category, concept_names, d_r, lse_owner, paths  # noqa: E402

READS = {42: "L24-35", 44: "L24-35"}
OFF_TRAINING = {44}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    P = paths(a.tag)
    assert RD.seq_logprob_from_hidden is RS._ORIG_SEQ_LP                 # 006's exact arithmetic (007 A2)
    t0 = time.time()
    cap = torch.load(P["captures"])
    items = {r["name"]: r for r in map(json.loads, open(P["items"])) if r["status"] == "evaluated"}
    names, arms, layers, X = cap["items"], cap["arms"], cap["layers"], cap["X"]
    h = lambda i, arm, L: X[i, arms.index(arm), layers.index(L)]
    nla = RD.NLArl()
    timing = {"load_s": round(time.time() - t0, 1)}
    rows, checks = [], {}
    try:
        t1 = time.time()
        for L, b in READS.items():
            for i, nm in enumerate(names):
                it = items[nm]
                base, seqs, owner = string_spellings(nla.tok, RS.PREFIX0["nla"], [it["src"], it["tgt"]])
                clean_seen = []
                for s in (f"J_{b}", f"rand_{b}"):
                    first = not checks.get(L)
                    out = nla.score_seq([h(i, "clean", L), h(i, s, L)], base, seqs, acts_per_batch=2, check=first).numpy()
                    if first:
                        checks[L] = {"own_logit_max_abs": nla.last_check, "item": nm, "last_write": nla.last_write}
                    llc, lle = lse_owner(out[0], owner, it), lse_owner(out[1], owner, it)
                    clean_seen.append(out[0])
                    rows.append({"item": nm, "reader": "nla", "layer": L, "band": b, "state": s,
                                 "off_training": L in OFF_TRAINING, "LL_src": lle["src"], "LL_tgt": lle["tgt"],
                                 "LL_src_clean": llc["src"], "LL_tgt_clean": llc["tgt"], "d_r": d_r(lle, llc),
                                 "spellings": {"owner": owner, "clean": out[0].tolist(), "edited": out[1].tolist()}})
                rows[-1]["clean_repeat_max_abs"] = float(abs(clean_seen[0] - clean_seen[1]).max())
        timing["nla_ll_s"] = round(time.time() - t1, 1)
        t1 = time.time()
        with open(P["readouts"]("nla"), "w") as f:
            for L, b in READS.items():
                st = ["clean", f"J_{b}", f"rand_{b}"]
                order = [(i, s) for i in range(len(names)) for s in st]
                outs = []
                for j in range(0, len(order), 32):
                    outs += nla.read_batch([h(i, s, L) for i, s in order[j:j + 32]], k=2, seed=C.SEED)
                for (i, s), smp in zip(order, outs):
                    it = items[names[i]]
                    f.write(json.dumps({"item": names[i], "reader": "nla", "layer": L, "band": b, "state": s,
                                        "off_training": L in OFF_TRAINING, "src": it["src"], "tgt": it["tgt"],
                                        "samples": smp, "category": [category(x, *concept_names(it)) for x in smp],
                                        "answer_category": [answer_category(x, it) for x in smp]},
                                       ensure_ascii=False) + "\n")
        timing["nla_readouts_s"] = round(time.time() - t1, 1)
    finally:
        nla.close()
    with open(P["ll"]("nla"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    P["summary"]("nla").write_text(json.dumps({"timing": timing, "nla_own_logit_check": checks, "n_items": len(names),
                                               "wall_s": round(time.time() - t0, 1)}, indent=1, default=str))
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
