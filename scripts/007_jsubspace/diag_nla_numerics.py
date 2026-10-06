"""007 engineering diagnostic (post hoc, no research outcome): why the NLA trips the max-based bf16 gates.

Measures, for the NLA L42 and for the Oracle L40 (comparison):
  D4  determinism: the same score_seq call twice
  D5  batch-composition sensitivity: score_seq with acts_per_batch 1 vs 2 vs 4 on the same activations
  D1  the G-score1 quantity (score_seq vs score_next, single-token spellings) on 32 activations at ':' and '</think>',
      its tail, and which spellings carry it (vs the spelling's log-prob)
  D3  (NLA) the 2-row fast scorer vs the 24-row path on all 231 ':' renders (the comparison G-repro saw), and the
      24-row rerun vs 006's saved likelihoods
Writes results/007_jsubspace/analysis/diag_nla_numerics.json.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import readers as RD                              # noqa: E402
import registry as R                              # noqa: E402
import p007 as C                                  # noqa: E402
import reader_split as RS                         # noqa: E402

assert RD.seq_logprob_from_hidden is RS._ORIG_SEQ_LP


def dist(d):
    d = np.abs(np.asarray(d, np.float64)).ravel()
    return {"n": int(d.size), "median": float(np.median(d)), "p90": float(np.quantile(d, .9)), "p99": float(np.quantile(d, .99)),
            "max": float(d.max()), "share_gt_0.25": float((d > 0.25).mean()), "share_gt_0.10": float((d > 0.10).mean())}


def acts_for(cname, L, idx):
    P = np.load(C.STAGE0 / f"pieces_{cname}_L{L}.npz")["whole"]
    return [torch.from_numpy(P[i]) for i in idx]


def g1(score_seq, score_next, hs, base, seqs, owner):
    single = [i for i, s in enumerate(seqs) if len(s) == 1]
    a = score_seq(hs, base, seqs, 2)[:, single].numpy()
    b = score_next(hs, base, [seqs[i][0] for i in single]).cpu().numpy()
    d = np.abs(a - b)
    # per-activation max, how the max grows with the number of activations, where the largest differences sit
    per_act = d.max(1)
    grow = {str(n): float(per_act[:n].max()) for n in (1, 2, 4, 8, 16, 32) if n <= len(per_act)}
    flat = [(float(d[i, j]), float(a[i, j]), owner[single[j]], i) for i in range(d.shape[0]) for j in range(d.shape[1])]
    flat.sort(reverse=True)
    corr = float(np.corrcoef(d.ravel(), -a.ravel())[0, 1])
    return {"dist": dist(d), "max_vs_n_activations": grow, "top5": flat[:5],
            "corr_absdiff_vs_neg_logprob": corr, "logprob_range": [float(a.min()), float(a.max())]}


def main():
    out = {"note": "engineering diagnostic, post hoc; no research outcome is computed or changed"}
    tok = R.make_tokenizer()
    # ------------------------------------------------ Oracle L40 (comparison)
    model = R.load_model()
    W32 = model.get_output_embeddings().weight.detach().float()
    ol = RD.OracleLens(model, tok)
    base, seqs, owner = RS.spellings(tok, RS.PREFIX0["oracle"])
    so = lambda hs, b, s, apb: ol.score_seq(hs, 40, b, s, W32, acts_per_batch=apb)
    sn = lambda hs, b, ci: ol.score_next(hs, 40, b, ci, W32)[0]
    o = {}
    for cname in ("colon", "think"):
        hs = acts_for(cname, 40, range(32))
        o[f"G_score1_{cname}"] = g1(so, sn, hs, base, seqs, owner)
    hs = acts_for("colon", 40, range(16))
    r2a, r2b = so(hs, base, seqs, 2).numpy(), so(hs, base, seqs, 2).numpy()
    o["determinism_same_call_twice"] = dist(r2a - r2b)
    o["batch_apb1_vs_apb2"] = dist(so(hs, base, seqs, 1).numpy() - r2a)
    o["batch_apb4_vs_apb2"] = dist(so(hs, base, seqs, 4).numpy() - r2a)
    out["oracle_L40"] = o
    print("oracle done", {k: v["dist"] if "dist" in v else v for k, v in o.items()}, flush=True)
    ol.unload()
    del ol, model, W32
    gc.collect()
    torch.cuda.empty_cache()
    # ------------------------------------------------ NLA L42
    nla = RD.NLArl()
    base, seqs, owner = RS.spellings(nla.tok, RS.PREFIX0["nla"])
    sq = lambda hs, b, s, apb: nla.score_seq(hs, b, s, acts_per_batch=apb)
    snn = lambda hs, b, ci: nla.score_next(hs, b, ci)[0]
    n = {}
    for cname in ("colon", "think"):
        hs = acts_for(cname, 42, range(32))
        n[f"G_score1_{cname}"] = g1(sq, snn, hs, base, seqs, owner)
    hs = acts_for("colon", 42, range(16))
    r2a, r2b = sq(hs, base, seqs, 2).numpy(), sq(hs, base, seqs, 2).numpy()
    n["determinism_same_call_twice"] = dist(r2a - r2b)
    n["batch_apb1_vs_apb2"] = dist(sq(hs, base, seqs, 1).numpy() - r2a)
    n["batch_apb4_vs_apb2"] = dist(sq(hs, base, seqs, 4).numpy() - r2a)
    # D3: the 2-row fast scorer vs the 24-row path on all 231 ':' renders, and the 24-row rerun vs 006
    hs = acts_for("colon", 42, range(231))
    fast = RS.nla_fast(nla, hs, base, seqs, acts_per_batch=16).numpy()
    exact = np.load(C.OUT / "stage1a" / "r2_nla_posthoc" / "ll" / "nla_L42_colon_whole.npy")
    ref006 = np.load(C.LL006 / "ll_nla_L42_p0.npz")["emit_colon"][:231]
    d = np.abs(fast - exact)
    i, j = np.unravel_index(d.argmax(), d.shape)
    n["fast2row_vs_exact24row_231"] = {**dist(d), "argmax": {"render": int(i), "spelling": owner[j] + (" (C)" if j % 2 else " (c)"),
                                                             "logprob_exact": float(exact[i, j]), "logprob_fast": float(fast[i, j])},
                                       "corr_absdiff_vs_neg_logprob": float(np.corrcoef(d.ravel(), -exact.ravel())[0, 1]),
                                       "by_spelling_max": {owner[k] + ("C" if k % 2 else "c"): float(d[:, k].max()) for k in range(24)}}
    n["exact24row_vs_006_231"] = dist(exact - ref006)
    out["nla_L42"] = n
    nla.close()
    C.dump(C.RES / "analysis" / "diag_nla_numerics.json", out)
    print("nla done", json.dumps({k: (v["dist"] if "dist" in v else {kk: vv for kk, vv in v.items() if kk in ("median", "p99", "max", "share_gt_0.25")})
                                  for k, v in n.items()}, default=float), flush=True)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"done in {time.time() - t:.0f}s")
