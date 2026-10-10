"""015 step 2 (design `docs/design_specs/015_oracle_hallucination.md` §8): R1 on all four cells (text source x scorer),
O versions only, clean first samples, no judge calls; plus determinism, the whole-readout check, G-repro (NLA side) and
the CW-20 consistency numbers. EXPLORATORY. READER-SIDE (both readers' reconstructors).

  CUDA_VISIBLE_DEVICES=0 python r1.py oracle_ar     # Oracle AR (L44) on Oracle bullets, Oracle whole readouts, NLA sentences
  CUDA_VISIBLE_DEVICES=1 python r1.py nla_ar        # NLA AR (L42, critic template) on NLA and Oracle texts, whole
  python r1.py analyze                              # -> results/015_oracle_hallucination/r1.json

Metric (§3a, 013's exact form): unit-normalised prediction and target, MSE = 2 - 2 cos, FVE = 1 - MSE / B, B = the
predict-the-mean baseline of the 261 normalised gold targets (Oracle AR: whitened with whitening_iolens_pooled_L44 and
centred at its mean; NLA AR: 013's normalisation). Shuffle: 013 de-risk's derangement (items sorted by name,
random.Random(1) shuffles until no item keeps its own text).
"""
import ast
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("scripts/common", "scripts/011_olens_swaps", "scripts/013_nla_hallucination"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, "/workspace/sources/repos/EasyNLA")

STATES = ["clean", "J_L24-35", "rand_L24-35"]
OUT = ROOT / "outputs/015_oracle_hallucination/r1"
RES = ROOT / "results/015_oracle_hallucination"
MAXTOK = 32


def oracle_rows():
    rows = []
    for l in open(ROOT / "outputs/012_reader_concordance/stageC/readouts_oracle.jsonl"):
        r = json.loads(l)
        if r["layer"] != 44 or r["state"] not in STATES:
            continue
        s = ast.literal_eval(r["samples"]) if isinstance(r["samples"], str) else r["samples"]
        for k, t in enumerate(s):
            rows.append({"eid": f"{r['item']}|{r['state']}|{k}", "item": r["item"], "state": r["state"], "sample": k, "text": t})
    rows.sort(key=lambda x: (x["item"], STATES.index(x["state"]), x["sample"]))
    assert len(rows) == 522, len(rows)
    return rows


def nla_rows():
    import hallucination as H
    return [{**r, "text": r["explanation"]} for r in H.explanations()]


def bullets(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        out.append(line[2:] if line.startswith("- ") else line)
    return out


def nla_sentences(text):
    import hallucination as H
    return H.sentences(text)


def to_pieces(texts, tok):
    """Pieces of <= 32 AR tokens: split at clause boundaries, then hard-split any clause still too long."""
    pieces, n_split = [], 0
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        if not ids:
            continue
        if len(ids) <= MAXTOK:
            pieces.append(ids)
            continue
        n_split += 1
        cur = []
        for cl in re.split(r"(?<=[,;:])\s+", t):
            cid = tok(cl if not cur else " " + cl, add_special_tokens=False).input_ids
            if cur and len(cur) + len(cid) > MAXTOK:
                pieces.append(cur)
                cur = []
                cid = tok(cl, add_special_tokens=False).input_ids
            cur = cur + cid
            while len(cur) > MAXTOK:
                pieces.append(cur[:MAXTOK])
                cur = cur[MAXTOK:]
        if cur:
            pieces.append(cur)
    return pieces, n_split


def gold(L):
    c = torch.load(ROOT / "outputs/012_reader_concordance/stageB/captures.pt", map_location="cpu", weights_only=False)
    li = c["layers"].index(L)
    return {(it, st): c["X"][n, c["arms"].index(st), li].double() for n, it in enumerate(c["items"]) for st in STATES}


# ------------------------------------------------------------------------------------------------ GPU: Oracle AR
def run_oracle_ar():
    import ar_vectors as AV
    import registry as R
    OUT.mkdir(parents=True, exist_ok=True)
    log = AV.Logger(OUT / "oracle_ar_log.txt")
    tok = R.make_tokenizer()
    model = R.load_model()
    pt, head, heads = AV.attach_ar(model, log)
    orc = [r for r in oracle_rows() if r["state"] == "clean"]
    nla = [r for r in nla_rows() if r["state"] == "clean" and r["sample"] == 0]

    def predict(piece_lists):
        flat = [p for pl in piece_lists for p in pl]
        y = AV.ar_forward(pt, head, heads, flat, 44)["ln_pre"].double()
        out, i = [], 0
        for pl in piece_lists:
            out.append(y[i:i + len(pl)].mean(0) if pl else None)
            i += len(pl)
        return out

    res, stats, keep_pls = {}, {}, {}
    for name, rows, splitter in (("oracle_bullets", orc, bullets), ("nla_sentences", nla, nla_sentences)):
        pls, ns, npieces = [], 0, 0
        for r in rows:
            pl, s_ = to_pieces(splitter(r["text"]), tok)
            pls.append(pl)
            ns += s_
            npieces += len(pl)
        preds = predict(pls)
        res[name] = {r["eid"]: p for r, p in zip(rows, preds)}
        keep_pls[name] = (rows, pls)
        stats[name] = {"n_texts": len(rows), "n_pieces": npieces, "n_split_pieces": ns}
        log(f"{name}: {stats[name]}")
    rows_, pls_ = keep_pls["oracle_bullets"]                    # determinism: the identical call (same pieces, same batches)
    res["oracle_bullets_rerun"] = {r["eid"]: p for r, p in zip(rows_, predict(pls_)) if r["sample"] == 0}
    s0 = [r for r in orc if r["sample"] == 0]                  # batch-layout diagnostic: the first samples alone
    pls = [to_pieces(bullets(r["text"]), tok)[0] for r in s0]
    res["oracle_bullets_layout"] = {r["eid"]: p for r, p in zip(s0, predict(pls))}
    res["oracle_whole"] = {r["eid"]: p for r, p in zip(s0, predict([[tok(r["text"], add_special_tokens=False).input_ids] for r in s0]))}
    torch.save({"preds": res, "stats": stats}, OUT / "oracle_ar_preds.pt")
    log("saved")


# ------------------------------------------------------------------------------------------------ GPU: NLA AR
def run_nla_ar():
    import hallucination as H
    OUT.mkdir(parents=True, exist_ok=True)
    recon, scale = H.critic_setup()
    orc = [r for r in oracle_rows() if r["state"] == "clean" and r["sample"] == 0]
    nla = [r for r in nla_rows() if r["state"] == "clean" and r["sample"] == 0]
    res = {"nla_whole": {r["eid"]: recon(r["text"]).double() for r in nla},
           "oracle_whole": {r["eid"]: recon(r["text"]).double() for r in orc}}
    torch.save({"preds": res, "scale": scale}, OUT / "nla_ar_preds.pt")
    print("saved", {k: len(v) for k, v in res.items()}, flush=True)


# ------------------------------------------------------------------------------------------------ CPU: analysis
def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def boot(vals, nb=2000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    ms = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(nb)]
    return {"mean": float(vals.mean()), "ci95": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))], "n": len(vals)}


def analyze():
    import ar_vectors as AV
    import wsb
    o = torch.load(OUT / "oracle_ar_preds.pt", weights_only=False)
    n_ = torch.load(OUT / "nla_ar_preds.pt", weights_only=False)
    g44, g42 = gold(44), gold(42)
    w = AV.whitener(44)
    tw = lambda x: (x.double() - w["mu"]) @ w["W"].T                                            # noqa: E731
    keys261 = sorted(g44)
    U44 = unit(torch.stack([tw(g44[k]) for k in keys261]))
    B_or = float(((U44 - U44.mean(0)) ** 2).sum(1).mean())
    U42 = unit(torch.stack([g42[k] for k in keys261]))
    B_nla = float(((U42 - U42.mean(0)) ** 2).sum(1).mean())
    items = sorted({k[0] for k in keys261})
    idx = list(range(len(items)))
    perm = idx[:]
    rng = random.Random(1)
    while any(a == b for a, b in zip(perm, idx)):
        rng.shuffle(perm)

    def cell(preds, reader, scorer):
        """preds: {eid: vector} for the clean first samples of `reader`; returns R1 statistics under `scorer`."""
        if scorer == "oracle":
            P = unit(torch.stack([tw(preds[f"{it}|clean|0"]) for it in items]))
            T = unit(torch.stack([tw(g44[(it, "clean")]) for it in items]))
            B = B_or
        else:
            P = unit(torch.stack([preds[f"{it}|clean|0"] for it in items]))
            T = unit(torch.stack([g42[(it, "clean")] for it in items]))
            B = B_nla
        C = P @ T.T                                                                               # [text, target] cosines
        fve = lambda c: 100 * (1 - (2 - 2 * c) / B)                                             # noqa: E731
        m = [float(fve(C[i, i])) for i in idx]
        s = [float(fve(C[perm[i], i])) for i in idx]
        d = boot([a - b for a, b in zip(m, s)])
        ret = float(np.mean([int(C[i].argmax()) == i for i in idx]))
        return {"reader_texts": reader, "scorer": scorer, "B": B, "matched_fve_mean": float(np.mean(m)),
                "shuffled_fve_mean": float(np.mean(s)), "matched_minus_shuffled": d,
                "matched_cos_mean": float(np.mean([float(C[i, i]) for i in idx])),
                "shuffled_cos_mean": float(np.mean([float(C[perm[i], i]) for i in idx])),
                "retrieval_at_1": ret, "ci_excludes_0": d["ci95"][0] > 0 or d["ci95"][1] < 0}

    out = {"label": "EXPLORATORY; 015 step 2: R1 on all four cells (O versions, clean first samples, no judge calls)",
           "B": {"oracle_ar": B_or, "nla_ar": B_nla}, "n_items": len(items), "derangement": "013 de-risk: random.Random(1)"}
    cells = {"oracle_texts|oracle_ar": cell(o["preds"]["oracle_bullets"], "oracle", "oracle"),
             "oracle_texts|nla_ar": cell(n_["preds"]["oracle_whole"], "oracle", "nla"),
             "nla_texts|nla_ar": cell(n_["preds"]["nla_whole"], "nla", "nla"),
             "nla_texts|oracle_ar": cell(o["preds"]["nla_sentences"], "nla", "oracle")}
    g = cells["oracle_texts|oracle_ar"]
    g["gate"] = {"rule": "matched - shuffled CI > 0 and >= 10 points; retrieval@1 >= 0.5",
                 "pass": bool(g["matched_minus_shuffled"]["ci95"][0] > 0 and g["matched_minus_shuffled"]["mean"] >= 10
                              and g["retrieval_at_1"] >= 0.5)}
    out["cells"] = cells
    out["stop_gate_pass"] = g["gate"]["pass"]
    # determinism (Oracle AR, Oracle texts, clean first samples, two passes)
    r1, r2 = o["preds"]["oracle_bullets"], o["preds"]["oracle_bullets_rerun"]
    out["determinism_max_abs_diff"] = float(max((r1[k] - r2[k]).abs().max() for k in r2))
    out["determinism_pass"] = out["determinism_max_abs_diff"] == 0.0
    r3 = o["preds"]["oracle_bullets_layout"]
    lay = []
    for k in r3:
        t = unit(tw(g44[(k.split("|")[0], "clean")]))
        lay.append(abs(100 * 2 * (float(unit(tw(r1[k])) @ t) - float(unit(tw(r3[k])) @ t)) / B_or))
    out["batch_layout_fve_noise_points"] = {"median": float(np.median(lay)), "p90": float(np.percentile(lay, 90)),
                                            "max": float(np.max(lay)), "note": "same texts scored in a different batch composition (bf16)"}
    out["whole_readout_check"] = cell(o["preds"]["oracle_whole"], "oracle (whole readout, one span)", "oracle")
    out["piece_stats"] = o["stats"]
    # G-repro (NLA AR on NLA texts vs 013 de-risk MSE, first samples)
    dr = {x["i"]: x["mse"] for x in map(json.loads, open(ROOT / "outputs/013_nla_hallucination/derisk/multihop_ar_scores.jsonl"))
          if x["sample"] == 0 and "mse" in x}
    mse = [float(((unit(n_["preds"]["nla_whole"][f"{it}|clean|0"]) - unit(g42[(it, "clean")])) ** 2).sum()) for it in items]
    out["G_repro_nla"] = {"max_abs_diff_mse": float(max(abs(mse[i] - dr[i]) for i in dr)), "n": len(dr), "pass_1e-6": None}
    out["G_repro_nla"]["pass_1e-6"] = out["G_repro_nla"]["max_abs_diff_mse"] <= 1e-6
    # CW-20 consistency: her metric on the shared items (WSB multihop items 31-100), both clean samples
    bank = [x for x in wsb.load_bank("multihop") if isinstance(x, list)][0]
    shared = sorted(set(it["name"] for it in bank[30:100]) & set(items))
    pr = o["preds"]["oracle_bullets"]
    keys = [k for k in pr if k.split("|")[0] in shared and k.split("|")[1] == "clean"]
    Pw = {k: tw(pr[k]) for k in keys}
    Tw = {k: tw(g44[(k.split("|")[0], "clean")]) for k in keys}
    a = float(sum((Pw[k] * Tw[k]).sum() for k in keys) / sum((Pw[k] * Pw[k]).sum() for k in keys))
    f_ = [1 - float(((Tw[k] - a * Pw[k]) ** 2).sum() / (Tw[k] ** 2).sum()) for k in keys]
    c_ = [float(unit(Pw[k]) @ unit(Tw[k])) for k in keys]
    out["cw20_consistency"] = {"n_shared_items": len(shared), "n_writeups": len(keys), "global_scale": a,
                               "scaled_fve_mean": float(np.mean(f_)), "cos_mean": float(np.mean(c_)),
                               "cw20": {"scaled_fve": 0.21, "cos": 0.46, "global_scale": 0.434, "n_writeups": 140}}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "r1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1, default=float)[:2500])
    for k, v in cells.items():
        d = v["matched_minus_shuffled"]
        print(f"{k:24s} matched {v['matched_fve_mean']:+.1f} (cos {v['matched_cos_mean']:.3f}) shuffled {v['shuffled_fve_mean']:+.1f} "
              f"(cos {v['shuffled_cos_mean']:.3f}) diff {d['mean']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] ret@1 {v['retrieval_at_1']:.2f}")


if __name__ == "__main__":
    {"oracle_ar": run_oracle_ar, "nla_ar": run_nla_ar, "analyze": analyze}[sys.argv[1]]()
