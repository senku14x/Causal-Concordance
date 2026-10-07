"""012 Stage E0 §17a (POST HOC amendment; EXPLORATORY; reader side): the four jfve candidates with ONE fitted
least-squares scale per layer, in each candidate's own space and centring:
  (a) raw J y vs J ŷ, batch-mean centred; (b) J(y - μ) vs J(ŷ - μ), centred on the whitener mean; (c) J-lens logits
  W_U·norm(J ·), batch-mean centred per vocab entry; (d) 007's dense J subspace coordinates, batch-mean centred.
  With T, T^ the centred target / prediction: a = <T, T^> / |T^|², scaled FVE = 1 - |T - a T^|² / |T|² = <T,T^>² / (|T|²|T^|²).
Same 1,024 spans as stageE0_ar_profile.py (regenerated deterministically from its saved texts and seed; parity checked
on the main whitened scaled FVE). Retrieval@1 is cosine-based (scale-free), so E0's ret values are reused.
Writes `jfve_scaled` into results/.../stageE0_ar_profile.json without touching existing fields.

  python stageE0a_jfve_scaled.py
"""
import json
import sys
import time

import numpy as np
import torch

from stageE0_ar_profile import AV, BASES, JL, OUT, RD, ROOT, R, TAGS, white_metrics


def sums(T, Th):
    return float((T * Th).sum()), float((T * T).sum()), float((Th * Th).sum())


def scaled(syo, syy, soo):
    return {"fve_scaled": syo * syo / (syy * soo), "scale": syo / soo}


def main():
    t0 = time.time()
    rp = ROOT / "results/012_reader_concordance/analysis/stageE0_ar_profile.json"
    res = json.loads(rp.read_text())
    assert "jfve_scaled" not in res
    n_spans = res["n_spans"]
    texts = json.loads((OUT / "texts.json").read_text())
    model = R.load_model()
    W32 = model.get_output_embeddings().weight.detach().float().clone()
    norm = R.final_norm(model)
    rng = np.random.default_rng(AV.SEED)
    per = -(-n_spans // len(texts))
    blocks = R.blocks(model)
    spans, Y = [], {L: [] for L in JL}
    with torch.no_grad():
        for prompt, resp in texts:
            full, store = prompt + resp, {}
            hs = [blocks[L].register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
                  for L in JL]
            try:
                model(torch.tensor([full], device=model.device), use_cache=False, logits_to_keep=1)
            finally:
                for h in hs:
                    h.remove()
            start = len(prompt) + 1
            for _ in range(per):
                n = int(rng.integers(1, 33))
                p = int(rng.integers(start, len(full) - n))
                spans.append(full[p:p + n])
                for L in JL:
                    Y[L].append(store[L][p - 1])
    spans = spans[:n_spans]
    Y = {L: torch.stack(v[:n_spans]).double() for L, v in Y.items()}
    log = AV.Logger(OUT / "log_17a.txt")
    pt, head, heads = AV.attach_ar(model, log)
    lens = RD.load_lenses(("J_NP",), layers=JL, device="cuda", verify_sha=True)["J_NP"]
    w44 = AV.whitener(44)
    out, parity = {}, None
    for L in JL:
        o = AV.ar_forward(pt, head, heads, spans, L)["ln_pre"]
        y = Y[L]
        if L == 44:
            parity = abs(white_metrics(o, y, w44)["fve_scaled"] - res["layers"]["44"]["white"]["fve_scaled"])
        mu = AV.whitener(L)["mu"].to("cuda")
        J = lens.jacobians[L].double()
        yd, od = y.to("cuda"), o.to("cuda")
        Jy, Jo = yd @ J.T, od @ J.T
        r = {"a_raw": scaled(*sums(Jy - Jy.mean(0), Jo - Jo.mean(0))),
             "b_mu_removed": scaled(*sums(Jy - mu @ J.T, Jo - mu @ J.T))}
        ny, no = norm(Jy.float()), norm(Jo.float())
        syo = syy = soo = 0.0
        for c in range(0, W32.shape[0], 32768):
            ly, lo = (ny @ W32[c:c + 32768].T).double(), (no @ W32[c:c + 32768].T).double()
            a_, b_, c_ = sums(ly - ly.mean(0), lo - lo.mean(0))
            syo, syy, soo = syo + a_, syy + b_, soo + c_
        r["c_lens_logits"] = scaled(syo, syy, soo)
        bp = BASES / f"J_raw_L{L}.pt"
        if bp.exists():
            Q = torch.load(bp, map_location="cpu", weights_only=False)["Q"][:, :1024].double().to("cuda")
            py, po = yd @ Q, od @ Q
            r["d_007_dense_J1024"] = scaled(*sums(py - py.mean(0), po - po.mean(0)))
        for k in r:
            r[k]["ret1_b24_from_E0"] = res["layers"][str(L)]["jfve"][k]["ret1_b24"]
        r["published"] = {"jfve": res["layers"][str(L)]["published"]["jfve"], "jret1": res["layers"][str(L)]["published"]["jret1"]}
        out[L] = r
        log(f"L{L}: " + " ".join(f"{k}:{v['fve_scaled']:.3f}" for k, v in r.items() if k != "published") + f" | pub {r['published']['jfve']}")
        del J, yd, od, Jy, Jo
        torch.cuda.empty_cache()
    cands = {}
    for c in ("a_raw", "b_mu_removed", "c_lens_logits", "d_007_dense_J1024"):
        Ls = [L for L in JL if c in out[L]]
        nf = sum(abs(out[L][c]["fve_scaled"] - out[L]["published"]["jfve"]) <= 0.04 for L in Ls)
        nr = sum(abs(out[L][c]["ret1_b24_from_E0"] - out[L]["published"]["jret1"]) <= 0.07 for L in Ls)
        cands[c] = {"n_layers": len(Ls), "n_fve_within_0.04": nf, "n_ret_within_0.07": nr, "identified": nf >= 9 and nr >= 9}
    res["jfve_scaled"] = {"amendment": "§17a POST HOC (written after E0's results)",
                          "definition": "one least-squares scale per layer in the candidate's own space and centring; "
                                        "scaled FVE = <T,T^>^2/(|T|^2 |T^|^2); retrieval reused from E0 (scale-free)",
                          "parity_L44_white_fve_scaled_abs_diff": parity, "layers": out, "candidates": cands,
                          "jfve": next((c for c, v in cands.items() if v["identified"]), "open"),
                          "wall_s": round(time.time() - t0, 1)}
    rp.write_text(json.dumps(res, indent=1, default=str))
    log("17a", json.dumps({"parity": parity, "candidates": cands}))


if __name__ == "__main__":
    main()
