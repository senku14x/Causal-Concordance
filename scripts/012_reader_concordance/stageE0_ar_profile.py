"""012 Stage E0 (design §17; EXPLORATORY; reader side): our Oracle-reconstructor (AR) loader against the published
per-layer profile (agu18dec/olens_and_ar ar_ptag_pooled/meta.json). Reuses 011's ar_vectors by import, unchanged:
whitener() (whitening_iolens_pooled_L{L}; the 9 missing layers are downloaded into HF_HOME), gen_texts() (the same
assistant-text source as 011's validation spans), attach_ar(), ar_forward() (011's accepted convention: tag + span, read
ln_pre, shared head, raw-activation output), fve / scaled_fve / ret_batches.
One dedicated process: subject captures first (block-L output at the token before each span, all 12 tag layers), then
the AR LoRA is attached to this process's model copy.
Spans: uniform 1-32 tokens (span_law uniform), positions uniform inside the assistant responses (seed 011's SEED).
Metrics per layer (whitened, W = Σ^-1/2 of the AR's own whitener): unscaled pooled FVE 1 - Σ|W(ŷ-y)|² / Σ|W(y-μ)|²
(primary; the batch-centred denominator is kept too), FVE after one global least-squares scale, per-sample cosine,
retrieval@1 within micro-batches of 24 (011's ret_batches) and over the full pool.
jfve candidates (L <= 60, J_NP = the AR config's jspace file; same sha256 as our registry pin): (a) J y vs J ŷ raw;
(b) J (y - μ) vs J (ŷ - μ); (c) J-lens logits W_U·norm(J ·) (vocab chunked); (d) coordinates in 007's dense J subspace
(top-1,024 of outputs/007_jsubspace/bases/J_raw_L{L}.pt; L40/44/56 only). Each with retrieval@1 in batches of 24.

  python stageE0_ar_profile.py [--n-spans 1024] [--tag smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/common"))
sys.path.insert(0, str(ROOT / "scripts/011_olens_swaps"))
import ar_vectors as AV                             # noqa: E402  (011, imported unchanged)
import readers as RD                                # noqa: E402
import registry as R                                # noqa: E402

TAGS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63]
JL = [L for L in TAGS if L <= 60]
OUT = ROOT / "outputs/012_reader_concordance/stageE0"
BASES = ROOT / "outputs/007_jsubspace/bases"


def ret_pool(yh, y):
    yn, hn = torch.nn.functional.normalize(y, dim=1), torch.nn.functional.normalize(yh, dim=1)
    return float((hn @ yn.T).argmax(1).eq(torch.arange(len(y))).double().mean())


def fve_mu(yh, y):                                   # y, yh already centred on the whitener / reference mean
    return float(1 - ((y - yh) ** 2).sum() / (y ** 2).sum())


def white_metrics(o, y, w):
    yw, ow = (y - w["mu"]) @ w["W"].T, (o - w["mu"]) @ w["W"].T
    a = float((ow * yw).sum() / (ow * ow).sum())
    cos = torch.nn.functional.cosine_similarity(ow, yw, dim=1)
    return {"fve_unscaled": fve_mu(ow, yw), "fve_scaled": fve_mu(a * ow, yw), "global_scale": a,
            "fve_unscaled_batchmean": AV.fve(ow, yw), "fve_scaled_batchmean": AV.scaled_fve(ow, yw)[0],
            "cos_mean": float(cos.mean()), "cos_median": float(cos.median()),
            "ret1_b24": AV.ret_batches(ow, yw), "ret1_pool": ret_pool(ow, yw)}


@torch.no_grad()
def logit_metrics(Jo, Jy, norm, W32, size=24, chunk=32768):
    """FVE and retrieval@1 (batches of 24, 011's permutation) of J-lens logits W_U·norm(J ·), vocab chunked."""
    dev = W32.device
    no, ny = norm(Jo.float().to(dev)), norm(Jy.float().to(dev))
    n = len(Jy)
    perm = torch.from_numpy(np.random.default_rng(AV.SEED).permutation(n))
    batches = [perm[j:j + size] for j in range(0, n - size + 1, size)]
    G = [torch.zeros(size, size, dtype=torch.float64, device=dev) for _ in batches]
    nno, nny = torch.zeros(n, dtype=torch.float64, device=dev), torch.zeros(n, dtype=torch.float64, device=dev)
    num = den = 0.0
    for c in range(0, W32.shape[0], chunk):
        Wc = W32[c:c + chunk]
        lo, ly = (no @ Wc.T).double(), (ny @ Wc.T).double()
        num += float(((lo - ly) ** 2).sum())
        den += float(((ly - ly.mean(0)) ** 2).sum())
        nno += (lo * lo).sum(1)
        nny += (ly * ly).sum(1)
        for g, b in zip(G, batches):
            g += lo[b.to(dev)] @ ly[b.to(dev)].T
    hits = [((g / (nno[b.to(dev)].sqrt()[:, None] * nny[b.to(dev)].sqrt()[None, :])).argmax(1).cpu() == torch.arange(size)).double().mean()
            for g, b in zip(G, batches)]
    return {"fve": 1 - num / den, "ret1_b24": float(torch.stack(hits).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spans", type=int, default=1024)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    assert not torch.backends.cuda.matmul.allow_tf32
    out = OUT / ("" if a.tag == "full" else a.tag)
    out.mkdir(parents=True, exist_ok=True)
    log = AV.Logger(out / "log.txt")
    t0 = time.time()
    wh = {}
    for L in TAGS:                                   # downloads the missing whiteners into HF_HOME
        w = AV.whitener(L)
        wh[L] = {"mu": w["mu"], "W": w["W"], "eig_min": w["eig_min"], "eig_max": w["eig_max"]}
    log(f"whiteners ready for {TAGS} ({time.time() - t0:.0f}s)")
    from huggingface_hub import hf_hub_download
    meta = json.load(open(hf_hub_download(AV.AR_REPO, "ar_ptag_pooled/meta.json", revision=AV.AR_REV)))
    tok = R.make_tokenizer()
    model = R.load_model()
    W32 = model.get_output_embeddings().weight.detach().float().clone()          # before the AR LoRA is attached
    norm = R.final_norm(model)
    tf = OUT / "texts.json"
    if tf.exists():
        texts = json.loads(tf.read_text())
    else:
        texts = [[list(p), list(r)] for p, r in AV.gen_texts(model, tok, log)]
        tf.write_text(json.dumps(texts))
    # spans (uniform 1-32) and stock targets at every tag layer
    rng = np.random.default_rng(AV.SEED)
    per = -(-a.n_spans // len(texts))
    blocks = R.blocks(model)
    spans, Y = [], {L: [] for L in TAGS}
    with torch.no_grad():
        for prompt, resp in texts:
            full, store = prompt + resp, {}
            hs = [blocks[L].register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
                  for L in TAGS]
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
                for L in TAGS:
                    Y[L].append(store[L][p - 1])
    spans = spans[:a.n_spans]
    Y = {L: torch.stack(v[:a.n_spans]).double() for L, v in Y.items()}
    log(f"{len(spans)} spans, lengths {min(map(len, spans))}-{max(map(len, spans))}, mean {np.mean([len(s) for s in spans]):.1f}; "
        f"response lengths {[len(r) for _, r in texts]}")
    pt, head, heads = AV.attach_ar(model, log)
    assert list(heads["layers"]) == TAGS, heads["layers"]
    O = {}
    for L in TAGS:
        O[L] = AV.ar_forward(pt, head, heads, spans, L)["ln_pre"]
    log(f"AR forward done ({time.time() - t0:.0f}s)")
    lens = RD.load_lenses(("J_NP",), layers=JL, device="cuda", verify_sha=True)["J_NP"]
    res = {"n_spans": len(spans), "span_law": "uniform 1-32", "source": "011 ar_vectors.gen_texts (12 VAL_PROMPTS, greedy 160)",
           "convention": "tag + span, ln_pre read, shared head, raw output; target block-L output at the token before the span",
           "jspace_file_note": "AR config jspace revision a4114d7 has the same sha256 as our J_NP pin 0731326e",
           "layers": {}}
    for L in TAGS:
        y, o, w = Y[L], O[L], wh[L]
        r = {"published": {k: meta.get(f"val_{k}_L{L}") for k in ("fve", "ret1", "jfve", "jret1")},
             "white": white_metrics(o, y, w), "whitener_eig_min": w["eig_min"]}
        if L in JL:
            J = lens.jacobians[L].double()
            yd, od = y.to("cuda"), o.to("cuda")
            Jy, Jo = yd @ J.T, od @ J.T
            mu = w["mu"].to("cuda")
            Jy_m, Jo_m = (yd - mu) @ J.T, (od - mu) @ J.T
            r["jfve"] = {"a_raw": {"fve": AV.fve(Jo.cpu(), Jy.cpu()), "ret1_b24": AV.ret_batches(Jo.cpu(), Jy.cpu())},
                         "b_mu_removed": {"fve": fve_mu(Jo_m.cpu(), Jy_m.cpu()), "ret1_b24": AV.ret_batches(Jo_m.cpu(), Jy_m.cpu())},
                         "c_lens_logits": logit_metrics(Jo, Jy, norm, W32)}
            bp = BASES / f"J_raw_L{L}.pt"
            if bp.exists():
                Q = torch.load(bp, map_location="cpu", weights_only=False)["Q"][:, :1024].double()
                py, po = y @ Q, o @ Q
                r["jfve"]["d_007_dense_J1024"] = {"fve": AV.fve(po, py), "ret1_b24": AV.ret_batches(po, py)}
            del J, yd, od, Jy, Jo, Jy_m, Jo_m
            torch.cuda.empty_cache()
        res["layers"][L] = r
        log(f"L{L}: fve {r['white']['fve_unscaled']:.3f} scaled {r['white']['fve_scaled']:.3f} (a {r['white']['global_scale']:.2f}) "
            f"ret24 {r['white']['ret1_b24']:.2f} | pub fve {r['published']['fve']} ret {r['published']['ret1']}"
            + (" | j " + " ".join(f"{k}:{v['fve']:.3f}/{v['ret1_b24']:.2f}" for k, v in r["jfve"].items())
               + f" | pub jfve {r['published']['jfve']} jret {r['published']['jret1']}" if "jfve" in r else ""))
    # §17 verdicts
    ly = res["layers"]
    within = lambda key, pub, tol, Ls: sum(abs(ly[L]["white"][key] - ly[L]["published"][pub]) <= tol for L in Ls)
    n_sf, n_ret, n_uf = within("fve_scaled", "fve", 0.04, TAGS), within("ret1_b24", "ret1", 0.05, TAGS), within("fve_unscaled", "fve", 0.04, TAGS)
    up_to_scale = n_sf >= 10 and n_ret >= 10
    cands = {}
    for c in ("a_raw", "b_mu_removed", "c_lens_logits", "d_007_dense_J1024"):
        Ls = [L for L in JL if c in ly[L]["jfve"]]
        nf = sum(abs(ly[L]["jfve"][c]["fve"] - ly[L]["published"]["jfve"]) <= 0.04 for L in Ls)
        nr = sum(abs(ly[L]["jfve"][c]["ret1_b24"] - ly[L]["published"]["jret1"]) <= 0.07 for L in Ls)
        cands[c] = {"n_layers": len(Ls), "n_fve_within_0.04": nf, "n_ret_within_0.07": nr, "identified": nf >= 9 and nr >= 9}
    res["verdicts"] = {"counts": {"scaled_fve_within_0.04": n_sf, "ret1_b24_within_0.05": n_ret, "unscaled_fve_within_0.04": n_uf},
                       "validated_up_to_global_scale": up_to_scale, "fully_validated": up_to_scale and n_uf >= 10,
                       "jfve_candidates": cands,
                       "jfve": next((c for c, v in cands.items() if v["identified"]), "open")}
    res["wall_s"] = round(time.time() - t0, 1)
    res["manifest"] = R.manifest_block()
    rp = ROOT / "results/012_reader_concordance/analysis" / ("stageE0_ar_profile.json" if a.tag == "full" else f"stageE0_ar_profile_{a.tag}.json")
    rp.write_text(json.dumps(res, indent=1, default=str))
    log("verdicts", json.dumps(res["verdicts"]))


if __name__ == "__main__":
    main()
