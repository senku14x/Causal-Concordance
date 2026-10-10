"""012 Stage E step 2c, reconstructor side (design `docs/design_specs/012E_ar_concept_swaps.md` §5): the Oracle Lens's
reconstructor (AR) outputs for every phrase, template and single word. READER-SIDE (the AR); never imported by label
code. One dedicated process: the AR's LoRA is attached to this process's copy of the model, which is marked modified.

Convention: 011 amendment 2 (accepted post hoc, then for L36-60 on 2026-10-09): `[Layer NN]` tag + span, read `ln_pre`
at the span's last token, raw head output (011 `ar_vectors.attach_ar` / `ar_forward`, unchanged). Inputs get a leading
space, as 011's " c". Tags: multihop family L36/40/44 (nearest-tag edits over L36-47), brew L40/44/48.

Regression check (before anything is used): the brew "word" vectors (" red", ...) must reproduce 011's v3 raw colour
vectors (`outputs/011_olens_swaps/v3/ar_colour_vectors.pt`, form "space"): median cosine >= 0.999 per layer.

  CUDA_VISIBLE_DEVICES=k python stageE_ar.py [--tag smoke]
      -> outputs/012_reader_concordance/stageE/ar/ar_outputs[_tag].pt   (per phrase, float16)
         results/012_reader_concordance/analysis/stageE_ar[_tag].json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "011_olens_swaps"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))
import ar_vectors as AV                                                     # noqa: E402
import registry as R                                                        # noqa: E402
import stageE_common as E                                                   # noqa: E402

TAGS = {"multihop": (36, 40, 44), "brew": (40, 44, 48)}
SETS = ("starts", "mentions", "template", "word")
READ = "ln_pre"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    ph = json.loads((E.OUT / "gen" / f"phrases{sfx}.json").read_text())
    out = E.OUT / "ar"
    out.mkdir(parents=True, exist_ok=True)
    log = AV.Logger(out / f"log{sfx}.txt")
    t0 = time.time()
    tok = R.make_tokenizer()
    model = R.load_model()
    pt, head, heads = AV.attach_ar(model, log)
    assert all(L in heads["layers"] for v in TAGS.values() for L in v), heads["layers"]
    res, meta = {}, {"label": "EXPLORATORY; 012 Stage E AR outputs (design 012E §5)", "read": READ, "tags": TAGS,
                     "space": "raw head output", "counts": {}}
    for fam, inv in ph.items():
        jobs = []                                             # (concept, set, index) per input, one forward per layer
        for c in sorted(inv):
            for s in SETS:
                if s in ("starts", "mentions") and inv[c][s].get("dropped"):
                    continue
                for i, p in enumerate(inv[c][s]["kept"]):
                    jobs.append((c, s, i, tok.encode(" " + p, add_special_tokens=False)))
        meta["counts"][fam] = len(jobs)
        for L in TAGS[fam]:
            y = AV.ar_forward(pt, head, heads, [j[3] for j in jobs], L)[READ].float()
            for (c, s, i, _ids), v in zip(jobs, y):
                res.setdefault(fam, {}).setdefault(c, {}).setdefault(s, {}).setdefault(L, []).append(v.half())
            log(f"{fam} L{L}: {len(jobs)} inputs ({time.time() - t0:.0f}s)")
    for fam in res:
        for c in res[fam]:
            for s in res[fam][c]:
                for L in res[fam][c][s]:
                    res[fam][c][s][L] = torch.stack(res[fam][c][s][L])
    torch.save({"tags": TAGS, "read": READ, "outputs": res}, out / f"ar_outputs{sfx}.pt")
    # regression vs 011 v3 (brew single words)
    ref = torch.load(ROOT / "outputs/011_olens_swaps/v3/ar_colour_vectors.pt", weights_only=False)
    reg = {}
    if "brew" in res:
        for L in TAGS["brew"]:
            ours = torch.stack([res["brew"][c]["word"][L][0].double() for c in ref["colours"] if c in res["brew"]])
            theirs = ref["vecs"][(L, "space")]["raw"].double()[[ref["colours"].index(c) for c in ref["colours"]
                                                                 if c in res["brew"]]]
            cs = torch.nn.functional.cosine_similarity(ours, theirs, dim=1)
            reg[str(L)] = {"median_cos": float(cs.median()), "min_cos": float(cs.min()), "n": len(cs)}
    meta["regression_vs_011_v3"] = reg
    meta["regression_pass"] = bool(reg) and all(v["median_cos"] >= 0.999 for v in reg.values())
    meta["seconds"] = round(time.time() - t0, 1)
    (E.RES / f"stageE_ar{sfx}.json").write_text(json.dumps(meta, indent=1))
    log("regression vs 011 v3", json.dumps(reg), "pass", meta["regression_pass"])


if __name__ == "__main__":
    main()
