"""012 Stage K smoke (spec §21, EXPLORATORY): does our path through the nla-rl reconstructor (AR) reproduce the card's
held-out FVE (76.6% at step 400)?

Two phases, run as separate processes so only one 27B model is on the GPU at a time:
  av  : av_base + iter_000400 (readers.NLArl, WSB's call; Karvonen add at the layer-1 output) verbalizes the 64
        L42 activations of `data/example_activations.parquet`, 4 samples per row, EasyNLA's eval sampling
        (T 1.0, top_p 1.0, top_k 0, 256 new tokens), seed 0. Raw responses -> outputs/.../stageK/av_samples.jsonl
  ar  : the final AR (ar_reconstructor/, saved_at_step 600) via EasyNLA's own NLACriticModel + critic_predict
        (EasyNLA @ 4d72847) scores every extracted explanation. FVE exactly as EasyNLA's eval/fve_pct:
        1 - mean_valid MSE(n(pred), n(gold)) / B, n = rescale to norm sqrt(5120), B = predict-the-mean baseline of
        the scored set (raw-variance variant). Rule (fixed before any number): primary FVE in [71.6, 81.6],
        extraction >= 0.90, matched - shuffled >= 20 points.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
EASYNLA = Path("/workspace/sources/repos/EasyNLA")      # pinned clone @ 4d72847 (rebuild_env.sh)
OUT = ROOT / "outputs/012_reader_concordance/stageK"
RES = ROOT / "results/012_reader_concordance"
SAMPLING_EASYNLA_EVAL = dict(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256)
K, SEED, CHUNK = 4, 0, 16
BAND, MIN_EXTRACT, MIN_GAP = (71.6, 81.6), 0.90, 20.0
REF_FVE_STEP400 = 76.6


def snapshot_root():
    import readers as RD
    root, _ = RD.nla_rl_paths()
    return Path(root)


def load_parquet(root):
    import pyarrow.parquet as pq
    t = pq.read_table(root / "data/example_activations.parquet").to_pylist()
    assert len(t) == 64 and all(int(r["activation_layer"]) == 42 for r in t), "unexpected example parquet"
    return t


def phase_av():
    import readers as RD
    root = snapshot_root()
    rows = load_parquet(root)
    hs = [torch.tensor(r["activation_vector"], dtype=torch.float32) for r in rows]
    av = RD.NLArl()
    assert av.scaling and "iter_000400" in str(RD.nla_rl_paths()[1]["adapter"]), "not the pinned adapter"
    t0, out = time.time(), []
    for c0 in range(0, len(hs), CHUNK):
        texts = av.read_batch(hs[c0:c0 + CHUNK], k=K, seed=SEED + c0, sampling=SAMPLING_EASYNLA_EVAL)
        w = av.last_write
        for i, ts in enumerate(texts):
            out.append({"row": c0 + i, "doc_id": rows[c0 + i]["doc_id"], "samples": ts})
        print(f"[av] rows {c0}-{c0 + len(texts) - 1} done ({time.time() - t0:.0f}s) write {w}", flush=True)
    assert av.injections > 0
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "av_samples.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    meta = {"sampling": SAMPLING_EASYNLA_EVAL, "k": K, "seed_rule": "SEED + chunk start", "chunk": CHUNK,
            "adapter": RD.nla_rl_paths()[1]["adapter"], "wall_s": time.time() - t0, "injections": av.injections}
    (OUT / "av_meta.json").write_text(json.dumps(meta, indent=1))
    print("[av] wrote", OUT / "av_samples.jsonl", meta, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def phase_ar():
    sys.path.insert(0, str(EASYNLA))
    import yaml
    from transformers import AutoTokenizer
    from nla.models import NLACriticModel
    from nla.schema import extract_explanation, normalize_activation, compute_predict_mean_baselines
    from nla.utils.critic import critic_predict
    root = snapshot_root()
    ar_dir = root / "ar_reconstructor"
    rows = load_parquet(root)
    av = [json.loads(l) for l in open(OUT / "av_samples.jsonl")]
    assert [r["row"] for r in av] == list(range(64))
    meta = yaml.safe_load(open(root / "nla_meta.yaml"))
    template = meta["prompt_templates"]["critic"]
    d = 5120
    scale = math.sqrt(d)                                  # mse_scale default sqrt_d_model (nla_meta sets none)
    tok = AutoTokenizer.from_pretrained(root / "av_base")
    t0 = time.time()
    critic = NLACriticModel.from_pretrained(str(ar_dir), dtype=torch.bfloat16, device_map="cuda")
    critic.eval()
    n_blocks = critic.backbone.config.num_hidden_layers
    gold = torch.stack([torch.tensor(r["activation_vector"], dtype=torch.float32) for r in rows])
    _, B = compute_predict_mean_baselines(gold, scale)
    gold_n = normalize_activation(gold, scale)

    @torch.no_grad()
    def predict(expl):
        ids = tok.encode(template.format(explanation=expl), add_special_tokens=False)
        if not 0 < len(ids) <= 1024:
            return None, len(ids)
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        return critic_predict(critic, x, None, scale)[0].float().cpu(), len(ids)

    per, preds = [], {}
    for r in av:
        for j, resp in enumerate(r["samples"]):
            expl = extract_explanation(resp)
            pred, ntok = predict(expl) if expl is not None else (None, 0)
            rec = {"row": r["row"], "sample": j, "extracted": expl is not None, "n_tok": ntok, "explanation": expl}
            if pred is not None:
                mse = float(((normalize_activation(pred[None], scale)[0] - gold_n[r["row"]]) ** 2).mean())
                rec.update(mse=mse, fve=1 - mse / B, pred_norm_raw=float(pred.norm()))
                preds[(r["row"], j)] = pred
            per.append(rec)
    prim = [p for p in per if p["sample"] == 0]
    val = [p for p in prim if "mse" in p]
    fve = lambda xs: 1 - float(np.mean([x["mse"] for x in xs])) / B if xs else float("nan")
    fve_primary = 100 * fve(val)
    rng = random.Random(SEED)
    boots = []
    for _ in range(2000):
        s = [val[rng.randrange(len(val))] for _ in val]
        boots.append(100 * fve(s))
    # loader sanity: score row i's activation against row perm[i]'s primary explanation (seeded derangement)
    rows_v = [p["row"] for p in val]
    perm = rows_v[:]
    rng2 = random.Random(SEED + 1)
    while any(a == b for a, b in zip(perm, rows_v)):
        rng2.shuffle(perm)
    sh = []
    for i, j in zip(rows_v, perm):
        pn = normalize_activation(preds[(j, 0)][None], scale)[0]
        sh.append({"mse": float(((pn - gold_n[i]) ** 2).mean())})
    fve_shuffled = 100 * fve(sh)
    all_val = [p for p in per if "mse" in p]
    extraction = sum(p["extracted"] for p in prim) / len(prim)
    res = {
        "label": "EXPLORATORY (012 Stage K smoke; rules in spec §21, fixed before any number)",
        "reference": {"fve_pct_step400": REF_FVE_STEP400, "definition": "EasyNLA eval/fve_pct; 128 held-out rows; AR as at step 400"},
        "n_rows": len(prim), "extraction_rate_primary": extraction, "n_valid_primary": len(val),
        "baseline_B": B, "mse_scale": scale, "ar_blocks": n_blocks,
        "fve_pct_primary": fve_primary, "fve_pct_primary_boot95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "fve_pct_all_samples": 100 * fve(all_val), "n_valid_all_samples": len(all_val),
        "extraction_rate_all_samples": sum(p["extracted"] for p in per) / len(per),
        "fve_pct_shuffled": fve_shuffled, "matched_minus_shuffled": fve_primary - fve_shuffled,
        "per_row_fve_primary_median": float(np.median([p["fve"] for p in val])),
        "pred_norm_raw_median": float(np.median([p["pred_norm_raw"] for p in all_val])),
        "gold_norm_median": float(gold.norm(dim=1).median()),
        "explanation_tokens_median": float(np.median([p["n_tok"] for p in all_val])),
        "rule": {"band": BAND, "min_extraction": MIN_EXTRACT, "min_matched_minus_shuffled": MIN_GAP},
        "ar_files_sha256": {f.name: sha256(f) for f in sorted(ar_dir.iterdir()) if f.is_file()},
        "ar_saved_at_step": (ar_dir / "saved_at_step.txt").read_text().strip(),
        "av_meta": json.loads((OUT / "av_meta.json").read_text()),
        "wall_s_ar": time.time() - t0,
    }
    res["pass_extraction"] = extraction >= MIN_EXTRACT
    res["pass_band"] = BAND[0] <= fve_primary <= BAND[1]
    res["pass_sanity"] = res["matched_minus_shuffled"] >= MIN_GAP
    res["verdict"] = ("extraction below 0.90: FVE not judged" if not res["pass_extraction"]
                      else "validated (in band, sanity passes)" if res["pass_band"] and res["pass_sanity"]
                      else "not validated")
    (RES / "analysis/stageK_ar_smoke.json").write_text(json.dumps(res, indent=1))
    with open(OUT / "ar_scores.jsonl", "w") as f:
        for p in per:
            f.write(json.dumps(p) + "\n")
    q = random.Random(7).sample(range(64), 8)
    with open(RES / "qualitative/stageK_samples.jsonl", "w") as f:
        for i in sorted(q):
            p = prim[i]
            f.write(json.dumps({"row": i, "doc_id": rows[i]["doc_id"], "source_text": rows[i]["detokenized_text_truncated"],
                                "explanation": p["explanation"], "fve": p.get("fve")}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("ar_files_sha256", "av_meta")}, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["av", "ar"])
    a = ap.parse_args()
    phase_av() if a.phase == "av" else phase_ar()
