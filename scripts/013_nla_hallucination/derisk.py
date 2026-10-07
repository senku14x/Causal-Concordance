"""013 de-risk (EXPLORATORY; rules fixed in /workspace/claude_suggestions/013_plan_nla_hallucination_2026-10-07.md §4
before the run): can the NLA reconstructor (AR) host the hallucination test on brew and on multihop?

  av      : nla-rl AV (readers.NLArl, iter_000400, WSB sampling as in the 012 readouts) reads all 231 brew `:` L42
            renders of 007 (whole activation, outputs/007_jsubspace/stage0/pieces_colon_L42.npz), 2 samples each.
            Every readout is saved.
  ar      : the final AR (EasyNLA NLACriticModel + critic_predict; the path validated in 012 Stage K) scores the first
            sample per activation for brew (231) and for multihop (012's saved clean L42 readouts, 87 items, against
            the exact captures). FVE as Stage K (B = predict-the-mean on each set's own activations), plus a seeded
            derangement control. Writes the JSON quietly (no FVE printed) so the brew hand-read can be coded blind.
  sample  : writes the 32 seeded brew member readouts for the hand-read (prompt tail, ground truth, first sample).
"""
import argparse
import ast
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
EASYNLA = Path("/workspace/sources/repos/EasyNLA")
OUT = ROOT / "outputs/013_nla_hallucination/derisk"
RES = ROOT / "results/013_nla_hallucination"
K, CHUNK, SEED = 2, 16, 0


def brew_inputs():
    z = np.load(ROOT / "outputs/007_jsubspace/stage0/pieces_colon_L42.npz")
    roles = json.loads((ROOT / "outputs/007_jsubspace/stage0/roles.json").read_text())
    X = torch.tensor(z["whole"], dtype=torch.float32)
    assert X.shape == (len(roles["renders"]), 5120)
    return X, roles["renders"], roles["t_of"]


def phase_av():
    import readers as RD
    X, renders, t_of = brew_inputs()
    av = RD.NLArl()
    OUT.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    for c0 in range(0, len(X), CHUNK):
        texts = av.read_batch(list(X[c0:c0 + CHUNK]), k=K, seed=SEED + c0, sampling=RD.SAMPLING_WSB)
        for i, ts in enumerate(texts):
            r = renders[c0 + i]
            rows.append({"idx": c0 + i, "table": t_of[c0 + i], **r, "samples": ts})
        print(f"[av] {c0 + len(texts)}/{len(X)} ({time.time() - t0:.0f}s) write {av.last_write}", flush=True)
    with open(OUT / "brew_readouts_nla_L42.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (OUT / "brew_av_meta.json").write_text(json.dumps({"sampling": RD.SAMPLING_WSB, "k": K, "seed_rule": "SEED + chunk start",
                                                       "chunk": CHUNK, "wall_s": time.time() - t0, "injections": av.injections}, indent=1))


def multihop_inputs():
    c = torch.load(ROOT / "outputs/012_reader_concordance/stageB/captures.pt", map_location="cpu", weights_only=False)
    li, ai = c["layers"].index(42), c["arms"].index("clean")
    acts = {it: c["X"][n, ai, li].float() for n, it in enumerate(c["items"])}
    rows = []
    for l in open(ROOT / "outputs/012_reader_concordance/stageC/readouts_nla.jsonl"):
        r = json.loads(l)
        if r["state"] == "clean" and str(r["layer"]) == "42" and r.get("off_training") not in (True, "True"):
            s = ast.literal_eval(r["samples"]) if isinstance(r["samples"], str) else r["samples"]
            rows.append({"item": r["item"], "samples": s})
    rows.sort(key=lambda r: r["item"])
    return torch.stack([acts[r["item"]] for r in rows]), rows


def phase_ar():
    sys.path.insert(0, str(EASYNLA))
    import yaml
    from transformers import AutoTokenizer
    from nla.models import NLACriticModel
    from nla.schema import extract_explanation, normalize_activation, compute_predict_mean_baselines
    from nla.utils.critic import critic_predict
    import readers as RD
    root = Path(RD.nla_rl_paths()[0])
    template = yaml.safe_load(open(root / "nla_meta.yaml"))["prompt_templates"]["critic"]
    scale = math.sqrt(5120)
    tok = AutoTokenizer.from_pretrained(root / "av_base")
    critic = NLACriticModel.from_pretrained(str(root / "ar_reconstructor"), dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def pred(expl):
        ids = tok.encode(template.format(explanation=expl), add_special_tokens=False)
        if not 0 < len(ids) <= 1024:
            return None
        return critic_predict(critic, torch.tensor([ids], device="cuda"), None, scale)[0].float().cpu()

    Xb, _, t_of = brew_inputs()
    brew = [json.loads(l) for l in open(OUT / "brew_readouts_nla_L42.jsonl")]
    Xm, mh = multihop_inputs()
    res = {"label": "EXPLORATORY (013 de-risk; rules in claude_suggestions/013_plan §4, fixed before the run)",
           "rule": {"R1_min_fve_pct": 40.0, "R1_min_matched_minus_shuffled": 20.0}}
    per_rows = {}
    for name, X, rows, clus in (("brew", Xb, brew, [r["table"] for r in brew]), ("multihop", Xm, mh, [r["item"] for r in mh])):
        _, B = compute_predict_mean_baselines(X, scale)
        gn = normalize_activation(X, scale)
        recs, P = [], {}
        for i, r in enumerate(rows):
            for j, s in enumerate(r["samples"]):
                e = extract_explanation(s)
                p = pred(e) if e is not None else None
                rec = {"i": i, "sample": j, "extracted": e is not None}
                if p is not None:
                    rec["mse"] = float(((normalize_activation(p[None], scale)[0] - gn[i]) ** 2).mean())
                    P[(i, j)] = p
                recs.append(rec)
        first = [x for x in recs if x["sample"] == 0 and "mse" in x]
        fve = lambda xs: 100 * (1 - float(np.mean([x["mse"] for x in xs])) / B)
        idx = [x["i"] for x in first]
        perm = idx[:]
        rng = random.Random(SEED + 1)
        while any(a == b for a, b in zip(perm, idx)):
            rng.shuffle(perm)
        sh = [{"mse": float(((normalize_activation(P[(j, 0)][None], scale)[0] - gn[i]) ** 2).mean())} for i, j in zip(idx, perm)]
        # cluster bootstrap (tables for brew, items for multihop) of the first-sample FVE
        cl = {}
        for x in first:
            cl.setdefault(clus[x["i"]], []).append(x)
        keys, rb, boots = sorted(cl, key=str), random.Random(SEED), []
        for _ in range(2000):
            boots.append(fve([x for k in (rb.choice(keys) for _ in keys) for x in cl[k]]))
        m, s_ = fve(first), fve(sh)
        res[name] = {"n_acts": len(rows), "n_valid_first": len(first), "baseline_B": B,
                     "extraction_rate_all": sum(x["extracted"] for x in recs) / len(recs),
                     "fve_pct_first": m, "fve_pct_first_boot95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                     "fve_pct_all_samples": fve([x for x in recs if "mse" in x]),
                     "fve_pct_shuffled": s_, "matched_minus_shuffled": m - s_,
                     "R1_pass": m >= 40.0 and m - s_ >= 20.0}
        per_rows[name] = recs
    if res["brew"]:
        res["brew"]["fve_by_role_first"] = {
            role: (lambda xs: 100 * (1 - float(np.mean([x["mse"] for x in xs])) / res["brew"]["baseline_B"]))(
                [x for x in per_rows["brew"] if x["sample"] == 0 and "mse" in x and brew[x["i"]]["role"] == role])
            for role in sorted({r["role"] for r in brew})}
    (RES / "analysis").mkdir(parents=True, exist_ok=True)
    (RES / "analysis/derisk.json").write_text(json.dumps(res, indent=1))
    for name, recs in per_rows.items():
        with open(OUT / f"{name}_ar_scores.jsonl", "w") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
    print("[ar] wrote", RES / "analysis/derisk.json", "(FVE not printed: hand-read first)", flush=True)


def phase_sample():
    import wsb
    items = {i["name"]: i for i in [x for x in wsb.load_bank("brew") if isinstance(x, list)][0]}
    brew = [json.loads(l) for l in open(OUT / "brew_readouts_nla_L42.jsonl")]
    mem = [r for r in brew if r["role"] == "member"]
    pick = sorted(random.Random(7).sample(range(len(mem)), 32))
    (RES / "qualitative").mkdir(parents=True, exist_ok=True)
    with open(RES / "qualitative/brew_handread_sample.jsonl", "w") as f:
        for n, k in enumerate(pick):
            r = mem[k]
            p = items[r["item"]]["prompt"]
            f.write(json.dumps({"n": n, "idx": r["idx"], "item": r["item"], "table": r["table"], "start": r["start"], "z": r["z"],
                                "answer": r["answer"], "prompt_tail": p[p.find("You stir"):][:300],
                                "readout": r["samples"][0]}, ensure_ascii=False) + "\n")
    print("[sample] wrote", RES / "qualitative/brew_handread_sample.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["av", "ar", "sample"])
    {"av": phase_av, "ar": phase_ar, "sample": phase_sample}[ap.parse_args().phase]()
