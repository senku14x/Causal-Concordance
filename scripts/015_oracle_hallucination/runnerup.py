"""015 follow-up (POST HOC, descriptive; ledger amendment 2026-10-11 "swap test and runner-up test"): are the Oracle's
wrong answers Qwen's own runner-up answers at the read cell?

For every wrong-answer opening (swap.py's rule; outputs/.../versions/wrong_openings.jsonl), Qwen's log-prob of the
string as the continuation of the prompt, at the read cell, in that readout's state (clean / J_L24-35 / rand_L24-35:
Stage B's exact edits, Jan's SwapEdit + BandEditor on prompt positions only, ModelRunner bf16, float32 final norm and
W_U), full sequence, log-sum-exp over spellings {' s', ' s.lower()', ' S...'}; plus the first token's rank at the read
cell (best spelling). Null: the same string scored at another item's read cell in the same state (a seeded derangement
of items within state, random.Random(0)). Positive control: the model's own answer (the continuation's first phrase)
scored the same way. Gate: the argmax at every read cell equals Stage B's greedy first token.

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src CUDA_VISIBLE_DEVICES=1 python runnerup.py score
  python runnerup.py analyze   ->  results/015_oracle_hallucination/analysis/runnerup.json, qualitative/runnerup_examples.md
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/012_reader_concordance"))
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]
import oh  # noqa: E402
import swap as SW  # noqa: E402
sys.argv = _argv

STATES = ["clean", "J_L24-35", "rand_L24-35"]
OUTF = oh.OUT / "runnerup_scores.jsonl"


def derangement(items, seed=0):
    idx = list(range(len(items)))
    perm, rng = idx[:], random.Random(seed)
    while any(a == b for a, b in zip(perm, idx)):
        rng.shuffle(perm)
    return {items[i]: items[perm[i]] for i in idx}


def spellings(s):
    out = []
    for v in (s, s.lower(), s[:1].upper() + s[1:]):
        if " " + v not in out:
            out.append(" " + v)
    return out


def score():
    import torch
    from phrase_lens.benchmarks import load_benchmark
    from phrase_lens.interventions import BandEditor, SwapEdit, random_directions
    from phrase_lens.lenses import DirectionBank
    from phrase_lens.modeling import ModelRunner
    from stageB_capture import ARM_ID, BANDS, BENCH, MODEL, REV, SA

    W = [json.loads(l) for l in open(oh.OUT / "versions/wrong_openings.jsonl")]
    O = {r["eid"]: r for r in oh.oracle_rows()}
    items_all = sorted({r["item"] for r in O.values()})
    rowsB = {r["name"]: r for r in map(json.loads, open(ROOT / "outputs/012_reader_concordance/stageB/items.jsonl"))}
    cont = {(r["item"], r["state"]): r["continuation"] for r in O.values()}
    jobs = {}                                                              # (item, state) -> list of (string, role, origin_item)
    for st in STATES:
        pi = derangement(items_all, 0)
        for w in W:
            if w["state"] != st:
                continue
            jobs.setdefault((w["item"], st), []).append((w["opening"], "wrong_own", w["item"]))
            jobs.setdefault((pi[w["item"]], st), []).append((w["opening"], "wrong_null", w["item"]))
        for it in items_all:
            ta = SW.true_answer(cont[(it, st)])
            jobs.setdefault((it, st), []).append((ta, "true_own", it))
            jobs.setdefault((pi[it], st), []).append((ta, "true_null", it))
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks, norm = runner.tokenizer, runner.blocks, runner.final_norm
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    bank = DirectionBank.load(SA / "bank_j")
    bench = {it["name"]: it for it in load_benchmark(BENCH)["items"]}
    layers = BANDS["L24-35"]
    t0, out, gate = time.time(), [], {"cells": 0, "argmax_mismatch": []}

    @torch.no_grad()
    def forward(ids, n, edit):
        grab = {}

        def pre(_m, args):
            grab["x"] = args[0][0].detach()
        hk = norm.register_forward_pre_hook(pre)
        ctx = BandEditor(blocks, layers, edit, n) if edit is not None else torch.no_grad()
        try:
            with ctx:
                runner.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            hk.remove()
        y = norm(grab["x"][n - 1:].float())
        return torch.log_softmax(y @ W32.T, -1)                           # positions n-1 .. end

    for k, ((item, st), js) in enumerate(sorted(jobs.items())):
        it = bench[item]
        ids = runner.encode(it["prompt"].rstrip())
        n = len(ids)
        edit = None
        if st != "clean":
            c = runner.capture_ids(ids, positions=list(range(n)), layers=layers)
            clean_acts = {L: c[i] for i, L in enumerate(layers)}
            s_dirs = bank.concept(it["src"], layers)
            t_dirs = bank.concept(it["tgt"], layers) if st.startswith("J_") else \
                random_directions(s_dirs.shape[-1], [BENCH, it["name"], ARM_ID, "rand_swap"], layers)
            edit = SwapEdit(s_dirs, t_dirs, clean_acts, layers, 1.0)
        lp0 = forward(ids, n, edit)[0]
        gate["cells"] += 1
        if int(lp0.argmax()) != rowsB[item]["arms"][st]["ids"][0]:
            gate["argmax_mismatch"].append([item, st])
        cache = {}
        for s, role, origin in js:
            if s in cache:
                res = cache[s]
            else:
                full, first, ranks = [], [], []
                for sp in spellings(s):
                    sid = tok.encode(sp, add_special_tokens=False)
                    lp = forward(ids + sid, n, edit)
                    full.append(float(sum(lp[t, sid[t]] for t in range(len(sid)))))
                    first.append(float(lp[0, sid[0]]))
                    ranks.append(int((lp0 > lp0[sid[0]]).sum()) + 1)
                m = max(full)
                res = {"logp": m + float(np.log(np.sum(np.exp(np.array(full) - m)))), "logp_first_best": max(first), "rank_first_best": min(ranks)}
                cache[s] = res
            out.append({"string": s, "role": role, "origin_item": origin, "scored_at_item": item, "state": st, **res})
        if (k + 1) % 20 == 0:
            print(f"[runnerup] {k + 1}/{len(jobs)} cells ({time.time() - t0:.0f}s)", flush=True)
    with open(OUTF, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate["pass"] = not gate["argmax_mismatch"]
    gate["wall_s"] = time.time() - t0
    (oh.OUT / "runnerup_gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps({k: v for k, v in gate.items() if k != "argmax_mismatch"} | {"n_mismatch": len(gate["argmax_mismatch"])}), flush=True)


def analyze():
    R = [json.loads(l) for l in open(OUTF)]
    gate = json.loads((oh.OUT / "runnerup_gate.json").read_text())
    W = [json.loads(l) for l in open(oh.OUT / "versions/wrong_openings.jsonl")]
    key = lambda r: (r["string"], r["origin_item"], r["state"])                                 # noqa: E731
    own = {key(r): r for r in R if r["role"].endswith("_own")}
    null = {key(r): r for r in R if r["role"].endswith("_null")}
    res = {"label": "POST HOC, descriptive (015 runner-up test; ledger amendment 2026-10-11)",
           "gate_argmax_equals_stageB_greedy": {"cells": gate["cells"], "n_mismatch": len(gate["argmax_mismatch"]), "pass": gate["pass"]},
           "by_state": {}}
    for role in ("wrong", "true"):
        rows = [(own[k], null[k]) for k in own if own[k]["role"] == f"{role}_own" and k in null]
        out = {}
        for st in STATES + ["all"]:
            rr = [(a, b) for a, b in rows if st == "all" or a["state"] == st]
            if not rr:
                continue
            d = [a["logp"] - b["logp"] for a, b in rr]
            cl = [a["origin_item"] for a, _ in rr]
            out[st] = {"n": len(d), "logp_own_minus_null": {"mean": float(np.mean(d)), "ci95": oh.H.boot(d, cl), "median": float(np.median(d))},
                       "logp_own_mean": float(np.mean([a["logp"] for a, _ in rr])), "logp_null_mean": float(np.mean([b["logp"] for _, b in rr])),
                       "rank_own_median": float(np.median([a["rank_first_best"] for a, _ in rr])),
                       "rank_null_median": float(np.median([b["rank_first_best"] for _, b in rr])),
                       "share_rank_le_10_own": float(np.mean([a["rank_first_best"] <= 10 for a, _ in rr])),
                       "share_rank_le_10_null": float(np.mean([b["rank_first_best"] <= 10 for _, b in rr]))}
        res["by_state"][role] = out
    (oh.RES / "analysis/runnerup.json").write_text(json.dumps(res, indent=1))
    # 20 random examples (wrong openings)
    rng = random.Random(20)
    wr = [k for k in own if own[k]["role"] == "wrong_own" and k in null]
    ex = rng.sample(wr, min(20, len(wr)))
    wmeta = {(w["opening"], w["item"], w["state"]): w for w in W}
    lines = ["# 015 runner-up test: 20 random wrong-answer openings (seed 20)\n",
             "log-prob = Qwen's full-sequence log-prob of the string as the continuation, LSE over spellings, at the read cell in that "
             "state; rank = rank of its first token at the read cell. Null = the same string at another item's read cell (same state).\n",
             "| item | state | model's answer | Oracle's wrong opening | log-prob own | rank own | log-prob null | rank null |", "|---|---|---|---|---|---|---|---|"]
    for k in sorted(ex, key=lambda k: (k[1], k[2])):
        a, b = own[k], null[k]
        w = wmeta.get((k[0], k[1], k[2]), {})
        hu = lambda x: str(__import__("decimal").Decimal(repr(x)).quantize(__import__("decimal").Decimal("0.01"), rounding="ROUND_HALF_UP")).replace("-", "−")  # noqa: E731
        lines.append(f"| {k[1]} | {k[2]} | {w.get('true_answer', '')} | {k[0]} | {hu(a['logp'])} | {a['rank_first_best']} | {hu(b['logp'])} | {b['rank_first_best']} |")
    (oh.RES / "qualitative/runnerup_examples.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    {"score": score, "analyze": analyze}[sys.argv[1]]()
