"""012 Stage E step 2c, subject side (design `docs/design_specs/012E_ar_concept_swaps.md` §4): gate the cue-varied
multihop prompts on Qwen, keep up to 16 per concept, and capture the class-mean material. EXPLORATORY. Label-side:
never imports reader code.

Regime: Jan's phrase-lens `ModelRunner` in bf16 (the regime of every 012 model-side stage; design §2). Batch-one per
prompt; the 10 samples are 10 copies of one prompt in a single generate call, so nothing is padded (Qwen3.6's
GatedDeltaNet layers make padded batches a risk).

Gate (as the bank, design §4): greedy correct, and >= 9/10 correct at T = 0.8, top_p 0.95, with Jan's
answer_generation_budget. Correct = the continuation starts with the answer (case-, punctuation- and leading-article-
insensitive). The concept word is already absent from every prompt (stageE_common.check_prompt).
Selection, in generator order among passing prompts: at most 4 per relation, at most 5 per answer (1/3 of 16), the first
16; a concept with fewer than 10 is dropped from the class-mean arm.
Captures: block outputs L36-47 (block L output = layer L) at the final prompt token (primary) and at the clue's last
token (sensitivity; found from the tokenizer's offsets; None if the clue string isn't found), float32.

  CUDA_VISIBLE_DEVICES=k PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src \
      python stageE_gate.py --shard k --nshard 2 [--only c1,c2] [--tag smoke]
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from phrase_lens.interventions import answer_generation_budget
from phrase_lens.modeling import ModelRunner

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stageE_common as E                                                  # noqa: E402

MODEL, REV = "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LAYERS = list(range(36, 48))
SEED = 20261010


NUM = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                         "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
MATCHER = "v2"


def norm(s):
    """Matcher v2 (2026-10-10, before any concept's kept set was used): accents folded (NFKD), number words 0-20 mapped
    to digits, a leading article dropped. v1 kept only ASCII letters, so an accented answer ("Brasília") could never
    match its unaccented form, and "5" never matched "five"."""
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    toks = [NUM.get(t, t) for t in re.findall(r"[a-z0-9]+", s.lower())]
    while toks and toks[0] in ("the", "a", "an"):
        toks = toks[1:]
    return toks


def correct(text, answer):
    """The continuation starts with the answer, token-wise (so "lyon" does not match "lyonnais")."""
    t, a = norm(text), norm(answer)
    return bool(a) and t[:len(a)] == a


def clue_last_token(tok, prompt, clue):
    i = prompt.lower().find(clue.lower())
    if i < 0:
        return None
    end = i + len(clue)
    enc = tok(prompt, add_special_tokens=True, return_offsets_mapping=True)
    last = None
    for k, (s, e) in enumerate(enc["offset_mapping"]):
        if e > s and s < end and e <= end + 1:
            last = k
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-tag", default="", help="suffix for this run's outputs (input stays prompts_checked[_tag])")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    osfx = sfx + (f"_{a.out_tag}" if a.out_tag else "")
    src = json.loads((E.OUT / "gen" / f"prompts_checked{sfx}.json").read_text())
    only = {x for x in a.only.split(",") if x}
    concepts = [c for c in sorted(src) if not only or c in only][a.shard::a.nshard]
    out = E.OUT / "gate"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, model = runner.tokenizer, runner.model
    print(f"loaded in {time.time() - t0:.0f}s; shard {a.shard}/{a.nshard}: {len(concepts)} concepts", flush=True)
    recs, caps = {}, {}
    for ci, c in enumerate(concepts):
        t1 = time.time()
        rows = []
        for j, r in enumerate(src[c]["checked"]):
            prompt, ans = r["prompt"], r["answer"]
            ids = runner.encode(prompt)
            budget = answer_generation_budget(tok, prompt, [ans])
            g = runner.generate(ids, max_new_tokens=budget, return_token_ids=True)
            gtext = tok.decode(g, skip_special_tokens=True)
            row = {**r, "greedy": gtext, "greedy_ok": correct(gtext, ans), "n_ok": None, "budget": budget}
            if row["greedy_ok"]:
                torch.manual_seed(SEED + 1000 * ci + j)
                x = torch.tensor([ids] * 10, device=runner.device)
                with torch.no_grad():
                    s = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=budget,
                                       do_sample=True, temperature=0.8, top_p=0.95, num_beams=1, use_cache=True,
                                       pad_token_id=tok.eos_token_id)
                texts = [tok.decode(t[len(ids):], skip_special_tokens=True) for t in s]
                row["n_ok"] = sum(correct(t, ans) for t in texts)
                row["samples"] = texts
            row["pass"] = bool(row["greedy_ok"] and (row["n_ok"] or 0) >= 9)
            rows.append(row)
        kept, per_rel, per_ans = [], {}, {}
        for row in rows:
            if not row["pass"] or len(kept) >= E.P_KEEP:
                continue
            rk, ak = row["relation"].lower(), " ".join(norm(row["answer"]))
            if per_rel.get(rk, 0) >= E.MAX_PER_REL or per_ans.get(ak, 0) >= int(E.P_KEEP * E.MAX_ANSWER_FRAC):
                row["skipped_by_cap"] = True
                continue
            per_rel[rk], per_ans[ak] = per_rel.get(rk, 0) + 1, per_ans.get(ak, 0) + 1
            kept.append(row)
        fin, clu, clue_ok = [], [], []
        for row in kept:
            ids = runner.encode(row["prompt"])
            cp = clue_last_token(tok, row["prompt"], row["clue"])
            pos = [len(ids) - 1] + ([cp] if cp is not None else [])
            X = runner.capture_ids(ids, positions=pos, layers=LAYERS)                 # [L, P, D] float32
            fin.append(X[:, 0])
            clu.append(X[:, 1] if cp is not None else None)
            row["clue_pos"], row["final_pos"] = cp, len(ids) - 1
            clue_ok.append(cp is not None)
        caps[c] = {"final": torch.stack(fin) if fin else None,
                   "clue": torch.stack([x for x in clu if x is not None]) if any(clue_ok) else None,
                   "clue_ok": clue_ok}
        recs[c] = {"n_checked": len(rows), "n_pass": sum(r["pass"] for r in rows), "n_kept": len(kept),
                   "dropped": len(kept) < E.P_MIN, "rows": rows, "kept_idx": [rows.index(r) for r in kept],
                   "seconds": round(time.time() - t1, 1)}
        print(f"[{ci + 1}/{len(concepts)}] {c}: pass {recs[c]['n_pass']}/{len(rows)} kept {len(kept)}"
              f"{' DROPPED' if recs[c]['dropped'] else ''} ({recs[c]['seconds']}s)", flush=True)
    torch.save({"layers": LAYERS, "caps": caps}, out / f"caps{osfx}_shard{a.shard}.pt")
    (out / f"gate{osfx}_shard{a.shard}.json").write_text(json.dumps(
        {"label": "EXPLORATORY; 012 Stage E cue-varied prompt gate (design 012E §4)", "matcher": MATCHER, "shard": a.shard,
         "nshard": a.nshard, "concepts": recs, "seconds": round(time.time() - t0, 1)}, indent=1, ensure_ascii=False))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
