"""012 Stage L: seeded qualitative sample of the Oracle's sampled readouts at H2, L56 (clean vs J_clue vs rand_clue),
for the results page. 4 holds, random.Random(7) over holds whose J_clue swap worked and 2 over holds where it failed.

  python stageL_samples.py  ->  results/012_reader_concordance/qualitative/stageL_samples.jsonl
"""
import ast
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "outputs/012_reader_concordance/stageJ"
tf = lambda x: x is True or x == "True"
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x


def main():
    hs = defaultdict(list)
    for l in open(J / "j2_items.jsonl"):
        t = json.loads(l)
        if tf(t["j1_gated"]) and tf(t["discriminating"]):
            hs[f"{t['clue_id']}|{t['carrier']}"].append(tf(lit(t["arms"])["J_clue"]["success"]))
    ok = {k: sum(v) / len(v) >= 0.5 for k, v in hs.items()}
    R = defaultdict(dict)
    for l in open(ROOT / "outputs/012_reader_concordance/stageL/l_readouts_oracle.jsonl"):
        r = json.loads(l)
        R[r["hold"]][r["state"]] = r
    prompts = {}
    for p in json.loads((ROOT / "outputs/012_reader_concordance/stageI/prompts.json").read_text())["items"]:
        prompts.setdefault(f"{p['clue_id']}|{p['carrier']}", p["prefix_H2"])
    rng = random.Random(7)
    pick = rng.sample(sorted(k for k in ok if ok[k] and k in R), 4) + rng.sample(sorted(k for k in ok if not ok[k] and k in R), 2)
    out = ROOT / "results/012_reader_concordance/qualitative/stageL_samples.jsonl"
    with open(out, "w") as f:
        for k in pick:
            rr = R[k]
            f.write(json.dumps({"hold": k, "swap_worked": ok[k], "src": rr["clean"]["src"], "tgt": rr["clean"]["tgt"], "prefix_H2": prompts[k],
                                "reader": "oracle", "layer": 56, "pos": "H2",
                                **{s: {"samples": rr[s]["samples"], "category": rr[s]["category"]} for s in ("clean", "J_clue", "rand_clue")}},
                               ensure_ascii=False) + "\n")
    print("wrote", out, pick)


if __name__ == "__main__":
    main()
