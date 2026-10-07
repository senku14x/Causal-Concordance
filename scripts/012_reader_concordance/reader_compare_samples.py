"""012: seeded side-by-side reader sample for the results page. For 8 clean-correct items (random.Random(11): 4 where the
J swap at L24-35 changed the answer, 4 where it did not), the first sampled readout of Oracle L40, Oracle L44 and NLA L42
(on-training) in the clean / J_L24-35 / rand_L24-35 states, with the model's own greedy continuation in each state.

  python reader_compare_samples.py  ->  results/012_reader_concordance/qualitative/reader_compare_samples.jsonl
"""
import ast
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "outputs/012_reader_concordance"
STATES = ["clean", "J_L24-35", "rand_L24-35"]
tf = lambda x: x is True or x == "True"
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x


def main():
    items = {}
    for l in open(O / "stageB/items.jsonl"):
        b = json.loads(l)
        if b.get("status") == "evaluated" and tf(b["clean_correct"]) and not tf(b.get("clean_contains_y_b")):
            items[b["name"]] = b
    rd = {}
    for f, reader in (("readouts_oracle.jsonl", "oracle"), ("readouts_nla.jsonl", "nla")):
        for l in open(O / "stageC" / f):
            r = json.loads(l)
            if r["state"] in STATES and r.get("off_training") not in (True, "True") and str(r["layer"]) in ("40", "42", "44"):
                rd[(r["item"], f"{reader} L{r['layer']}", r["state"])] = lit(r["samples"])[0]
    readers = ["oracle L40", "oracle L44", "nla L42"]
    ok = lambda n: tf(lit(items[n]["arms"])["J_L24-35"]["success"])
    full = [n for n in sorted(items) if all((n, r, s) in rd for r in readers for s in STATES)]
    rng = random.Random(11)
    pick = [(n, True) for n in rng.sample([n for n in full if ok(n)], 4)] + [(n, False) for n in rng.sample([n for n in full if not ok(n)], 4)]
    out = ROOT / "results/012_reader_concordance/qualitative/reader_compare_samples.jsonl"
    with open(out, "w") as f:
        for n, worked in pick:
            b = items[n]
            arms = lit(b["arms"])
            f.write(json.dumps({"item": n, "prompt": b["prompt"], "src": b["src"], "tgt": b["tgt"], "swap_worked": worked,
                                "continuation": {s: arms[s]["text"] for s in STATES},
                                "readouts": {r: {s: rd[(n, r, s)] for s in STATES} for r in readers}}, ensure_ascii=False) + "\n")
    print("wrote", out, [n for n, _ in pick])


if __name__ == "__main__":
    main()
