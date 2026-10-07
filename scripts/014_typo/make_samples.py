"""014 qualitative samples for the page: 4 random typo items (seed 5); for each, the first sample of Oracle L40, Oracle L44
and NLA L42 on the whole activation and on the rest (whole minus J).

  python make_samples.py  ->  results/014_typo/qualitative/samples.jsonl
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/014_typo"
meta = {json.loads(l)["item"]: json.loads(l) for l in open(D / "meta.jsonl")}
rd = {}
for f in ("readouts_oracle.jsonl", "readouts_nla.jsonl"):
    for l in open(D / f):
        r = json.loads(l)
        rd[(f"{'Oracle' if r['reader'] == 'oracle' else 'NLA'} L{r['layer']}", r["piece"], r["item"])] = r["samples"][0]
out = []
for it in sorted(random.Random(5).sample(sorted(meta), 4)):
    m = meta[it]
    out.append({"item": it, "prompt": m["prompt"], "target": m["target"], "continuation": m["continuation"],
                "readouts": {f"{R}|{pc}": rd[(R, pc, it)] for R in ("Oracle L40", "Oracle L44", "NLA L42") for pc in ("whole", "rest_J")}})
p = ROOT / "results/014_typo/qualitative/samples.jsonl"
p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
print("wrote", p, [x["item"] for x in out])
