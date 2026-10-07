"""014 POST HOC (EXPLORATORY; written after reading the full Part A table): paired per-item contrasts on naming.
Per item, naming = share of the 2 samples naming the target. Contrasts: whole − rest_J (what removing J costs),
J − R0 (J alone vs a random 1,024 slice), and Oracle − NLA on the cost of removing J (the paired "reader-specific" test,
spec 012 §20 style). Item bootstrap, 2,000 draws, seed 0.

  python paired.py [--tag smoke]  ->  results/014_typo/analysis/paired_<tag>.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "012_reader_concordance"))
from c012 import _hit  # noqa: E402

READERS = ("oracle L40", "oracle L44", "nla L42")


def boot(d, nb=2000, seed=0):
    rng, d = random.Random(seed), np.asarray(d, float)
    b = [d[[rng.randrange(len(d)) for _ in d]].mean() for _ in range(nb)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n_items": len(d)}


def main(tag):
    D = ROOT / "outputs/014_typo" / ("" if tag == "full" else tag)
    T = {json.loads(l)["item"]: json.loads(l)["target"] for l in open(D / "meta.jsonl")}
    h = {}
    for f in ("readouts_oracle.jsonl", "readouts_nla.jsonl"):
        for l in open(D / f):
            r = json.loads(l)
            h[(f"{r['reader']} L{r['layer']}", r["piece"], r["item"])] = float(np.mean([_hit(s, [T[r["item"]]]) for s in r["samples"]]))
    items = sorted(T)
    d = lambda R, a, b: [h[(R, a, i)] - h[(R, b, i)] for i in items]
    out = {"label": f"POST HOC, EXPLORATORY (014, tag {tag}): paired per-item naming contrasts", "contrasts": {}}
    for R in READERS:
        out["contrasts"][f"{R}|whole_minus_rest_J"] = boot(d(R, "whole", "rest_J"))
        out["contrasts"][f"{R}|J_minus_R0"] = boot(d(R, "J", "R0"))
    for R in READERS[:2]:
        out["contrasts"][f"{R}_minus_nla L42|whole_minus_rest_J"] = boot([a - b for a, b in zip(d(R, "whole", "rest_J"), d("nla L42", "whole", "rest_J"))])
    p = ROOT / f"results/014_typo/analysis/paired_{tag}.json"
    p.write_text(json.dumps(out, indent=1))
    for k, v in out["contrasts"].items():
        print(f"{k:44s} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    main(ap.parse_args().tag)
