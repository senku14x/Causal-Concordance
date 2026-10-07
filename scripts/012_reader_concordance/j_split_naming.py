"""POST HOC (EXPLORATORY): naming by activation piece, re-scored from 007's saved multihop readouts (stage 3, Oracle
L40 / L44 and the post hoc NLA L42 run). For each reader x piece (whole / J part / rest / random / PCA / mean-removed):
the share of sampled readouts that name the hidden step (WSB intermediates, whole word, case-insensitive; c012's
matcher) and the answer (WSB target). Item-cluster bootstrap 95% (2,000 draws, seed 0). Test half (items 51-100, 40
items) and dev (27 items) reported separately.

  python j_split_naming.py  ->  results/012_reader_concordance/analysis/j_split_naming.json
"""
import ast
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
import wsb  # noqa: E402
from c012 import NUM, _hit  # noqa: E402

FILES = {"Oracle L40": "stage3/m1/restatement_oracle_L40.jsonl", "Oracle L44": "stage3/m1/restatement_oracle_L44.jsonl",
         "NLA L42": "stage3/m1_nla_posthoc/restatement_nla_L42.jsonl"}
PIECES = {"whole": "whole activation", "J": "J part", "rest_J": "rest (whole minus J)", "R0": "random part (same size as J)",
          "PCA": "PCA part (same size)", "MR": "mean removed"}
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x


def boot(vals, clus, nb=2000, seed=0):
    g = {}
    for v, c in zip(vals, clus):
        g.setdefault(c, []).append(v)
    keys, rng = sorted(g), random.Random(seed)
    b = [np.mean([v for k in (rng.choice(keys) for _ in keys) for v in g[k]]) for _ in range(nb)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    items = {it["name"]: it for it in [x for x in wsb.load_bank("multihop") if isinstance(x, list)][0]}
    names = {}
    for n, it in items.items():
        br = list(lit(it["intermediates"]))
        br += [NUM[x] for x in br if x in NUM]
        names[n] = (br, [it["target"]])
    out = {"label": "POST HOC, EXPLORATORY: 007 multihop readouts re-scored by piece (word match)", "pieces": PIECES, "cells": {}}
    for reader, f in FILES.items():
        for l in open(ROOT / "outputs/007_jsubspace" / f):
            r = json.loads(l)
            br, ans = names[r["item"]]
            for s in lit(r["samples"]):
                key = (reader, r["piece"], r["split"])
                c = out["cells"].setdefault("|".join(key), {"bridge": [], "answer": [], "item": []})
                c["bridge"].append(bool(_hit(s, br)))
                c["answer"].append(bool(_hit(s, ans)))
                c["item"].append(r["item"])
    for k, c in out["cells"].items():
        out["cells"][k] = {"n_readouts": len(c["item"]), "n_items": len(set(c["item"])),
                           "bridge": float(np.mean(c["bridge"])), "bridge_ci95": boot(c["bridge"], c["item"]),
                           "answer": float(np.mean(c["answer"])), "answer_ci95": boot(c["answer"], c["item"])}
    p = ROOT / "results/012_reader_concordance/analysis/j_split_naming.json"
    p.write_text(json.dumps(out, indent=1))
    for k, v in sorted(out["cells"].items()):
        if k.endswith("|test"):
            print(f"{k:28s} n={v['n_readouts']:3d} bridge {v['bridge']:.2f} answer {v['answer']:.2f}")


if __name__ == "__main__":
    main()
