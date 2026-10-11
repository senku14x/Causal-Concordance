"""015 POST HOC, descriptive (written after reading oh_all.json): Fc − Pc and Fc − O by state, Oracle texts, both
scorers. Question: is the small cost of correcting wrong answers in place specific to swapped states?
  python posthoc_fc_by_state.py  ->  results/015_oracle_hallucination/analysis/posthoc_fc_by_state.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = sys.argv[:1]
import oh  # noqa: E402

A = json.loads((oh.RES / "analysis/oh_all.json").read_text())
B = A["B"]
V = {x["eid"]: x for x in map(json.loads, open(oh.OUT / "versions/versions_all.jsonl")) if x["reader"] == "oracle"}
out = {"label": "POST HOC, descriptive (015 full run; written after reading oh_all.json)", "cells": {}}
for s in ("oracle_ar", "nla_ar"):
    files = [oh.OUT / f"score/{s}_all.jsonl"] if (oh.OUT / f"score/{s}_all.jsonl").exists() else sorted((oh.OUT / "score").glob(f"{s}_all.shard*.jsonl"))
    cos = {}
    for fp in files:
        for x in map(json.loads, open(fp)):
            if x["reader"] == "oracle" and x["version"] in ("Fc", "Pc", "O") and not x["degenerate"]:
                cos.setdefault((x["eid"], x["version"]), []).append(x["cos"])
    fve = lambda e, n: float(np.mean([100 * (1 - (2 - 2 * c) / B[s]) for c in cos[(e, n)]]))    # noqa: E731
    res = {}
    for st in oh.STATES:
        es = [e for e, v in V.items() if v["state"] == st and (e, "Fc") in cos and (e, "Pc") in cos]
        for name, (a, b) in (("Fc_minus_Pc", ("Fc", "Pc")), ("Fc_minus_O", ("Fc", "O"))):
            dd = [fve(e, a) - fve(e, b) for e in es]
            res.setdefault(name, {})[st] = {"n": len(dd), "mean": float(np.mean(dd)), "ci95": oh.H.boot(dd, [V[e]["item"] for e in es])}
    out["cells"][f"oracle_texts|{s}"] = res
(oh.RES / "analysis/posthoc_fc_by_state.json").write_text(json.dumps(out, indent=1))
for k, r in out["cells"].items():
    for name, by in r.items():
        print(k, name, {st: (round(v["mean"], 2), [round(x, 2) for x in v["ci95"]], v["n"]) for st, v in by.items()})
