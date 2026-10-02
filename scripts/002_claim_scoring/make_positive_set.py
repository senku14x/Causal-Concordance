"""Design 002 v2.2 (review point 5): positive-claim validation supplement.

Draws 30 judged readouts (10 per reader) where the judge assigned a candidate colour to the intermediate role, excluding
the pilot and the 60-readout validation set. Writes text + candidates only; the judge's labels are NOT written, so the
author can code blind to them (selection uses the judge's output; coding does not see it).

    python scripts/002_claim_scoring/make_positive_set.py  ->  results/002_claim_scoring/validation/positive.jsonl
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claim_scoring as C  # noqa: E402

SEED = 20261005


def main():
    vdir = C.RES / "validation"
    used = {json.loads(l)["id"] for f in ("pilot.jsonl", "validation.jsonl") for l in open(vdir / f)}
    rows, _ = C.all_readouts()
    text = {r["id"]: r for r in rows}
    claims = [json.loads(l) for l in open(C.JDIR / "claims.jsonl")]
    pool = [c for c in claims if c["status"] == "ok" and c["id"] not in used and c["judge"]["intermediate"] not in C.SPECIAL]
    rng = np.random.default_rng(SEED)
    out = []
    for rd in ("nla", "oracle", "oracle_wsb"):
        p = [c for c in pool if c["reader"] == rd]
        out += [p[i] for i in rng.permutation(len(p))[:10]]
    with open(vdir / "positive.jsonl", "w") as f:
        for c in out:
            r = text[c["id"]]
            f.write(json.dumps({"id": c["id"], "reader": r["reader"], "arm": r["arm"], "candidates": r["candidates"],
                                "text": r["text"]}, ensure_ascii=False) + "\n")
    print(len(out), "positive-claim readouts ->", vdir / "positive.jsonl",
          "| pool sizes:", {rd: sum(c["reader"] == rd for c in pool) for rd in ("nla", "oracle", "oracle_wsb")})


if __name__ == "__main__":
    main()
