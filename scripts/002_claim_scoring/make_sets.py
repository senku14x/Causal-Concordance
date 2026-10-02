"""Design 002 §3/§6: draw the 20-readout judge-prompt pilot and the 60-readout validation set (stratified by reader x arm)
from the saved 001 snapshot. Writes texts + shuffled candidates only; the truth key is kept in a separate file so the
author can role-code blind to the trajectory.

    python scripts/002_claim_scoring/make_sets.py  ->  results/002_claim_scoring/validation/{pilot,validation,key}.jsonl
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import registry as R  # noqa: E402

SNAP = R.PROJECT / "results" / "001_leverage_profile" / "analysis" / "snapshot.json"
OUTD = R.out_dir("results", "002_claim_scoring", "validation")
SEED = 20261004
STRATA = {("nla", "real"): 12, ("nla", "shuffled"): 6, ("nla", "none"): 4, ("oracle", "real"): 12, ("oracle", "shuffled"): 6,
          ("oracle", "none"): 4, ("oracle_wsb", "real"): 12, ("oracle_wsb", "shuffled"): 4}   # 60


def readouts(sn):
    """Every verbalizer readout as (id, reader, arm, table, text). 'none' arms are table-independent prompts but are
    scored against each table's candidates, as the spec's no-activation baseline."""
    rows = []
    for t in sn["tables"]:
        for key, v in t["readouts"].items():
            p = key.split("|")
            if len(p) == 5 and p[3] in ("nla", "oracle", "oracle_wsb"):
                reader, arm = p[3], p[4]
            elif len(p) == 4 and p[0] == "none" and p[2] in ("nla", "oracle_wsb"):   # per-table no-activation runs
                reader, arm = p[2], "none"
            else:
                continue
            texts = v["explanations"] if reader == "nla" else v
            for k, x in enumerate(texts):
                if x:
                    rows.append({"id": f"{t['item']}|{key}|{k}", "reader": reader, "arm": arm, "table": t["item"], "text": x})
    return rows


def variant_truth(t, rid):
    """Truth for the render a readout was taken from (review of design 002, point 2). Member, substituted (off-path
    substitution keeps the trajectory) and no-activation readouts take the member trajectory; partner readouts take the
    partner's. Shuffled readouts are scored against the recipient variant named in their key (the primary shuffled
    control); donor truth is not used."""
    variant = rid.split("|")[1]
    if variant == "partner":
        return {"variant": "partner", "start": t["partner"]["start"], "z": t["partner"]["z"], "answer": t["partner"]["answer"]}
    return {"variant": variant, "start": t["start"], "z": t["z"], "answer": t["answer"]}


def table_candidates(t):
    """One closed set per table (design 002 v2.1): the member's path, the partner's path, then WSB's off-path options for
    the item. Both pair members' z are candidates for every readout of the table, so start-swap tracking can be scored
    on roles. 6-8 colours."""
    out = []
    for c in [t["start"], t["z"], t["answer"], t["partner"]["start"], t["partner"]["z"], t["partner"]["answer"]] + \
            [c for c in t["candidates"] if c not in (t["start"], t["z"], t["answer"])]:
        if c not in out:
            out.append(c)
    return out


def main():
    sn = json.load(open(SNAP))
    tab = {t["item"]: t for t in sn["tables"]}
    rows = readouts(sn)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    pilot = [rows[i] for i in order[:20]]
    used = {r["id"] for r in pilot}
    val, need = [], dict(STRATA)
    for i in order[20:]:
        r = rows[i]
        k = (r["reader"], r["arm"])
        if need.get(k, 0) > 0 and r["id"] not in used:
            val.append(r)
            need[k] -= 1
    assert not any(need.values()), need
    for name, rs in (("pilot", pilot), ("validation", val)):
        with open(OUTD / f"{name}.jsonl", "w") as f:
            for r in rs:
                cands = [str(c) for c in np.random.default_rng(SEED + sum(map(ord, r["id"]))).permutation(table_candidates(tab[r["table"]]))]
                f.write(json.dumps({"id": r["id"], "reader": r["reader"], "arm": r["arm"], "candidates": cands, "text": r["text"]}, ensure_ascii=False) + "\n")
    with open(OUTD / "key.jsonl", "w") as f:
        for r in pilot + val:
            f.write(json.dumps({"id": r["id"], **variant_truth(tab[r["table"]], r["id"])}) + "\n")
    print(len(pilot), "pilot,", len(val), "validation ->", OUTD)


if __name__ == "__main__":
    main()
