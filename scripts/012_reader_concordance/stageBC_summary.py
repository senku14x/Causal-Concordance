"""012 Stage B/C summary (CPU; reads saved outputs only). EXPLORATORY.

Primary cohort (§12 contract): evaluated items that are clean-correct and whose clean continuation does not already
contain y_b. Reports dm (float32; Jan's bf16 alongside) per arm, edit_direction counts x Stage A success, d_r per
reader x layer x arm split by Stage A success (the item's J-swap success in that band; the random arm is split by the
same item label) and by edit_direction, readout naming counts, the identity-clamp floor.

  python stageBC_summary.py [--tag smoke]
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from c012 import ROOT, paths


def stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    q = np.percentile(xs, [25, 50, 75])
    return {"n": len(xs), "median": round(float(q[1]), 3), "iqr": [round(float(q[0]), 3), round(float(q[2]), 3)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    P = paths(a.tag)
    res_dir = ROOT / "results/012_reader_concordance/analysis"
    gates = json.loads((res_dir / ("stageB_gates.json" if a.tag == "full" else f"stageB_gates_{a.tag}.json")).read_text())
    ev = [r for r in map(json.loads, open(P["items"])) if r["status"] == "evaluated"]
    coh = {r["name"]: r for r in ev if r["clean_correct"] and not r["clean_contains_y_b"]}
    out = {"tag": a.tag, "n_evaluated": len(ev), "n_primary": len(coh),
           "excluded": {"clean_incorrect": [r["name"] for r in ev if not r["clean_correct"]],
                        "clean_contains_y_b": [r["name"] for r in ev if r["clean_correct"] and r["clean_contains_y_b"]]},
           "gates": {k: {kk: vv for kk, vv in v.items() if kk not in ("mismatches", "not_in_wsb")} if isinstance(v, dict) else v
                     for k, v in gates.items()}}
    band = lambda arm: arm.split("_", 1)[1]
    succ = lambda r, b: bool(r["parity"][b]["stageA_success"])
    out["dm"] = {arm: {"f32": stat([r["arms"][arm]["dm"] for r in coh.values()]),
                       "bf16_jan": stat([r["arms"][arm]["dm_bf16_jan"] for r in coh.values()]),
                       "f32_by_stageA_success": {str(s): stat([r["arms"][arm]["dm"] for r in coh.values() if succ(r, band(arm)) == s])
                                                 for s in (True, False)}}
                 for arm in ("J_L24-35", "rand_L24-35", "J_L24-47", "rand_L24-47")}
    out["edit_direction"] = {}
    for b in ("L24-35", "L24-47"):
        ct = Counter((r["edit_direction"][b], "success" if succ(r, b) else "fail") for r in coh.values())
        out["edit_direction"][b] = {"counts": dict(Counter(r["edit_direction"][b] for r in coh.values())),
                                    "x_stageA": {f"{d}|{s}": n for (d, s), n in sorted(ct.items())},
                                    "dm_J_by_direction": {d: stat([r["arms"][f"J_{b}"]["dm"] for r in coh.values()
                                                                   if r["edit_direction"][b] == d])
                                                          for d in ("toward_tgt", "toward_src")}}
    out["floor_identity_clamp"] = stat([r["floor_identity_clamp"] for r in coh.values()])
    rows = [json.loads(l) for f in ("oracle_lenses", "nla") if P["ll"](f).exists() for l in open(P["ll"](f))]
    g = defaultdict(list)
    for r in rows:
        it = coh.get(r["item"])
        if it is None:
            continue
        b, arm = r["band"], r["state"].split("_")[0]
        key = f"{r['reader']}|L{r['layer']}|{arm}"
        g[(key, "all")].append(r["d_r"])
        g[(key, "stageA_success" if succ(it, b) else "stageA_fail")].append(r["d_r"])
        g[(key, it["edit_direction"][b])].append(r["d_r"])
    out["d_r"] = defaultdict(dict)
    for (key, split), v in sorted(g.items()):
        out["d_r"][key][split] = stat(v)
    nm = defaultdict(Counter)
    for f in ("oracle", "nla"):
        if not P["readouts"](f).exists():
            continue
        for r in map(json.loads, open(P["readouts"](f))):
            if r["item"] in coh:
                key = f"{r['reader']}|L{r['layer']}|{r['state'].split('_')[0]}"
                nm[key].update(f"concept:{c}" for c in r["category"])
                nm[key].update(f"answer:{c}" for c in r.get("answer_category", []))
    out["naming"] = {k: dict(sorted(v.items())) for k, v in sorted(nm.items())}
    f = res_dir / ("stageBC_summary.json" if a.tag == "full" else f"stageBC_summary_{a.tag}.json")
    f.write_text(json.dumps(out, indent=1))
    print(f)


if __name__ == "__main__":
    main()
