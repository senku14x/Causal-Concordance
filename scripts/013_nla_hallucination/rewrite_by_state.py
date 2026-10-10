"""013 A2 rewrite, split by state (CPU, EXPLORATORY, POST HOC; asked for 2026-10-09 to line up with Cathryn's CW-16b).

rewrite_all.json reports F - P per state but F - O and P - O only pooled. CW-16b's "corrected - orig" is on clean
(unswapped) multihop readouts, items 31-100 of the WSB bank. This adds, from the same scored rows and the same
item-cluster bootstrap (hallucination.boot, 2,000 draws, seed 0):
  - per state: FVE means and F - O, P - O, F - P;
  - clean state on items 31-100 (bank order; CW-14 / CW-16b's 70), and on the remaining items 1-30.
Parity with rewrite_all.json (pooled F - P, F - O, P - O and per-state F - P: mean and ci95) is asserted first.

Not like for like with CW-16b: our F is a minimal factual rewrite that also deletes invented context and unsupported
claims (judge A, 013 spec §12); CW-16b corrects wrong spans in place at the same length. Our FVE baseline B comes from
the 87 items x 3 states gold set; CW-16b centres on its 70 targets. Compare signs and rough sizes, not levels.

  python rewrite_by_state.py  ->  results/013_nla_hallucination/analysis/rewrite_by_state.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination as H  # noqa: E402

sys.path.insert(0, str(H.ROOT / "scripts" / "common"))
import wsb  # noqa: E402


def bank_index():
    items = [x for x in wsb.load_bank("multihop") if isinstance(x, list)][0]
    return {it["name"]: i for i, it in enumerate(items)}


def stat(vals, clus):
    if not vals:
        return None
    return {"n": len(vals), "n_items": len(set(clus)), "mean": float(np.mean(vals)), "ci95": H.boot(vals, clus)}


def main():
    rw = H.OUT / "rewrite"
    B = json.loads((rw / "score_meta_all.json").read_text())["baseline_B"]
    W = [json.loads(l) for l in open(rw / "scored_all.jsonl")]
    ok = [s for s in W if all(k in s["mse"] for k in ("O", "F", "P"))]
    fve = lambda s, k: 100 * (1 - s["mse"][k] / B)
    diffs = {"F_minus_O": lambda s: fve(s, "F") - fve(s, "O"), "P_minus_O": lambda s: fve(s, "P") - fve(s, "O"),
             "F_minus_P": lambda s: fve(s, "F") - fve(s, "P")}
    block = lambda rows: {"fve_mean": {k: float(np.mean([fve(s, k) for s in rows])) for k in ("O", "F", "P")} if rows else None,
                          **{k: stat([f(s) for s in rows], [s["item"] for s in rows]) for k, f in diffs.items()}}

    # parity with the committed rewrite_all.json
    ref = json.loads((H.RES / "analysis/rewrite_all.json").read_text())
    pooled = block(ok)
    for k, rk in (("F_minus_P", "dFVE_F_minus_P"), ("F_minus_O", "dFVE_F_minus_O"), ("P_minus_O", "dFVE_P_minus_O")):
        assert pooled[k]["n"] == ref[rk]["n"], k
        assert abs(pooled[k]["mean"] - ref[rk]["mean"]) < 1e-9, k
        assert all(abs(a - b) < 1e-9 for a, b in zip(pooled[k]["ci95"], ref[rk]["ci95"])), k
    for st in H.STATES:
        rows = [s for s in ok if s["state"] == st]
        got, want = stat([diffs["F_minus_P"](s) for s in rows], [s["item"] for s in rows]), ref["per_state"][st]["dFVE_F_minus_P"]
        assert got["n"] == want["n"] and abs(got["mean"] - want["mean"]) < 1e-9, st
        assert all(abs(a - b) < 1e-9 for a, b in zip(got["ci95"], want["ci95"])), st

    idx = bank_index()
    clean = [s for s in ok if s["state"] == "clean"]
    out = {"label": "EXPLORATORY, POST HOC (2026-10-09): 013 A2 rewrite split by state and by CW-16b's item range; "
                    "same rows, FVE baseline and bootstrap as rewrite_all.json (parity asserted)",
           "baseline_B": B, "n_ok": len(ok), "pooled": pooled,
           "per_state": {st: block([s for s in ok if s["state"] == st]) for st in H.STATES},
           "clean_items_31_100": block([s for s in clean if idx[s["item"]] >= 30]),
           "clean_items_1_30": block([s for s in clean if idx[s["item"]] < 30]),
           "cw16b_reference": {"source": "LittleMeHere/nla-vs-lenses runs/cw16b/analysis.txt",
                               "corrected_minus_orig_FVE_points": 5.4, "ci95": [2.6, 8.0], "n_prompts": 70,
                               "note": "FVE fraction x 100; corrects wrong spans in place at the same length; centred on its 70 targets"}}
    dst = H.RES / "analysis/rewrite_by_state.json"
    dst.write_text(json.dumps(out, indent=1))
    r1 = lambda v: f"{v['mean']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] (n {v['n']}, {v['n_items']} items)"
    for name, blk in [("pooled", pooled)] + list(out["per_state"].items()) + [("clean 31-100", out["clean_items_31_100"]),
                                                                              ("clean 1-30", out["clean_items_1_30"])]:
        print(f"{name:14s} F-O {r1(blk['F_minus_O'])}  P-O {r1(blk['P_minus_O'])}  F-P {r1(blk['F_minus_P'])}")
    print("parity with rewrite_all.json: ok")


if __name__ == "__main__":
    main()
