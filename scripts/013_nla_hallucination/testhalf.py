"""013, POST HOC (labelled): the deletion and rewrite statistics restricted to 007's multihop split.
The test half is items 51-100 of the WSB bank order (matches 007's own split labels 67/67), so the numbers line up
with the 007 multihop numbers shared earlier. Same statistics and bootstrap as hallucination.py / rewrite.py.

  python testhalf.py   ->  results/013_nla_hallucination/analysis/testhalf.json
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


def split_of():
    items = [x for x in wsb.load_bank("multihop") if isinstance(x, list)][0]
    return {it["name"]: ("dev" if i < 50 else "test") for i, it in enumerate(items)}


def stat(vals, clus):
    if not vals:
        return None
    return {"n": len(vals), "n_items": len(set(clus)), "mean": float(np.mean(vals)), "ci95": H.boot(vals, clus)}


def main():
    sp = split_of()
    out = {"label": "EXPLORATORY, POST HOC: 013 restricted to 007's multihop dev/test halves"}
    # deletion
    B = json.loads((H.OUT / "score_meta_all.json").read_text())["baseline_B"]
    S = [json.loads(l) for l in open(H.OUT / "scored_all.jsonl")]
    f = lambda s, k: 100 * (1 - s["mse"][k] / B)
    for half in ("test", "dev"):
        rows = [s for s in S if sp[s["item"]] == half]
        el = [s for s in rows if s["A_ok"] and not s["nothing_to_clean"] and not s["cleaned_away"]]
        d = {"primary_R1_minus_C1": stat([f(s, "R1") - f(s, "C1") for s in el], [s["item"] for s in el])}
        for short, kname in (("i", "invented_context"), ("ii", "misquote")):
            ek = [s for s in rows if s["A_ok"] and s.get(f"has_{short}") and s.get(f"tok_C1{short}", 0) >= H.MIN_TOK]
            d[f"{kname}_clause"] = stat([f(s, f"R{short}") - f(s, f"C1{short}") for s in ek], [s["item"] for s in ek])
            d[f"{kname}_sentence"] = stat([f(s, f"RS{short}") - f(s, f"C1{short}") for s in ek], [s["item"] for s in ek])
        out[f"deletion_{half}"] = d
    # rewrite (if the full run exists)
    rw = H.OUT / "rewrite"
    if (rw / "scored_all.jsonl").exists():
        Bw = json.loads((rw / "score_meta_all.json").read_text())["baseline_B"]
        W = [json.loads(l) for l in open(rw / "scored_all.jsonl")]
        g = lambda s, k: 100 * (1 - s["mse"][k] / Bw)
        for half in ("test", "dev"):
            ok = [s for s in W if sp[s["item"]] == half and all(k in s["mse"] for k in ("O", "F", "P"))]
            out[f"rewrite_{half}"] = {
                "fve_mean": {k: float(np.mean([g(s, k) for s in ok])) for k in ("O", "F", "P")} if ok else None,
                "dFVE_F_minus_P": stat([g(s, "F") - g(s, "P") for s in ok], [s["item"] for s in ok]),
                "dJ_F_minus_P": stat([s["share_J"]["F"] - s["share_J"]["P"] for s in ok], [s["item"] for s in ok])}
    (H.RES / "analysis/testhalf.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
