"""013: qualitative examples for the results page (CPU). Two illustrative explanations (named) plus four seeded random
ones (random.Random(5)): the original with judge A's contradicted spans and B's kinds, the factual rewrite F, the
paraphrase P, and per-version FVE (deletion C1 and rewrite F / P), all from the full-run outputs.

  python make_examples.py  ->  results/013_nla_hallucination/qualitative/examples.jsonl
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination as H  # noqa: E402

ILLUSTRATIVE = ["b3-mh-kremlin-language|J_L24-35|1", "bf-bordeaux-fruit|clean|0"]


def main():
    rows = {r["eid"]: r for r in H.explanations()}
    J = {json.loads(l)["eid"]: json.loads(l) for l in open(H.OUT / "judge/judged_all.jsonl")}
    RW = {json.loads(l)["eid"]: json.loads(l) for l in open(H.OUT / "rewrite/rewrites_all.jsonl")}
    SD = {json.loads(l)["eid"]: json.loads(l) for l in open(H.OUT / "scored_all.jsonl")}
    SR = {json.loads(l)["eid"]: json.loads(l) for l in open(H.OUT / "rewrite/scored_all.jsonl")}
    Bd = json.loads((H.OUT / "score_meta_all.json").read_text())["baseline_B"]
    Br = json.loads((H.OUT / "rewrite/score_meta_all.json").read_text())["baseline_B"]
    pool = [e for e in rows if e not in ILLUSTRATIVE and J[e]["A_ok"] and J[e]["K_ok"]]
    pick = [(e, "illustrative") for e in ILLUSTRATIVE] + [(e, "random (seed 5)") for e in random.Random(5).sample(sorted(pool), 4)]
    out = []
    for e, why in pick:
        r, j, w, sd, sr = rows[e], J[e], RW[e], SD[e], SR[e]
        spans = [{"text": s, "kind": k} for s, k in zip(j["A_raw"]["contradicted"], j["span_kinds"] or [])]
        out.append({"eid": e, "selection": why, "prompt": r["prompt"], "continuation": r["continuation"], "src": r["src"], "tgt": r["tgt"],
                    "state": r["state"], "explanation": r["explanation"], "contradicted_spans": spans, "F": w["F"], "P": w["P"],
                    "fve": {"O": 100 * (1 - sd["mse"]["O"] / Bd), "C1": 100 * (1 - sd["mse"]["C1"] / Bd) if "C1" in sd["mse"] else None,
                            "F": 100 * (1 - sr["mse"]["F"] / Br), "P": 100 * (1 - sr["mse"]["P"] / Br)}})
    p = H.RES / "qualitative/examples.jsonl"
    with open(p, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("wrote", p, [o["eid"] for o in out])


if __name__ == "__main__":
    main()
