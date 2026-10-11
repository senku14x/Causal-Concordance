"""015 qualitative samples for the results page: 2 illustrative items (the same two as the 013 section of the 012 page)
and 12 drawn at random (random.Random(5); 4 clean, 4 J swap, 4 random swap). EXPLORATORY, descriptive.

For each sample and reader: the seed-0 text of every version, the marked spans (judge A's contradicted spans by kind,
judge B's answer spans), and each version's FVE under both reconstructors (mean over seeds, 013's metric).

  python make_samples.py  ->  results/015_oracle_hallucination/qualitative/samples.json
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = sys.argv[:1]
import oh  # noqa: E402

ILLUSTRATIVE = ["b3-mh-kremlin-language|J_L24-35|1", "bf-bordeaux-fruit|clean|0"]
VERSIONS = ["O", "C1", "R1", "CA", "RA", "PA", "F", "P", "Fc", "Pc", "Pstar"]


def main():
    A = json.loads((oh.RES / "analysis/oh_all.json").read_text())
    B = A["B"]
    O = {r["eid"]: r for r in oh.oracle_rows()}
    N = {r["eid"]: r for r in oh.nla_rows()}
    V = {(x["eid"], x["reader"]): x for x in map(json.loads, open(oh.OUT / "versions/versions_all.jsonl"))}
    jo = {x["eid"]: x for x in map(json.loads, open(oh.OUT / "judge/judged_oracle_all.jsonl"))}
    jn = {x["eid"]: x for x in map(json.loads, open(oh.OUT / "judge/judged_nla_all.jsonl"))}
    j13 = {x["eid"]: x for x in map(json.loads, open(oh.H.OUT / "judge/judged_all.jsonl"))}
    rng = random.Random(5)
    pick = []
    for st in oh.STATES:
        pool = sorted(e for e in O if O[e]["state"] == st and e not in ILLUSTRATIVE)
        pick += rng.sample(pool, 4)
    eids = [(e, "illustrative") for e in ILLUSTRATIVE] + [(e, "random (seed 5)") for e in pick]
    want = {e for e, _ in eids}
    sc = {}
    for s in ("oracle_ar", "nla_ar"):
        files = [oh.OUT / f"score/{s}_all.jsonl"] if (oh.OUT / f"score/{s}_all.jsonl").exists() else \
            sorted((oh.OUT / "score").glob(f"{s}_all.shard*.jsonl"))
        for fp in files:
            for x in map(json.loads, open(fp)):
                if x["eid"] in want and x["version"] in VERSIONS and not x["degenerate"]:
                    sc.setdefault((s, x["eid"], x["reader"], x["version"]), []).append(100 * (1 - (2 - 2 * x["cos"]) / B[s]))

    def marks(text, judged, kinds_ok, ans):
        lab = [None] * len(text)
        kr = judged.get("kind_ranges") if kinds_ok else None
        if kr:
            for k, rr in kr.items():
                for a, b in rr:
                    for i in range(a, b):
                        lab[i] = k
        else:
            for a, b in judged.get("contradicted_ranges") or []:
                for i in range(a, b):
                    lab[i] = "contradicted"
        for a, b in (oh.locate_seq(ans.get("spans") or [], text) if ans.get("ok") else []):
            for i in range(a, b):
                lab[i] = "answer"
        runs, i = [], 0
        while i < len(text):
            j = i
            while j < len(text) and lab[j] == lab[i]:
                j += 1
            if lab[i]:
                runs.append([i, j, lab[i]])
            i = j
        return runs

    out = []
    for e, why in eids:
        r = O[e]
        rec = {"eid": e, "why": why, "item": r["item"], "state": r["state"], "sample": r["sample"], "prompt": r["prompt"],
               "continuation": r["continuation"], "src": r["src"], "tgt": r["tgt"], "readers": {}}
        for rd, rows, judged, ans in (("oracle", O, jo[e], jo[e]["answer"]), ("nla", N, j13[e], jn[e]["answer"])):
            v = V[(e, rd)]
            text = rows[e]["explanation"]
            rec["readers"][rd] = {
                "texts": {n: v["texts"][n][0] for n in VERSIONS if n in v["texts"]},
                "marks": marks(text, judged, bool(judged.get("K_ok")), ans),
                "fve": {s: {n: float(np.mean(sc[(s, e, rd, n)])) for n in VERSIONS if (s, e, rd, n) in sc} for s in ("oracle_ar", "nla_ar")},
                "info": {k: v["info"].get(k) for k in ("n_ans", "PA_computable", "ans_eligible", "nothing_to_clean", "cleaned_away", "A_ok",
                                                         "non_english_O", "words_changed_F", "words_changed_P", "words_changed_Fc")}}
        out.append(rec)
    p = oh.RES / "qualitative/samples.json"
    p.write_text(json.dumps({"label": "EXPLORATORY, descriptive: 015 qualitative samples (2 illustrative + 12 random, seed 5)",
                             "B": B, "samples": out}, indent=1, ensure_ascii=False))
    print("wrote", p, len(out), "samples:", [e for e, _ in eids])


if __name__ == "__main__":
    main()
