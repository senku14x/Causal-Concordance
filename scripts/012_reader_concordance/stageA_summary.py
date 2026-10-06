"""012 Stage A summary (CPU, EXPLORATORY): Jan's per-item causal outputs -> rates on clean-correct items, with no-op and
contamination categories, per-target / per-relation breakdowns, src/tgt cosine, and seeded qualitative samples.

Categories per item (swap runs):
  uncovered     no direction for src or tgt (not evaluated)
  contaminated  clean answer wrong (clean_correct False); excluded from rates, counted
  clean_target_present  clean continuation already contains y_b (excluded; review 2026-10-06)
  unchanged     intervention continuation identical to the clean continuation (an answer-level no-op; says
                nothing about the size of the activation edit)
  success       counterfactual answer y_b in the intervention continuation (Jan's matcher)
  other         changed continuation without y_b
Ablation runs: `success` is Jan's outcome (original answer survives; lower = more disruption).

  python stageA_summary.py
"""
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/012_reader_concordance/stageA/runs"
RES = ROOT / "results/012_reader_concordance"
BANKS = Path("/workspace/sources/ext/phrase-lens-src/src/phrase_lens/data/benchmarks/causal")


def norm_words(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^\w\s]", " ", s).split()


def contains(text, answers):
    w = norm_words(text)
    for a in answers:
        aw = norm_words(a)
        if aw and any(w[i:i + len(aw)] == aw for i in range(len(w) - len(aw) + 1)):
            return True
    return False


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def rate(k, n):
    return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": wilson(k, n)}


def tf(x):
    return x is True or x == "True"


def main():
    out, samples = {}, []
    bank = {i["name"]: i for b in ("multihop_st", "broadcast_st")
            for i in json.loads((BANKS / f"{b}.json").read_text())["items"]}
    rng = random.Random(20261006)
    for f in sorted(RUNS.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        name = f.stem
        is_abl = "ablation" in name
        cat, by_tgt, by_rel, cos_med = Counter(), defaultdict(Counter), defaultdict(Counter), []
        rnd = Counter()
        for r in rows:
            res = r.get("result") or {}
            if r.get("status") != "evaluated":
                cat["uncovered"] += 1
                continue
            if not tf(r.get("clean_correct")):
                cat["contaminated"] += 1
                continue
            it = bank.get(r["name"], {})
            if not is_abl and contains(res.get("clean"), [it.get("y_b", "")] + list(it.get("y_b_alts", []))):
                cat["clean_target_present"] += 1
                continue
            cs = (res.get("flags") or {}).get("cos_st")
            if cs:
                cos_med.append(sorted(cs)[len(cs) // 2])
            if is_abl:
                c = "survives" if tf(r.get("success")) else "disrupted"
            elif res.get("intervention") == res.get("clean"):
                c = "unchanged"
            else:
                c = "success" if tf(r.get("success")) else "other"
            cat[c] += 1
            rnd["random_" + ("hit" if tf(r.get("random_success")) else "miss")] += 1
            tgt = res.get("target", "?")
            by_tgt["Japan" if tgt == "Japan" else "other_targets"][c] += 1
            by_rel[r["name"].split("-")[-1]][c] += 1
            samples.append({"run": name, "name": r["name"], "src": res.get("source"), "tgt": tgt, "category": c,
                            "clean": res.get("clean"), "intervention": res.get("intervention"),
                            "random_success": tf(r.get("random_success"))})
        n = sum(v for k, v in cat.items() if k not in ("uncovered", "contaminated", "clean_target_present"))
        key = "survives" if is_abl else "success"
        out[name] = {"n_rows": len(rows), "categories": dict(cat), "n_clean_correct": n,
                     key: rate(cat[key], n), "random_control": rate(rnd["random_hit"], n),
                     "unchanged": None if is_abl else rate(cat["unchanged"], n),
                     "by_target": {k: dict(v) for k, v in by_tgt.items()},
                     "by_relation": {k: dict(v) for k, v in sorted(by_rel.items())},
                     "cos_src_tgt_median_over_layers": {"median": sorted(cos_med)[len(cos_med) // 2] if cos_med else None,
                                                        "n": len(cos_med)}}
    RES.joinpath("analysis").mkdir(parents=True, exist_ok=True)
    RES.joinpath("qualitative").mkdir(parents=True, exist_ok=True)
    meta = {"label": "EXPLORATORY; Jan's phrase-lens code unmodified (ea85583); rates on clean-correct items only",
            "runs": out}
    (RES / "analysis/stageA_summary.json").write_text(json.dumps(meta, indent=1))
    by_run = defaultdict(list)
    for s in samples:
        by_run[s["run"]].append(s)
    keep = [x for run in sorted(by_run) for x in rng.sample(by_run[run], min(12, len(by_run[run])))]
    (RES / "qualitative/stageA_samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in keep))
    for k, v in out.items():
        s = v.get("success") or v.get("survives")
        print(f"{k:22s} n={v['n_clean_correct']:3d} {('surv' if 'ablation' in k else 'succ')} {s['rate'] if s['rate'] is None else round(s['rate'], 3)}"
              f"  rand {v['random_control']['rate'] and round(v['random_control']['rate'], 3)}  cats {v['categories']}")


if __name__ == "__main__":
    main()
