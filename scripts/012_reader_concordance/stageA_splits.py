"""012 lever splits (CPU, EXPLORATORY, POST HOC: groupings chosen after seeing the per-item outcomes).

What the pooled lever numbers (J-row swap at every prompt position flips 48/82 at L24-47, 38/82 at L24-35; random
directions 5/82 and 1/82) hide, from the Stage B per-item records:
  1. split by target (Japan vs other), for the J and the random arms;
  2. split by bridge type (the bridge is a country vs anything else; country list fixed below) and, within country
     bridges, by target; and within the relations that have both Japan and non-Japan targets;
     2b. (added 2026-10-10, PI) split by named place: the bridge is a country or one of the geographic places fixed
     below (GEO_PLACES), vs anything else;
  3. per relation (the name's last '-' field, as stageD_analysis.rel);
  4. split by the reflection's direction (edit_direction, as Stage D);
  5. edit size at the read position, ||dh|| / ||h|| averaged over band layers, J vs random arm (Stage B logs the read
     position only; the swap is applied at every prompt position);
  6. AUC of the model's own read-position bridge shift for swap success within toward_tgt items (as stageD_answers),
     with Japan targets excluded and by bridge type.
Parity checks against stageA_summary.json (success / random_control k) and stageD_answers.json (model_output AUCs)
are asserted before anything is written.

  python stageA_splits.py [--stageB DIR] [--analysis DIR]
  -> results/012_reader_concordance/analysis/stageA_splits.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from stageD_analysis import auc, rel  # noqa: E402  (same AUC and relation key as Stage D)

BANDS = ("L24-35", "L24-47")
# Bridges (src) that are countries. Fixed by reading the 82 src values; Scotland and Vatican City counted as countries.
COUNTRIES = {"Argentina", "Australia", "Brazil", "Canada", "China", "Denmark", "Egypt", "Finland", "France", "Greece",
             "India", "Ireland", "Italy", "Japan", "Mexico", "Nepal", "Netherlands", "Norway", "Portugal", "Russia",
             "Scotland", "Spain", "Sweden", "Switzerland", "Vatican"}
# Non-country bridges that are geographic places (continents, cities, rivers, mountains). Fixed 2026-10-10 by reading
# the 40 non-country src values, after their per-item outcomes had been seen (POST HOC). Mars (a planet), people,
# months, numbers, elements, directions and common nouns are not counted.
GEO_PLACES = {"Africa", "Antarctica", "Chicago", "Everest", "Nile", "Olympus", "Sydney"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, float(c - h)), min(1.0, float(c + h))]


def kn(rows, arm):
    k, n = sum(r[arm]["success"] for r in rows), len(rows)
    return {"k": int(k), "n": n, "wilson95": wilson(k, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageB", default=str(ROOT / "outputs/012_reader_concordance/stageB"))
    ap.add_argument("--analysis", default=str(ROOT / "results/012_reader_concordance/analysis"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SB, AN = Path(a.stageB), Path(a.analysis)

    rows = []
    for l in open(SB / "items.jsonl"):
        r = json.loads(l)
        if r.get("status") != "evaluated" or not r["clean_correct"] or r["clean_contains_y_b"]:
            continue                                        # Stage A / D primary cohort (82)
        x = {"name": r["name"], "rel": rel(r["name"]), "src": r["src"], "tgt": r["tgt"], "japan": r["tgt"] == "Japan",
             "country_bridge": r["src"] in COUNTRIES, "named_place": r["src"] in COUNTRIES | GEO_PLACES}
        for b in BANDS:
            for arm in ("J", "rand"):
                A = r["arms"][f"{arm}_{b}"]
                sz = [v["dh"] / v["h"] for v in A["diag"].values() if v.get("h")]
                x[f"{arm}_{b}"] = {"success": bool(A["success"]), "size": float(np.mean(sz)), "dm": float(A["dm"]),
                                   "text_changed": A["text"] != r["arms"]["clean"]["text"]}
            x[f"dir_{b}"] = r["edit_direction"][b]
        rows.append(x)

    # parity with the committed Stage A summary
    SA = json.load(open(AN / "stageA_summary.json"))["runs"]
    for b in BANDS:
        run = SA[f"j_swap_{b}"]
        assert len(rows) == run["success"]["n"] == run["random_control"]["n"], (b, len(rows))
        assert sum(r[f"J_{b}"]["success"] for r in rows) == run["success"]["k"], b
        assert sum(r[f"rand_{b}"]["success"] for r in rows) == run["random_control"]["k"], b
        bt = run["by_target"]
        assert sum(r[f"J_{b}"]["success"] for r in rows if r["japan"]) == bt["Japan"].get("success", 0), b
        assert sum(r[f"J_{b}"]["success"] for r in rows if not r["japan"]) == bt["other_targets"].get("success", 0), b

    out = {"label": "EXPLORATORY, POST HOC (groupings chosen after seeing per-item outcomes); 012 Stage B per-item "
                    "records; cohort = clean-correct items whose clean continuation lacks y_b (as Stage A/D)",
           "n": len(rows), "countries": sorted(COUNTRIES), "geographic_places": sorted(GEO_PLACES),
           "named_place_note": "named place = country or geographic place; grouping added 2026-10-10 (PI), POST HOC",
           "bands": {}}
    for b in BANDS:
        J, R = f"J_{b}", f"rand_{b}"
        g = lambda f: [r for r in rows if f(r)]
        jp, oth = g(lambda r: r["japan"]), g(lambda r: not r["japan"])
        cb, ncb = g(lambda r: r["country_bridge"]), g(lambda r: not r["country_bridge"])
        cb_oth = g(lambda r: r["country_bridge"] and not r["japan"])
        npl, nnpl = g(lambda r: r["named_place"]), g(lambda r: not r["named_place"])
        pl_nc = g(lambda r: r["named_place"] and not r["country_bridge"])
        shared = sorted({r["rel"] for r in jp} & {r["rel"] for r in oth})
        sh_jp = g(lambda r: r["japan"] and r["rel"] in shared)
        sh_oth = g(lambda r: not r["japan"] and r["rel"] in shared)
        per_rel = defaultdict(list)
        for r in rows:
            per_rel[r["rel"]].append(r)
        Js, Rs = np.array([r[J]["size"] for r in rows]), np.array([r[R]["size"] for r in rows])
        sJ = lambda rs: float(np.median([r[J]["size"] for r in rs])) if rs else None
        d = {
            "by_target": {"Japan": {"J": kn(jp, J), "rand": kn(jp, R)}, "other": {"J": kn(oth, J), "rand": kn(oth, R)}},
            "by_bridge": {"country": {"J": kn(cb, J), "rand": kn(cb, R), "n_japan_target": len([r for r in cb if r["japan"]])},
                          "not_country": {"J": kn(ncb, J), "rand": kn(ncb, R),
                                          "n_japan_target": len([r for r in ncb if r["japan"]])},
                          "country_not_japan": {"J": kn(cb_oth, J), "rand": kn(cb_oth, R)},
                          "named_place": {"J": kn(npl, J), "rand": kn(npl, R),
                                          "n_japan_target": len([r for r in npl if r["japan"]])},
                          "not_named_place": {"J": kn(nnpl, J), "rand": kn(nnpl, R)},
                          "place_not_country": {"J": kn(pl_nc, J), "rand": kn(pl_nc, R),
                                                "srcs": sorted({r["src"] for r in pl_nc})}},
            "shared_relations": {"relations": shared, "Japan": {"J": kn(sh_jp, J)}, "other": {"J": kn(sh_oth, J)}},
            "by_relation": {k: {"n": len(v), "J": int(sum(r[J]["success"] for r in v)),
                                "rand": int(sum(r[R]["success"] for r in v)),
                                "n_japan": len([r for r in v if r["japan"]]),
                                "n_country_bridge": len([r for r in v if r["country_bridge"]]),
                                "n_named_place": len([r for r in v if r["named_place"]])}
                            for k, v in sorted(per_rel.items())},
            "by_direction": {dd: {"J": kn(g(lambda r, dd=dd: r[f"dir_{b}"] == dd), J),
                                  "n_japan": len(g(lambda r, dd=dd: r[f"dir_{b}"] == dd and r["japan"]))}
                             for dd in sorted({r[f"dir_{b}"] for r in rows})},
            "edit_size_read_position": {
                "J_median": float(np.median(Js)), "J_mean": float(np.mean(Js)),
                "rand_median": float(np.median(Rs)), "rand_mean": float(np.mean(Rs)),
                "n_rand_larger": int((Rs > Js).sum()),
                "J_median_success": sJ([r for r in rows if r[J]["success"]]),
                "J_median_fail": sJ([r for r in rows if not r[J]["success"]]),
                "note": "||dh||/||h|| at the read position only, mean over band layers; the swap edits every prompt position"},
            "arm_effects": {arm: {"n_text_changed": int(sum(r[f"{arm}_{b}"]["text_changed"] for r in rows)),
                                  "median_dm": float(np.median([r[f"{arm}_{b}"]["dm"] for r in rows])),
                                  "note": "text = greedy continuation vs the clean arm's; dm = change in the y_b - y_a "
                                          "margin (nats, positive = toward the target's answer)"}
                            for arm in ("J", "rand")},
        }
        # model's own read-position output (stageD_answers' AUC), within toward_tgt
        mo = {}
        for l in open(SB / "model_output.jsonl"):
            m = json.loads(l)
            arms = m["arms"] if isinstance(m["arms"], dict) else json.loads(m["arms"])
            mo[m["name"]] = arms.get(J, {}).get("d_model_bridge")
        tt = [r for r in rows if r[f"dir_{b}"] == "toward_tgt" and mo.get(r["name"]) is not None]
        A = lambda rs: {"n": len(rs), "n_success": int(sum(r[J]["success"] for r in rs)),
                        "auc_d_model_bridge": auc([mo[r["name"]] for r in rs], [r[J]["success"] for r in rs])}
        d["model_output_auc_toward_tgt"] = {"all": A(tt), "no_japan": A([r for r in tt if not r["japan"]]),
                                            "country_bridge": A([r for r in tt if r["country_bridge"]]),
                                            "not_country_bridge": A([r for r in tt if not r["country_bridge"]]),
                                            "named_place": A([r for r in tt if r["named_place"]]),
                                            "not_named_place": A([r for r in tt if not r["named_place"]])}
        ref = json.load(open(AN / "stageD_answers.json"))["model_output"][b]
        assert d["model_output_auc_toward_tgt"]["all"]["n"] == ref["n"], b
        assert abs(d["model_output_auc_toward_tgt"]["all"]["auc_d_model_bridge"] - ref["auc_d_model_bridge"]) < 1e-12, b
        out["bands"][b] = d

    dst = Path(a.out) if a.out else ROOT / "results/012_reader_concordance/analysis/stageA_splits.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    for b, d in out["bands"].items():
        bt, bb, sz = d["by_target"], d["by_bridge"], d["edit_size_read_position"]
        print(f"{b}: Japan J {bt['Japan']['J']['k']}/{bt['Japan']['J']['n']} rand {bt['Japan']['rand']['k']} | "
              f"other J {bt['other']['J']['k']}/{bt['other']['J']['n']} rand {bt['other']['rand']['k']} | "
              f"country-bridge J {bb['country']['J']['k']}/{bb['country']['J']['n']} (non-Japan "
              f"{bb['country_not_japan']['J']['k']}/{bb['country_not_japan']['J']['n']}) | not-country "
              f"{bb['not_country']['J']['k']}/{bb['not_country']['J']['n']} | named place "
              f"{bb['named_place']['J']['k']}/{bb['named_place']['J']['n']} (not country "
              f"{bb['place_not_country']['J']['k']}/{bb['place_not_country']['J']['n']}) | other "
              f"{bb['not_named_place']['J']['k']}/{bb['not_named_place']['J']['n']} | size J {sz['J_median']:.4f} rand "
              f"{sz['rand_median']:.4f} rand>J {sz['n_rand_larger']}/{out['n']} | AUC {d['model_output_auc_toward_tgt']}")
    print("parity with stageA_summary.json and stageD_answers.json: ok")


if __name__ == "__main__":
    main()
