"""Analysis for design 006 (docs/design_specs/006_reader_likelihood.md §4–§6). CPU only.

  python reader_likelihood_analysis.py RUN_ID

Writes results/006_reader_likelihood/analysis/summary.json: gates (G-score1, G-score2, G-pc), tracking D_t per
reader x layer x prefix x cell (t = z, start, answer; 002's statistic), forced choice among 5 candidates, the shuffled
and no-activation controls, Holm over the primary family, and 002's lens tracking for comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import registry as R                              # noqa: E402

COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
SEED, NB, ALPHA = 20261004, 10000, 0.025
PRIMARY = [("nla", 42), ("oracle", 40), ("oracle", 44)]


def lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def per_colour(arr, owner):
    """[n, spellings] -> [n, 12]: log-sum-exp over each colour's spellings."""
    return np.stack([lse(arr[:, [i for i, o in enumerate(owner) if o == c]], 1) for c in COLOURS], 1)


def boot(x, rng):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(NB, len(x)), replace=True).mean(1)
    return {"n": int(len(x)), "mean": float(x.mean()), "ci": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            "p_one_sided_gt0": float(np.mean(m <= 0)), "pos": int((x > 0).sum()), "neg": int((x < 0).sum()),
            "zero": int((x == 0).sum())}


def main():
    run = sys.argv[1]
    D = R.PROJECT / "outputs" / "006_reader_likelihood" / run
    st = json.loads((D / "state.json").read_text())
    meta = json.loads((R.PROJECT / "outputs/001_leverage_profile/run/captures_brew_meta.json").read_text())
    renders = [r for r in meta["renders"] if r["role"] in ("member", "partner")]
    assert [r["key"] for r in renders] == st["material"]["render_keys"]
    items = sorted({r["item"] for r in renders})
    tab = {it: {"m": [n for n, r in enumerate(renders) if r["item"] == it and r["role"] == "member"][0],
                "p": [n for n, r in enumerate(renders) if r["item"] == it and r["role"] == "partner"]} for it in items}
    ci = {c: i for i, c in enumerate(COLOURS)}

    def cands(r):                                  # 002's 5-candidate rule (seeded per member render key)
        off = sorted(set(r["present"]) - set(r["trajectory"]))
        off = [str(x) for x in np.random.default_rng(SEED + sum(map(ord, r["key"]))).permutation(off)[:2]]
        return [r["start"], r["z"], r["answer"]] + off

    S = {"run_id": run, "design": "docs/design_specs/006_reader_likelihood.md", "gates": {}, "blocks": {}}
    S["gates"]["G_score1_seq_vs_next_maxabs"] = st.get("G_score1_seq_vs_next_maxabs")
    g2 = st.get("G_score2_own_logits_maxabs", {})
    S["gates"]["G_score2_own_logits_maxabs"] = g2
    S["gates"]["G_score_pass"] = (all(v <= 1e-3 for v in (st.get("G_score1_seq_vs_next_maxabs") or {}).values())
                                  and all(v <= 1.0 for d in g2.values() for v in d.values()))
    rng = np.random.default_rng(SEED)
    for f in sorted(D.glob("ll_*.npz")):
        z = np.load(f)
        owner = [str(o) for o in z["owner"]]
        name, layer, pk = f.stem.split("_")[1], int(f.stem.split("_")[2][1:]), f.stem.split("_")[3]
        pos = per_colour(z["positive"], owner)
        gpc = float(np.mean(pos.argmax(1) == np.arange(12)))
        blk = {"G_pc_ID": gpc, "G_pc_pass": gpc >= 0.5, "positive_argmax": [COLOURS[i] for i in pos.argmax(1)], "cells": {}}
        for cell in ("emit_colon", "emit_asst", "emit_think"):
            LL = per_colour(z[cell], owner)                            # [232, 12]; last row = dev-mean reference
            ref = LL[-1]
            res = {}
            for t in ("z", "start", "answer"):
                d, dsh = [], []
                for k, it in enumerate(items):
                    m, ps = tab[it]["m"], tab[it]["p"]
                    nxt = tab[items[(k + 1) % len(items)]]
                    ps_used = [p for p in ps if renders[p][t] != renders[m][t]]
                    if not ps_used:
                        continue
                    cm = ci[renders[m][t]]
                    vals, svals = [], []
                    for j, p in enumerate(ps_used):
                        cp = ci[renders[p][t]]
                        vals.append((LL[m, cm] - LL[m, cp]) - (LL[p, cm] - LL[p, cp]))
                        p2 = nxt["p"][j % len(nxt["p"])]
                        svals.append((LL[nxt["m"], cm] - LL[nxt["m"], cp]) - (LL[p2, cm] - LL[p2, cp]))
                    d.append(float(np.mean(vals)))
                    dsh.append(float(np.mean(svals)))
                fc = [max(cands(renders[tab[it]["m"]]), key=lambda c: LL[tab[it]["m"], ci[c]]) == renders[tab[it]["m"]][t]
                      for it in items]
                fc_ref = [max(cands(renders[tab[it]["m"]]), key=lambda c: ref[ci[c]]) == renders[tab[it]["m"]][t] for it in items]
                res[t] = {"tracking": boot(d, rng), "shuffled": boot(dsh, rng),
                          "forced_choice_member": float(np.mean(fc)), "forced_choice_reference": float(np.mean(fc_ref))}
            blk["cells"][cell] = res
        S["blocks"][f"{name}_L{layer}_{pk}"] = blk
    # primary family (emit_colon, primary prefix p0): Holm on one-sided bootstrap p
    fam = {f"{n}_L{l}": S["blocks"][f"{n}_L{l}_p0"]["cells"]["emit_colon"]["z"]["tracking"]["p_one_sided_gt0"]
           for n, l in PRIMARY if f"{n}_L{l}_p0" in S["blocks"]}
    order = sorted(fam, key=fam.get)
    holm, stop = {}, False
    for i, k in enumerate(order):
        thr = ALPHA / (len(order) - i)
        ok = (not stop) and fam[k] <= thr
        stop = stop or not ok
        holm[k] = {"p": fam[k], "threshold": thr, "rejects": ok}
    S["holm_primary"] = holm

    def blk(n, l):
        return S["blocks"].get(f"{n}_L{l}_p0")

    def informative(n, l):
        b = blk(n, l)
        return b is not None and b["G_pc_pass"] and S["gates"]["G_score_pass"]
    S["rules"] = {
        "T1_nla_tracks_z": (holm.get("nla_L42", {}).get("rejects") if informative("nla", 42) else None),
        "T2_oracle_tracks_z": (any(holm.get(f"oracle_L{l}", {}).get("rejects") for l in (40, 44) if informative("oracle", l))
                               if any(informative("oracle", l) for l in (40, 44)) else None),
        "T3_start_not_tracked": all(blk(n, l)["cells"]["emit_colon"]["start"]["tracking"]["ci"][0] <= 0
                                    for n, l in PRIMARY if informative(n, l)),
        "T4_forced_choice_z_gt_0.4": any(blk(n, l)["cells"]["emit_colon"]["z"]["forced_choice_member"] > 0.4
                                         for n, l in PRIMARY if informative(n, l)),
        "T5_G_pc": {f"{n}_L{l}": (blk(n, l) or {}).get("G_pc_pass") for n, l in PRIMARY}}
    lens = R.PROJECT / "results" / "002_claim_scoring" / "analysis" / "scores.json"
    if lens.exists():
        lp = json.loads(lens.read_text()).get("lens_post_hoc", {})
        tr = lp.get("tracking", lp)
        S["lens_002_tracking_emit_colon"] = {k: v for k, v in tr.items() if "|emit_colon|" in k and k.split("|")[-1] in ("L40", "L42", "L44")
                                             or ("|emit_colon|" in k and k.endswith(("|start", "|answer")) and "|L4" in k)}
    out = R.PROJECT / "results" / "006_reader_likelihood" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
