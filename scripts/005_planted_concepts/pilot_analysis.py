"""Answer the pilot's pre-registered questions Q1-Q9 (design 005 §16.1) from outputs/005_planted_concepts/pilot/<run_id>
and write results/005_planted_concepts/analysis/pilot_summary.json. CPU only. Dev concepts only; descriptive.
Works on a partial run (uses whatever blocks exist)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p005 as P                                   # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "pilot1"
D = P.ROOT / "outputs/005_planted_concepts/pilot" / run
OUT = P.ROOT / "results/005_planted_concepts/analysis/pilot_summary.json"
GRID9 = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
st = json.loads((D / "state.json").read_text())
rng = np.random.default_rng(P.SEED)
S = {"run_id": run, "note": "dev concepts only; descriptive; answers §16.1 Q1-Q9 by their pre-registered rules"}


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


blocks = {}
for f in sorted(D.glob("rows_*_*.json")):
    _, band, pre = f.stem.split("_")
    blocks[(band, pre)] = json.loads(f.read_text())
S["blocks"] = [f"{b}_{p}" for b, p in blocks]


def null95(rows, key, s):
    v = [r[key] for r in rows if r["cond"] == 7 and r["s"] == s]
    return float(np.quantile(v, 0.95)) if v else float("nan")


def cond_s(rows, cond, s, seed0=True):
    return [r for r in rows if r["cond"] == cond and r["s"] == s and (not seed0 or r.get("seed", 0) == 0)]


# ---------------------------------------------------------------- per-block curves (medians over the 10 dev concepts)
curves = {}
for (band, pre), rows in blocks.items():
    c = {}
    for cond in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "10r"):
        for s in GRID9:
            rs = cond_s(rows, cond, s)
            if not rs:
                continue
            r2 = [r["R2"] for r in rs if r.get("R2") is not None]
            c[f"{cond}|{s}"] = {"R1": med([r["R1"] for r in rs]), "dR1p": med([r["dR1p"] for r in rs]),
                                "R2": float(np.mean(r2)) if r2 else None, "B1": med([r["B1"] for r in rs]),
                                "B2": med([r["B2"] for r in rs]), "B4": med([r["B4"] for r in rs]),
                                "B3": med([r["B3"] for r in rs])}
    for s in GRID9:
        c[f"null95|{s}"] = {k: null95(rows, k, s) for k in ("dR1p", "B1", "B2", "B4")}
    curves[f"{band}_{pre}"] = c
S["curves"] = curves


def above_null(rows, cond, key="dR1p"):
    return [s for s in GRID9 if cond_s(rows, cond, s) and
            med([r[key] for r in cond_s(rows, cond, s)]) > null95(rows, key, s)]


# ---------------------------------------------------------------------------------------------------------- Q1
q1 = {}
for (band, pre), rows in blocks.items():
    a1, a2 = above_null(rows, 1), above_null(rows, 2)
    q1[f"{band}_{pre}"] = {"cond1_strengths_above_null": a1, "cond2_strengths_above_null": a2,
                           "rule_met": len(a1) >= 3 and len(a2) >= 3}
S["Q1"] = {"by_block": q1, "answer": "yes" if any(v["rule_met"] for v in q1.values()) else "no",
           "rule": "median dR1p above the rand-dir 95th pct at >= 3 strengths for condition 1 and condition 2",
           "agent_prior_yes": 0.90}

# ---------------------------------------------------------------------------------------------------------- Q2


def best_r2(rows, cond):
    best = (-1.0, None)
    for s in GRID9:
        r2 = [r["R2"] for r in cond_s(rows, cond, s) if r.get("R2") is not None]
        if r2 and np.mean(r2) > best[0]:
            best = (float(np.mean(r2)), s)
    return best


q2 = {}
for (band, pre), rows in blocks.items():
    b1, b2 = best_r2(rows, 1), best_r2(rows, 2)
    q2[f"{band}_{pre}"] = {"cond1_best_R2": b1[0], "cond1_best_s": b1[1], "cond2_best_R2": b2[0], "cond2_best_s": b2[1]}
word_ok = [k for k, v in q2.items() if k.endswith("_word") and max(v["cond1_best_R2"], v["cond2_best_R2"]) >= 0.25]
S["Q2"] = {"by_block": q2, "answer": "yes" if word_ok else "no", "blocks_meeting": word_ok,
           "rule": "word prefill: mean R2 at the best strength for condition 1 or 2 >= 0.25", "agent_prior_yes": 0.30}

# ---------------------------------------------------------------------------------------------------------- Q3
q3 = {}
for pre in ("default", "word"):
    if ("B", pre) not in blocks or ("B2", pre) not in blocks:
        continue
    out = {}
    for band in ("B", "B2"):
        rows = blocks[(band, pre)]
        best = max(GRID9, key=lambda s: med([r["dR1p"] for r in cond_s(rows, 2, s)]) if cond_s(rows, 2, s) else -9e9)
        out[band] = {"cond2_best_s": best, "cond2_best_dR1p": med([r["dR1p"] for r in cond_s(rows, 2, best)]),
                     "B1_at_best": med([r["B1"] for r in cond_s(rows, 2, best)]),
                     "cond1_best_dR1p": max(med([r["dR1p"] for r in cond_s(rows, 1, s)]) for s in GRID9)}
    choose_b2 = (out["B2"]["cond2_best_dR1p"] - out["B"]["cond2_best_dR1p"] >= 1.0
                 and out["B2"]["B1_at_best"] <= out["B"]["B1_at_best"])
    q3[pre] = {**out, "rule_choice": "B2" if choose_b2 else "B"}
S["Q3"] = {"by_prefill": q3, "rule": "B unless B2's condition-2 dR1p exceeds B's by >= 1 nat without a larger B1",
           "agent_prior_B2": 0.50}

# ---------------------------------------------------------------------------------------------------------- Q4
q4 = {}
for (band, pre), rows in blocks.items():
    s_star, trace = P.onset([r for r in rows if r["cond"] in (4, 7)], GRID9)
    q4[f"{band}_{pre}"] = {"s_star": s_star, "inside_grid_above_bottom": s_star is not None and s_star > GRID9[0],
                           "trace": trace, "text_strengths_rule": P.text_strengths(s_star, GRID9)}
S["Q4"] = {"by_block": q4, "agent_prior_above_bottom_in_one_band": 0.60,
           "answer": "yes" if any(v["inside_grid_above_bottom"] for v in q4.values()) else "no"}

# ---------------------------------------------------------------------------------------------------------- Q6
q6 = {}
for (band, pre), rows in blocks.items():
    diffs, diffs_r1 = [], []
    for c in P.DEV:
        perp = {r["s"]: r for r in rows if r["concept"] == c and r["cond"] == 4}
        if not perp:
            continue
        s_b = max(perp, key=lambda s: perp[s]["R1"])                        # u_perp's best strength by R1 (M2)
        if perp[s_b]["dR1p"] <= null95(rows, "dR1p", s_b):
            continue                                                         # u_perp has no effect: not evaluable
        cc = [r for r in rows if r["concept"] == c and r["cond"] == 10 and r["s"] == s_b][0]
        rc = [r for r in rows if r["concept"] == c and r["cond"] == "10r" and r["s"] == s_b][0]
        diffs.append((perp[s_b]["dR1p"] - cc["dR1p"]) - (perp[s_b]["dR1p"] - rc["dR1p"]))
        diffs_r1.append((perp[s_b]["R1"] - cc["R1"]) - (perp[s_b]["R1"] - rc["R1"]))
    ci = boot_ci(diffs)
    q6[f"{band}_{pre}"] = {"n_evaluable": len(diffs), "extra_reduction_dR1p_mean": float(np.mean(diffs)) if diffs else None,
                           "ci": ci, "extra_reduction_R1_mean": float(np.mean(diffs_r1)) if diffs_r1 else None,
                           "distinguishable": bool(diffs) and ci[0] > 0}
S["Q6"] = {"by_block": q6, "agent_prior_distinguishable": 0.25,
           "rule": "c-clamp lowers u_perp's report more than the random clamp (bootstrap CI over concepts excludes 0)"}

# ---------------------------------------------------------------------------------------------------- Q7 / Q8 / Q9
sq = st.get("split_quality", {})
S["Q7"] = {"leaky": sq.get("leaky"), "rule": sq.get("leaky_rule"), "agent_prior_le5": 0.60}
cos = list(sq.get("cos_loo_base", {}).values())
S["Q8"] = {"median": med(cos), "min": float(min(cos)) if cos else None, "agent_prior_below_0.9": 0.70}
js = list(sq.get("J_share", {}).values())
S["J_share_all55"] = {"median": med(js), "min": float(min(js)) if js else None, "max": float(max(js)) if js else None}
S["Q9"] = {"timings": st.get("timings"), "memory": st.get("memory")}

# ---------------------------------------------------------------------------------------- Q5: readers (preview)


def naming_table(reads):
    tab, clean = defaultdict(list), {}
    for q in reads:
        if q["cond"] == "clean":
            clean[(q["layer"], q["kind"])] = {c: float(np.mean([P.names(t, c) for t in q["samples"]])) for c in P.CONCEPTS}
            continue
        tab[(q.get("band"), q["layer"], q["kind"], str(q["cond"]), q["s"])].append(
            float(np.mean([P.names(t, q["concept"]) for t in q["samples"]])))
    return tab, clean


q5 = {}
for name in ("nla", "oracle"):
    f = D / f"{name}_reads.json"
    if not f.exists():
        continue
    tab, clean = naming_table(json.loads(f.read_text()))
    rate = {"|".join(map(str, k)): float(np.mean(v)) for k, v in tab.items()}
    best = {}
    for (band, L, kind, cond, s), v in tab.items():
        key = f"{band}|L{L}|{kind}|{cond}"
        if float(np.mean(v)) > best.get(key, (-1, None))[0]:
            best[key] = (float(np.mean(v)), s)
    q5[name] = {"mean_naming": rate, "best_by_cond": best,
                "clean_over_5pct": {f"L{k[0]}|{k[1]}": sorted(c for c, x in d.items() if x > 0.05) for k, d in clean.items()}}
S["Q5"] = q5
S["Q5_rules_preview"] = {
    "NLA names c at carried C_o for the full vector at some strength (mean over dev > 0.10 and > rand-dir)":
        {b: any(v[0] > 0.10 for k, v in q5.get("nla", {}).get("best_by_cond", {}).items()
                if k.startswith(f"{b}|L42|carried_co|2")) for b in ("B", "B2")} if "nla" in q5 else None,
    "agent_priors": {"carried_full": 0.40, "direct_full": 0.80}}

# ------------------------------------------------------------------- descriptive previews (labelled, decide nothing)
prev = {}
for (band, pre), rows in blocks.items():
    j = {s: med([r["dR1p"] for r in cond_s(rows, 3, s)]) for s in GRID9}
    p_ = {s: med([r["dR1p"] for r in cond_s(rows, 4, s)]) for s in GRID9}
    jn = {s: med([r["dR1p"] for r in cond_s(rows, 5, s)]) for s in GRID9}
    pn = {s: med([r["dR1p"] for r in cond_s(rows, 6, s)]) for s in GRID9}
    prev[f"{band}_{pre}"] = {"M1_like_J_minus_perp_dR1p": {s: j[s] - p_[s] for s in GRID9},
                             "M3_like_Jnat_minus_perpnat_dR1p": {s: jn[s] - pn[s] for s in GRID9}}
S["previews_not_tests"] = prev

# ------------------------------------------------------------------------- step 7b (§16.3): Q10, Q11, R-lens, SEC
D7 = D / "step7b"
S["SEC_crosscheck"] = st.get("SEC_crosscheck")
if D7.exists():
    q10, q11, sweep_curves = {}, {}, {}
    for band in ("B", "B2"):
        f = D7 / f"reentry_{band}.json"
        base = blocks.get((band, "default"))
        if f.exists() and base:
            rows = json.loads(f.read_text())
            out = {}
            for pair in (("10", "10r"), ("12", "12r"), ("13", "13r")):
                diffs, n_eval = [], 0
                for c in P.DEV:
                    perp = {r["s"]: r for r in rows if r["concept"] == c and r["cond"] == 4}
                    s_b = max(perp, key=lambda s: perp[s]["R1"])
                    if perp[s_b]["dR1p"] <= null95(base, "dR1p", s_b):
                        continue
                    n_eval += 1
                    cc = [r for r in rows if r["concept"] == c and str(r["cond"]) == pair[0] and r["s"] == s_b][0]
                    rc = [r for r in rows if r["concept"] == c and str(r["cond"]) == pair[1] and r["s"] == s_b][0]
                    diffs.append(rc["dR1p"] - cc["dR1p"])           # extra reduction by the c-clamp over its null
                ci = boot_ci(diffs)
                med_share = {k: med([r.get(k) for r in rows if str(r["cond"]) in pair and r.get(k) is not None])
                             for k in ("removed_share_band", "removed_share_above", "max_residual")}
                out[f"{pair[0]}_vs_{pair[1]}"] = {"n_evaluable": n_eval, "extra_reduction_mean": float(np.mean(diffs))
                                                  if diffs else None, "ci": ci,
                                                  "specific": bool(diffs) and ci[0] > 0, "clamp_logs": med_share}
            curve = {}
            for cond in (4, 10, "10r", 12, "12r", 13, "13r"):
                curve[str(cond)] = {str(s): med([r["dR1p"] for r in rows if str(r["cond"]) == str(cond) and r["s"] == s])
                                    for s in GRID9}
            q10[band] = {"pairs": out, "dR1p_curves": curve}
        f = D7 / f"sweep_{band}.json"
        if f.exists() and base:
            rows = json.loads(f.read_text())
            s_eval = [s for s in GRID9 if med([r["dR1p"] for r in cond_s(base, 4, s)]) > null95(base, "dR1p", s)]
            first, hits = None, []
            curves7 = {}
            for reader in ("J_NP", "logit", "R_CB"):
                cr = {}
                for cond in (2, 3, 4, 7, "clean"):
                    for sv in GRID9 + [0.0]:
                        rs = [r for r in rows if str(r["cond"]) == str(cond) and r["s"] == sv
                              and (cond != 7 or True)]
                        if not rs:
                            continue
                        cr[f"{cond}|{sv}"] = {str(l): med([r["ranks"][str(l)][reader] if str(l) in r["ranks"]
                                                           else r["ranks"][l][reader] for r in rs])
                                              for l in range(32, 63)}
                curves7[reader] = cr
            for sv in s_eval:
                for l in range(32, 63):
                    perp = [r["ranks"].get(str(l), r["ranks"].get(l))["J_NP"] for r in rows
                            if r["cond"] == 4 and r["s"] == sv]
                    rnd = [r["ranks"].get(str(l), r["ranks"].get(l))["J_NP"] for r in rows
                           if r["cond"] == 7 and r["s"] == sv]
                    m, p5 = med(perp), float(np.quantile(rnd, 0.05)) if rnd else float("nan")
                    if m <= 10 and m < p5:
                        hits.append({"s": sv, "layer": l, "median_rank": m, "rand_p05": p5})
            if hits:
                first = min(hits, key=lambda h: h["layer"])
            q11[band] = {"strengths_evaluable": s_eval, "answer": "yes" if hits else "no", "first_hit": first,
                         "n_hits": len(hits)}
            sweep_curves[band] = curves7
    S["Q10"] = {"by_band": q10, "agent_priors": {"12_specific": 0.40, "13_specific": 0.25},
                "rule": "c-clamp lowers u_perp's dR1p more than its random-clamp null at u_perp's best strength "
                        "(bootstrap CI over evaluable dev concepts excludes 0)"}
    S["Q11"] = {"by_band": q11, "agent_prior_yes": 0.55,
                "rule": "under u_perp at strengths above the null, median J_NP rank of c at C_o <= 10 at some layer "
                        "L32-62 and below the rand-dir 5th-percentile rank there"}
    S["sweep_curves"] = sweep_curves
    f = D7 / "rlens_reads.json"
    if f.exists() and (D / "token_probe_reads.json").exists():
        rl = json.loads(f.read_text())
        tp = json.loads((D / "token_probe_reads.json").read_text())
        key = lambda q: (q["band"], q["layer"], q["kind"], q["concept"], str(q["cond"]), q["seed"], q["s"])
        jn = {key(q): q["token"]["J_NP"]["rank"] for q in tp}
        lg = {key(q): q["token"]["logit"]["rank"] for q in tp}
        pairs = [(jn[key(q)], q["R_CB_rank"], lg[key(q)]) for q in rl if key(q) in jn]
        from scipy.stats import spearmanr
        a = np.array(pairs, float)
        S["rlens_consistency"] = {"n": len(pairs),
                                  "spearman_JNP_RCB": float(spearmanr(a[:, 0], a[:, 1]).correlation),
                                  "spearman_RCB_logit": float(spearmanr(a[:, 1], a[:, 2]).correlation),
                                  "top10_agree_JNP_RCB": float(np.mean((a[:, 0] <= 10) == (a[:, 1] <= 10))),
                                  "top10_rate": {"J_NP": float(np.mean(a[:, 0] <= 10)), "R_CB": float(np.mean(a[:, 1] <= 10)),
                                                 "logit": float(np.mean(a[:, 2] <= 10))}}
    st7 = json.loads((D7 / "state.json").read_text()) if (D7 / "state.json").exists() else {}
    S["cos_atom_vs_output"] = st7.get("cos_atom_vs_output")
    S["timings_7b"] = st7.get("timings")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(S, indent=1, default=str))
print("blocks", S["blocks"])
for q in ("Q1", "Q2", "Q4"):
    print(q, S[q]["answer"], json.dumps(S[q]["by_block"], default=str)[:900])
print("Q3", json.dumps(S["Q3"], default=str)[:900])
print("Q6", json.dumps(S["Q6"], default=str)[:900])
print("Q7", S["Q7"]["leaky"], "Q8", S["Q8"])
for b, c in curves.items():
    print("==", b)
    for cond in (1, 2, 3, 4, 7):
        print(f"  c{cond} dR1p:", " ".join(f"{c.get(f'{cond}|{s}', {}).get('dR1p', float('nan')):+.1f}" for s in GRID9),
              "| R2:", " ".join("-" if c.get(f'{cond}|{s}', {}).get('R2') is None else f"{c[f'{cond}|{s}']['R2']:.2f}"
                               for s in GRID9))
    print("  null95 dR1p:", " ".join(f"{c[f'null95|{s}']['dR1p']:+.1f}" for s in GRID9))
