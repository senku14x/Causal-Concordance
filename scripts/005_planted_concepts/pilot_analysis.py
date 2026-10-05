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

# ---------------------------------------------------------------------------- token readers + probe (Q5 baseline)
f = D / "token_probe_reads.json"
if f.exists():
    tp = json.loads(f.read_text())
    rnd = defaultdict(list)                    # (band, kind, layer, s, concept) -> probe deltas under rand-dir
    for q in tp:
        if q["cond"] == 7:
            rnd[(q["band"], q["kind"], q["layer"], q["s"], q["concept"])].append(q["probe"] - q["probe_clean"])
    agg = defaultdict(lambda: {"jnp10": [], "logit10": [], "probe_det": []})
    for q in tp:
        k = (q["band"], q["kind"], q["layer"], str(q["cond"]), q["s"])
        a = agg[k]
        a["jnp10"].append(q["token"]["J_NP"]["rank"] <= 10)
        a["logit10"].append(q["token"]["logit"]["rank"] <= 10)
        nul = rnd.get((q["band"], q["kind"], q["layer"], q["s"], q["concept"]))
        if nul and q["cond"] != 7:
            a["probe_det"].append((q["probe"] - q["probe_clean"]) > max(nul))
    for q in tp:
        agg[(q["band"], q["kind"], q["layer"], str(q["cond"]), q["s"])].setdefault("probe_abs", []).append(q["probe"] > 0)
    S["token_probe"] = {"|".join(map(str, k)): {kk: (float(np.mean(v)) if v else None) for kk, v in a.items()}
                        for k, a in agg.items()}
    S["token_probe_note"] = ("probe_det: above the strongest of 5 random directions at the same dose (pre-registered; "
                             "scale-free in a deterministic forward, so it fires at any dose); probe_abs: the probe "
                             "classifies the read as the concept (decision > 0), a post-hoc absolute criterion")

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

# --------------------------------------------------------------------------------- step 8 (§16.4): Q12, Q13, power
D8 = D / "step8"
READ_S8 = [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0]
if D8.exists():
    q12 = {}
    for band in ("B", "B2"):
        f = D8 / f"probe_sweep_{band}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        base = blocks.get((band, "default"))
        window = [s for s in READ_S8 if base and cond_s(base, 4, s) and
                  med([r["dR1p"] for r in cond_s(base, 4, s)]) <= null95(base, "dR1p", s)]
        res = {"strict_P11_window_doses": window, "curves": {}, "first_arrival": {}}
        for cond in (2, 3, 4):
            for sv in READ_S8:
                rate = {}
                for l in range(32, 63):
                    det = []
                    for c in P.DEV:
                        x = [r for r in rows if r["concept"] == c and r["cond"] == cond and r["s"] == sv]
                        rn = [r for r in rows if r["concept"] == c and r["cond"] == 7 and r["s"] == sv]
                        if not x or not rn:
                            continue
                        thr = max(r["dprobe"][str(l)] if str(l) in r["dprobe"] else r["dprobe"][l] for r in rn)
                        v = x[0]["dprobe"][str(l)] if str(l) in x[0]["dprobe"] else x[0]["dprobe"][l]
                        det.append(v > thr)
                    rate[l] = float(np.mean(det)) if det else None
                first = next((l for l in range(32, 63) if rate.get(l) is not None and rate[l] >= 0.5), None)
                res["curves"][f"{cond}|{sv}"] = rate
                res["first_arrival"][f"{cond}|{sv}"] = first
        # post-hoc sensitivity (not pre-registered): absolute criterion, the probe classifies x as c (decision > 0).
        # Reason: the forward pass is deterministic and nearly linear at small doses, so the relative criterion
        # (above the strongest random direction at the same dose) is scale-free and fires at any dose.
        res["curves_absolute"], res["first_arrival_absolute"] = {}, {}
        for cond in (2, 3, 4, 7):
            for sv in READ_S8:
                rate = {}
                for l in range(32, 63):
                    x = [r for r in rows if r["cond"] == cond and r["s"] == sv and (cond != 7 or r["seed"] == 0)
                         and "probe" in r]
                    if x:
                        rate[l] = float(np.mean([(r["probe"][str(l)] if str(l) in r["probe"] else r["probe"][l]) > 0 for r in x]))
                res["curves_absolute"][f"{cond}|{sv}"] = rate
                res["first_arrival_absolute"][f"{cond}|{sv}"] = next((l for l in range(32, 63) if rate.get(l, 0) >= 0.5), None)
        fa = res["first_arrival"]
        res["full_by_L44_at_1/4"] = fa.get(f"2|{0.25}") is not None and fa[f"2|{0.25}"] <= 44
        jf, pf = fa.get(f"3|{0.25}"), fa.get(f"4|{0.25}")
        res["perp_no_later_than_J_at_1/4"] = (pf if pf is not None else 63) <= (jf if jf is not None else 63)
        res["window_any_by_L44"] = any(fa.get(f"{c}|{sv}") is not None and fa[f"{c}|{sv}"] <= 44
                                       for c in (2, 3, 4) for sv in window)
        q12[band] = res
    S["Q12"] = {"by_band": q12, "agent_priors": {"full_by_L44_at_1/4": 0.50, "perp_no_later_than_J_at_1/4": 0.60,
                                                 "window_any_by_L44": 0.20},
                "rule": "first layer where >= 50% of dev concepts are probe-detected at C_o above the max of 5 rand-dir seeds"}
    f = D8 / "oracle_late.json"
    if f.exists():
        reads = json.loads(f.read_text())
        tab, clean = naming_table(reads)
        rate = {"|".join(map(str, k)): float(np.mean(v)) for k, v in tab.items()}
        out13 = {}
        for cond in ("2", "4"):
            hits = []
            for (band, L, kind, cnd, sv), v in tab.items():
                if cnd != cond:
                    continue
                rnd = tab.get((band, L, kind, "7", sv))
                m, mr = float(np.mean(v)), (float(np.mean(rnd)) if rnd else 0.0)
                if m > 0.10 and m > mr:
                    hits.append({"band": band, "layer": L, "s": sv, "rate": m, "rand": mr})
            out13[cond] = {"answer": "yes" if hits else "no", "hits": hits}
        S["Q13"] = {"full": out13["2"], "perp": out13["4"], "rates": rate,
                    "clean_over_5pct": {f"L{k[0]}": sorted(c for c, x in d.items() if x > 0.05) for k, d in clean.items()},
                    "agent_priors": {"full": 0.45, "perp": 0.25}}

# ------------------------------------------------------------------------------- step 9 (§16.5): R9a, R9b robustness
D9 = D / "step9"
base_rows = blocks.get(("B", "default"))
if D9.exists() and base_rows:
    out9 = {}
    for f in sorted(D9.glob("rows_*.json")):
        name = f.stem.split("_", 1)[1]
        rows = json.loads(f.read_text())

        def dmed(cond, s_):
            return med([r["dR1p"] for r in rows if str(r["cond"]) == str(cond) and r["s"] == s_])
        diff = {s_: (dmed(3, s_) - dmed(4, s_)) for s_ in GRID9}
        mid = [diff[1 / 16], diff[1 / 8]]
        hi = [diff[1 / 2], diff[1.0]]
        peaks = {"J": max(dmed(3, s_) for s_ in GRID9), "perp": max(dmed(4, s_) for s_ in GRID9)}
        r9a = all(x > 0 for x in mid) and all(x <= 0 for x in hi)
        pairs = {}
        for pair in (("12", "12r"), ("13", "13r")):
            diffs = []
            for c in P.DEV:
                perp = {r["s"]: r for r in rows if r["concept"] == c and r["cond"] == 4}
                s_b = max(perp, key=lambda x: perp[x]["R1"])
                if perp[s_b]["dR1p"] <= null95(base_rows, "dR1p", s_b):
                    continue
                cc = [r for r in rows if r["concept"] == c and str(r["cond"]) == pair[0] and r["s"] == s_b][0]
                rc = [r for r in rows if r["concept"] == c and str(r["cond"]) == pair[1] and r["s"] == s_b][0]
                diffs.append(rc["dR1p"] - cc["dR1p"])
            ci = boot_ci(diffs)
            pairs[pair[0]] = {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else None, "ci": ci,
                              "specific": bool(diffs) and ci[0] > 0}
        r9b = pairs["12"]["specific"] and not pairs["13"]["specific"]
        out9[name] = {"J_minus_perp_by_dose": diff, "peaks": peaks, "R9a_pattern_holds": r9a, "clamp": pairs,
                      "R9b_pattern_holds": r9b}
    st9 = json.loads((D9 / "state.json").read_text()) if (D9 / "state.json").exists() else {}
    S["step9"] = {"variants": out9, "J_share_median": st9.get("J_share_median"),
                  "R9a_holds_in": sum(v["R9a_pattern_holds"] for v in out9.values()),
                  "R9b_holds_in": sum(v["R9b_pattern_holds"] for v in out9.values()), "n_variants": len(out9),
                  "agent_priors": {"R9a_ge4_of_5": 0.55, "R9b_ge4_of_5": 0.65}}

# 8b power analysis (planning only)
pw = {}
for band in ("B", "B2"):
    rows = blocks.get((band, "default"))
    if not rows:
        continue
    d = []
    for c in P.DEV:
        j = [r["dR1p"] for r in rows if r["concept"] == c and r["cond"] == 3 and r["s"] == 0.25]
        p_ = [r["dR1p"] for r in rows if r["concept"] == c and r["cond"] == 4 and r["s"] == 0.25]
        if j and p_:
            d.append(j[0] - p_[0])
    if len(d) > 2:
        sd = float(np.std(d, ddof=1))
        pw[f"M1_like_dR1p_{band}"] = {"dev_mean": float(np.mean(d)), "dev_sd": sd, "ci_halfwidth_n45": 1.96 * sd / np.sqrt(45)}
for name, reads in (("nla", D / "nla_reads.json"), ("oracle", D / "oracle_reads.json")):
    if not reads.exists():
        continue
    rr = json.loads(reads.read_text())
    for band in ("B", "B2"):
        for sv in (1 / 16, 1 / 4, 1.0):
            per = {}
            for q in rr:
                if q["cond"] == 4 and q.get("band") == band and q["s"] == sv and q["kind"] == "direct_cq" and q["layer"] in (42, 44):
                    per.setdefault(q["concept"], []).append(float(np.mean([P.names(t, q["concept"]) for t in q["samples"]])))
            v = [float(np.mean(x)) for x in per.values()]
            if len(v) > 2:
                obs = float(np.var(v, ddof=1))
                within4 = float(np.mean([p_ * (1 - p_) for p_ in v])) / 4
                true_var = max(obs - within4, 0.0)
                pw[f"{name}_perp_direct_{band}_{sv}"] = {
                    "dev_mean_rate": float(np.mean(v)), "obs_sd": float(np.sqrt(obs)),
                    "ci_halfwidth_n45_k4": 1.96 * np.sqrt(true_var + within4) / np.sqrt(45),
                    "ci_halfwidth_n45_k8": 1.96 * np.sqrt(true_var + within4 / 2) / np.sqrt(45)}
S["power_8b"] = pw

# ---------------------------------------------- step 10 (§16.6): gate G10, Q14 (clamp end layer), Q15 (specificity)
D10 = D / "step10"
if D10.exists():
    s10 = {"agent_priors": {"Q14_B": {"56-62": 0.55, "52-54": 0.20, "46-50": 0.15, "none_or_other": 0.10},
                            "Q15": {"cond2_specific_B": 0.65, "cond2_specific_B2": 0.70, "cond4_specific_B": 0.35,
                                    "cond4_specific_B2": 0.45, "cond1_specific_both": 0.90,
                                    "B2_at_least_as_specific": 0.60, "random_at_chance": 0.90}}}
    # G10: rows that repeat stored rows must reproduce their stored dR1p
    g10 = {}
    same = {"4": "4", "E62": "12", "E62r": "12r", "E44": "13", "E44r": "13r"}
    for band in ("B", "B2"):
        d_spec, d_clamp = [], []
        f, base = D10 / f"spec_{band}.json", blocks.get((band, "default"))
        if f.exists() and base:
            ref = {(r["concept"], str(r["cond"]), r["s"], r.get("seed", 0)): r["dR1p"] for r in base}
            for r in json.loads(f.read_text()):
                k = (r["concept"], str(r["cond"]), r["s"], r.get("seed", 0))
                if k in ref:
                    d_spec.append(abs(r["dR1p"] - ref[k]))
        f, f7 = D10 / f"clampsweep_{band}.json", D / "step7b" / f"reentry_{band}.json"
        if f.exists() and f7.exists():
            ref = {(r["concept"], str(r["cond"]), r["s"]): r["dR1p"] for r in json.loads(f7.read_text())}
            for r in json.loads(f.read_text()):
                k = (r["concept"], same.get(str(r["cond"])), r["s"])
                if k in ref:
                    d_clamp.append(abs(r["dR1p"] - ref[k]))
        for name, d in (("spec", d_spec), ("clamp", d_clamp)):
            if d:
                g10[f"{name}_{band}"] = {"n": len(d), "median_abs": med(d), "p95_abs": float(np.quantile(d, 0.95)),
                                         "max_abs": float(max(d))}
    g10_pass = bool(g10) and all(v["median_abs"] <= 0.05 and v["p95_abs"] <= 0.25 for v in g10.values())
    s10["G10"] = {"checks": g10, "pass": g10_pass, "rule": "median |diff| <= 0.05 and 95th pct <= 0.25 nats"}

    # Q14: extra reduction (r-clamp minus c-clamp) at each evaluable concept's best non-J dose, by end layer E
    cosd = S.get("cos_atom_vs_output") or {}
    q14 = {}
    for band in ("B", "B2"):
        f, base = D10 / f"clampsweep_{band}.json", blocks.get((band, "default"))
        if not (f.exists() and base):
            continue
        rows = json.loads(f.read_text())
        ev = {}
        for c in P.DEV:
            perp = {r["s"]: r for r in rows if r["concept"] == c and r["cond"] == 4}
            s_b = max(perp, key=lambda x: perp[x]["R1"])
            if perp[s_b]["dR1p"] > null95(base, "dR1p", s_b):
                ev[c] = s_b
        by_e = {}
        for E in range(44, 63, 2):
            diffs, rem, un, cv, rv = [], [], [], [], []
            for c, s_b in ev.items():
                cc = [r for r in rows if r["concept"] == c and r["cond"] == f"E{E}" and r["s"] == s_b][0]
                rc = [r for r in rows if r["concept"] == c and r["cond"] == f"E{E}r" and r["s"] == s_b][0]
                diffs.append(rc["dR1p"] - cc["dR1p"])
                rem.append(cc.get("removed_share_above"))
                # descriptive decomposition (added after the results were read; no rule uses it)
                un.append([r for r in rows if r["concept"] == c and r["cond"] == 4 and r["s"] == s_b][0]["dR1p"])
                cv.append(cc["dR1p"])
                rv.append(rc["dR1p"])
            ci = boot_ci(diffs)
            by_e[E] = {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else None, "ci": ci,
                       "specific": bool(diffs) and ci[0] > 0,
                       "mean_dR1p_unclamped": float(np.mean(un)) if un else None,
                       "mean_dR1p_c_clamp": float(np.mean(cv)) if cv else None,
                       "mean_dR1p_r_clamp": float(np.mean(rv)) if rv else None,
                       "c_removed_share_above_median": med([x for x in rem if x is not None]),
                       "cos_atom_vs_output_median": med([cosd[c][str(E)] for c in cosd]) if cosd else None,
                       "dR1p_curve_c": {str(sv): med([r["dR1p"] for r in rows if r["cond"] == f"E{E}" and r["s"] == sv]) for sv in GRID9},
                       "dR1p_curve_r": {str(sv): med([r["dR1p"] for r in rows if r["cond"] == f"E{E}r" and r["s"] == sv]) for sv in GRID9}}
        es = sorted(by_e)
        estar = next((E for E in es if all(by_e[e]["specific"] for e in es if e >= E)), None)
        cat = "none" if estar is None else ("56-62" if estar >= 56 else "52-54" if estar >= 52 else
                                            "46-50" if estar >= 46 else "44")
        amend = {"56-62": "A8 as drafted", "52-54": "intermediate: reported; A8 left to the PI",
                 "46-50": f"M2 end layer becomes E* = {estar} (replacing L44); A8 reworded",
                 "44": "contradicts step 7b: stop and investigate", "none": "contradicts step 7b: stop and investigate"}[cat]
        q14[band] = {"evaluable": ev, "by_E": by_e, "E_star": estar, "category": cat, "amendment": amend,
                     "dR1p_curve_unclamped": {str(sv): med([r["dR1p"] for r in rows if r["cond"] == 4 and r["s"] == sv]) for sv in GRID9}}
    s10["Q14"] = q14

    # Q15: concept specificity of dR1p. SM statistic = the mean over concepts (its bootstrap CI is of the mean);
    # ID = rate over the 10 concepts (over the 50 concept x seed rows for condition 7).
    q15 = {}
    for band in ("B", "B2"):
        f = D10 / f"spec_{band}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        for r in rows:
            r.update(P.specificity(r["dW"], r["concept"], P.DEV))
        nul = {sv: float(np.quantile([r["dR1p"] for r in rows if r["cond"] == 7 and r["s"] == sv], 0.95)) for sv in GRID9}
        out = {}
        for cond in (1, 2, 3, 4, 7):
            per_s, n_ok = {}, 0
            for sv in GRID9:
                rs = [r for r in rows if r["cond"] == cond and r["s"] == sv]
                if cond == 7:
                    sm_c = [float(np.mean([r["SM"] for r in rs if r["concept"] == c])) for c in P.DEV]
                else:
                    sm_c = [r["SM"] for r in rs]
                ci = boot_ci(sm_c)
                idr = float(np.mean([r["ID"] for r in rs]))
                above = cond != 7 and med([r["dR1p"] for r in rs]) > nul[sv]
                ok = bool(above and idr >= 0.5 and ci[0] > 0)
                n_ok += ok
                per_s[str(sv)] = {"dR1p_median": med([r["dR1p"] for r in rs]), "above_null": bool(above),
                                  "ID_rate": idr, "SM_mean": float(np.mean(sm_c)), "SM_ci": ci,
                                  "SM_median": med(sm_c), "pct_median": med([r["pct"] for r in rs]),
                                  "generic_median": med([r["generic"] for r in rs]), "meets": ok}
            res = {"by_dose": per_s}
            if cond == 7:
                res["at_chance"] = all(v["ID_rate"] <= 0.3 for v in per_s.values())
            else:
                res["n_doses_meeting"] = n_ok
                res["specific"] = n_ok >= 3
            out[str(cond)] = res
        q15[band] = out
    if "B" in q15 and "B2" in q15:
        cmp_ = {}
        for sv in (1 / 16, 1 / 4):
            b, b2 = q15["B"]["2"]["by_dose"][str(sv)], q15["B2"]["2"]["by_dose"][str(sv)]
            cmp_[str(sv)] = {"SM_mean_B": b["SM_mean"], "SM_mean_B2": b2["SM_mean"], "ID_B": b["ID_rate"],
                             "ID_B2": b2["ID_rate"], "B2_ge_B": b2["SM_mean"] >= b["SM_mean"] and b2["ID_rate"] >= b["ID_rate"]}
        q15["A3_B2_vs_B"] = {"by_dose": cmp_, "B2_at_least_as_specific": all(v["B2_ge_B"] for v in cmp_.values())}
    s10["Q15"] = q15
    # raw readouts: clean top-10 and, per band, condition 2 at s = 1/4 for every dev concept
    st10 = json.loads((D10 / "state.json").read_text()) if (D10 / "state.json").exists() else {}
    s10["clean_top10_at_quote"] = st10.get("clean_top10_at_quote")
    s10["W_size"] = len(st10.get("W", {}))
    ex = {}
    for band in ("B", "B2"):
        f = D10 / f"spec_{band}.json"
        if f.exists():
            ex[band] = {r["concept"]: {"top10": r["top10"][:6], "dR1p": r["dR1p"]}
                        for r in json.loads(f.read_text()) if r["cond"] == 2 and r["s"] == 0.25}
    s10["examples_cond2_s0.25"] = ex
    s10["timings"] = st10.get("timings")
    S["step10"] = s10

# ------------------------- step 11 (§16.7): consistency, Q16 (output-direction clamp), G11 + Q17 (reader likelihoods)
D11 = D / "step11"
if D11.exists():
    s11 = {"agent_priors": {"Q16_J_specific_B": 0.50, "G11_pass": {"oracle": 0.80, "nla": 0.55},
                            "Q17_perp_at_least_as_sensitive": {"nla": 0.35, "oracle": 0.45},
                            "Q17_full_at_least_as_sensitive": {"nla": 0.40, "oracle": 0.50}}}

    def mean_(x):
        return float(np.mean(x)) if len(x) else None
    # consistency: the unclamped condition-4 rows repeat step 10's
    cons = {}
    for band in ("B", "B2"):
        f, f10 = D11 / f"outclamp_{band}.json", D / "step10" / f"clampsweep_{band}.json"
        if f.exists() and f10.exists():
            ref = {(r["concept"], r["s"]): r["dR1p"] for r in json.loads(f10.read_text()) if r["cond"] == 4}
            d = [abs(r["dR1p"] - ref[(r["concept"], r["s"])]) for r in json.loads(f.read_text()) if r["cond"] == 4]
            cons[band] = {"n": len(d), "median_abs": med(d), "p95_abs": float(np.quantile(d, 0.95)), "max_abs": float(max(d))}
    s11["consistency"] = {"checks": cons, "rule": "G10's: median |diff| <= 0.05 and 95th pct <= 0.25 nats",
                          "pass": bool(cons) and all(v["median_abs"] <= 0.05 and v["p95_abs"] <= 0.25 for v in cons.values())}

    # Q16: J-atom clamp (step 10) vs output-direction clamp (step 11), same tokens, rank and best doses
    q14s = (S.get("step10") or {}).get("Q14", {})
    q16 = {}
    for band in ("B", "B2"):
        f, f10 = D11 / f"outclamp_{band}.json", D / "step10" / f"clampsweep_{band}.json"
        if not (f.exists() and f10.exists() and band in q14s):
            continue
        rows, rows10 = json.loads(f.read_text()), json.loads(f10.read_text())
        ev = q14s[band]["evaluable"]

        def at(rs, c, cond, s_b):
            return [r for r in rs if r["concept"] == c and str(r["cond"]) == cond and r["s"] == s_b][0]["dR1p"]
        by_e = {}
        for E in range(44, 63, 2):
            xo, xj, dd, oc, orr = [], [], [], [], []
            for c, s_b in ev.items():
                o_c, o_r = at(rows, c, f"O{E}", s_b), at(rows, c, f"O{E}r", s_b)
                ej = at(rows10, c, f"E{E}r", s_b) - at(rows10, c, f"E{E}", s_b)
                xo.append(o_r - o_c)
                xj.append(ej)
                dd.append(ej - (o_r - o_c))
                oc.append(o_c)
                orr.append(o_r)
            ci_o, ci_d = boot_ci(xo), boot_ci(dd)
            by_e[E] = {"n": len(dd), "extra_out_mean": mean_(xo), "extra_out_ci": ci_o, "out_specific": bool(xo) and ci_o[0] > 0,
                       "extra_J_mean": mean_(xj), "D_mean": mean_(dd), "D_ci": ci_d, "D_above_0": bool(dd) and ci_d[0] > 0,
                       "D_below_0": bool(dd) and ci_d[1] < 0, "mean_dR1p_O_c": mean_(oc), "mean_dR1p_O_r": mean_(orr)}
        k = sum(by_e[E]["D_above_0"] for E in (48, 50, 52))
        verdict = "J-specific" if k >= 2 else ("not J-specific" if k == 0 else "inconclusive")
        amend = {"J-specific": "M2 as drafted: the J clamp to L50 tests routing through J coordinates",
                 "not J-specific": "M2 reports the output-direction clamp as a mandatory control beside the J clamp; "
                                   "wording 'routing through the concept's output-related coordinates'",
                 "inconclusive": "both clamps reported; the PI decides"}[verdict]
        q16[band] = {"by_E": by_e, "n_mid_D_above_0": k, "verdict": verdict, "amendment": amend}
    s11["Q16"] = q16

    # Q17: reader likelihoods at a forced answer position vs the model's report (absolute onsets on the median SM)
    def onset(curve, th):
        for s_ in GRID9:
            v = curve.get(str(s_))
            if v is not None and v >= th:
                return s_
        return None
    mq = ((S.get("step10") or {}).get("Q15") or {}).get("B", {})
    model_on = {}
    for cond in ("2", "3", "4"):
        cm = {k: v["SM_median"] for k, v in (mq.get(cond) or {}).get("by_dose", {}).items()}
        model_on[cond] = {"onset_1": onset(cm, 1.0), "onset_0.5": onset(cm, 0.5), "SM_median_by_dose": cm}
    readers_ = {}
    for f in sorted(D11.glob("reads_*_L*.json")):
        d = json.loads(f.read_text())
        W, clean = d["W"], d["clean"]
        nat_id = {x["concept"]: max(P.DEV, key=lambda w: x["lpW"][w]) == x["concept"] for x in d["natural"]}
        g11 = float(np.mean(list(nat_id.values())))
        rows = d["rows"]
        for r in rows:
            r["dW"] = {w: r["lpW"][w] - clean[w] for w in W}
            r.update(P.specificity(r["dW"], r["concept"], P.DEV))
        per = {}
        for cond in (2, 3, 4, 7):
            by = {}
            for s_ in GRID9:
                rs = [r for r in rows if r["cond"] == cond and r["s"] == s_]
                sm_c = ([float(np.mean([r["SM"] for r in rs if r["concept"] == c])) for c in P.DEV] if cond == 7
                        else [r["SM"] for r in rs])
                by[str(s_)] = {"ID_rate": float(np.mean([r["ID"] for r in rs])), "SM_median": med(sm_c),
                               "SM_mean": mean_(sm_c), "SM_ci": boot_ci(sm_c),
                               "dc_median": med([r["dW"][r["concept"]] for r in rs]),
                               "generic_median": med([r["generic"] for r in rs])}
            sm_curve = {k: v["SM_median"] for k, v in by.items()}
            per[str(cond)] = {"by_dose": by, "onset_1": onset(sm_curve, 1.0), "onset_0.5": onset(sm_curve, 0.5)}
        readers_[f"{d['reader']}_L{d['layer']}"] = {"reader": d["reader"], "layer": d["layer"], "prefix": d["prefix"],
                                                   "W_size": len(W), "G11_ID_natural": g11, "G11_pass": g11 >= 0.7,
                                                   "natural_ID_by_concept": nat_id, "by_cond": per}
    verdicts = {}
    for name in ("nla", "oracle"):
        mine = [k for k, v in readers_.items() if v["reader"] == name]
        ok_layers = [k for k in mine if readers_[k]["G11_pass"]]
        res = {"layers": mine, "layers_passing_G11": ok_layers}
        for cond in ("2", "3", "4"):
            out = {}
            for th in ("1", "0.5"):
                ons = [readers_[k]["by_cond"][cond][f"onset_{th}"] for k in ok_layers
                       if readers_[k]["by_cond"][cond][f"onset_{th}"] is not None]
                s_r, s_m = (min(ons) if ons else None), model_on[cond][f"onset_{th}"]
                out[th] = {"reader_onset": s_r, "model_onset": s_m,
                           "at_least_as_sensitive": (s_r is not None and (s_m is None or s_r <= s_m)) if ok_layers else None}
            res[cond] = out
        res["uninformative"] = not ok_layers
        verdicts[name] = res
    withdrawn = [n for n, v in verdicts.items() if not v["uninformative"] and v["4"]["1"]["at_least_as_sensitive"]]
    s11["Q17"] = {"model_onsets_bandB": model_on, "readers": readers_, "verdicts": verdicts,
                  "preview_withdrawn_for": withdrawn,
                  "amendment": ("P11′ uses reader likelihoods as the readers' primary measure (naming secondary); the "
                                "naming-based 'readers are less sensitive' preview is withdrawn for " + ", ".join(withdrawn))
                  if withdrawn else ("the preview stands under a matched measure; P11′ as drafted with a likelihood column"),
                  "rule": "s_R(1 nat) <= s_M(1 nat) on the median SM; readers failing G11 are uninformative"}
    st11 = json.loads((D11 / "state.json").read_text()) if (D11 / "state.json").exists() else {}
    s11["scoring_check_maxabs"] = st11.get("scoring_check_maxabs")
    s11["timings"] = st11.get("timings")
    S["step11"] = s11

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
