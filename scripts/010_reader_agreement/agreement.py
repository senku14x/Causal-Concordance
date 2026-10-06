"""010 Task 1: reader agreement at the brew cells (EXPLORATORY, post hoc, CPU only; no model, no GPU, no API).

  CUDA_VISIBLE_DEVICES="" /venv/main/bin/python scripts/010_reader_agreement/agreement.py

Reads saved log-likelihood matrices only:
  - 006 readers (prefix p0; p1 for prefix reliability): outputs/006_reader_likelihood/run1/ll_*_p{0,1}.npz
  - 006b model output: outputs/006_reader_likelihood/model_m1/ll_model.npz
  - 007 per-piece reader LLs: outputs/007_jsubspace/stage1a/{r2, r2_nla_posthoc}/ll (`:`),
    stage1b/{r1, r1_nla_posthoc_A3}/ll (`</think>`), brew_test/{t1, t1_nla_posthoc}/ll (49 test tables)
  - 002 token-lens colour scores: results/002_claim_scoring/lens/colour_scores.json
  - 001 whole-cell patch shares S: outputs/001_leverage_profile/run/profile_brew.jsonl (82 gated dev pairs)
D statistic: 006's, via 007's dstat (start/answer collision-free per 006's census; z always scorable).
Writes results/010_reader_agreement/analysis/{agreement.json, per_pair_D.json}.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np                                                        # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "scripts" / "007_jsubspace"))
import agree_lib as A                                                     # noqa: E402
import dstat                                                              # noqa: E402

O6 = PROJECT / "outputs" / "006_reader_likelihood"
O7 = PROJECT / "outputs" / "007_jsubspace"
RES = PROJECT / "results" / "010_reader_agreement" / "analysis"
CELLS = {"colon": "emit_colon", "think": "emit_think", "asst": "emit_asst"}
ROLES = ("z", "start", "answer")
READERS006 = {"NLA L42": "nla_L42", "Oracle L40": "oracle_L40", "Oracle L44": "oracle_L44", "Oracle L56": "oracle_L56",
              "Oracle L60": "oracle_L60"}
LAYER = {"NLA L42": 42, "Oracle L40": 40, "Oracle L44": 44, "Oracle L56": 56, "Oracle L60": 60}
MODEL = "Model output"
PIECES = ("whole", "J", "rest_J", "MR")
R007 = {"Oracle L40": "oracle_L40", "Oracle L44": "oracle_L44", "NLA L42": "nla_L42"}
RUN007 = {"colon": ("stage1a", {"oracle_L40": "r2", "oracle_L44": "r2", "nla_L42": "r2_nla_posthoc"}),
          "think": ("stage1b", {"oracle_L40": "r1", "oracle_L44": "r1", "nla_L42": "r1_nla_posthoc_A3"})}
TEST007 = {"oracle_L40": "t1", "oracle_L44": "t1", "nla_L42": "t1_nla_posthoc"}
LENSES = {"J_NP_cosine": "Lens J_NP-cos", "J_CB_standard": "Lens J_CB", "logit": "Lens logit"}
LENS_LAYERS = (40, 42, 44)
S_LAYERS = (40, 42, 44, 56, 60)


def name007(r, pc):
    return f"007 {r} {pc}"


def layer_of(nm):
    for tok in nm.split():
        if tok.startswith("L") and tok[1:].isdigit():
            return int(tok[1:])
    return None


# ------------------------------------------------------------------------------------------------------- load
def load_dev():
    roles = json.loads((O7 / "stage0" / "roles.json").read_text())
    pairs = roles["pairs"]
    meta = json.loads((PROJECT / "outputs/001_leverage_profile/run/captures_brew_meta.json").read_text())
    RS = [r for r in meta["renders"] if r["role"] in ("member", "partner")]
    assert [r["key"] for r in RS] == [r["key"] for r in roles["renders"]]
    LL = {}                                                               # (instrument, cell) -> [231, 12]
    for nm, f in READERS006.items():
        for pk in ("p0", "p1"):
            z = np.load(O6 / "run1" / f"ll_{f}_{pk}.npz")
            assert z["emit_colon"].shape[0] == 232                         # last row = dev-mean reference
            for c, cn in CELLS.items():
                LL[(nm if pk == "p0" else f"{nm} (p1)", c)] = dstat.colour_ll(z[cn][:231], z["owner"])
    z = np.load(O6 / "model_m1" / "ll_model.npz")
    assert [str(k) for k in z["keys"]] == [r["key"] for r in RS]
    cl = [str(c) for c in z["cells"]]
    for c, cn in CELLS.items():
        LL[(MODEL, c)] = dstat.colour_ll(z["LL"][:, cl.index(cn)], z["owner"])
    for c, (stage, runs) in RUN007.items():
        cname = {"colon": "colon", "think": "think"}[c]
        for r, tag in R007.items():
            d = O7 / stage / runs[tag] / "ll"
            own = np.load(d / "owner.npy")
            for pc in PIECES:
                LL[(name007(r, pc), c)] = dstat.colour_ll(np.load(d / f"{tag}_{cname}_{pc}.npy"), own)
    ls = json.loads((PROJECT / "results/002_claim_scoring/lens/colour_scores.json").read_text())
    assert list(ls["colours"]) == list(dstat.COLOURS)
    keys = [r["key"] for r in ls["renders"]]
    ridx = [keys.index(r["key"]) for r in RS]
    for rd, nm in LENSES.items():
        for L in LENS_LAYERS:
            for c, cn in CELLS.items():
                S = np.array(ls["scores"][f"{rd}|{ls['cells'].index(cn)}|{ls['layers'].index(L)}"])[ridx]
                LL[(f"{nm} L{L}", c)] = np.log(np.clip(S, 1e-30, None)) if rd != "J_NP_cosine" else S   # 002's transform
    return pairs, RS, LL


def load_test():
    roles = json.loads((O7 / "brew_test" / "roles.json").read_text())
    LL = {}
    for r, tag in R007.items():
        d = O7 / "brew_test" / TEST007[tag] / "ll"
        own = np.load(d / "owner.npy")
        for pc in PIECES:
            LL[name007(r, pc)] = dstat.colour_ll(np.load(d / f"{tag}_colon_{pc}.npy"), own)
    return roles["pairs"], LL


def load_S(pairs):
    """001 whole-cell patch share S (p->m) per gated dev pair, cell and layer."""
    S = {}
    for l in open(PROJECT / "outputs/001_leverage_profile/run/profile_brew.jsonl"):
        p = json.loads(l)
        if p["direction"] == "p->m" and p["region"] in CELLS.values():
            S[(p["pair_id"], p["region"], p["layer"])] = p["S"]
    pid = [f"{p['item']}__{p['p_roles']['start']}" for p in pairs]
    gated = [i for i, x in enumerate(pid) if (x, "emit_colon", 42) in S]
    gate = {}
    for l in open(PROJECT / "results/001_leverage_profile/analysis/gate_pairs.jsonl"):
        g = json.loads(l)
        gate[g["pair_id"]] = bool(g["pass"])
    assert sorted(gated) == sorted(i for i, x in enumerate(pid) if gate.get(x, False))
    out = {}
    for c, cn in CELLS.items():
        for L in S_LAYERS:
            out[(c, L)] = np.array([S[(pid[i], cn, L)] for i in gated])
    return np.array(gated), out, pid


# ------------------------------------------------------------------------------------------------------- helpers
def pair_vals(LLc, pairs, role):
    return dstat.pair_D(LLc, pairs, role)[0]


def inst_summary(v, tabs):
    ok = ~np.isnan(v)
    tm, ts = dstat.table_means(v, tabs)
    idx = A.boot_idx(len(ts))
    return {"table": A.summarize(tm, idx),
            "pairs": {"n": int(ok.sum()), "mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                      "pos": int((v[ok] > 0).sum()), "neg": int((v[ok] < 0).sum()),
                      "abs_max": float(np.nanmax(np.abs(v)))}}


def block(names, vals, tabs, control=None, keep=False):
    """vals: name -> [n_pairs] with a common NaN pattern; returns agreement_block over the scorable pairs."""
    X = np.stack([vals[n] for n in names])
    ok = ~np.isnan(X).any(0)
    assert np.isnan(X[:, ok]).sum() == 0 and np.isfinite(X[:, ok]).all()
    return A.agreement_block(names, X[:, ok], np.asarray(tabs)[ok], control=control, keep=keep)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    pairs, RS, LL = load_dev()
    tabs = np.array([p["table"] for p in pairs])
    insts = sorted({k[0] for k in LL})
    D = {}                                                                 # (inst, cell, role) -> [183]
    for (nm, c), M in LL.items():
        for role in ROLES:
            D[(nm, c, role)] = pair_vals(M, pairs, role)
    core = list(READERS006) + [MODEL]
    p007 = [name007(r, pc) for r in R007 for pc in PIECES]
    lens = [f"{LENSES[rd]} L{L}" for rd in LENSES for L in LENS_LAYERS]
    res = {"label": "EXPLORATORY, post hoc (010, 2026-10-06); CPU only; saved LL matrices; dev brew (48 tables, 183 pairs) "
                    "unless stated; no multiplicity correction; wording: decodable through the reader's likelihood",
           "conventions": {"D": "006's per-pair D_t = [LL_m(t_m) - LL_m(t_p)] - [LL_p(t_m) - LL_p(t_p)]; table = mean over "
                                "its scorable pairs; start/answer collision-free (006 census: 63 collisions each, 120 "
                                "scorable pairs, 44 tables)",
                           "bootstrap": "table-cluster, B=10,000, default_rng(20261004).integers (007 boot_idx); pair-level "
                                        "statistics resample tables and keep all their pairs; percentile 95% CI",
                           "derangement_null": "1,000 derangements of tables (default_rng(1); first 200 = 006's); the second "
                                               "instrument's per-table values are permuted; one-sided p = (1 + #null >= obs) / 1001",
                           "partial": "partial correlation of two instruments' per-pair (or per-table) D controlling for the "
                                      "model's own D at the same cell and role (006b); model_cov_share = cov(a,m) cov(b,m) / "
                                      "(var(m) cov(a,b)) = share of the a-b covariance carried by the linear model term",
                           "lens_scores": "002's per-render colour scores; log(clip(p, 1e-30)) for softmax lenses, raw cosine "
                                          "for J_NP-cos (002's transform); probabilities are stored rounded to 1e-7, so J_CB "
                                          "log-scores are coarse for small p; J_NP standard excluded (thousands of exact zeros)",
                           "007_scorers": "007 Oracle pieces use the exact 2-row scorer (bf16 layout differs from 006's; "
                                          "whole reproduces 006's D_z to 0.001); 007 NLA pieces use 006's 24-row scorer (A2/A3)"},
           "instruments": {"006_readers_p0": list(READERS006), "model": MODEL, "007_pieces": p007, "lenses": lens}}

    # ---------------------------------------------------------------------------------------------- reproduction
    rep = {}
    s006 = json.loads((PROJECT / "results/006_reader_likelihood/analysis/summary.json").read_text())
    ph = json.loads((PROJECT / "results/006_reader_likelihood/analysis/posthoc.json").read_text())
    mo = json.loads((PROJECT / "results/006_reader_likelihood/analysis/model_output.json").read_text())

    def chk(key, expected, got, tol=1e-6):          # 006 took the log-sum-exp in float32, 010 (007's dstat) in float64
        rep[key] = {"reported": float(expected), "recomputed": float(got), "abs_diff": float(abs(expected - got)),
                    "pass": bool(abs(expected - got) <= tol)}

    def tmean(nm, c, role):
        return float(dstat.table_means(D[(nm, c, role)], tabs)[0].mean())
    for nm, f in READERS006.items():
        for pk in ("p0", "p1"):
            nmk = nm if pk == "p0" else f"{nm} (p1)"
            for c, cn in CELLS.items():
                chk(f"006 {nm} {pk} {cn} D_z", s006["blocks"][f"{f}_{pk}"]["cells"][cn]["z"]["tracking"]["mean"], tmean(nmk, c, "z"))
    for nm in ("NLA L42", "Oracle L40", "Oracle L44"):
        for role in ("start", "answer"):
            chk(f"006 posthoc {nm} emit_colon D_{role} collision-free", ph["collisions"][nm][role]["mean"], tmean(nm, "colon", role))
    for c, cn in CELLS.items():
        for role in ROLES:
            chk(f"006b model {cn} D_{role}", mo["model_D"][cn][role]["mean"], tmean(MODEL, c, role))
    for f_, c in (("posthoc_A2_stage1a", "colon"), ("posthoc_A2_stage1b", "think")):
        j = json.loads((PROJECT / f"results/007_jsubspace/analysis/{f_}.json").read_text())
        for r, tag in R007.items():
            for pc in PIECES:
                chk(f"007 {f_} {tag} {pc} D_z", j["readers"][tag]["pieces"][pc]["z"]["mean"], tmean(name007(r, pc), c, "z"))
    sc = json.loads((PROJECT / "results/002_claim_scoring/analysis/scores.json").read_text())["lens_post_hoc"]["start_swap_tracking_logscore"]
    for rd, nm in LENSES.items():
        for L in LENS_LAYERS:
            for c, cn in CELLS.items():
                chk(f"002 lens {rd} {cn} L{L} D_z", sc[f"{rd}|{cn}|L{L}"]["mean"], tmean(f"{nm} L{L}", c, "z"))
    # per-pair agreement already reported (006 posthoc §6, 006b B2 r, 007 A2 agreement r)
    for o in ("Oracle L40", "Oracle L44"):
        a, b = D[("NLA L42", "colon", "z")], D[(o, "colon", "z")]
        chk(f"006 posthoc agreement NLA L42 vs {o} r", ph["agreement"][f"NLA L42 vs {o}"]["r"], np.corrcoef(a, b)[0, 1])
        chk(f"006 posthoc agreement NLA L42 vs {o} same_sign", ph["agreement"][f"NLA L42 vs {o}"]["same_sign"],
            np.mean(np.sign(a) == np.sign(b)))
    for nm in ("NLA L42", "Oracle L40", "Oracle L44"):
        chk(f"006 posthoc {nm} p0 vs p1 r", ph["agreement"][f"{nm} p0 vs p1"]["r"],
            np.corrcoef(D[(nm, "colon", "z")], D[(f"{nm} (p1)", "colon", "z")])[0, 1])
        chk(f"006b B2 r reader vs model {nm}", mo["B2_reader_beyond_model"][nm]["r"],
            np.corrcoef(D[(MODEL, "colon", "z")], D[(nm, "colon", "z")])[0, 1])
    for f_, c in (("posthoc_A2_stage1a", "colon"), ("posthoc_A2_stage1b", "think")):
        j = json.loads((PROJECT / f"results/007_jsubspace/analysis/{f_}.json").read_text())["agreement"]
        for o in ("Oracle L40", "Oracle L44"):
            for pc in PIECES:
                chk(f"007 {f_} agreement NLA vs {o} {pc} r", j[f"nla_L42_vs_{R007[o]}"][pc]["r"],
                    np.corrcoef(D[(name007("NLA L42", pc), c, "z")], D[(name007(o, pc), c, "z")])[0, 1])
    gated, S, pid = load_S(pairs)
    for c, cn, med in (("colon", "emit_colon", 0.48), ("asst", "emit_asst", 0.29), ("think", "emit_think", 0.01)):
        chk(f"001 median S {cn} L42 (reported to 2 dp)", med, A.rhu(np.median(S[(c, 42)]), 2), tol=1e-9)
    rep["_n_gated_pairs"] = int(len(gated))
    rep["_n_gated_tables"] = int(len(np.unique(tabs[gated])))
    rep["_all_pass"] = bool(all(v["pass"] for k, v in rep.items() if isinstance(v, dict)))
    res["reproduction"] = rep
    print("reproduction all pass:", rep["_all_pass"], "| failures:", [k for k, v in rep.items() if isinstance(v, dict) and not v["pass"]])

    # ---------------------------------------------------------------------------------------------- (a) per instrument
    per = {}
    for c in CELLS:
        for role in ROLES:
            for nm in insts:
                if (nm, c, role) in D:
                    per.setdefault(c, {}).setdefault(role, {})[nm] = inst_summary(D[(nm, c, role)], tabs)
    res["per_instrument"] = per
    ppd = {"pair_ids": pid, "tables": tabs.tolist(), "items": [p["item"] for p in pairs],
           "note": "per-pair D (NaN = role collision); keys 'instrument|cell|role'",
           "D": {f"{k[0]}|{k[1]}|{k[2]}": [None if np.isnan(x) else float(x) for x in v] for k, v in D.items()}}
    (RES / "per_pair_D.json").write_text(json.dumps(ppd))

    # ---------------------------------------------------------------------------------------------- (b, c) agreement
    ag, kept = {}, {}
    plan = {"colon|z": core + p007 + lens, "colon|start": core + p007 + lens, "colon|answer": core + p007 + lens,
            "think|z": core + p007 + lens, "asst|z": core + lens}
    for key, names in plan.items():
        c, role = key.split("|")
        recs, kb = block(names, {n: D[(n, c, role)] for n in names}, tabs, control=MODEL, keep=key in ("colon|z", "think|z"))
        ag[key] = recs
        if kb:
            kept[key] = kb
        print("agreement block", key, len(recs))
    # prefix reliability (p0 vs p1; p1 = the colour-forcing prefixes "- The colour is" / "<explanation>\nThe colour is"), 006 vs 007 re-score
    rel = {}
    for c in CELLS:
        names = []
        for nm in READERS006:
            names += [nm, f"{nm} (p1)"]
        rlens = ["Lens J_CB L40", "Lens J_CB L42", "Lens J_CB L44", "Lens J_NP-cos L42"]
        names += [MODEL] + rlens
        recs, kb = block(names, {n: D[(n, c, "z")] for n in names}, tabs, control=MODEL, keep=True)
        rel[f"{c}|z|p0_vs_p1"] = {nm: recs[f"{nm} || {nm} (p1)"] for nm in READERS006}
        rel[f"{c}|z|p1_records"] = {k: v for k, v in recs.items() if "(p1)" in v["a"] and "(p1)" not in v["b"]
                                    and not v["b"].startswith(v["a"].replace(" (p1)", ""))} | \
            {k: v for k, v in recs.items() if "(p1)" in v["a"] and "(p1)" in v["b"]}
        # does reader-reader agreement depend on the reader prefix? paired (same draws) p1 - p0 difference
        dd = {}
        rl = list(READERS006)
        for i, a in enumerate(rl):
            for b in rl[i + 1:]:
                dd[f"{a} || {b}"] = {L: {s: A.diff_record(kb["bs"][L], kb["pt"][L], kb["names"], s, (f"{a} (p1)", f"{b} (p1)"), (a, b))
                                         for s in ("pearson", "spearman", "partial_pearson")} for L in ("pairs", "tables")}
            for o in [MODEL] + rlens:
                dd[f"{a} || {o}"] = {L: {s: A.diff_record(kb["bs"][L], kb["pt"][L], kb["names"], s, (f"{a} (p1)", o), (a, o))
                                         for s in ("pearson", "spearman")} for L in ("pairs", "tables")}
        rel[f"{c}|z|p1_minus_p0_agreement"] = dd
    for c in ("colon", "think"):
        rel[f"{c}|z|006_vs_007_whole"] = {nm: ag[f"{c}|z"][f"{nm} || {name007(nm, 'whole')}"]
                                          for nm in R007 if f"{nm} || {name007(nm, 'whole')}" in ag[f"{c}|z"]}
    # descriptive disattenuation (prefix reliability as the reliability of each reader's per-pair D_z at ':')
    dis = {}
    for i, a in enumerate(READERS006):
        for b in list(READERS006)[i + 1:]:
            r = ag["colon|z"][f"{a} || {b}"]["pairs"]["pearson"]
            ra, rb = rel["colon|z|p0_vs_p1"][a]["pairs"]["pearson"], rel["colon|z|p0_vs_p1"][b]["pairs"]["pearson"]
            dis[f"{a} || {b}"] = {"r": r, "rel_a": ra, "rel_b": rb, "disattenuated": r / np.sqrt(ra * rb)}
    rel["colon|z|disattenuated_pearson_pairs"] = dis
    res["agreement"] = ag
    res["reliability"] = rel

    # ---------------------------------------------------------------------------------------------- (d) leverage
    lev = {"n_pairs": int(len(gated)), "n_tables": int(len(np.unique(tabs[gated]))),
           "S_quantiles_colon_L42": [float(q) for q in np.quantile(S[("colon", 42)], [0, .25, .5, .75, 1])],
           "note": "001 whole-cell patch share S (p->m) at ':' on the 82 gated dev pairs; layer matched to the reader "
                   "(model output and the NLA: L42). Low S = no single-cell leverage at that cell, not 'not used'."}
    gv = {n: D[(n, "colon", "z")][gated] for n in core + p007 + lens}
    for L in S_LAYERS:
        gv[f"S : L{L}"] = S[("colon", L)]
    names = core + p007 + lens + [f"S : L{L}" for L in S_LAYERS]
    recs, kb = block(names, gv, tabs[gated], control=MODEL, keep=True)
    lev["S_cross_layer"] = {k: {"pearson": v["pairs"]["pearson"], "spearman": v["pairs"]["spearman"]}
                            for k, v in recs.items() if v["a"].startswith("S") and v["b"].startswith("S")}
    rows = {}
    bsP, ptP = kb["bs"]["pairs"], kb["pt"]["pairs"]
    for n in core + p007 + lens:
        Lm = 42 if n == MODEL else layer_of(n)
        sname = f"S : L{Lm}"
        r = recs[f"{n} || {sname}"]["pairs"] if f"{n} || {sname}" in recs else recs[f"{sname} || {n}"]["pairs"]
        row = {"S_layer": Lm, "pearson": r["pearson"], "pearson_ci": r["pearson_ci"], "spearman": r["spearman"],
               "spearman_ci": r["spearman_ci"], "sign_counts_note": "S > 0 for every gated pair at ':'",
               "partial_pearson_on_model": r.get("partial_pearson"), "partial_pearson_on_model_ci": r.get("partial_pearson_ci")}
        if n != MODEL:
            for s in ("pearson", "spearman"):
                row[f"{s}_minus_model"] = A.diff_record(bsP, ptP, kb["names"], s, (n, sname), (MODEL, sname))
        rows[n] = row
    lev["reader_Dz_vs_S"] = rows
    lev["model_Dz_vs_S_by_layer"] = {L: {k: (recs[f"{MODEL} || S : L{L}"]["pairs"][k]) for k in
                                         ("pearson", "pearson_ci", "spearman", "spearman_ci")} for L in S_LAYERS}
    # agreement in high- vs low-S halves (median split on S at ':' L42)
    s42 = S[("colon", 42)]
    hi = s42 >= np.median(s42)
    WP = kb["WP"]
    halves = {}
    X = np.stack([gv[n] for n in core + p007])
    nm_h = core + p007
    for lab, msk in (("high", hi), ("low", ~hi)):
        halves[lab] = {"pt": A.point_stats(X[:, msk]), "bs": A.boot_stats(X[:, msk], WP[:, msk]),
                       "n": int(msk.sum()), "n_tables": int(len(np.unique(tabs[gated][msk])))}
    sel_pairs = [(a, b) for i, a in enumerate(core) for b in core[i + 1:]] + \
                [(name007(o, pc), name007("NLA L42", pc)) for o in ("Oracle L40", "Oracle L44") for pc in PIECES]
    conc = {"split": "S at ':' L42 >= median (high) vs < median (low)", "median_S": float(np.median(s42)),
            "n_high": halves["high"]["n"], "n_low": halves["low"]["n"],
            "tables_high": halves["high"]["n_tables"], "tables_low": halves["low"]["n_tables"], "pairs": {}, "mean_Dz": {}}
    for a, b in sel_pairs:
        i, j = nm_h.index(a), nm_h.index(b)
        rec = {}
        for s in ("pearson", "spearman", "sign", "sign_kappa"):
            h, l_ = halves["high"]["bs"][s][:, i, j], halves["low"]["bs"][s][:, i, j]
            rec[s] = {"high": float(halves["high"]["pt"][s][i, j]), "high_ci": A.ci(h),
                      "low": float(halves["low"]["pt"][s][i, j]), "low_ci": A.ci(l_),
                      "high_minus_low": float(halves["high"]["pt"][s][i, j] - halves["low"]["pt"][s][i, j]),
                      "high_minus_low_ci": A.ci(h - l_)}
        conc["pairs"][f"{a} || {b}"] = rec
    for k_, n in enumerate(nm_h):
        mh = (WP[:, hi] @ X[k_, hi]) / WP[:, hi].sum(1)
        ml = (WP[:, ~hi] @ X[k_, ~hi]) / WP[:, ~hi].sum(1)
        conc["mean_Dz"][n] = {"high": float(X[k_, hi].mean()), "low": float(X[k_, ~hi].mean()),
                              "high_minus_low": float(X[k_, hi].mean() - X[k_, ~hi].mean()), "high_minus_low_ci": A.ci(mh - ml),
                              "pos_high": int((X[k_, hi] > 0).sum()), "pos_low": int((X[k_, ~hi] > 0).sum())}
    lev["agreement_by_S_half"] = conc
    lev["block_S_records"] = {k: v for k, v in recs.items() if v["b"].startswith("S :")}   # every instrument vs S, all layers
    res["leverage"] = lev

    # ---------------------------------------------------------------------------------------------- (e) across cells
    xc = {"note": "per-pair difference of D_z between two cells; ΔS = difference of 001's S (p->m) at the reader's "
                  "layer on the 82 gated pairs (model output and NLA: L42). ΔS(think - colon) is about -S(colon), "
                  "since S(think) is about 0.", "contrasts": {}}
    both = core + lens
    for a_, b_ in (("think", "colon"), ("asst", "colon"), ("think", "asst")):
        key = f"{a_}-{b_}"
        names = both + (p007 if "asst" not in key else [])
        dv = {n: D[(n, a_, "z")] - D[(n, b_, "z")] for n in names}
        recs_all, _ = block(names, dv, tabs, control=None)
        summ = {n: inst_summary(dv[n], tabs) for n in names}
        gvd = {n: dv[n][gated] for n in names}
        for L in S_LAYERS:
            gvd[f"ΔS L{L}"] = S[(a_, L)] - S[(b_, L)]
        recs_g, _ = block(names + [f"ΔS L{L}" for L in S_LAYERS], gvd, tabs[gated], control=None)
        rows = {}
        for n in names:
            Lm = 42 if n == MODEL else layer_of(n)
            rm = recs_all.get(f"{n} || {MODEL}") or recs_all.get(f"{MODEL} || {n}")
            rs = recs_g[f"{n} || ΔS L{Lm}"]
            rows[n] = {"mean_delta_tables": summ[n]["table"], "pairs_pos_neg": [summ[n]["pairs"]["pos"], summ[n]["pairs"]["neg"]],
                       "vs_model_delta": None if n == MODEL else {k: rm["pairs"][k] for k in
                                                                  ("pearson", "pearson_ci", "spearman", "spearman_ci", "sign", "sign_ci", "sign_expected",
                                                                   "sign_kappa", "sign_kappa_ci", "sign_counts")},
                       "vs_deltaS": {"S_layer": Lm, **{k: rs["pairs"][k] for k in ("pearson", "pearson_ci", "spearman",
                                                                                     "spearman_ci", "sign", "sign_ci", "sign_expected",
                                                                                     "sign_kappa", "sign_kappa_ci", "sign_counts")}}}
        xc["contrasts"][key] = {"instruments": rows,
                                "deltaS_tables": {f"L{L}": A.summarize(dstat.table_means(np.r_[gvd[f'ΔS L{L}']], tabs[gated])[0],
                                                                         A.boot_idx(len(np.unique(tabs[gated]))))
                                                  for L in S_LAYERS},
                                "deltaS_pairs_pos_neg": {f"L{L}": [int((gvd[f'ΔS L{L}'] > 0).sum()), int((gvd[f'ΔS L{L}'] < 0).sum())]
                                                         for L in S_LAYERS}}
    res["across_cells"] = xc

    # ---------------------------------------------------------------------------------------------- (f) 007 pieces
    pcs = {}
    rpairs = [("Oracle L40", "NLA L42"), ("Oracle L44", "NLA L42"), ("Oracle L40", "Oracle L44")]
    for key in ("colon|z", "think|z"):
        kb = kept[key]
        out = {}
        for a, b in rpairs:
            rec = {}
            for pc in PIECES:
                r = ag[key][f"{name007(a, pc)} || {name007(b, pc)}"]
                rec[pc] = {L: {k: r[L][k] for k in ("n", "pearson", "pearson_ci", "spearman", "spearman_ci", "sign", "sign_ci",
                                                    "sign_kappa", "sign_kappa_ci", "partial_pearson", "partial_pearson_ci", "model_cov_share", "model_cov_share_ci")
                               if k in r[L]} for L in ("pairs", "tables")}
                rec[pc]["tables"]["derangement_null_pearson_p"] = r["tables"]["derangement_null"]["pearson"]["p_one_sided_ge"]
                if pc != "whole":
                    rec[pc]["minus_whole"] = {L: {s: A.diff_record(kb["bs"][L], kb["pt"][L], kb["names"], s,
                                                                   (name007(a, pc), name007(b, pc)),
                                                                   (name007(a, "whole"), name007(b, "whole")))
                                                  for s in ("pearson", "spearman", "sign_kappa", "partial_pearson")}
                                              for L in ("pairs", "tables")}
            out[f"{a} vs {b}"] = rec
        pcs[f"dev|{key}"] = out
    tpairs, TLL = load_test()
    ttabs = np.array([p["table"] for p in tpairs])
    tv = {n: pair_vals(M, tpairs, "z") for n, M in TLL.items()}
    names = list(TLL)
    recs, kb = block(names, tv, ttabs, control=None, keep=True)
    out = {}
    for a, b in rpairs:
        rec = {}
        for pc in PIECES:
            r = recs[f"{name007(a, pc)} || {name007(b, pc)}"]
            rec[pc] = {L: {k: r[L][k] for k in ("n", "pearson", "pearson_ci", "spearman", "spearman_ci", "sign", "sign_ci",
                                                "sign_kappa", "sign_kappa_ci")}
                       for L in ("pairs", "tables")}
            if pc != "whole":
                rec[pc]["minus_whole"] = {L: {s: A.diff_record(kb["bs"][L], kb["pt"][L], kb["names"], s,
                                                               (name007(a, pc), name007(b, pc)),
                                                               (name007(a, "whole"), name007(b, "whole")))
                                              for s in ("pearson", "spearman", "sign_kappa")} for L in ("pairs", "tables")}
        out[f"{a} vs {b}"] = rec
    j = json.loads((PROJECT / "results/007_jsubspace/analysis/posthoc_A2_brew_test.json").read_text())["agreement"]
    tchk = {f"NLA vs {o} {pc}": {"reported": j[f"nla_L42_vs_{R007[o]}"][pc]["r"],
                                 "recomputed": float(np.corrcoef(tv[name007("NLA L42", pc)], tv[name007(o, pc)])[0, 1])}
            for o in ("Oracle L40", "Oracle L44") for pc in PIECES}
    for v in tchk.values():
        v["pass"] = bool(abs(v["reported"] - v["recomputed"]) < 1e-9)
    res["reproduction"]["_brew_test_agreement_r"] = tchk
    res["reproduction"]["_brew_test_D_z"] = {
        n: {"reported": json.loads((PROJECT / "results/007_jsubspace/analysis/posthoc_A2_brew_test.json").read_text())
            ["readers"][R007[" ".join(n.split()[1:3])]]["pieces"][n.split()[-1]]["z"]["mean"],
            "recomputed": float(dstat.table_means(tv[n], ttabs)[0].mean())} for n in names}
    for v in res["reproduction"]["_brew_test_D_z"].values():
        v["pass"] = bool(abs(v["reported"] - v["recomputed"]) < 1e-6)
    pcs["test|colon|z"] = out
    pcs["test_n"] = {"pairs": int(len(tpairs)), "tables": int(len(np.unique(ttabs)))}
    pcs["test_per_instrument"] = {n: inst_summary(tv[n], ttabs) for n in names}
    res["pieces_007"] = pcs

    # ---------------------------------------------------------------------------------------------- reader-lens vs reader-model
    # paired (same draws): does each reader's per-pair D_z agree more with a token lens at its own layer than with the
    # model's own output D_z at the same cell?
    rl = {}
    for key in ("colon|z", "think|z"):
        kb = kept[key]
        for rd in ("NLA L42", "Oracle L40", "Oracle L44", name007("Oracle L40", "J"), name007("Oracle L44", "J"),
                   name007("NLA L42", "J")):
            L = layer_of(rd)
            for ln in ("Lens J_CB", "Lens J_NP-cos", "Lens logit"):
                rl[f"{key}|{rd}|{ln} L{L}"] = {lv: {s: A.diff_record(kb["bs"][lv], kb["pt"][lv], kb["names"], s, (rd, f"{ln} L{L}"),
                                                                     (rd, MODEL)) for s in ("pearson", "spearman")}
                                               for lv in ("pairs", "tables")}
    res["reader_lens_minus_reader_model"] = rl
    (RES / "agreement.json").write_text(json.dumps(res, separators=(",", ":"), default=float))   # compact: ~350 records/block
    print("wrote", RES / "agreement.json")


if __name__ == "__main__":
    main()
