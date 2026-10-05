"""007 analysis (CPU; design §3.3-3.6, §4.4 + amendment A1). Reader-side; never imported by label code.

  python analysis.py stage1 --stage stage1a --run-id r1      # also stage1b, brew_test (same rules)

Writes results/007_jsubspace/analysis/<stage>.json. Pure functions (D statistics, bootstrap, Holm, TOST, categories,
the three-model mapping) are unit-tested in tests/test_007_analysis.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dstat                                                               # noqa: E402
import p007 as C                                                           # noqa: E402

B_BOOT = 10000
MODELS = ("Raw", "Visible", "H_J")
# design §2.2 (P5 Visible prediction set per cell from Stage 0 geometry, A1); None = no prediction
PRED = {"Raw": {"P1": "≈", "P2": "≈", "P3": "≈", "P4": "≈", "P5": "≈"},
        "Visible": {"P1": "−", "P2": "+", "P3": "≈", "P4": "+", "P5": None},
        "H_J": {"P1": "+", "P2": "+", "P3": "+", "P4": None, "P5": "+"}}


# ------------------------------------------------------------------------------------------------ pure functions
def boot_idx(n, B=B_BOOT, seed=C.BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def summarize(x, idx):
    x = np.asarray(x, float)
    m = x[idx].mean(1)
    return {"n": int(len(x)), "mean": float(x.mean()), "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))],
            "pos": int((x > 0).sum()), "neg": int((x < 0).sum()), "zero": int((x == 0).sum())}


def contrast(a, b, idx, delta=C.DELTA):
    """Paired per-table difference a - b. Returns estimate, CI, two-sided bootstrap p and the TOST p for |diff| < delta."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m = d[idx].mean(1)
    p_two = max(2 * min((m <= 0).mean(), (m >= 0).mean()), 1.0 / len(m))
    p_lo, p_hi = (m <= -delta).mean(), (m >= delta).mean()        # H0: diff <= -delta ; H0: diff >= delta
    p_tost = max(max(p_lo, p_hi), 1.0 / len(m))
    return {"estimate": float(d.mean()), "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))],
            "ci90": [float(np.quantile(m, .05)), float(np.quantile(m, .95))], "p_two": float(min(p_two, 1.0)),
            "p_tost": float(min(p_tost, 1.0)), "n": int(len(d))}


def holm(ps, alpha=0.05):
    """Holm step-down: returns a list of booleans (rejected) in the input order."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    rej = np.zeros(len(ps), bool)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            rej[i] = True
        else:
            break
    return rej.tolist()


def category(est, diff_rejected, tost_rejected, informative, delta=C.DELTA):
    if not informative:
        return "uninformative"
    if diff_rejected and est >= delta:
        return "+"
    if diff_rejected and est <= -delta:
        return "−"
    if tost_rejected:
        return "≈"
    return "unresolved"


def model_mapping(cats, preds=PRED):
    """cats: {P1..P5: category}. Returns {model: supported | contradicted | unresolved} (design §4.4)."""
    out = {}
    for mdl, pr in preds.items():
        entries = [(k, v) for k, v in pr.items() if v is not None]
        if any(cats[k] not in (v, "unresolved", "uninformative") for k, v in entries):
            out[mdl] = "contradicted"
        elif all(cats[k] == v for k, v in entries):
            out[mdl] = "supported"
        else:
            out[mdl] = "unresolved"
    return out


def derangements(n, k=200, seed=1):
    """006's derangement draws (reanalyse_006: default_rng(1), permutations with no fixed point)."""
    r2 = np.random.default_rng(seed)
    perms = []
    while len(perms) < k:
        p_ = r2.permutation(n)
        if not (p_ == np.arange(n)).any():
            perms.append(p_)
    return perms


def deranged_D(LLc, pairs, role, perms):
    """006's derangement null: table k's labels, table perm[k]'s activations (member; partners cycled). Returns the
    mean-over-tables D per derangement."""
    tabs = sorted({p["table"] for p in pairs})
    by_t = {t: [p for p in pairs if p["table"] == t] for t in tabs}
    vals = []
    for perm in perms:
        per = []
        for k, t in enumerate(tabs):
            o = by_t[tabs[perm[k]]]
            om, ops = o[0]["m"], [q["p"] for q in o]
            v = []
            for j, p in enumerate(by_t[t]):
                if role == "off":
                    if p["off"] is None:
                        continue
                    tm, tp = p["off"]
                else:
                    if role != "z" and p["collide"][role]:
                        continue
                    tm, tp = p["m_roles"][role], p["p_roles"][role]
                ci_m, ci_p = dstat.CI[tm], dstat.CI[tp]
                a, b = om, ops[j % len(ops)]
                val = (LLc[a, ci_m] - LLc[a, ci_p]) - (LLc[b, ci_m] - LLc[b, ci_p])
                if np.isfinite(val):                                       # template lens: teal is NaN (absent)
                    v.append(val)
            if v:
                per.append(np.mean(v))
        vals.append(float(np.mean(per)) if per else np.nan)
    return np.array(vals)


def pc_stats(P):
    """P [12, 12] colour LL of the 12 colour activations (rows: activation colour). LOO-prior ID and pairwise double
    differences (006 post hoc). Teal-free variants are handled by the caller (NaN columns)."""
    ok = ~np.isnan(P).any(0)
    idx = np.where(ok)[0]
    Q = P[np.ix_(idx, idx)]
    loo = np.array([Q[i] - np.delete(Q, i, 0).mean(0) for i in range(len(idx))])
    n = len(idx)
    pos = sum(((Q[i, i] - Q[i, j]) - (Q[j, i] - Q[j, j])) > 0 for i in range(n) for j in range(i + 1, n))
    return {"loo_prior_ID": float(np.mean(loo.argmax(1) == np.arange(n))), "raw_ID": float(np.mean(Q.argmax(1) == np.arange(n))),
            "pairwise_positive": int(pos), "pairs": int(n * (n - 1) // 2), "n_colours": int(n)}


def pc_pass(st):
    need = int(np.ceil(50 / 66 * st["pairs"]))
    return bool(st["loo_prior_ID"] >= 0.5 and st["pairwise_positive"] >= need)


# ------------------------------------------------------------------------------------------------ stage 1 analysis
def load_blocks(stage, run_id, pieces):
    d = C.OUT / stage / run_id / "ll"
    cname = {"stage1a": "colon", "stage1b": "think", "brew_test": "colon"}[stage]
    owner = np.load(d / "owner.npy")
    out = {}
    for rd, L in C.READERS:
        tag = f"{rd}_L{L}"
        if not (d / f"{tag}_{cname}_whole.npy").exists():
            continue
        blk = {"pieces": {}, "none": None, "positive": {}}
        for nm in pieces:
            f = d / f"{tag}_{cname}_{nm}.npy"
            if f.exists():
                blk["pieces"][nm] = dstat.colour_ll(np.load(f), owner)
        fn = d / f"{tag}_{cname}_none.npy"
        if fn.exists():
            blk["none"] = dstat.colour_ll(np.load(fn), owner)
        fp = d / f"{tag}_{cname}_positive.npz"
        if fp.exists():
            z = np.load(fp)
            blk["positive"] = {k: dstat.colour_ll(z[k], owner) for k in z.files if k != "owner"}
        out[tag] = blk
    return out


def stage1(stage, run_id, rand_draws=None, pieces=None, roles_path=None, geom_key=None, tag_out=None):
    pieces = pieces or C.PIECE_ORDER
    roles = json.loads(Path(roles_path or (C.STAGE0 / "roles.json")).read_text())
    pairs = roles["pairs"]
    st0 = json.loads((C.RES / "analysis" / "stage0.json").read_text())
    cname = {"stage1a": "colon", "stage1b": "think", "brew_test": "colon"}[stage]
    blocks = load_blocks(stage, run_id, pieces)
    tabs = sorted({p["table"] for p in pairs})
    perms = derangements(len(tabs))
    res = {"stage": stage, "run_id": run_id, "design": "docs/design_specs/007_jsubspace.md (v2 + A1)", "readers": {}}
    rand_all = [nm for nm in pieces if nm.startswith("R") and nm[1:].isdigit()]
    rand = rand_draws if rand_draws is not None else rand_all
    restr = [nm for nm in pieces if nm.startswith("rest_R")]
    randc = [nm for nm in pieces if nm.startswith("R") and nm.endswith("_c")]
    for tag, blk in blocks.items():
        rd, L = tag.split("_L")[0], int(tag.split("_L")[1])
        P = blk["pieces"]
        R = {"pieces": {}, "positive_control": {}, "contrasts": {}}
        tm = {}                                                            # (piece, role) -> table-mean array
        idxs = {}
        for nm, LLc in P.items():
            rec = {}
            for role in ("z", "start", "answer", "off"):
                v, tb = dstat.pair_D(LLc, pairs, role)
                m, ts = dstat.table_means(v, tb)
                key = (role, tuple(ts))
                if key not in idxs:
                    idxs[key] = boot_idx(len(ts))
                tm[(nm, role)] = (m, ts)
                rec[role] = summarize(m, idxs[key])
                nul = deranged_D(LLc, pairs, role, perms)
                rec[role]["derangement_null"] = {"mean": float(nul.mean()), "p2.5": float(np.quantile(nul, .025)),
                                                 "p97.5": float(np.quantile(nul, .975))}
                if role == "off":
                    ok = ~np.isnan(v)
                    rec[role]["abs_pair_D_median"] = float(np.median(np.abs(v[ok])))
                    rec[role]["abs_mean_vs_null"] = {"abs_mean": abs(float(m.mean())),
                                                     "null_abs_p95": float(np.quantile(np.abs(nul), .95))}
            # role-specificity flags (design §3.3; A1 wording)
            dz, ds, da, do = rec["z"], rec["start"], rec["answer"], rec["off"]
            off_ok = (do["ci"][0] <= 0 <= do["ci"][1]) or abs(do["mean"]) < C.DELTA / 2
            broad = all(x["ci"][0] > 0 for x in (dz, ds, da)) and \
                (max(dz["mean"], ds["mean"], da["mean"]) - min(dz["mean"], ds["mean"], da["mean"]) <= C.DELTA)
            rec["flags"] = {"off_trajectory_alignment_ok": bool(off_ok),
                            "broad_colour_tracking_without_demonstrated_z_selectivity_or_role_binding": bool(broad)}
            # colour prior
            am = LLc.argmax(1)
            dist = np.bincount(am, minlength=12) / len(am)
            rec["colour_prior"] = {"top_colour_share": {c: float(x) for c, x in zip(C.COLOURS, dist)},
                                   "mean_LL": {c: float(x) for c, x in zip(C.COLOURS, LLc.mean(0))}}
            if blk["none"] is not None:
                rec["engagement_median_max_abs_vs_none"] = float(np.median(np.abs(LLc - blk["none"]).max(1)))
                rec["engagement_flag_ignored"] = rec["engagement_median_max_abs_vs_none"] < 0.1
            R["pieces"][nm] = rec
        if "whole" in P:
            dw = np.bincount(P["whole"].argmax(1), minlength=12) / P["whole"].shape[0]
            for nm in P:
                d_ = np.array(list(R["pieces"][nm]["colour_prior"]["top_colour_share"].values()))
                R["pieces"][nm]["colour_prior"]["tv_from_whole"] = float(0.5 * np.abs(d_ - dw).sum())
        # positive control: colour-mean version decides informativeness (pre-registered); brew-operator version reported
        pcs = {}
        for k, Pk in blk["positive"].items():
            s = pc_stats(Pk)
            s["pass"] = pc_pass(s)
            pcs[k] = s
        R["positive_control"] = pcs
        # informativeness per piece (A1): PC fail -> null uninformative; a clear positive D stands, flagged
        info = {}
        for nm in P:
            pk = nm if nm in pcs else None
            passed = pcs[pk]["pass"] if pk else None
            clear_pos = R["pieces"][nm]["z"]["ci"][0] > 0
            info[nm] = {"pc_pass": passed, "clear_positive_Dz": bool(clear_pos),
                        "status": "informative" if passed in (True, None) else ("stands_flagged" if clear_pos else "null_uninformative")}
            R["pieces"][nm]["informativeness"] = info[nm]

        def tmean(names, role="z"):
            arrs = [tm[(n, role)][0] for n in names if (n, role) in tm]
            return None if not arrs else np.mean(arrs, 0), arrs

        def avail(*names):
            return all((n, "z") in tm for n in names)
        Rs = rand if all((n, "z") in tm for n in rand) else [n for n in rand if (n, "z") in tm]
        defs = {"P1": (["J"], ["PCA"]), "P2": (["J"], Rs), "P3": (restr, ["rest_J"]), "P4": (["MR"], ["whole"]),
                "P5": (["J_c"], randc)}
        R["contrast_pieces"] = {k: {"a": a, "b": b} for k, (a, b) in defs.items()}
        R["contrasts"] = {}
        for k, (a, b) in defs.items():
            if not a or not b or not all((n, "z") in tm for n in a + b):
                continue
            A, _ = tmean(a)
            Bm, _ = tmean(b)
            ts = tm[(a[0], "z")][1]
            R["contrasts"][k] = contrast(A, Bm, idxs[("z", tuple(ts))])
            pieces_k = a + b
            R["contrasts"][k]["pieces_status"] = {n: info[n]["status"] for n in pieces_k}
            R["contrasts"][k]["informative"] = all(info[n]["status"] != "null_uninformative" for n in pieces_k)
            R["contrasts"][k]["flagged"] = any(info[n]["status"] == "stands_flagged" for n in pieces_k)
        res["readers"][tag] = R
    # Holm/TOST across the primary family (all reader-layers x P1-P5 present)
    fam = [(tag, k) for tag in res["readers"] for k in res["readers"][tag]["contrasts"]]
    if fam:
        # the pre-registered family is P1-P5 x the three reader-layers (15); contrasts that could not be computed (a
        # disabled reader, A1) enter as p = 1, so the familywise level stays the one defined for 15 tests
        n_missing = max(0, 5 * len(C.READERS) - len(fam))
        pd_ = [res["readers"][t]["contrasts"][k]["p_two"] for t, k in fam]
        pt_ = [res["readers"][t]["contrasts"][k]["p_tost"] for t, k in fam]
        rej = holm(pd_ + [1.0] * n_missing)[:len(fam)]
        trej = holm(pt_ + [1.0] * n_missing)[:len(fam)]
        rej_s, trej_s = holm(pd_), holm(pt_)                              # sensitivity: Holm over the available tests
        for (t, k), r1, r2, s1, s2 in zip(fam, rej, trej, rej_s, trej_s):
            c = res["readers"][t]["contrasts"][k]
            c["holm_diff_rejected"], c["holm_tost_rejected"] = bool(r1), bool(r2)
            c["category"] = category(c["estimate"], r1, r2, c["informative"])
            c["category_holm_available_only"] = category(c["estimate"], s1, s2, c["informative"])
        res["family_size"] = len(fam) + n_missing
        res["family_missing_as_p1"] = n_missing
    p5geo = st0.get("P5_geometry", {}).get(geom_key or f"{cname}_L42", {})
    for tag, R in res["readers"].items():
        L = int(tag.split("_L")[1])
        g = st0.get("P5_geometry", {}).get(f"{cname}_L{L}", p5geo)
        preds = json.loads(json.dumps(PRED))
        preds["Visible"]["P5"] = "≈" if g.get("visible_model_P5_prediction") == "≈" else None
        R["predictions"] = preds
        cats = {k: R["contrasts"].get(k, {}).get("category", "uninformative") for k in ("P1", "P2", "P3", "P4", "P5")}
        R["categories"] = cats
        R["model_mapping"] = model_mapping(cats, preds)
    # secondary: content curve, Spearman, random spread, GMR vs MR, prompt-only, agreement
    geo = st0["geometry"].get(f"{cname}_L42", {})
    for tag, R in res["readers"].items():
        L = int(tag.split("_L")[1])
        geoL = st0["geometry"].get(f"{cname}_L{L}", {}).get("pieces", {})
        D = {nm: R["pieces"][nm]["z"]["mean"] for nm in R["pieces"]}
        sec = {}
        if all(n in D for n in ["whole"] + rand + restr) and geoL:
            xr = float(np.mean([geoL[n]["delta_share"] for n in rand]))
            xq = float(np.mean([geoL[n]["delta_share"] for n in restr]))
            yr, yq = float(np.mean([D[n] for n in rand])), float(np.mean([D[n] for n in restr]))
            xs, ys = [0.0, xr, xq, 1.0], [0.0, yr, yq, D["whole"]]
            sec["content_curve"] = {"knots": {"x": xs, "y": ys},
                                    "excess": {nm: float(D[nm] - np.interp(geoL[nm]["delta_share"], xs, ys))
                                               for nm in D if nm in geoL}}
        if geoL:
            from scipy.stats import spearmanr
            nms = [n for n in D if n in geoL]
            sec["spearman_Dz_vs_visible_ratio"] = float(spearmanr([D[n] for n in nms], [geoL[n]["visible_ratio"] for n in nms])[0])
            sec["spearman_Dz_vs_delta_share"] = float(spearmanr([D[n] for n in nms], [geoL[n]["delta_share"] for n in nms])[0])
        rr = [D[n] for n in rand_all if n in D]
        if rr:
            sec["random_spread"] = {"sd_across_draws": float(np.std(rr, ddof=1)) if len(rr) > 1 else None,
                                    "min": float(min(rr)), "max": float(max(rr)), "J": D.get("J")}
        if all(n in R["pieces"] for n in ("GMR", "MR", "whole")):
            sec["generic_vs_template"] = {role: {n: R["pieces"][n][role]["mean"] for n in ("whole", "MR", "GMR")}
                                          for role in ("z", "start", "answer", "off")}
        R["secondary"] = sec
    # A1 replication comparator: dev P2 with random draws 0-2 only (others identical)
    res["agreement"] = agreement(blocks, pairs, tabs)
    out = C.RES / "analysis" / f"{tag_out or stage}.json"
    C.dump(out, res)
    return res


def agreement(blocks, pairs, tabs):
    """Per piece: pair-level D_z agreement between NLA L42 and each Oracle layer (same-sign share, Pearson r, table
    bootstrap CI)."""
    out = {}
    if "nla_L42" not in blocks:
        return out
    tb = np.array([p["table"] for p in pairs])
    ut = sorted(set(tb))
    idx = boot_idx(len(ut))
    for o in ("oracle_L40", "oracle_L44"):
        if o not in blocks:
            continue
        for nm in blocks["nla_L42"]["pieces"]:
            if nm not in blocks[o]["pieces"]:
                continue
            a, _ = dstat.pair_D(blocks["nla_L42"]["pieces"][nm], pairs, "z")
            b, _ = dstat.pair_D(blocks[o]["pieces"][nm], pairs, "z")
            rs = []
            for row in idx[:2000]:
                sel = np.concatenate([np.where(tb == ut[i])[0] for i in row])
                if np.std(a[sel]) > 0 and np.std(b[sel]) > 0:
                    rs.append(np.corrcoef(a[sel], b[sel])[0, 1])
            out.setdefault(f"nla_L42_vs_{o}", {})[nm] = {
                "same_sign": float(np.mean(np.sign(a) == np.sign(b))), "r": float(np.corrcoef(a, b)[0, 1]),
                "r_ci": [float(np.quantile(rs, .025)), float(np.quantile(rs, .975))], "n_pairs": int(len(a))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage1"])
    ap.add_argument("--stage", default="stage1a")
    ap.add_argument("--run-id", default="r1")
    a = ap.parse_args()
    if a.what == "stage1":
        pieces = C.TEST_PIECES if a.stage == "brew_test" else C.PIECE_ORDER
        res = stage1(a.stage, a.run_id, pieces=pieces,
                     roles_path=(C.OUT / "brew_test" / "roles.json") if a.stage == "brew_test" else None)
        if a.stage == "stage1a":                                           # A1: the 3-draw comparator for the replication
            stage1(a.stage, a.run_id, rand_draws=["R0", "R1", "R2"], tag_out="stage1a_draws0-2")
        for tag, R in res["readers"].items():
            print(tag, R["categories"], R["model_mapping"])


if __name__ == "__main__":
    main()
