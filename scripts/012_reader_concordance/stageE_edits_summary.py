"""012 Stage E step 4 summary (design 012E §15): G-parity, G0 and identity floors, edit sizes at every position, the
class-mean lever rule, final class-mean coverage (by bridge type and Japan share), the style control, and the smoke.
CPU only. EXPLORATORY; the coverage rescue and everything built on it are POST HOC (§15).

  python stageE_edits_summary.py   -> results/012_reader_concordance/analysis/stageE_step4.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stageE_common as E                                                   # noqa: E402
import stageA_splits as S                                                   # noqa: E402

ED = E.OUT / "edits"
BAND = list(range(36, 48))
PLACES = S.COUNTRIES | S.GEO_PLACES


def rows(prefix):
    out = []
    for f in sorted(ED.glob(f"{prefix}_shard*.jsonl")):
        out += [json.loads(l) for l in open(f)]
    return out


def size_summary(rs, arm):
    """Realised ||dh|| / ||h|| per layer: median over items of (median over positions), read position, and max."""
    per_L = {}
    for L in BAND:
        med_pos, read, mx = [], [], []
        for r in rs:
            a = r["arms"].get(arm, {})
            lg = (a.get("edit_log") or {}).get(str(L))
            if not lg:
                continue
            x = np.array(lg["rel_dh"])
            med_pos.append(float(np.median(x)))
            read.append(float(x[r["read_pos"]]))
            mx.append(float(x.max()))
        if med_pos:
            per_L[str(L)] = {"median_over_positions": float(np.median(med_pos)), "read_position": float(np.median(read)),
                             "max_any_position": float(np.max(mx)), "n_items": len(med_pos)}
    errs = [v.get("coord_err_bf16") for r in rs for v in ((r["arms"].get(arm, {}).get("edit_log") or {}).values())
            if v.get("coord_err_bf16") is not None]
    return {"by_layer": per_L, "coord_err_bf16_max": float(max(errs)) if errs else None}


def rate(rs, arm):
    ok = [r["arms"][arm]["success"] for r in rs if arm in r["arms"] and "success" in r["arms"][arm]]
    return {"k": int(sum(ok)), "n": len(ok)}


def lever(rs, arm, rand):
    a, b = rate(rs, arm), rate(rs, rand)
    pa, pb = (a["k"] / a["n"] if a["n"] else None), (b["k"] / b["n"] if b["n"] else None)
    return {arm: a, rand: b, "pass_rule": bool(pa is not None and pa >= 0.30 and pa - pb >= 0.20),
            "rule": ">= 30% flips and >= 20 points above the random control (012 lever rule)"}


NB, SEED = 10000, 11


def boot_rel(vals, clus):
    vals, clus = np.asarray(vals, float), np.asarray(clus)
    keys = sorted(set(clus))
    idx = {k: np.where(clus == k)[0] for k in keys}
    rng = np.random.default_rng(SEED)
    ms = [np.mean(np.concatenate([vals[idx[keys[i]]] for i in rng.choice(len(keys), len(keys))])) for _ in range(NB)]
    return {"mean": float(vals.mean()), "median": float(np.median(vals)), "n": int(len(vals)), "n_clusters": len(keys),
            "ci95": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]}


def full_analysis(F, P, C):
    """012E §16 rules on the full run. Δm in nats toward y_b; flips = y_b in the greedy output; relation clusters."""
    J = {r["name"]: r for r in P}
    CMr = {r["name"]: r for r in C}
    dm = lambda r, a: r["arms"][a]["dm"]                                                     # noqa: E731
    fl = lambda r, a: bool(r["arms"][a].get("success"))                                     # noqa: E731
    have = lambda r, *arms: all(a in r["arms"] and "dm" in r["arms"][a] for a in arms)      # noqa: E731

    def paired(rs, a, b):
        """b: arm name or list of arms (their mean is the comparator)."""
        bl = b if isinstance(b, list) else [b]
        xs = [(dm(r, a) - np.mean([dm(r, x) for x in bl]), E.rel(r["name"])) for r in rs if have(r, a, *bl)]
        return boot_rel([x for x, _ in xs], [c for _, c in xs]) if xs else None

    def flips(rs, a):
        v = [fl(r, a) for r in rs if a in r["arms"] and "success" in r["arms"][a]]
        return {"k": int(sum(v)), "n": len(v), "rate": float(np.mean(v)) if v else None}

    def mean_flips(rs, arms):
        return float(np.mean([np.mean([fl(r, a) for a in arms]) for r in rs if have(r, *arms)]))

    def absolute(rs, a, nulls):
        f_, nf = flips(rs, a), mean_flips(rs, nulls)
        return {"flips": f_, "null_mean_flip_rate": nf,
                "pass": bool(f_["rate"] is not None and f_["rate"] >= 0.30 and f_["rate"] - nf >= 0.20)}

    def block(rs, sz):
        rf, rj = [f"rand_full_{i}@{sz}" for i in range(2)], [f"rand_Jcomp_{i}@{sz}" for i in range(2)]
        pw = {a: paired(rs, f"{a}@{sz}", rf) for a in ("AR_full", "AR_answer")}
        power = any(v and v["ci95"][0] > 0 for v in pw.values())
        rel = paired(rs, f"AR_Jrem@{sz}", rj)
        ab = absolute(rs, f"AR_Jrem@{sz}", rj)
        rel_ok = bool(rel and rel["ci95"][0] > 0)
        verdict = ("no power at this size" if not power else "lever" if rel_ok and ab["pass"]
                   else "weak handle" if rel_ok else "not shown")
        full_abs = absolute(rs, f"AR_full@{sz}", rf)
        jr = paired(rs, f"AR_rrem@{sz}", f"AR_Jrem@{sz}")      # (full - Jrem) - (full - rrem) = rrem - Jrem
        j_ok = bool(jr and jr["ci95"][0] > 0)
        return {"n": len(rs), "power": {"AR_full_minus_rand_full": pw["AR_full"], "AR_answer_minus_rand_full": pw["AR_answer"],
                                        "has_power": power},
                "primary_relative_AR_Jrem_minus_rand_Jcomp": rel, "primary_absolute": ab, "primary_verdict": verdict,
                "AR_full_absolute": full_abs,
                "J_reading": {"full_minus_Jrem": paired(rs, f"AR_full@{sz}", f"AR_Jrem@{sz}"),
                              "full_minus_rrem": paired(rs, f"AR_full@{sz}", f"AR_rrem@{sz}"),
                              "contrast_rrem_minus_Jrem": jr,
                              "verdict": (("J-attribution on the margin" if not full_abs["pass"] else "J-attribution (lever)")
                                          if j_ok else "not attributed to J"),
                              "rule": "§9: J share >= 0.30 (report 41: 0.53-0.62) and (full - Jrem) > (full - rrem), CI > 0"},
                "arm_table": {a: {"median_dm": float(np.median([dm(r, a) for r in rs if have(r, a)])), **flips(rs, a)}
                              for a in sorted({a for r in rs for a in r["arms"] if a.endswith(f"@{sz}")})}}

    out = {"n_items": len(F), "J_size": {}, "C_size": {}}
    subsets = {"all": F, "covered": [r for r in F if r.get("covered")], "uncovered": [r for r in F if not r.get("covered")],
               "rescued_concept": [r for r in F if r.get("rescued_concept")],
               "covered_not_rescued": [r for r in F if r.get("covered") and not r.get("rescued_concept")],
               "J_toward_tgt": [r for r in F if r["j_edit_direction"] == "toward_tgt"],
               "CM_toward_tgt": [r for r in F if CMr.get(r["name"], {}).get("cm_edit_direction") == "toward_tgt"],
               "named_place": [r for r in F if r.get("bridge_type") == "named place"],
               "other_bridge": [r for r in F if r.get("bridge_type") == "other"],
               "japan_target": [r for r in F if r["tgt"] == "Japan"], "not_japan": [r for r in F if r["tgt"] != "Japan"]}
    for nm_, rs in subsets.items():
        if len(rs) >= 3:
            out["J_size"][nm_] = block(rs, "J")
            out["C_size"][nm_] = block(rs, "C")
    # references (joined from the parity and class-mean lever runs)
    cov = {r["name"] for r in subsets["covered"]}
    cmtt = [n for n, r in CMr.items() if r.get("cm_edit_direction") == "toward_tgt"]
    jref = lambda names: {"flips": int(sum(J[n]["arms"]["J_L36-47"]["success"] for n in names)), "n": len(names),   # noqa: E731
                          "median_dm": float(np.median([J[n]["arms"]["J_L36-47"]["dm"] for n in names]))}
    cref = lambda names: {"flips": int(sum(CMr[n]["arms"]["CM_swap"]["success"] for n in names)), "n": len(names),  # noqa: E731
                          "median_dm": float(np.median([CMr[n]["arms"]["CM_swap"]["dm"] for n in names]))}
    out["like_for_like_lever"] = {"covered_51": {"J_row": jref(sorted(cov)), "CM_swap": cref(sorted(cov))},
                                  "CM_toward_tgt": {"J_row": jref(cmtt), "CM_swap": cref(cmtt)},
                                  "note": "012E §16 item 4; J-row and CM results from the parity and lever runs"}
    out["secondary_J_size"] = {
        "AR_full_minus_AR_word": paired(F, "AR_full@J", "AR_word@J"),
        "AR_starts_minus_AR_mentions": paired(F, "AR_starts@J", "AR_mentions@J"),
        "LLM_mean_minus_template": boot_rel(
            [np.mean([dm(r, "AR_starts@J"), dm(r, "AR_mentions@J")]) - dm(r, "AR_template@J") for r in F
             if have(r, "AR_starts@J", "AR_mentions@J", "AR_template@J")],
            [E.rel(r["name"]) for r in F if have(r, "AR_starts@J", "AR_mentions@J", "AR_template@J")]),
        "AR_mismatched_minus_rand_full": paired(F, "AR_mismatched@J", [f"rand_full_{i}@J" for i in range(2)])}
    out["C_white"] = {"median_dm": float(np.median([dm(r, "C_white") for r in F if have(r, "C_white")])), **flips(F, "C_white")}
    out["CM_swap_Jrem"] = {"median_dm": float(np.median([dm(r, "CM_swap_Jrem") for r in F if have(r, "CM_swap_Jrem")])),
                           **flips(F, "CM_swap_Jrem")}
    out["sizes"] = {"Kj_rel_median_by_layer": {L: float(np.median([r["Kj_rel_median"][L] for r in F])) for L in ("36", "40", "44", "47")},
                    "Kc_rel_median_by_layer": {L: float(np.median([r["Kc_rel_median"][L] for r in F])) for L in ("36", "40", "44", "47")}}
    return out


def overlap_analysis(P, F):
    """012E report-43 item 2 (CPU): per-item flip-set overlaps."""
    J = {r["name"]: bool(r["arms"]["J_L36-47"]["success"]) for r in P}
    names = [r["name"] for r in F]
    sets = {"J_row": {n for n in names if J.get(n)}}
    for a in ("AR_full@J", "AR_full@C", "AR_answer@J", "C_white"):
        sets[a] = {r["name"] for r in F if r["arms"][a].get("success")}
    keys = list(sets)
    pair = {f"{x} & {y}": {"both": len(sets[x] & sets[y]), "only_" + x: len(sets[x] - sets[y]), "only_" + y: len(sets[y] - sets[x])}
            for i, x in enumerate(keys) for y in keys[i + 1:]}
    A, B = sets["AR_full@J"], sets["AR_answer@J"]
    two = {"full_flip_answer_flip": len(A & B), "full_flip_answer_no": len(A - B), "full_no_answer_flip": len(B - A),
           "neither": len(set(names) - A - B)}
    tgt = {r["name"]: r["tgt"] for r in F}
    return {"sizes": {k: len(v) for k, v in sets.items()}, "pairs": pair, "AR_full_J_vs_AR_answer_J_2x2": two,
            "AR_full_J_flips_japan_target": sum(tgt[n] == "Japan" for n in A),
            "AR_full_J_flips_not_in_J_row": sorted(A - sets["J_row"]), "J_row_flips_not_in_AR_full_J": sorted(sets["J_row"] - A)}


def energy_analysis(EN, F):
    """012E §17: energy-matched (random) and Δ-matched PCA-k removal vs J removal, pre-stated reading."""
    Fm = {r["name"]: r for r in F}
    out = {"determinism_max_abs_diff_AR_full_J": float(max(abs(r["arms"]["AR_full@J"]["dm"] - Fm[r["name"]]["arms"]["AR_full@J"]["dm"])
                                                       for r in EN))}
    diag = {k: [] for k in ("s_J", "emrem_removed_share", "emrem_cos_to_u", "pca_k", "pca_removed_share", "pca_cos_to_u",
                            "pca_overlap_with_J")}
    unmatched = 0
    for r in EN:
        for v in r["energy_diag"].values():
            for k in diag:
                diag[k].append(v[k])
            unmatched += not v["pca_matched"]
    out["diagnostics_median"] = {k: float(np.median(v)) for k, v in diag.items()}
    out["pca_unmatched_item_layers"] = {"n": unmatched, "of": len(diag["s_J"])}
    grp = {"all": EN, "named_place": [r for r in EN if r["bridge_type"] == "named place"],
           "other_bridge": [r for r in EN if r["bridge_type"] == "other"],
           "japan_target": [r for r in EN if r["tgt"] == "Japan"], "not_japan": [r for r in EN if r["tgt"] != "Japan"],
           "covered": [r for r in EN if r["covered"]], "uncovered": [r for r in EN if not r["covered"]],
           "rescued_concept": [r for r in EN if r["rescued_concept"]]}
    res = {}
    for sz in ("J", "C"):
        for X in ("emrem", "pcarem"):
            for g, rs in grp.items():
                if len(rs) < 3:
                    continue
                dX = [r["arms"][f"AR_{X}@{sz}"]["dm"] for r in rs]
                dJ = [Fm[r["name"]]["arms"][f"AR_Jrem@{sz}"]["dm"] for r in rs]
                dF = [Fm[r["name"]]["arms"][f"AR_full@{sz}"]["dm"] for r in rs]
                b = boot_rel([x - j for x, j in zip(dX, dJ)], [E.rel(r["name"]) for r in rs])
                ci_ok = b["ci95"][0] > 0
                keep = float(np.median(dX)) >= 0.5 * float(np.median(dF))
                res[f"{X}@{sz}|{g}"] = {
                    "n": len(rs), "X_minus_Jrem": b, "median_dm_X": float(np.median(dX)), "median_dm_full": float(np.median(dF)),
                    "median_dm_Jrem": float(np.median(dJ)), "keeps_half_of_full": keep,
                    "flips_X": int(sum(r["arms"][f"AR_{X}@{sz}"].get("success", False) for r in rs)),
                    "flips_full": int(sum(Fm[r["name"]]["arms"][f"AR_full@{sz}"].get("success", False) for r in rs)),
                    "reading": ("J-specific" if ci_ok and keep else "partly energy" if ci_ok
                                else "the handle breaks under any removal of this size")}
    out["readings"] = res
    return out


def main():
    out = {"label": "EXPLORATORY; 012 Stage E step 4 (design 012E §15). Coverage rescue and class-mean results POST HOC."}
    cohort = {r["name"]: r for r in E.cohort()}

    # ---------------- parity, G0, floors ----------------
    P = rows("parity")
    if P:
        coh = [r for r in P if r["name"] in cohort]
        par = [r for r in coh if "parity" in r]
        agree = [r["parity"]["stageA_success"] == r["parity"]["ours_success"] for r in par]
        ragree = [r["parity"]["stageA_random_success"] == r["parity"]["ours_random_success"] for r in par]
        out["G_parity"] = {
            "cohort_n": len(coh),
            "ours_J": rate(coh, "J_L36-47"), "ours_rand": rate(coh, "rand_L36-47"),
            "stageA_J": {"k": int(sum(r["parity"]["stageA_success"] for r in par)), "n": len(par)},
            "stageA_rand": {"k": int(sum(r["parity"]["stageA_random_success"] for r in par)), "n": len(par)},
            "flip_agreement": float(np.mean(agree)), "random_flip_agreement": float(np.mean(ragree)),
            "pass": bool(np.mean(agree) >= 0.90),
            "text_identical": {k: float(np.mean([r["parity"][k] for r in par])) for k in ("clean_same", "J_same", "rand_same")},
            "disagreements": [{"name": r["name"], **{k: r["parity"][k] for k in ("stageA_success", "ours_success")}}
                              for r in par if r["parity"]["stageA_success"] != r["parity"]["ours_success"]]}
        out["J_lever_L36_47"] = lever(coh, "J_L36-47", "rand_L36-47")
        tt = [r for r in coh if r["j_edit_direction"] == "toward_tgt"]
        out["J_toward_tgt"] = {"n": len(tt), **{k: rate(tt, k) for k in ("J_L36-47", "rand_L36-47")}}
        out["G0"] = {"alpha0_max_abs_dlp": max(r["G0"]["alpha0"]["max_abs_dlp"] for r in P),
                     "alpha0_greedy_same_all": all(r["G0"]["alpha0"]["greedy_same"] for r in P)}
        for k in ("identity_swap", "identity_additive"):
            dm = sorted(abs(r["G0"][k]["dm"]) for r in P)
            out[f"floor_{k}"] = {"median_abs_dm": float(np.median(dm)), "p90_abs_dm": float(np.percentile(dm, 90)),
                                 "max_abs_dm": float(dm[-1]), "greedy_same_frac": float(np.mean([r["G0"][k]["greedy_same"] for r in P])),
                                 "n": len(dm)}
        out["edit_sizes_parity"] = {arm: size_summary(coh, arm) for arm in ("J_L36-47", "rand_L36-47")}

    # ---------------- coverage (final, after the POST HOC rescue) and the style control ----------------
    G = E.OUT / "gate"
    cov = {}
    for pat, src in (("gate_shard*.json", "v1"), ("gate_v2*_shard*.json", "v2"), ("gate_easy_shard*.json", "easy")):
        for f in sorted(G.glob(pat)):
            for c, v in json.loads(f.read_text())["concepts"].items():
                if src == "easy" and c not in ("Brazil", "Canada", "China", "Denmark", "Fuji", "Miami", "Nile", "Olympus", "Sydney"):
                    continue
                cov[c] = {"covered": not v["dropped"], "source": src}
    covered = {c for c, v in cov.items() if v["covered"]}
    both = [r for r in cohort.values() if r["src"] in covered and r["tgt"] in covered]
    grp = lambda r: "named place" if r["src"] in PLACES else "other"           # noqa: E731
    out["coverage_final"] = {
        "concepts_covered": len(covered), "concepts_total": len(cov),
        "rescued_post_hoc": sorted(c for c, v in cov.items() if v["source"] == "easy" and v["covered"]),
        "rescue_failed": sorted(c for c, v in cov.items() if v["source"] == "easy" and not v["covered"]),
        "items_both_covered": len(both), "items_total": len(cohort),
        "by_bridge": {g: {"covered": sum(grp(r) == g for r in both), "total": sum(grp(r) == g for r in cohort.values())}
                      for g in ("named place", "other")},
        "countries_sub": {"covered": sum(r["src"] in S.COUNTRIES for r in both),
                          "total": sum(r["src"] in S.COUNTRIES for r in cohort.values())},
        "japan_target": {"covered": sum(r["tgt"] == "Japan" for r in both),
                         "total": sum(r["tgt"] == "Japan" for r in cohort.values())}}
    easy = {}
    for f in sorted(G.glob("caps_easy_shard*.pt")):
        easy.update(torch.load(f, weights_only=False)["caps"])
    orig = {}
    for pat in ("caps_shard*.pt", "caps_v2*_shard*.pt"):
        for f in sorted(G.glob(pat)):
            orig.update(torch.load(f, weights_only=False)["caps"])
    sc = {}
    ctrl = [c for c in ("Japan", "Egypt", "Italy") if c in easy and easy[c].get("final") is not None
            and orig.get(c, {}).get("final") is not None]
    if ctrl:
        allc = [c for c in orig if orig[c].get("final") is not None]
        for L in (36, 40, 44, 47):
            li = BAND.index(L)
            mu = torch.stack([orig[c]["final"][:, li].float().mean(0) for c in allc]).mean(0)
            cen = lambda X: torch.nn.functional.normalize(X[:, li].float().mean(0) - mu, dim=0)   # noqa: E731
            same = {c: float(cen(easy[c]["final"]) @ cen(orig[c]["final"])) for c in ctrl}
            cross = [float(cen(easy[c]["final"]) @ cen(orig[d]["final"])) for c in ctrl for d in allc if d != c]
            sc[str(L)] = {"same_concept": same, "cross_concept_median": float(np.median(cross)),
                          "cross_concept_max": float(np.max(cross))}
    out["style_control"] = {"label": "descriptive (012E §15): cos(easy-prompt class mean, original class mean), centred",
                            "concepts": ctrl, "by_layer": sc}

    # ---------------- class-mean lever ----------------
    C = rows("cmlever")
    if C:
        out["CM_lever"] = lever(C, "CM_swap", "CM_rand")
        out["CM_lever"]["n_items"] = len(C)
        out["CM_lever"]["by_bridge"] = {g: lever([r for r in C if grp(r) == g], "CM_swap", "CM_rand") for g in ("named place", "other")}
        out["CM_lever"]["japan_target"] = rate([r for r in C if r["tgt"] == "Japan"], "CM_swap")
        out["CM_lever"]["non_japan"] = rate([r for r in C if r["tgt"] != "Japan"], "CM_swap")
        tt = [r for r in C if r.get("cm_edit_direction") == "toward_tgt"]
        out["CM_lever"]["toward_tgt"] = {"n": len(tt), "CM_swap": rate(tt, "CM_swap"), "CM_rand": rate(tt, "CM_rand")}
        out["CM_lever"]["dm_median"] = {a: float(np.median([r["arms"][a]["dm"] for r in C])) for a in ("CM_swap", "CM_rand")}
        out["CM_lever"]["involves_rescued_concept"] = sum(r.get("cm_source", {}).get("src") == "easy"
                                                          or r.get("cm_source", {}).get("tgt") == "easy" for r in C)
        out["edit_sizes_cm"] = {a: size_summary(C, a) for a in ("CM_swap", "CM_rand")}
        out["CM_size_reference_status"] = ("class-mean swap is the size reference" if out["CM_lever"]["pass_rule"]
                                           else "fails the lever rule: primary uses J-row-matched sizes (fix 2)")

    # ---------------- smoke ----------------
    Sm = rows("smoke")
    if Sm:
        arms = [a for a in Sm[0]["arms"] if a != "clean"]
        out["smoke"] = {"items": [r["name"] for r in Sm], "size_reference": {r["name"]: r.get("size_reference") for r in Sm},
                        "per_arm": {a: {"dm": [round(r["arms"][a]["dm"], 3) if "dm" in r["arms"].get(a, {}) else None for r in Sm],
                                        "success": [r["arms"][a].get("success") for r in Sm],
                                        "rel_dh_median_all_pos_by_layer": {
                                            L: float(np.median([np.median(r["arms"][a]["edit_log"][L]["rel_dh"]) for r in Sm
                                                                if L in (r["arms"][a].get("edit_log") or {})]))
                                            for L in ("36", "40", "44", "47")
                                            if any(L in (r["arms"][a].get("edit_log") or {}) for r in Sm)},
                                        "coord_err_bf16_max": max([v.get("coord_err_bf16", 0) for r in Sm
                                                                   for v in (r["arms"][a].get("edit_log") or {}).values()] or [0])}
                                    for a in arms},
                        "ref_size_rel_median_all_pos": {r["name"]: r.get("ref_size_rel_median_all_pos") for r in Sm}}
    # ---------------- full run (012E §16) ----------------
    F = rows("full")
    if F:
        out["full"] = full_analysis(F, rows("parity"), rows("cmlever"))
        out["overlap"] = overlap_analysis(rows("parity"), F)
    EN = rows("energy")
    if EN and F:
        out["energy"] = energy_analysis(EN, F)
    E.RES.mkdir(parents=True, exist_ok=True)
    p = E.RES / "stageE_step4.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
