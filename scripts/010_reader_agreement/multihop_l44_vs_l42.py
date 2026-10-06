"""010 Task 2: Oracle L44 vs NLA L42 on 007 Stage 3 multihop (SECONDARY; 9 key clusters; EXPLORATORY, post hoc, CPU only).

  CUDA_VISIBLE_DEVICES="" /venv/main/bin/python scripts/010_reader_agreement/multihop_l44_vs_l42.py

Reads saved reader LLs and restatement samples only (no model, no GPU, no judge):
  outputs/007_jsubspace/stage3/m1/ll/oracle_L{40,44}_*.npz, restatement_oracle_L{40,44}.jsonl (pre-registered run)
  outputs/007_jsubspace/stage3/m1_nla_posthoc/ll/nla_L42_*.npz, restatement_nla_L42.jsonl (post hoc A2 run)
Bridge D per pair (i, j; same key group, different bridge and target) = [LL_i(b_i) - LL_i(b_j)] - [LL_j(b_i) - LL_j(b_j)],
LL = log-sum-exp over a string's spellings; item value = mean over the pairs it belongs to (analysis_secondary.stage3).
Key-cluster bootstrap: 10,000 draws, default_rng(20261004).choice(keys) per statistic (stage3's procedure).
Restatement word match: analysis_secondary.py lines 115-155 (deterministic regex; no judge).
Writes results/010_reader_agreement/analysis/multihop_l44_vs_l42.json.
"""
import itertools
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np                                                        # noqa: E402
from scipy import stats as sst                                            # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import agree_lib as A                                                     # noqa: E402

S3 = PROJECT / "outputs" / "007_jsubspace" / "stage3"
RES = PROJECT / "results" / "010_reader_agreement" / "analysis"
BANK = Path("/workspace/sources/repos/workspace-bench/evals/multihop/items.json")
READERS = {"oracle_L40": "m1", "oracle_L44": "m1", "nla_L42": "m1_nla_posthoc"}
PIECES = ("whole", "J", "rest_J", "MR", "PCA")
COMPARISONS = (("oracle_L44", "nla_L42", "primary"), ("oracle_L40", "nla_L42", "sensitivity"),
               ("oracle_L40", "oracle_L44", "context: same reader, two layers"))
RESTATE_PIECES = ("whole", "J", "rest_J", "R0", "PCA", "MR")
B = 10000
LABEL = "secondary; 9 clusters"


def lse_by(arr, owner, names):
    owner = np.asarray(owner)
    out = np.empty((arr.shape[0], len(names)))
    for j, s in enumerate(names):
        cols = arr[:, owner == s]
        m = cols.max(1, keepdims=True)
        out[:, j] = (m + np.log(np.exp(cols - m).sum(1, keepdims=True)))[:, 0]
    return out


def load_ll(tag):
    d = S3 / READERS[tag] / "ll"
    S = {}
    for f in sorted(d.glob(f"{tag}_*__*.npz")):
        z = np.load(f)
        idx, owner = list(z["items"]), list(z["owner"])
        names = list(dict.fromkeys(owner))
        for nm in z.files:
            if nm in ("items", "owner", "none"):
                continue
            M = lse_by(z[nm], owner, names)
            for a, i in enumerate(idx):
                for b, s in enumerate(names):
                    S.setdefault(nm, {})[(int(i), s)] = float(M[a, b])
    return S


def pair_D(tab, pairs, items):
    return np.array([(tab[(p["i"], items[p["i"]]["intermediates"][0])] - tab[(p["i"], items[p["j"]]["intermediates"][0])])
                     - (tab[(p["j"], items[p["i"]]["intermediates"][0])] - tab[(p["j"], items[p["j"]]["intermediates"][0])])
                     for p in pairs])


def item_means(v, pairs):
    pv = {}
    for x, p in zip(v, pairs):
        pv.setdefault(p["i"], []).append(x)
        pv.setdefault(p["j"], []).append(x)
    its = sorted(pv)
    return np.array([np.mean(pv[i]) for i in its]), its


def key_boot_mean(x, keys):
    """stage3's key-cluster bootstrap of a mean (fresh default_rng(20261004) per statistic)."""
    ug = sorted(set(keys))
    rng = np.random.default_rng(A.BOOT_SEED)
    bm = []
    for _ in range(B):
        gs = rng.choice(ug, len(ug))
        bm.append(np.mean(np.concatenate([x[keys == g] for g in gs])))
    return [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))]


def key_draws(keys):
    ug = sorted(set(keys))
    rng = np.random.default_rng(A.BOOT_SEED)
    return ug, [rng.choice(ug, len(ug)) for _ in range(B)]


def summ(x, keys):
    ug = sorted(set(keys))
    gm = np.array([x[keys == g].mean() for g in ug])
    flips = np.array([np.mean(gm * np.array(s)) for s in itertools.product((1, -1), repeat=len(ug))])
    return {"mean": float(x.mean()), "ci": key_boot_mean(x, keys), "n": int(len(x)), "n_keys": len(ug),
            "pos": int((x > 0).sum()), "neg": int((x < 0).sum()),
            "keys_pos": int((gm > 0).sum()), "keys_neg": int((gm < 0).sum()),
            "key_means": {g: float(m) for g, m in zip(ug, gm)},
            "signflip_p_two_sided_keys": float(np.mean(np.abs(flips) >= abs(gm.mean()) - 1e-12))}


def demean(x, keys):
    x = np.asarray(x, float).copy()
    for g in set(keys):
        x[keys == g] -= x[keys == g].mean()
    return x


def corr_boot(a, b, keys):
    """Pearson / Spearman between two readers' values, plus Pearson within key (each key's mean removed: agreement on
    which items of a key carry more, not on which keys do). Key-cluster bootstrap CIs."""
    ug, draws = key_draws(keys)
    rp, rs, rw = [], [], []
    for gs in draws:
        sel = np.concatenate([np.where(keys == g)[0] for g in gs])
        if np.std(a[sel]) > 0 and np.std(b[sel]) > 0:
            rp.append(np.corrcoef(a[sel], b[sel])[0, 1])
            rs.append(sst.spearmanr(a[sel], b[sel])[0])
        wa, wb = demean(a[sel], keys[sel]), demean(b[sel], keys[sel])
        if np.std(wa) > 0 and np.std(wb) > 0:
            rw.append(np.corrcoef(wa, wb)[0, 1])
    wa, wb = demean(a, keys), demean(b, keys)
    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "pearson_ci": A.ci(rp),
            "spearman": float(sst.spearmanr(a, b)[0]), "spearman_ci": A.ci(rs),
            "pearson_within_key": float(np.corrcoef(wa, wb)[0, 1]), "pearson_within_key_ci": A.ci(rw),
            "n": int(len(a)), "n_keys": len(ug), "valid_draws": len(rp)}


# ---------------------------------------------------------------------------------------------- restatement
def flags(tag, items, cues):
    out = {}
    for l in open(S3 / READERS[tag] / f"restatement_{tag}.jsonl"):
        r = json.loads(l)
        it = items[r["idx"]]
        b = it["intermediates"][0]
        bw = [b] + ([b.split()[-1]] if len(b.split()) > 1 else [])
        for k, s in enumerate(r["samples"]):
            t = s.lower()
            nb = any(re.search(r"\b" + re.escape(w.lower()) + r"\b", t) for w in bw)
            nt = bool(re.search(r"\b" + re.escape(it["target"].lower()) + r"\b", t))
            cw = cues.get(r["idx"], [])
            rc = any(re.search(r"\b" + re.escape(w.lower()) + r"\b", t) for w in cw) if cw else None
            out[(r["piece"], r["split"], r["idx"], k)] = {"nb": bool(nb), "nt": nt, "rc": rc}
    return out


def mcnemar(b, c):
    n = b + c
    return float(min(1.0, 2 * sst.binom.cdf(min(b, c), n, 0.5))) if n else None


def ratio_boot(num, den, keys):
    """Key-cluster bootstrap of sum(num)/sum(den) (per-item counts)."""
    ug, draws = key_draws(keys)
    vals = []
    for gs in draws:
        sel = np.concatenate([np.where(keys == g)[0] for g in gs])
        if den[sel].sum() > 0:
            vals.append(num[sel].sum() / den[sel].sum())
    return A.ci(vals)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    items = json.loads(BANK.read_text())["items"]
    dm = json.loads((S3 / "m1" / "design_material.json").read_text())
    assert dm == json.loads((S3 / "m1_nla_posthoc" / "design_material.json").read_text())
    cues = {int(k): v for k, v in dm["cue_words"].items()}
    LL = {t: load_ll(t) for t in READERS}
    res = {"label": f"{LABEL}; EXPLORATORY, post hoc (010, 2026-10-06); CPU only; test split primary (items 51-100, "
                    "untouched), dev secondary (touched); the NLA run is itself post hoc (A2)",
           "conventions": {"bridge_D": "per pair [LL_i(b_i) - LL_i(b_j)] - [LL_j(b_i) - LL_j(b_j)]; item = mean over its pairs",
                           "bootstrap": "key-cluster, 10,000 draws, default_rng(20261004).choice per statistic (stage3)",
                           "signflip": "exact, over the 2^n_keys sign patterns of the key means, two-sided",
                           "random_mean": "per-pair mean over R0-R9 (10 dimension-matched random slices)",
                           "caveat": "D cannot separate reading the bridge from restating a cue (design §1.5); 9 key "
                                     "clusters, one of which (capital) holds 66 of 104 test pairs and 12 of 40 test items"}}

    # ------------------------------------------------------------------------------------------ reproduction
    rep = {}
    for tag, f in (("oracle_L40", "stage3"), ("oracle_L44", "stage3"), ("nla_L42", "posthoc_A2_stage3")):
        j = json.loads((PROJECT / f"results/007_jsubspace/analysis/{f}.json").read_text())["readers"][tag]["pieces"]
        for sp in ("test", "dev"):
            prs = [p for p in dm["pairs"] if p["split"] == sp]
            for pc in ("whole", "J", "rest_J", "MR", "PCA", "R0"):
                x, its = item_means(pair_D(LL[tag][pc], prs, items), prs)
                keys = np.array([items[i]["name"].split("-")[-1] for i in its])
                ci_ = key_boot_mean(x, keys)
                rr = j[pc][f"bridge|{sp}"]
                rep[f"{tag} {pc} bridge|{sp}"] = {"reported_mean": rr["mean"], "recomputed_mean": float(x.mean()),
                                                  "reported_ci": rr["ci"], "recomputed_ci": ci_,
                                                  "pass": bool(abs(rr["mean"] - x.mean()) < 1e-9 and
                                                               max(abs(a - b) for a, b in zip(rr["ci"], ci_)) < 1e-9)}
            rmean = np.mean([j[f"R{r}"][f"bridge|{sp}"]["mean"] for r in range(10)])
            rep[f"{tag} random_mean bridge|{sp}"] = {"reported_mean_of_10": float(rmean)}
    res["reproduction"] = rep
    print("bridge-D reproduction pass:", all(v["pass"] for v in rep.values() if "pass" in v))

    # ------------------------------------------------------------------------------------------ bridge D, paired
    out = {}
    for sp in ("test", "dev"):
        prs = [p for p in dm["pairs"] if p["split"] == sp]
        pk = np.array([p["key"] for p in prs])
        V = {}
        for tag in READERS:
            for pc in PIECES:
                V[(tag, pc)] = pair_D(LL[tag][pc], prs, items)
            V[(tag, "random_mean")] = np.mean([pair_D(LL[tag][f"R{r}"], prs, items) for r in range(10)], 0)
            V[(tag, "J_minus_whole")] = V[(tag, "J")] - V[(tag, "whole")]
            V[(tag, "J_minus_random_mean")] = V[(tag, "J")] - V[(tag, "random_mean")]
        _, its = item_means(V[("nla_L42", "whole")], prs)
        ik = np.array([items[i]["name"].split("-")[-1] for i in its])
        sp_out = {"n_pairs": len(prs), "n_items": len(its), "n_keys": len(set(pk)),
                  "pairs_per_key": {k: int((pk == k).sum()) for k in sorted(set(pk))},
                  "items_per_key": {k: int((ik == k).sum()) for k in sorted(set(ik))}, "per_reader": {}, "comparisons": {}}
        for tag in READERS:
            sp_out["per_reader"][tag] = {}
            for pc in PIECES + ("random_mean", "J_minus_whole", "J_minus_random_mean"):
                xi, _ = item_means(V[(tag, pc)], prs)
                sp_out["per_reader"][tag][pc] = {"items": summ(xi, ik), "pairs": summ(V[(tag, pc)], pk)}
        for a, b, role in COMPARISONS:
            cmp = {"role": role}
            for pc in PIECES + ("random_mean", "J_minus_whole", "J_minus_random_mean"):
                d = V[(a, pc)] - V[(b, pc)]
                di, _ = item_means(d, prs)
                xa, _ = item_means(V[(a, pc)], prs)
                xb, _ = item_means(V[(b, pc)], prs)
                cmp[pc] = {f"{a}_mean_items": float(xa.mean()), f"{b}_mean_items": float(xb.mean()),
                           "diff_items": summ(di, ik), "diff_pairs": summ(d, pk),
                           "corr_pairs": corr_boot(V[(a, pc)], V[(b, pc)], pk),
                           "corr_items": corr_boot(xa, xb, ik)}
            sp_out["comparisons"][f"{a} vs {b}"] = cmp
        out[sp] = sp_out
    res["bridge_D"] = out

    # ------------------------------------------------------------------------------------------ restatement
    F = {t: flags(t, items, cues) for t in READERS}
    rst = {"note": "word match of analysis_secondary.py 115-155; nb = names the bridge (bridge or its last word), rc = "
                   "restates a cue word (None when the item has no cue words); 2 samples per item, reader and piece; "
                   "sample pairing by sample index is arbitrary (independent draws), so item-level counts are also given"}
    for sp in ("test", "dev"):
        idxs = sorted({k[2] for k in F["nla_L42"] if k[1] == sp})
        for t in READERS:
            assert sorted({k[2] for k in F[t] if k[1] == sp}) == idxs
        keys = np.array([items[i]["name"].split("-")[-1] for i in idxs])
        cue_items = np.array([bool(cues.get(i)) for i in idxs])
        spr = {"n_items": len(idxs), "n_keys": len(set(keys)), "n_items_with_cue_words": int(cue_items.sum()),
               "n_keys_with_cue_words": len(set(keys[cue_items])), "shares": {}, "paired": {}}
        for t in READERS:
            for pc in RESTATE_PIECES:
                nbc = np.array([sum(F[t][(pc, sp, i, k)]["nb"] for k in (0, 1)) for i in idxs])
                nb_c = np.array([sum(F[t][(pc, sp, i, k)]["nb"] for k in (0, 1)) if cues.get(i) else 0 for i in idxs])
                rs_c = np.array([sum(F[t][(pc, sp, i, k)]["nb"] and bool(F[t][(pc, sp, i, k)]["rc"]) for k in (0, 1))
                                 if cues.get(i) else 0 for i in idxs])
                ntc = np.array([sum(F[t][(pc, sp, i, k)]["nt"] for k in (0, 1)) for i in idxs])
                spr["shares"][f"{t}|{pc}"] = {
                    "names_bridge": f"{int(nbc.sum())}/{2 * len(idxs)}", "names_bridge_share": float(nbc.sum() / (2 * len(idxs))),
                    "names_bridge_ci": ratio_boot(nbc.astype(float), np.full(len(idxs), 2.0), keys),
                    "names_target": f"{int(ntc.sum())}/{2 * len(idxs)}",
                    "restates_among_bridge_naming_cue_items": f"{int(rs_c.sum())}/{int(nb_c.sum())}",
                    "restate_share": float(rs_c.sum() / nb_c.sum()) if nb_c.sum() else None,
                    "restate_share_ci": ratio_boot(rs_c.astype(float), nb_c.astype(float), keys),   # non-cue items add 0/0
                    "_counts": {"nb_items": nbc.tolist(), "nb_cue": nb_c.tolist(), "restate_cue": rs_c.tolist()}}
        for a, b, role in COMPARISONS:
            cmp = {"role": role}
            for pc in RESTATE_PIECES:
                tab = {"both": 0, f"{a}_only": 0, f"{b}_only": 0, "neither": 0}
                for i in idxs:
                    for k in (0, 1):
                        x, y = F[a][(pc, sp, i, k)]["nb"], F[b][(pc, sp, i, k)]["nb"]
                        tab["both" if x and y else f"{a}_only" if x else f"{b}_only" if y else "neither"] += 1
                ca = np.array([sum(F[a][(pc, sp, i, k)]["nb"] for k in (0, 1)) for i in idxs])
                cb = np.array([sum(F[b][(pc, sp, i, k)]["nb"] for k in (0, 1)) for i in idxs])
                cross = {f"{u}{v}": int(((ca == u) & (cb == v)).sum()) for u in (0, 1, 2) for v in (0, 1, 2)}
                anya, anyb = ca > 0, cb > 0
                d = (ca - cb) / 2.0
                # restatement on sample pairs where both name the bridge, cue-word items only
                rt = {"both": 0, f"{a}_only": 0, f"{b}_only": 0, "neither": 0}
                for i in idxs:
                    if not cues.get(i):
                        continue
                    for k in (0, 1):
                        fa, fb = F[a][(pc, sp, i, k)], F[b][(pc, sp, i, k)]
                        if fa["nb"] and fb["nb"]:
                            x, y = bool(fa["rc"]), bool(fb["rc"])
                            rt["both" if x and y else f"{a}_only" if x else f"{b}_only" if y else "neither"] += 1
                sa, sb = spr["shares"][f"{a}|{pc}"]["_counts"], spr["shares"][f"{b}|{pc}"]["_counts"]
                ug, draws = key_draws(keys)
                dd = []
                for gs in draws:
                    sel = np.concatenate([np.where(keys == g)[0] for g in gs])
                    na_, nb_ = np.sum(np.array(sa["nb_cue"])[sel]), np.sum(np.array(sb["nb_cue"])[sel])
                    if na_ > 0 and nb_ > 0:
                        dd.append(np.sum(np.array(sa["restate_cue"])[sel]) / na_ - np.sum(np.array(sb["restate_cue"])[sel]) / nb_)
                cmp[pc] = {"sample_pairs_names_bridge": tab, "mcnemar_exact_p": mcnemar(tab[f"{a}_only"], tab[f"{b}_only"]),
                           f"P({a} names | {b} names)": tab["both"] / max(1, tab["both"] + tab[f"{b}_only"]),
                           f"P({a} names | {b} does not)": tab[f"{a}_only"] / max(1, tab[f"{a}_only"] + tab["neither"]),
                           f"P({b} names | {a} names)": tab["both"] / max(1, tab["both"] + tab[f"{a}_only"]),
                           "items_count_crosstab_a_b": cross,
                           "items_any_naming": {"both": int((anya & anyb).sum()), f"{a}_only": int((anya & ~anyb).sum()),
                                                f"{b}_only": int((~anya & anyb).sum()), "neither": int((~anya & ~anyb).sum()),
                                                "mcnemar_exact_p": mcnemar(int((anya & ~anyb).sum()), int((~anya & anyb).sum()))},
                           "naming_rate_diff_items": summ(d, keys),
                           "restatement_when_both_name_bridge_cue_items": {**rt, "mcnemar_exact_p": mcnemar(rt[f"{a}_only"], rt[f"{b}_only"])},
                           "restate_share_diff": {"estimate": (spr["shares"][f"{a}|{pc}"]["restate_share"] or 0) -
                                                  (spr["shares"][f"{b}|{pc}"]["restate_share"] or 0), "ci": A.ci(dd)}}
            spr["paired"][f"{a} vs {b}"] = cmp
        for v in spr["shares"].values():
            v.pop("_counts")
        rst[sp] = spr
    known = {"oracle_L44|J": "13/54", "nla_L42|J": "8/53", "oracle_L40|J": "15/53",
             "oracle_L40|whole": "27/48", "oracle_L44|whole": "12/48", "nla_L42|whole": "28/48"}
    rst["reproduction_report26"] = {k: {"reported": v, "recomputed": rst["test"]["shares"][k]["restates_among_bridge_naming_cue_items"],
                                        "pass": v == rst["test"]["shares"][k]["restates_among_bridge_naming_cue_items"]}
                                    for k, v in known.items()}
    res["restatement"] = rst
    print("restatement reproduction:", {k: v["pass"] for k, v in rst["reproduction_report26"].items()})
    (RES / "multihop_l44_vs_l42.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", RES / "multihop_l44_vs_l42.json")


if __name__ == "__main__":
    main()
