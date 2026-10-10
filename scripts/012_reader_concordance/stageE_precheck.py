"""012 Stage E step 2d (design `docs/design_specs/012E_ar_concept_swaps.md` §6): J-share pre-check, nearest-tag check and
the vector-quality gate. CPU only; no edits to the model. READER-SIDE (AR outputs). EXPLORATORY.

Vectors: per (concept, set, tag) the mean of the AR outputs, centred across the family's concepts, unit-normalised.
"Phrase" vectors pool the two LLM sets (starts + mentions); each set is also reported alone. Differences are
unit(AR(z') - AR(z)) (brew, Track B's gated pairs) or unit(AR(tgt) - AR(src)) (multihop, the 82-item cohort).

Gate (design §6c; thresholds fixed in the design):
  stability       median split-half cosine across concepts >= 0.8 (phrases split in two halves, seed 11)
  specificity 1   cos(AR diff, class-mean diff) matched > mismatched: mean paired difference, cluster bootstrap CI > 0
                  (brew: 008's leave-table-out class means at emit_colon, table clusters; multihop: Stage E class means
                  at the final token, relation clusters). Mismatched = the AR diffs of 5 seeded pairs disjoint from the
                  item's concepts.
  specificity 2   phrase matched median cosine - single-word matched median cosine >= 0.05
  decodability    nearest-concept accuracy (centred cosine) >= 2 x chance and above the single-word vectors
                  (brew: 007's held-out test captures at emit_colon, L40/L44; multihop: the gated class-mean prompts'
                  final-token captures at the tag layers)
  stop rule       a family stops if phrases beat single words on neither specificity 2 nor decodability
Pre-checks: (a) J share of each difference at each edit layer vs the 0.20 random expectation (design threshold 0.30
for the J-attribution reading); (b) median cosine of the multihop AR differences between adjacent tags (>= 0.7).

  python stageE_precheck.py [--tag smoke]  -> results/012_reader_concordance/analysis/stageE_precheck[_tag].json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "008_trackb"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))
import stageE_common as E                                                   # noqa: E402

SEED = 11
NB = 10000
BASES = E.OUT / "bases"
EDIT = {"multihop": list(range(36, 48)), "brew": [40, 44, 48]}
TAG_OF = {L: (36 if L < 40 else 40 if L < 44 else 44) for L in range(36, 48)}   # nearest tag at or below (design §2)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def Jb(L):
    return torch.load(BASES / f"J_raw_L{L}.pt", weights_only=False)["Q"][:, :1024].double()


def boot(vals, clus, nb=NB, seed=SEED):
    vals, clus = np.asarray(vals, float), np.asarray(clus)
    keys = sorted(set(clus))
    idx = {k: np.where(clus == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(nb):
        pick = rng.choice(len(keys), len(keys))
        ms.append(np.mean(np.concatenate([vals[idx[keys[i]]] for i in pick])))
    return {"mean": float(vals.mean()), "median": float(np.median(vals)), "n": int(len(vals)),
            "n_clusters": len(keys), "ci95": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]}


def concept_vecs(out, fam, sets, L, concepts, half=None):
    """mean over the given sets' outputs, centred across `concepts`, unit. half: None, 0 or 1 (seeded split)."""
    M = {}
    for c in concepts:
        xs = [out[fam][c][s][L].double() for s in sets if s in out[fam][c] and L in out[fam][c][s]]
        if not xs:
            continue
        X = torch.cat(xs)
        if half is not None:
            g = torch.Generator().manual_seed(SEED + hash(c) % 100000)
            perm = torch.randperm(len(X), generator=g)
            X = X[perm[half::2]]
        M[c] = X.mean(0)
    mu = torch.stack(list(M.values())).mean(0)
    return {c: unit(v - mu) for c, v in M.items()}


def stability(out, fam, sets, L, concepts):
    a, b = concept_vecs(out, fam, sets, L, concepts, 0), concept_vecs(out, fam, sets, L, concepts, 1)
    cs = [float(a[c] @ b[c]) for c in a if c in b]
    return {"median_cos": float(np.median(cs)), "n": len(cs)}


def diff(V, x, y):
    return unit(V[y] - V[x]) if x in V and y in V else None


def specificity(pairs, V, CM, clusters, seed=SEED):
    """pairs: list of (x, y, cm_key); CM(cm_key) -> class-mean diff (unit) or None. Mismatched: 5 seeded pairs disjoint
    from {x, y}."""
    rng = np.random.default_rng(seed)
    matched, paired, cl = [], [], []
    pool = [(x, y) for x, y, _ in pairs]
    for (x, y, k), c in zip(pairs, clusters):
        d, a = CM(k), diff(V, x, y)
        if d is None or a is None:
            continue
        m = float(a @ d)
        others = [p for p in pool if not ({p[0], p[1]} & {x, y}) and diff(V, *p) is not None]
        pick = [others[i] for i in rng.choice(len(others), min(5, len(others)), replace=False)]
        mm = float(np.mean([float(diff(V, *p) @ d) for p in pick]))
        matched.append(m)
        paired.append(m - mm)
        cl.append(c)
    return {"matched_median": float(np.median(matched)) if matched else None,
            "matched_minus_mismatched": boot(paired, cl) if paired else None}


def decodability(V, X, labels):
    """X [n, D] activations (centred here by their mean), labels per row; nearest concept by cosine."""
    keys = [c for c in V]
    A = torch.stack([V[c] for c in keys])
    Xc = unit(X.double() - X.double().mean(0))
    pred = (Xc @ A.T).argmax(1)
    ok = [keys[int(p)] == l for p, l in zip(pred, labels) if l in V]
    return {"acc": float(np.mean(ok)) if ok else None, "n": len(ok), "chance": 1 / len(keys), "n_concepts": len(keys)}


def jshare(u, Q):
    return float(((Q.T @ u.double()) ** 2).sum() / (u.double() @ u.double()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--families", default="brew,multihop")
    a = ap.parse_args()
    fams = set(a.families.split(","))
    sfx = f"_{a.tag}" if a.tag else ""
    arz = torch.load(E.OUT / "ar" / f"ar_outputs{sfx}.pt", weights_only=False)
    out, tags = arz["outputs"], arz["tags"]
    res = {"label": "EXPLORATORY; 012 Stage E pre-checks and vector-quality gate (design 012E §6)",
           "phrase_vector": "pooled LLM phrases (starts + mentions); starts, mentions, template and word also reported"}
    SETS = {"phrases": ("starts", "mentions"), "starts": ("starts",), "mentions": ("mentions",),
            "template": ("template",), "word": ("word",)}

    # ---------------- brew ----------------
    if "brew" in out and "brew" in fams:
        import trackb as TB
        import registry as R
        D = torch.load(ROOT / "outputs/008_trackb/directions.pt", weights_only=False)
        _rend, _keys, tables, gated = TB.material(R.make_tokenizer())
        pairs = [(e["pair"].m.z, e["pair"].p.z, e) for e in gated]
        tab = [e["item"]["name"] for e in gated]
        cols = list(E.COLOURS)
        caps = np.load(ROOT / "outputs/007_jsubspace/brew_test/captures_test.npz")["X"]
        meta = json.loads((ROOT / "outputs/007_jsubspace/brew_test/captures_test_meta.json").read_text())
        zlab = [r["z"] for r in meta["renders"]]
        B = {}
        for L in tags["brew"]:
            Tix = lambda e: tables.index(e["item"]["name"])                                       # noqa: E731
            def CM(e, L=L):
                k1, k2 = ("emit_colon", L, Tix(e), e["pair"].m.z), ("emit_colon", L, Tix(e), e["pair"].p.z)
                return unit(D["dirs"][k2].double() - D["dirs"][k1].double()) if k1 in D["dirs"] and k2 in D["dirs"] else None
            row = {}
            for nm, ss in SETS.items():
                V = concept_vecs(out, "brew", ss, L, cols)
                r = {"specificity": specificity([(x, y, e) for x, y, e in pairs], V, CM, tab)}
                if nm in ("phrases", "starts", "mentions"):
                    r["stability"] = stability(out, "brew", ss, L, cols)
                if L in meta["layers"]:
                    li, ci = meta["layers"].index(L), meta["cells"].index("emit_colon")
                    r["decodability"] = decodability(V, torch.from_numpy(caps[:, ci, li]), zlab)
                if nm == "phrases":
                    Q = Jb(L)
                    sh = [jshare(diff(V, x, y), Q) for x, y, _ in pairs]
                    cm = [jshare(CM(e), Q) for _, _, e in pairs if CM(e) is not None]
                    r["jshare_ar_diff_median"], r["jshare_classmean_diff_median"] = float(np.median(sh)), float(np.median(cm))
                row[nm] = r
            B[str(L)] = row
        res["brew"] = B

    # ---------------- multihop ----------------
    if "multihop" in out and "multihop" in fams:
        items = E.cohort()
        concepts = sorted(out["multihop"])
        caps, gate = {}, {}
        for f in sorted((E.OUT / "gate").glob(f"caps{sfx}_shard*.pt")):
            caps.update(torch.load(f, weights_only=False)["caps"])
        for f in sorted((E.OUT / "gate").glob(f"gate{sfx}_shard*.json")):
            gate.update(json.loads(f.read_text())["concepts"])
        # matcher v2 re-gates (2026-10-10) replace the first pass for the concepts whose rows v2 changes; every other
        # concept's rows are identical under v1 and v2 (checked by the classification before the re-gate)
        regated = []
        for f in sorted((E.OUT / "gate").glob(f"caps{sfx}_v2*_shard*.pt")):
            caps.update(torch.load(f, weights_only=False)["caps"])
        for f in sorted((E.OUT / "gate").glob(f"gate{sfx}_v2*_shard*.json")):
            g2 = json.loads(f.read_text())["concepts"]
            gate.update(g2)
            regated += list(g2)
        layers = list(range(36, 48))
        kept = {c: g for c, g in gate.items() if not g["dropped"] and caps.get(c, {}).get("final") is not None}
        CMV = {L: {} for L in layers}
        if kept:
            for L in layers:
                li = layers.index(L)
                means = {c: caps[c]["final"][:, li].double().mean(0) for c in kept}
                mu = torch.stack(list(means.values())).mean(0)
                CMV[L] = {c: unit(v - mu) for c, v in means.items()}
        M = {"gate_summary": {"n_concepts_generated": len(gate), "n_kept": len(kept), "regated_matcher_v2": sorted(regated),
                              "dropped": sorted(c for c, g in gate.items() if g["dropped"])}}
        mc = [c for c in concepts if any(it["src"] == c or it["tgt"] == c for it in items)]   # src/tgt concepts
        for L in tags["multihop"]:
            row = {}
            for nm, ss in SETS.items():
                V = concept_vecs(out, "multihop", ss, L, mc)
                CM = lambda k, L=L: diff(CMV[L], k[0], k[1])                                     # noqa: E731
                r = {"specificity": specificity([(it["src"], it["tgt"], (it["src"], it["tgt"])) for it in items], V, CM,
                                                [E.rel(it["name"]) for it in items])}
                if nm in ("phrases", "starts", "mentions"):
                    r["stability"] = stability(out, "multihop", ss, L, mc)
                if kept:
                    li = layers.index(L)
                    X = torch.cat([caps[c]["final"][:, li] for c in kept])
                    lab = [c for c in kept for _ in range(len(caps[c]["final"]))]
                    r["decodability"] = decodability({c: V[c] for c in kept if c in V}, X, lab)
                row[nm] = r
            res.setdefault("multihop", {})[str(L)] = row
        # (a) J share at every edit layer (nearest-tag AR vector), (b) adjacent-tag agreement
        js, jcm = {}, {}
        for L in layers:
            V = concept_vecs(out, "multihop", SETS["phrases"], TAG_OF[L], mc)
            Q = Jb(L)
            js[str(L)] = float(np.median([jshare(diff(V, it["src"], it["tgt"]), Q) for it in items
                                          if diff(V, it["src"], it["tgt"]) is not None]))
            if CMV[L]:
                v = [jshare(diff(CMV[L], it["src"], it["tgt"]), Q) for it in items if diff(CMV[L], it["src"], it["tgt"]) is not None]
                jcm[str(L)] = float(np.median(v)) if v else None
        Vt = {L: concept_vecs(out, "multihop", SETS["phrases"], L, mc) for L in tags["multihop"]}
        adj = {}
        for L1, L2 in zip(tags["multihop"][:-1], tags["multihop"][1:]):
            cs = [float(diff(Vt[L1], it["src"], it["tgt"]) @ diff(Vt[L2], it["src"], it["tgt"])) for it in items]
            adj[f"{L1}-{L2}"] = float(np.median(cs))
        M.update({"jshare_ar_diff_median_by_layer": js, "jshare_classmean_diff_median_by_layer": jcm,
                  "nearest_tag_adjacent_cos_median": adj})
        res["multihop_prechecks"] = M

    E.RES.mkdir(parents=True, exist_ok=True)
    p = E.RES / f"stageE_precheck{sfx}{'' if fams == {'brew', 'multihop'} else '_' + '_'.join(sorted(fams))}.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
