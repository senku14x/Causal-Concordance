"""007 secondary analyses (CPU; design §8.4-8.6, §9.3, §10 + A1). Reader-side; descriptive (no Holm).

  python analysis_secondary.py stage3 --run-id m1
  python analysis_secondary.py stage4b --run-id q1
  python analysis_secondary.py stage5 --run-id c1
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as AN                                                      # noqa: E402
import p007 as C                                                           # noqa: E402

CW12 = {"whole": (0.45, 0.43), "J": (0.48, 0.42), "rest_J": (0.20, 0.12), "R0": (0.17, 0.35)}   # Oracle, NLA (her L42)


def lse_by(arr, owner, names):
    owner = np.asarray(owner)
    out = np.empty((arr.shape[0], len(names)))
    for j, s in enumerate(names):
        cols = arr[:, owner == s]
        m = cols.max(1, keepdims=True)
        out[:, j] = (m + np.log(np.exp(cols - m).sum(1, keepdims=True)))[:, 0]
    return out


def stage3(run_id):
    d = C.OUT / "stage3" / run_id
    dm = json.loads((d / "design_material.json").read_text())
    items = json.loads(Path("/workspace/sources/repos/workspace-bench/evals/multihop/items.json").read_text())["items"]
    pairs = dm["pairs"]
    res = {"run_id": run_id, "readers": {}, "note": "secondary; D cannot separate reading the bridge from restating the cue (design §1.5)"}
    for rd, L in C.READERS:
        tag = f"{rd}_L{L}"
        files = sorted((d / "ll").glob(f"{tag}_*__*.npz"))
        if not files:
            continue
        S = {}                                                             # piece -> {(item, string): LL}
        for f in files:
            z = np.load(f)
            idx, owner = list(z["items"]), list(z["owner"])
            names = list(dict.fromkeys(owner))
            for nm in z.files:
                if nm in ("items", "owner"):
                    continue
                arr = z[nm]
                if nm == "none":
                    continue
                M = lse_by(arr, owner, names)
                for a, i in enumerate(idx):
                    for b, s in enumerate(names):
                        S.setdefault(nm, {})[(int(i), s)] = float(M[a, b])
        R = {"pieces": {}}
        for nm, tab in S.items():
            rec = {}
            for role, key in (("bridge", lambda i: items[i]["intermediates"][0]), ("target", lambda i: items[i]["target"])):
                for sp in ("dev", "test"):
                    pv = {}
                    for p in pairs:
                        if p["split"] != sp:
                            continue
                        i, j = p["i"], p["j"]
                        bi, bj = key(i), key(j)
                        v = (tab[(i, bi)] - tab[(i, bj)]) - (tab[(j, bi)] - tab[(j, bj)])
                        pv.setdefault(i, []).append(v)
                        pv.setdefault(j, []).append(v)
                    its = sorted(pv)
                    x = np.array([np.mean(pv[i]) for i in its])
                    keys = np.array([items[i]["name"].split("-")[-1] for i in its])
                    ug = sorted(set(keys))
                    # key-cluster bootstrap
                    rng = np.random.default_rng(C.BOOT_SEED)
                    bm = []
                    for _ in range(10000):
                        gs = rng.choice(ug, len(ug))
                        bm.append(np.mean(np.concatenate([x[keys == g] for g in gs])))
                    gmeans = np.array([x[keys == g].mean() for g in ug])
                    flips = [np.mean(gmeans * np.array(s)) for s in itertools.product((1, -1), repeat=len(ug))]
                    p_sf = float(np.mean(np.array(flips) >= np.mean(gmeans) - 1e-12))
                    rec[f"{role}|{sp}"] = {"mean": float(x.mean()), "ci": [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))],
                                           "n_items": int(len(x)), "n_groups": len(ug), "pos": int((x > 0).sum()),
                                           "neg": int((x < 0).sum()), "signflip_p_one_sided": p_sf}
            R["pieces"][nm] = rec
        # positive control: bridge activations vs the bridges of the same key group (within-group pairs)
        fpc = d / "ll" / f"{tag}_positive.npz"
        if fpc.exists():
            z = np.load(fpc)
            br, owner = list(z["bridges"]), list(z["owner"])
            grp = {}
            for it in items:
                grp.setdefault(it["intermediates"][0], set()).add(it["name"].split("-")[-1])
            pcs = {}
            for nm in z.files:
                if nm in ("bridges", "owner"):
                    continue
                M = lse_by(z[nm], owner, br)                               # [n_bridges, n_bridges]
                pos, tot, ids = 0, 0, []
                for a in range(len(br)):
                    same = [b for b in range(len(br)) if b != a and grp[br[a]] & grp[br[b]]]
                    if not same:
                        continue
                    cand = [a] + same
                    loo = [M[a, c] - np.mean([M[o, c] for o in range(len(br)) if o != a]) for c in cand]
                    ids.append(int(np.argmax(loo) == 0))
                    for b in same:
                        if b > a:
                            tot += 1
                            pos += int(((M[a, a] - M[a, b]) - (M[b, a] - M[b, b])) > 0)
                pcs[nm] = {"loo_prior_ID": float(np.mean(ids)) if ids else None, "pairwise_positive_share": pos / tot if tot else None,
                           "pass": bool(ids and np.mean(ids) >= 0.5 and tot and pos / tot >= 0.70)}
            R["positive_control"] = pcs
        # restatement flags (deterministic word match; no judge)
        fq = d / f"restatement_{tag}.jsonl"
        if fq.exists():
            cues = {int(k): v for k, v in dm["cue_words"].items()}
            flags = {}
            for l in open(fq):
                r = json.loads(l)
                it = items[r["idx"]]
                b = it["intermediates"][0]
                bw = [b] + ([b.split()[-1]] if len(b.split()) > 1 else [])
                for s in r["samples"]:
                    t = s.lower()
                    nb = any(re.search(r"\b" + re.escape(w.lower()) + r"\b", t) for w in bw)
                    nt = bool(re.search(r"\b" + re.escape(it["target"].lower()) + r"\b", t))
                    cw = cues.get(r["idx"], [])
                    rc = any(re.search(r"\b" + re.escape(w.lower()) + r"\b", t) for w in cw) if cw else None
                    key = (r["piece"], r["split"])
                    fl = flags.setdefault(key, {"n": 0, "names_bridge": 0, "bridge_without_cue": 0, "restates_cue": 0,
                                                "names_target": 0, "cue_undeterminable": 0, "first30_names_bridge": [0, 0]})
                    fl["n"] += 1
                    fl["names_bridge"] += nb
                    fl["names_target"] += nt
                    if rc is None:
                        fl["cue_undeterminable"] += 1
                    else:
                        fl["restates_cue"] += rc
                        fl["bridge_without_cue"] += int(nb and not rc)
                    if r["idx"] < 30:
                        fl["first30_names_bridge"][0] += nb
                        fl["first30_names_bridge"][1] += 1
            R["restatement"] = {f"{k[0]}|{k[1]}": {**v, **{f"P({kk})": v[kk] / v["n"] for kk in
                                                         ("names_bridge", "bridge_without_cue", "restates_cue", "names_target")}}
                                for k, v in flags.items()}
            R["cw12_comparison"] = {nm: {"ours_first30_names_bridge": (flags[(nm, "dev")]["first30_names_bridge"][0] /
                                                                       max(1, flags[(nm, "dev")]["first30_names_bridge"][1]))
                                         if (nm, "dev") in flags else None,
                                         "cw12_oracle_L42_offcontract": CW12[nm][0], "cw12_nla_L42": CW12[nm][1]}
                                    for nm in CW12}
        res["readers"][tag] = R
    st = json.loads((d / "state.json").read_text())
    res["gates"] = {k: st.get(k) for k in ("G_cap", "G_cw", "pieces_meta")}
    res["gates"]["readers"] = st.get("readers")
    out_name = "stage3.json" if run_id == "m1" else f"posthoc_A2_stage3.json"   # never overwrite the pre-registered run
    res["label"] = "pre-registered run" if run_id == "m1" else "post hoc (A2): NLA on 006's exact scorer"
    C.dump(C.RES / "analysis" / out_name, res)
    for tag, R in res["readers"].items():
        print(tag, {nm: (round(v["bridge|test"]["mean"], 2), v["bridge|test"]["ci"]) for nm, v in R["pieces"].items()
                    if nm in ("whole", "J", "rest_J", "PCA", "MR", "R0")})
    return res


def stage4b(run_id):
    d = C.OUT / "stage4" / run_id
    f4a = C.RES / "analysis" / "stage4a.json"
    a4 = json.loads(f4a.read_text())
    if a4.get("stop", True) or not (d / "parta_material.json").exists():
        C.dump(C.RES / "analysis" / "stage4b.json", {"run": False, "reason": "4a did not yield both positions or 4b not run"})
        return
    pm = json.loads((d / "parta_material.json").read_text())
    rend = pm["renders"]
    na = len(rend) // 2
    import relational as RL
    import registry as R
    tok = R.make_tokenizer()
    mats = {e["pair"].pair_id: e for e in RL.material(tok)}
    res = {"positions": pm["positions"], "readers": {}}
    for rd, L in C.READERS:
        for k in pm["positions"]:
            f = d / f"ll_{rd}_L{L}_{k}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            owner = list(z["owner"])
            rec = {}
            for nm in z.files:
                if nm in ("owner", "none"):
                    continue
                arr = z[nm]
                vals = []
                for s in range(na):
                    pid = rend[s][0]
                    e = mats[pid]
                    Xa, Xb = e["pair"].m.z[0], e["pair"].p.z[0]
                    ia, ib = owner.index(Xa), owner.index(Xb)
                    vals.append((arr[s, ia] - arr[s, ib]) - (arr[s + na, ia] - arr[s + na, ib]))
                rec[nm] = AN.summarize(np.array(vals), AN.boot_idx(na))
            res["readers"].setdefault(f"{rd}_L{L}", {})[k] = rec
    C.dump(C.RES / "analysis" / "stage4b.json", res)
    for tag, v in res["readers"].items():
        print(tag, {k: {nm: round(x["mean"], 2) for nm, x in rec.items() if nm in ("whole", "J", "rest_J", "MR")} for k, rec in v.items()})


def stage5(run_id):
    d = C.OUT / "stage5" / run_id
    f = d / "ll" / "oracle_L56.npz"
    if not f.exists():
        C.dump(C.RES / "analysis" / "stage5.json", {"run": False})
        return
    mat = json.loads((d / "material.json").read_text())
    z = np.load(f)
    owner = list(z["owner"])
    digits = [str(i) for i in range(10)]
    res = {"n_pairs": len(mat["pairs"]), "PCA_k": mat["PCA_k"], "pieces": {}}
    for nm in z.files:
        if nm in ("owner", "none"):
            continue
        M = lse_by(z[nm], owner, digits)
        out = {}
        for role, km, kp in (("z1", "z_m", "z_p"), ("answer", "a_m", "a_p")):
            per = {}
            for p in mat["pairs"]:
                tm, tp = p[km], p[kp]
                if tm == tp:
                    continue
                v = (M[p["m"], tm] - M[p["m"], tp]) - (M[p["p"], tm] - M[p["p"], tp])
                per.setdefault(p["table"], []).append(v)
            x = np.array([np.mean(v) for v in per.values()])
            out[role] = AN.summarize(x, AN.boot_idx(len(x)))
        res["pieces"][nm] = out
    C.dump(C.RES / "analysis" / "stage5.json", res)
    print({nm: round(v["z1"]["mean"], 2) for nm, v in res["pieces"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage3", "stage4b", "stage5"])
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    rid = a.run_id or {"stage3": "m1", "stage4b": "q1", "stage5": "c1"}[a.what]
    {"stage3": stage3, "stage4b": stage4b, "stage5": stage5}[a.what](rid)


if __name__ == "__main__":
    main()
