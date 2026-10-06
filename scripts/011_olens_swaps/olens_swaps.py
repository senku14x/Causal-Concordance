"""011 arm C: Oracle-space swaps on brew dev (Camila's suggestion via the PI, 2026-10-06). Model-side intervention along
directions from the Oracle Lens's own reconstructor (AR); experiments about reader geometry, never labels.

  python olens_swaps.py --vec-run v3 --run-id s1 [--limit-pairs N]   (resumes: finished pair files are skipped)

SUPERSEDED in part (report 31): v1 failed validation; runs use v3 (amendment 2). Amendment 3 added the C_raw_* and
D_* arms (additive reading). Gate bookkeeping is rebuilt from every pair file, so resumed runs report all pairs.

Per pair (Track B's 82 gated dev pairs, member <- swap, p -> m margin), clamped per layer at the AR's layers inside the
band (L40, L44, L48) over the three emit cells, with Track B's clamp primitive (c = R (h - mu); h <- h + A (t - c),
R A = I; t from the clean run's coordinates):
  C_white       (primary, "Camila's swap") V~ = [a_z, a_z'] = the AR colour vectors centred over the 12 colours, in the
                AR's ZCA-whitened space; coordinates c~ = V~+ W (h - mu); A = W^-1 V~; target = sigma(c)
  C_white_a2    alpha 2;  C_white_unc  uncentred vectors;  C_rawcoord  the same directions, raw coordinates (A = V, R = V+)
  controls      answer colours (a <-> a') [positive], two off-trajectory colours [negative], 4 random whitened pairs
                (edit norm matched to C_white's per layer and cell)
  reference     Track B's class-mean full-direction swap at the same three layers (the lever)
Gates: G0 (zero clamp exact), G1 fidelity, G-clamp, G-RA, G-repro vs 001 (Track B's). Power: the answer-colour swap must
reach median S >= 0.15 before the z-swap is read (Track B's rule for its arm A).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "008_trackb"))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "004_span_assay"))
import ar_vectors as AV                            # noqa: E402
import c008 as C8                                  # noqa: E402
import regime as RG                                # noqa: E402
import registry as R                               # noqa: E402
import span_assay as SA                            # noqa: E402
import stats as ST                                 # noqa: E402
import trackb as TB                                # noqa: E402

LAYERS = (40, 44, 48)
POWER_MIN = 0.15


def fin(A, Rm, mu):
    RA = Rm @ A
    err = float((RA - torch.eye(RA.shape[0], dtype=RA.dtype)).abs().max())
    return (A.float().cuda(), Rm.float().cuda(), mu.float().cuda()), err


def build(P, AR, WH, D):
    pr = P.pair
    z, z2, a, a2 = pr.m.z, pr.p.z, pr.m.a, pr.p.a
    traj = {pr.m.inp, z, a, pr.p.inp, z2, a2}
    off = [c for c in AV.COLOURS if c not in traj]
    rng = np.random.default_rng(SA.seed_of(pr.pair_id, "011_off"))
    o1, o2 = list(rng.choice(off, 2, replace=False))
    ci = {c: i for i, c in enumerate(AV.COLOURS)}
    errs = []

    def vec(L, c, centred=True):
        Z = AR["vecs"][(L, "space")]["zca_white"].double()
        v = Z[ci[c]] - (Z.mean(0) if centred else 0)
        return v

    def white_mats(L, Vt):
        w = WH[L]
        (A, Rm, mu), e = fin(w["Wi"] @ Vt, torch.linalg.pinv(Vt) @ w["W"], w["mu"])
        errs.append(e)
        return A, Rm, mu

    def sw(alpha=1.0):
        return lambda c, L, p: TB.swap_target(c, ((0, 1),), alpha)

    conds = []

    def add(name, make, **meta):
        specs = {}
        for L in LAYERS:
            specs[L] = []
            for cell in C8.EMIT:
                p = P.pos[cell]
                A, Rm, mu, tf = make(L, cell, p)
                specs[L].append((p, A, Rm, mu, tf))
        conds.append({"name": name, "site": "emit3", "specs": specs, **meta})

    def colour_make(c1, c2, centred=True, alpha=1.0, coords="white"):
        def make(L, cell, p):
            Vt = torch.stack([vec(L, c1, centred), vec(L, c2, centred)], 1)
            if coords == "white":
                A, Rm, mu = white_mats(L, Vt)
            else:
                V = WH[L]["Wi"] @ Vt
                (A, Rm, mu), e = fin(V, torch.linalg.pinv(V), torch.zeros(5120, dtype=torch.float64))
                errs.append(e)
            return A, Rm, mu, sw(alpha)
        return make

    add("C_white", colour_make(z, z2), arm="C", role="z", coords="white", centred=True, alpha=1.0)
    add("C_white_a2", colour_make(z, z2, alpha=2.0), arm="C", role="z", coords="white", centred=True, alpha=2.0)
    # C_white_unc dropped by amendment 2: uncentred vectors depend on the AR's unexplained output scale and offset
    add("C_rawcoord", colour_make(z, z2, coords="raw"), arm="C", role="z", coords="raw", centred=True, alpha=1.0)
    add("Cctrl_answer", colour_make(a, a2), arm="C_ctrl_answer", role="answer", coords="white", centred=True, alpha=1.0)
    add("Cctrl_off", colour_make(o1, o2), arm="C_ctrl_off", role="off", coords="white", centred=True, alpha=1.0,
        off=[o1, o2])
    for i in range(4):
        def make(L, cell, p, i=i):
            g = torch.from_numpy(np.stack([SA.unit(SA.seed_of(pr.pair_id, "011_rand", i, k)) for k in (0, 1)], 1)).double()
            A, Rm, mu = white_mats(L, g)
            Ap, Rp, mup = white_mats(L, torch.stack([vec(L, z), vec(L, z2)], 1))

            def tf(c, L_, p_, A=A, Ap=Ap, Rp=Rp, mup=mup):
                h = P.clean[()][L_][p_]
                cp = Rp @ (h - mup)
                ep = Ap @ (TB.swap_target(cp, ((0, 1),)) - cp)
                ex = A @ (TB.swap_target(c, ((0, 1),)) - c)
                return c + float(ep.norm() / ex.norm().clamp_min(1e-12)) * (TB.swap_target(c, ((0, 1),)) - c)
            return A, Rm, mu, tf
        add(f"Crand_{i}", make, arm="C_rand", role="rand", coords="white", draw=i)

    def ref_make(L, cell, p):
        V = torch.stack([D["dirs"][(cell, L, P.T, z)].double(), D["dirs"][(cell, L, P.T, z2)].double()], 1)
        (A, Rm, mu), e = fin(V, torch.linalg.pinv(V), D["mu_dev"][(cell, L, P.T)].double())
        errs.append(e)
        return A, Rm, mu, sw()
    add("Bref_full_3L", ref_make, arm="B_ref", role="z", coords="raw")

    # ---- amendment 3 (POST HOC, after the 2-pair smoke s0): controls for the raw-coordinate swap, and the additive
    # reading of the tip ("subtract z's AR vector, add z' 's") at the class-mean reference's edit norm ----
    def h_clean(L, p):
        return P.clean[()][L][p].double().cpu()

    def swap_norm(V, Rm, mu, h):
        c = Rm @ (h - mu)
        return float((V @ (TB.swap_target(c, ((0, 1),)) - c)).norm())

    def raw_colour(L, c1, c2):
        return WH[L]["Wi"] @ torch.stack([vec(L, c1), vec(L, c2)], 1)

    def ref_norm(L, cell, p):
        V = torch.stack([D["dirs"][(cell, L, P.T, z)].double(), D["dirs"][(cell, L, P.T, z2)].double()], 1)
        return swap_norm(V, torch.linalg.pinv(V), D["mu_dev"][(cell, L, P.T)].double(), h_clean(L, p))

    add("C_raw_answer", colour_make(a, a2, coords="raw"), arm="C_raw_ctrl_answer", role="answer", coords="raw",
        centred=True, alpha=1.0)
    for i in range(2):
        def make(L, cell, p, i=i):
            g = torch.from_numpy(np.stack([SA.unit(SA.seed_of(pr.pair_id, "011_rawrand", i, k)) for k in (0, 1)], 1)).double()
            Rg = torch.linalg.pinv(g)
            Vz = raw_colour(L, z, z2)
            target = swap_norm(Vz, torch.linalg.pinv(Vz), torch.zeros(5120, dtype=torch.float64), h_clean(L, p))
            own = swap_norm(g, Rg, torch.zeros(5120, dtype=torch.float64), h_clean(L, p))
            r = target / max(own, 1e-12)
            (A, Rm, mu), e = fin(g, Rg, torch.zeros(5120, dtype=torch.float64))
            errs.append(e)
            return A, Rm, mu, (lambda c, L_, p_, r=r: c + r * (TB.swap_target(c, ((0, 1),)) - c))
        add(f"C_raw_rand_{i}", make, arm="C_raw_rand", role="rand", coords="raw", draw=i)

    def additive(dir_fn):
        def make(L, cell, p):
            d = dir_fn(L, cell, p).double()
            k = ref_norm(L, cell, p) / float(d.norm())
            (A, Rm, mu), e = fin(d[:, None], torch.linalg.pinv(d[:, None]), torch.zeros(5120, dtype=torch.float64))
            errs.append(e)
            return A, Rm, mu, (lambda c, L_, p_, k=k: c + k)
        return make

    def ar_diff(c1, c2):
        return lambda L, cell, p: WH[L]["Wi"] @ (vec(L, c2) - vec(L, c1))          # raw AR(c2) - AR(c1)

    add("D_ar", additive(ar_diff(z, z2)), arm="D", role="z", coords="additive")
    add("D_ar_answer", additive(ar_diff(a, a2)), arm="D_ctrl_answer", role="answer", coords="additive")
    add("D_classmean", additive(lambda L, cell, p: D["dirs"][(cell, L, P.T, z2)].double() - D["dirs"][(cell, L, P.T, z)].double()),
        arm="D_ref", role="z", coords="additive")
    for i in range(2):
        add(f"D_rand_{i}", additive(lambda L, cell, p, i=i: torch.from_numpy(SA.unit(SA.seed_of(pr.pair_id, "011_addrand", i, L, cell))).double()),
            arm="D_rand", role="rand", coords="additive", draw=i)
    return conds, max(errs)


def zero_cond(P, AR, WH):
    ci = {c: i for i, c in enumerate(AV.COLOURS)}
    specs = {}
    for L in LAYERS:
        Z = AR["vecs"][(L, "space")]["zca_white"].double()
        Vt = torch.stack([Z[ci[P.pair.m.z]] - Z.mean(0), Z[ci[P.pair.p.z]] - Z.mean(0)], 1)
        (A, Rm, mu), _ = fin(WH[L]["Wi"] @ Vt, torch.linalg.pinv(Vt) @ WH[L]["W"], WH[L]["mu"])
        specs[L] = [(P.pos[cell], A, Rm, mu, (lambda c, L_, p_: c.clone())) for cell in C8.EMIT]
    return {"name": "G0_zero", "site": "emit3", "arm": "G0", "specs": specs}


def summarize(rows):
    by = {}
    for r in rows:
        if "arm" in r and r["arm"] != "G0" and not r.get("quarantined"):
            by.setdefault(r["name"], []).append(r)
    out = {}
    for nm, rs in sorted(by.items()):
        S = np.array([r["S"] for r in rs])
        T = np.array([r["table"] for r in rs])
        ci = ST.cluster_bootstrap(S, T, stat=np.median, B=10000, seed=C8.BOOT_SEED)["ci"]
        out[nm] = {"n": len(rs), "median_S": float(np.median(S)), "ci": ci, "mean_S": float(S.mean()),
                   "flip_rate": float(np.mean([r["flip"] for r in rs])), "pos": int((S > 0).sum()), "neg": int((S < 0).sum())}
    return out


def bookkeeping(rows, log_path, expected):
    """Gate bookkeeping over EVERY pair file, so a resumed run reports all pairs, not only the newly run ones
    (report 31). Pairs written before the per-pair `ra_err` field existed take it from the run log (1 significant
    digit), and the record says so."""
    import re
    done = sorted({r["pair_id"] for r in rows})
    ra = {r["pair_id"]: r["ra_err"] for r in rows if r.get("name") == "G_repro" and "ra_err" in r}
    from_log = {}
    if log_path.exists():
        for ln in open(log_path):
            m = re.search(r"\] (\S+) \d+s repro .* RA ([0-9.eE+-]+)", ln)
            if m and m.group(1) not in ra:
                from_log[m.group(1)] = max(from_log.get(m.group(1), 0.0), float(m.group(2)))
    missing_ra = [p for p in done if p not in ra and p not in from_log]
    errs = list(ra.values()) + list(from_log.values())
    return {"n_pairs_done": len(done), "n_pairs_expected": len(expected),
            "complete": set(done) >= set(expected), "missing_pairs": sorted(set(expected) - set(done)),
            "quarantined": sum(1 for r in rows if r.get("quarantined")),
            "repro_fail": sorted(r["pair_id"] for r in rows if r.get("name") == "G_repro" and not r["pass"]),
            "G0": {r["pair_id"]: {"dm": r["dm"], "pass": r["pass"]} for r in rows if r.get("name") == "G0_zero"},
            "ra_err_max": max(errs) if errs else None,
            "ra_err_source": {"rows": len(ra), "log_1sigfig": len(from_log), "missing": missing_ra}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-run", default="v1")
    ap.add_argument("--run-id", default="s1")
    ap.add_argument("--limit-pairs", type=int, default=0)
    a = ap.parse_args()
    assert not torch.backends.cuda.matmul.allow_tf32
    val = json.loads((AV.RES / f"ar_validation_{a.vec_run}.json").read_text())
    AV.WH_PREFIX[0] = val["whitener"]
    assert val.get("acceptance_basis"), "AR validation did not pass; no Oracle-space swaps"
    d = AV.OUT / a.run_id
    (d / "pairs").mkdir(parents=True, exist_ok=True)
    log = AV.Logger(d / "log.txt")
    AR = torch.load(AV.OUT / a.vec_run / "ar_colour_vectors.pt", weights_only=False)
    WH = {L: AV.whitener(L) for L in LAYERS}
    D = torch.load(C8.OUT / "directions.pt", weights_only=False)
    tok = R.make_tokenizer()
    rend, keys, tables, gated = TB.material(tok)
    groups = SA.spelling_groups(tok)
    ref001 = TB.p001()
    ents = gated[: a.limit_pairs] if a.limit_pairs else gated
    state = {"quarantined": 0, "ra_err_max": 0.0, "repro_fail": [], "G0": {}, "convention": val["convention"],
             "acceptance_basis": val["acceptance_basis"], "vec_run": a.vec_run,
             "label": "POST HOC, exploratory (011 amendment 2)" if val["acceptance_basis"] != "preset" else "exploratory"}
    model = R.load_model()
    t_all = time.time()
    with RG.SecEngine(model, l0=39) as sec:
        for n, e in enumerate(ents):
            pid = e["pair"].pair_id
            f = d / "pairs" / f"{pid}.jsonl"
            if f.exists():
                continue
            t0 = time.time()
            P = TB.Pair(sec, tok, e, groups, keys, tables)
            rows = [{**P.base, "name": "clean"}]
            S_rep = P.repro()
            rows.append({**P.base, "name": "G_repro", "S": S_rep, "S_001": ref001[pid], "pass": abs(S_rep - ref001[pid]) <= 0.01})
            if not rows[-1]["pass"]:
                state["repro_fail"].append(pid)
            if n < 2:
                r0 = P.evaluate(zero_cond(P, AR, WH))
                state["G0"][pid] = {"dm": r0["dm"], "pass": abs(r0["dm"]) <= C8.FLOOR and r0["fid_ok"]}
                rows.append({**r0, "pass": state["G0"][pid]["pass"]})
            conds, err = build(P, AR, WH, D)
            rows[1]["ra_err"] = float(err)
            for c in conds:
                r = P.evaluate(c)
                if not (r["fid_ok"] and r["clamp_ok"]):
                    r["quarantined"] = True
                    state["quarantined"] += 1
                rows.append(r)
            f.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
            sec.clear()
            log(f"[{n + 1}/{len(ents)}] {pid} {time.time() - t0:.0f}s repro {S_rep:.3f}/{ref001[pid]:.3f} RA {err:.1e}")
    rows = [json.loads(l) for f in sorted((d / "pairs").glob("*.jsonl")) for l in open(f)]
    state.update(bookkeeping(rows, d / "log.txt", [e["pair"].pair_id for e in ents]))
    summ = summarize(rows)
    ans = summ.get("Cctrl_answer", {}).get("median_S", 0.0)
    state.update({"minutes": (time.time() - t_all) / 60,
                  "G_RA_pass": state["ra_err_max"] is not None and state["ra_err_max"] <= 1e-5 and not state["ra_err_source"]["missing"],
                  "power_answer_swap": ans, "power_pass": ans >= POWER_MIN, "summary": summ})
    AV.RES.mkdir(parents=True, exist_ok=True)
    (AV.RES / f"olens_swaps_{a.run_id}.json").write_text(json.dumps(state, indent=1, default=float))
    log("done", json.dumps({k: (round(v["median_S"], 3), round(v["flip_rate"], 2)) for k, v in summ.items()}),
        "power", ans >= POWER_MIN)


if __name__ == "__main__":
    main()
