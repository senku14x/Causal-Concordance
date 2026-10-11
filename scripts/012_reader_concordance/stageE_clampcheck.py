"""012E §18: clamp mechanism check (POST HOC, descriptive; spec 012E §18, written before this ran). EXPLORATORY.

Does a per-layer additive clamp along u' = cos u + sin w accumulate the persistent part, so that (h - h_clean).u settles
near k / cos(u', u)? 10 seeded items, arms AR_full@J, AR_rrem@J, AR_emrem@J, AR_Jrem@J built exactly as in
stageE_edits.py (full / energy modes). Prefill only, through the identical BandEditor path; the block outputs after the
clamp are captured at every band layer and position.

  CUDA_VISIBLE_DEVICES=k PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageE_clampcheck.py run
  python stageE_clampcheck.py analyze   ->  results/012_reader_concordance/analysis/stageE_clampcheck.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
OUTD = ROOT / "outputs/012_reader_concordance/stageE/clampcheck"
RESF = ROOT / "results/012_reader_concordance/analysis/stageE_clampcheck.json"
ARMS = ["AR_full@J", "AR_rrem@J", "AR_emrem@J", "AR_Jrem@J"]


def run():
    import torch
    from phrase_lens.benchmarks import load_benchmark
    from phrase_lens.interventions import BandEditor
    from phrase_lens.lenses import DirectionBank
    from phrase_lens.modeling import ModelRunner
    import stageE_common as E
    import stageE_edits as SE

    BAND, D = SE.BAND, SE.D
    cohort = sorted(r["name"] for r in E.cohort())
    names = sorted(np.random.default_rng(20261011).choice(cohort, 10, replace=False).tolist())
    arz, Q, _ = SE.load_material(with_cm=False)
    runner = ModelRunner.from_pretrained(SE.MODEL, device="cuda", dtype="bfloat16", revision=SE.REV, local_files_only=True)
    blocks = runner.blocks
    bank = DirectionBank.load(SE.SA / "bank_j")
    bench = {it["name"]: it for it in load_benchmark(SE.BENCH)["items"]}

    class Capture:
        """Block outputs at every band layer, all positions, prefill only; registered after the BandEditor hooks."""
        def __init__(self, n):
            self.n, self.acts, self.h = n, {}, []

        def __enter__(self):
            for L in BAND:
                def f(_m, _a, out, L=L):
                    x = out[0] if isinstance(out, tuple) else out
                    if x.shape[1] == self.n and L not in self.acts:
                        self.acts[L] = x[0].detach().float().cpu()
                self.h.append(blocks[L].register_forward_hook(f))
            return self

        def __exit__(self, *exc):
            for h in self.h:
                h.remove()

    def prefill(ids, edit):
        n = len(ids)
        if edit is None:
            with Capture(n) as cp:
                runner.generate(ids, edit=None, max_new_tokens=1, return_token_ids=True)
            return cp.acts
        with BandEditor(blocks, edit.band, edit, n), Capture(n) as cp:
            runner.generate(ids, edit=None, max_new_tokens=1, return_token_ids=True)
        return cp.acts

    OUTD.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for nm in names:
        it = bench[nm]
        ids = runner.encode(it["prompt"].rstrip())
        n, pos = len(ids), len(ids) - 1
        cap = runner.capture_ids(ids, positions=list(range(n)), layers=BAND)              # as stageE_edits.py (targets)
        clean = {L: cap[i] for i, L in enumerate(BAND)}
        s_dirs, t_dirs = bank.concept(it["src"], BAND), bank.concept(it["tgt"], BAND)
        Kj = SE.swap_sizes(s_dirs, t_dirs, clean)
        full = SE.ar_diff(arz, it["src"], it["tgt"], SE.PHRASES)
        u = {L: SE.unit(v) for L, v in full.items()}
        em, em_cos = {}, {}
        for L in BAND:                                                                      # exactly energy mode
            uu = SE.unit(full[L])
            sJ = float(((Q[L].T @ uu) ** 2).sum())
            d = int(round(sJ * D))
            g = torch.Generator(device="cuda").manual_seed(SE.seed_of(nm, "emrem", L))
            P, _ = torch.linalg.qr(torch.randn(D, d, generator=g, device="cuda"))
            uc = uu.cuda()
            em[L] = SE.unit((uc - P @ (P.T @ uc)).cpu())
            em_cos[L] = float(em[L] @ uu)
        dirs = {"AR_full@J": u, "AR_rrem@J": SE.rremove(full, nm), "AR_emrem@J": em, "AR_Jrem@J": SE.jremove(full, Q)}
        hc = prefill(ids, None)
        uJ = {L: SE.unit(Q[L] @ (Q[L].T @ u[L])) for L in BAND}
        rec = {"name": nm, "src": it["src"], "tgt": it["tgt"], "n_prompt": n, "read_pos": pos, "emrem_cos_rebuilt": em_cos, "arms": {}}
        for arm in ARMS:
            e = SE.additive(dirs[arm], Kj, clean, arm)
            lg = SE.Logged(e)
            h = prefill(ids, lg)
            per = {}
            for L in BAND:
                dh = h[L] - hc[L]                                                           # [P, D]
                pu = dh @ u[L]
                r = {"cos_u": float(dirs[arm][L] @ u[L]), "proj_u_read": float(pu[pos]), "proj_u_median": float(pu.median()),
                     "k_read": float(Kj[L][pos]), "k_median": float(Kj[L].median()),
                     "rel_dh_read": lg.log[L]["rel_dh"][pos], "rel_dh_median": float(np.median(lg.log[L]["rel_dh"])),
                     "rel_dh_all": lg.log[L]["rel_dh"]}
                if arm == "AR_Jrem@J":
                    pj = dh @ uJ[L]
                    r.update(proj_uJ_read=float(pj[pos]), proj_uJ_median=float(pj.median()))
                per[str(L)] = r
            rec["arms"][arm] = per
        rows.append(rec)
        print(f"{nm} done ({time.time() - t0:.0f}s)", flush=True)
    with open(OUTD / "clampcheck.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("saved", len(rows), flush=True)


def analyze():
    rows = [json.loads(l) for l in open(OUTD / "clampcheck.jsonl")]
    E_ = ROOT / "outputs/012_reader_concordance/stageE/edits"
    rec_full, rec_en = {}, {}
    for f in sorted(E_.glob("full_shard*.jsonl")):
        for r in map(json.loads, open(f)):
            rec_full[r["name"]] = r
    for f in sorted(E_.glob("energy_shard*.jsonl")):
        for r in map(json.loads, open(f)):
            rec_en[r["name"]] = r
    # reproduction gate: realised rel_dh equals the recorded runs; emrem cos equals energy_diag
    gate = {"rel_dh_max_abs_diff": {}, "emrem_cos_max_abs_diff": 0.0}
    for r in rows:
        for arm in ARMS:
            src = rec_en if arm == "AR_emrem@J" else rec_full
            rl = src[r["name"]]["arms"][arm]["edit_log"]
            dmax = max(max(abs(a - b) for a, b in zip(r["arms"][arm][L]["rel_dh_all"], rl[L]["rel_dh"])) for L in rl)
            gate["rel_dh_max_abs_diff"][arm] = max(gate["rel_dh_max_abs_diff"].get(arm, 0.0), dmax)
        for L, c in r["emrem_cos_rebuilt"].items():
            gate["emrem_cos_max_abs_diff"] = max(gate["emrem_cos_max_abs_diff"], abs(c - rec_en[r["name"]]["energy_diag"][L]["emrem_cos_to_u"]))
    gate["pass"] = all(v == 0.0 for v in gate["rel_dh_max_abs_diff"].values()) and gate["emrem_cos_max_abs_diff"] < 1e-6
    L47 = "47"
    table, summ = [], {}
    for arm in ARMS[1:]:
        q_read, q_med, path = [], [], {}
        for r in rows:
            a, f = r["arms"][arm][L47], r["arms"]["AR_full@J"][L47]
            ratio_read = a["proj_u_read"] / f["proj_u_read"]
            ratio_med = a["proj_u_median"] / f["proj_u_median"]
            inv = 1 / a["cos_u"]
            table.append({"item": r["name"], "arm": arm, "cos_u_L47": a["cos_u"], "inv_cos": inv, "ratio_read": ratio_read,
                          "ratio_median_pos": ratio_med, "ratio_over_invcos_read": ratio_read / inv,
                          "full_proj_over_k_read": f["proj_u_read"] / f["k_read"]})
            q_read.append(ratio_read / inv)
            q_med.append(ratio_med / inv)
            for L in r["arms"][arm]:
                path.setdefault(L, []).append(r["arms"][arm][L]["proj_u_read"] / r["arms"]["AR_full@J"][L]["proj_u_read"])
        summ[arm] = {"median_ratio_over_invcos_read": float(np.median(q_read)), "median_ratio_over_invcos_median_pos": float(np.median(q_med)),
                     "median_ratio_read": float(np.median([t["ratio_read"] for t in table if t["arm"] == arm])),
                     "median_inv_cos": float(np.median([t["inv_cos"] for t in table if t["arm"] == arm])),
                     "path_median_ratio_read_by_layer": {L: float(np.median(v)) for L, v in sorted(path.items(), key=lambda x: int(x[0]))},
                     "within_15pct": bool(0.85 <= float(np.median(q_read)) <= 1.15)}
    jr = {L: {"proj_u_read_med": float(np.median([r["arms"]["AR_Jrem@J"][L]["proj_u_read"] for r in rows])),
              "proj_uJ_read_med": float(np.median([r["arms"]["AR_Jrem@J"][L]["proj_uJ_read"] for r in rows])),
              "full_proj_u_read_med": float(np.median([r["arms"]["AR_full@J"][L]["proj_u_read"] for r in rows]))} for L in rows[0]["arms"]["AR_Jrem@J"]}
    reapp = {arm: {L: {"rel_dh_read_med": float(np.median([r["arms"][arm][L]["rel_dh_read"] for r in rows])),
                       "rel_dh_median_med": float(np.median([r["arms"][arm][L]["rel_dh_median"] for r in rows]))}
                   for L in rows[0]["arms"][arm]} for arm in ARMS}
    holds = summ["AR_rrem@J"]["within_15pct"] and summ["AR_emrem@J"]["within_15pct"]
    out = {"label": "POST HOC, descriptive (012E §18 clamp mechanism check)", "items": [r["name"] for r in rows], "gate": gate,
           "summary": summ, "mechanism_holds": holds,
           "reading": ("mechanism holds: removal arms accumulate the persistent part to about k / cos(u', u)" if holds else
                       "mechanism does not track 1/cos within 15%: 'random removal makes the edit stronger' stays unexplained"),
           "per_item_L47": table, "AR_Jrem_paths": jr, "re_application_by_layer": reapp}
    RESF.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "mechanism_holds", "reading")}, indent=1))
    for arm, s in summ.items():
        print(arm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "path_median_ratio_read_by_layer"})
        print("   path", {L: round(v, 3) for L, v in s["path_median_ratio_read_by_layer"].items()})


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
