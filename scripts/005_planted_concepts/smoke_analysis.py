"""Summarise the 005 smoke (outputs/005_planted_concepts/smoke/<run_id>) into
results/005_planted_concepts/analysis/smoke_summary.json and print the tables used by the smoke note. CPU only."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p005 as P                       # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "smoke1"
D = P.ROOT / "outputs/005_planted_concepts/smoke" / run
OUT = P.ROOT / "results/005_planted_concepts/analysis/smoke_summary.json"
st = json.loads((D / "state.json").read_text())
S = {"run_id": run}

# gates
S["determinism"] = st["determinism"]
S["G1"] = st["G1"]
S["final_band"], S["final_grid"] = st["final_band"], st["final_grid"]
S["onset"] = st["onset"]
g3 = list(st["G3"].values())
S["G3_J_share"] = {"median": float(np.median(g3)), "min": float(min(g3)), "max": float(max(g3)), "n": len(g3)}
g4 = st["G4_probe_validity"]
la = [v["loto_auc"] for v in g4.values() if v["loto_auc"] is not None]
pc = [v["perm_ctrl_auc"] for v in g4.values() if v["perm_ctrl_auc"] is not None]
S["G4"] = {"loto_auc_median": float(np.median(la)), "loto_auc_min": float(min(la)),
           "perm_ctrl_median": float(np.median(pc)), "perm_ctrl_max": float(max(pc)), "n": len(la),
           "pass_rule_on_medians": bool(np.median(la) >= 0.9 and np.median(pc) <= 0.6)}
g5 = st["G5"]
S["G5"] = {"share_top10": float(np.mean([r <= 10 for r in g5.values()])), "n": len(g5),
           "leaky_pairs": {k: r for k, r in g5.items() if r <= 10}}
cs = list(st["cos_loo_vs_base"].values())
S["cos_loo_base"] = {"median": float(np.median(cs)), "min": float(min(cs))}
sens = st["sensitivity_J_share"]
S["sensitivity_J_share_median"] = {k: float(np.median([v[k] for v in sens.values()]))
                                   for k in ("k10", "k25", "ls", "base", "raw_atoms")}

# fidelity + clamp
fid = json.loads((D / "fidelity_e2e.json").read_text())
inj = [v for d in fid["inject"] for v in d.values()]
S["G0_inject"] = {"n_layer_batches": len(inj), "rho_min": min(x["rho_min"] for x in inj),
                  "rho_max": max(x["rho_max"] for x in inj), "kappa_min": min(x["kappa_min"] for x in inj)}
rows_e = json.loads((D / "rows_e2e.json").read_text())
band = list(range(st["final_band"][0], st["final_band"][1] + 1))
share = defaultdict(list)
resid = []
for chunk_start, d in ((int(k_), v) for blk in fid["clamp"] for k_, v in blk.items()):
    r = rows_e[chunk_start]
    for l, e in d.items():
        resid.append(e["residual"])
        where = "band" if int(l) in band else "above"
        share[(str(r["cond"]), where)].append(e["removed_share"])
S["G0_clamp"] = {"max_residual": max(resid), "n": len(resid),
                 "removed_share_median": {f"{c}|{w}": float(np.median(v)) for (c, w), v in share.items()}}

# model side (end to end, seed 0)
tab = {}
for r in rows_e:
    if r.get("seed", 0) != 0:
        continue
    tab[f"{r['concept']}|{r['cond']}|{r['s']}"] = {k: r.get(k) for k in ("R1", "R1p", "dR1p", "R2", "B1", "B2", "B3",
                                                                         "B4", "top1_at_quote")}
S["e2e"] = tab

# readers
ts = st["onset"]["text_strengths"]


def naming(reads):
    out = defaultdict(list)
    clean = {}
    for q in reads:
        if q["cond"] == "clean":
            clean[(q["layer"], q["kind"])] = {c: float(np.mean([P.names(t, c) for t in q["samples"]]))
                                              for c in P.CONCEPTS}
            continue
        out[(q["layer"], q["kind"], q["concept"], str(q["cond"]), q["s"])].append(
            float(np.mean([P.names(t, q["concept"]) for t in q["samples"]])))
    return out, clean


for name in ("oracle", "nla"):
    f = D / f"{name}_reads.json"
    if not f.exists():
        continue
    reads = json.loads(f.read_text())
    nm, clean = naming(reads)
    S[f"{name}_naming"] = {f"L{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}": v[0] for k, v in nm.items()}
    S[f"{name}_clean"] = {f"L{k[0]}|{k[1]}": {"over_5pct": sorted(c for c, x in d.items() if x > 0.05),
                                             "max": max(d.values()), "smoke_concepts": {c: d[c] for c in P.SMOKE}}
                          for k, d in clean.items()}
    S[f"{name}_examples"] = [{"layer": q["layer"], "kind": q["kind"], "concept": q["concept"], "cond": q["cond"],
                              "s": q["s"], "sample": q["samples"][0][:400]}
                             for q in reads if (q["cond"] in (2, 4, "clean") and q["s"] in (ts[1], 0.0))][:12]

tp = json.loads((D / "token_probe_reads.json").read_text())
tr = defaultdict(list)
for q in tp:
    if q.get("seed", 0) != 0:
        continue
    tr[(q["layer"], q["kind"], str(q["cond"]), q["s"])].append((q["token"]["J_NP"]["rank"], q["token"]["logit"]["rank"],
                                                                q["probe"] - q["probe_clean"]))
S["token_probe"] = {f"L{k[0]}|{k[1]}|{k[2]}|{k[3]}": {"J_NP_rank_median": float(np.median([x[0] for x in v])),
                                                      "logit_rank_median": float(np.median([x[1] for x in v])),
                                                      "probe_delta_median": float(np.median([x[2] for x in v]))}
                    for k, v in tr.items()}

# throughput + memory
S["timings"], S["memory"] = st["timings"], st.get("memory")
t = st["timings"]
rate = {}
for k, v in t.items():
    if v.get("n"):
        rate[k] = v["n"] / v["seconds"]
S["rates_per_s"] = rate
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(S, indent=1, default=str))
print(json.dumps({k: S[k] for k in ("determinism", "final_band", "G3_J_share", "G4", "cos_loo_base",
                                    "sensitivity_J_share_median", "G0_inject")}, indent=1))
print("G5", S["G5"]["share_top10"], S["G5"]["leaky_pairs"])
print("clamp", json.dumps(S["G0_clamp"], indent=1))
for name in ("oracle", "nla"):
    if f"{name}_clean" in S:
        print(name, "clean", json.dumps(S[f"{name}_clean"], indent=1))
        nm = S[f"{name}_naming"]
        for c in P.SMOKE:
            for kind in ("direct_cq", "carried_co"):
                for L in ((40, 44) if name == "oracle" else (42,)):
                    line = []
                    for cond in ("2", "3", "4", "7"):
                        vals = [nm.get(f"L{L}|{kind}|{c}|{cond}|{s}") for s in ts]
                        line.append(f"c{cond}:" + "/".join("-" if v is None else f"{v:.2f}" for v in vals))
                    print(f"  {name} L{L} {kind:10s} {c:6s}", "  ".join(line))
print("token/probe (median over gel, earth) at text strengths:")
for kind in ("direct_cq", "carried_co", "injected_cq"):
    for cond in ("1", "2", "3", "4", "7"):
        vals = [S["token_probe"].get(f"L42|{kind}|{cond}|{s}") for s in ts]
        print(f"  L42 {kind:11s} c{cond}", " | ".join("-" if v is None else
                                                      f"J{v['J_NP_rank_median']:.0f} lg{v['logit_rank_median']:.0f} "
                                                      f"pr{v['probe_delta_median']:+.1f}" for v in vals))
print("rates/s", json.dumps({k: round(v, 2) for k, v in rate.items()}))
print("memory", S["memory"])
