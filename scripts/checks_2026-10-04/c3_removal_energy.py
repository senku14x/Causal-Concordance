"""C3 (pre-registered in docs/design_specs/checks_2026-10-04_no_gpu.md): removal energy of the CW pilot's
J-neighbourhood removal ('strong') vs its random-vocabulary control ('rand'). Read-only on the CW repo; CPU only.
Usage: python c3_removal_energy.py /workspace/nla-vs-lenses"""
import json, sys
from pathlib import Path
import numpy as np
import torch

CW = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[2] / "results/checks_2026-10-04/c3_removal_energy.json"
SETS = {"multihop_CW4": ("runs/abl50/resid_L42.pt", "runs/abl3/resid_abl2.pt"),
        "brew_CW9": ("runs/brew50/resid_L42.pt", "runs/brew50_abl/resid_abl2.pt")}
rng = np.random.default_rng(20261004)


def boot_median(x, n=10000):
    x = np.asarray(x)
    b = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


res = {"design": "docs/design_specs/checks_2026-10-04_no_gpu.md#c3", "cw_commit": "58d34a8", "sets": {}}
for name, (orig_p, edit_p) in SETS.items():
    orig = {(c["id"], c["layer"]): c for c in torch.load(CW / orig_p, weights_only=False)}
    edits = torch.load(CW / edit_p, weights_only=False)
    per, gate_fail = {}, []
    for e in edits:
        if e["cond"] not in ("strong", "rand"):
            continue
        o = orig.get((e["id"], e["layer"]))
        if o is None:
            gate_fail.append((e["id"], e["cond"], "no original")); continue
        h, he = o["h"].double(), e["h"].double()
        d = h - he
        cos = float(d @ he / (d.norm() * he.norm() + 1e-30))
        pyth = float(abs(h.norm() ** 2 - he.norm() ** 2 - d.norm() ** 2) / h.norm() ** 2)
        if abs(cos) >= 1e-3 or pyth >= 1e-4:
            gate_fail.append((e["id"], e["cond"], f"cos={cos:.2e} pyth={pyth:.2e}")); continue
        per.setdefault(e["id"], {})[e["cond"]] = {"r": float(d.norm() ** 2 / h.norm() ** 2),
                                                 "rel_norm": float(d.norm() / h.norm()), "n_dirs": int(e["n_dirs"]),
                                                 "cos_gate": cos, "pyth_gate": pyth}
    both = {k: v for k, v in per.items() if "strong" in v and "rand" in v}
    rs = np.array([v["strong"]["r"] for v in both.values()]); rr = np.array([v["rand"]["r"] for v in both.values()])
    R = rs / rr
    res["sets"][name] = {
        "n_items": len(both), "gate_failures": gate_fail,
        "r_strong_median": float(np.median(rs)), "r_strong_ci": boot_median(rs),
        "r_rand_median": float(np.median(rr)), "r_rand_ci": boot_median(rr),
        "R_median": float(np.median(R)), "R_ci": boot_median(R),
        "R_quartiles": [float(np.quantile(R, q)) for q in (0.25, 0.5, 0.75)],
        "strong_gt_rand": int((rs > rr).sum()),
        "n_dirs_quartiles": [float(np.quantile([v["strong"]["n_dirs"] for v in both.values()], q)) for q in (0, 0.25, 0.5, 0.75, 1)],
        "per_item": both}
    s = res["sets"][name]
    print(f"{name}: n={s['n_items']} gate_fail={len(gate_fail)} | removed energy share: strong median {s['r_strong_median']:.4f} "
          f"{[round(x,4) for x in s['r_strong_ci']]}, rand median {s['r_rand_median']:.5f} {[round(x,5) for x in s['r_rand_ci']]} | "
          f"R median {s['R_median']:.1f} {[round(x,1) for x in s['R_ci']]} | strong>rand {s['strong_gt_rand']}/{s['n_items']} | n_dirs q {s['n_dirs_quartiles']}")
OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT)
