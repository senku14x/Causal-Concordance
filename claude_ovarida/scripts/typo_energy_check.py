"""Energy check for 014's J contrast (EXPLORATORY, CPU only, no model). Written before running.

Question: does 007's dense J subspace (J_raw_L{L}, first 1,024 cols) hold more of the typo activations' energy than a
random 1,024-dim subspace? If so, 014's "removing J costs the Oracle more than the NLA" may reflect energy removed,
not J, and an energy-matched control (rest-of-random / matched-PCA) is needed (report 36, question 1).

Data: outputs/014_typo/captures.pt (the exact activations 014 read; 100 items, L40/42/44, stock bf16 captures).
Measures per layer, on the uncentred activations the readers read (primary) and on centred ones (secondary):
  share_J = mean over items of ||P_J h||^2 / ||h||^2; same for 20 random 1,024-dim bases (007's r1024 seeds 0..19).
  Also: the PCA-k (on the 100 typo activations) whose share matches J's, as a size reference.
Decision rule (fixed now): step 2 (GPU control reads) is warranted if, at L40 or L44, share_J exceeds the maximum
random share by a factor >= 1.5. Otherwise 014's contrast already removes roughly random-sized energy, and the
energy explanation is weak.
"""
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/007_jsubspace"), str(ROOT / "scripts/common")]
import p007 as C
from pieces import rand_basis_f32

cap = torch.load(ROOT / "outputs/014_typo/captures.pt", weights_only=False)
out = {"label": "EXPLORATORY, CPU (claude_ovarida): energy share of 007's J subspace in 014 typo activations",
       "rule": "step 2 warranted if share_J >= 1.5 x max random share at L40 or L44 (uncentred)", "layers": {}}
R = [rand_basis_f32(C.rand_seed("r1024", r), C.K)[0].double() for r in range(20)]


def share(H, Q):
    return float((((H @ Q) ** 2).sum(1) / (H ** 2).sum(1)).mean())


for L in cap["layers"]:
    H = cap["H"][L].double()
    QJ = torch.linalg.qr(torch.load(ROOT / f"outputs/007_jsubspace/bases/J_raw_L{L}.pt", weights_only=False)["Q"][:, :C.K].double())[0]
    rec = {}
    for name, X in (("uncentred", H), ("centred", H - H.mean(0, keepdim=True))):
        sj = share(X, QJ)
        sr = [share(X, Q) for Q in R]
        # PCA on the 100 activations (rank <= 99): smallest k whose share >= share_J
        U, S, Vt = torch.linalg.svd(X - (0 if name == "centred" else 0), full_matrices=False)
        cum = [share(X, Vt[:k].T) for k in range(1, Vt.shape[0] + 1)]
        k_match = next((k for k, s in enumerate(cum, 1) if s >= sj), None)
        rec[name] = {"share_J": sj, "random_min": min(sr), "random_median": sorted(sr)[10], "random_max": max(sr),
                     "ratio_J_to_random_max": sj / max(sr), "pca_k_matching_J": k_match}
    out["layers"][str(L)] = rec

r = [out["layers"][str(L)]["uncentred"]["ratio_J_to_random_max"] for L in (40, 44)]
out["step2_warranted"] = bool(max(r) >= 1.5)
p = ROOT / "claude_ovarida/results/typo_energy_check.json"
p.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
