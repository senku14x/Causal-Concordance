"""012 Stage E §17: cross-fitted PCA bases of clean multihop activations, for the Δ-matched PCA-k removal arm. CPU.
EXPLORATORY, POST HOC (design 012E §17).

007's construction adapted to multihop: rows are clean block-L outputs at every prompt position of the cohort's items,
centred within item, each row weighted 1/sqrt(n_item); 4 folds of relation clusters (sorted, round-robin); the basis
for fold f is fit on the items of the other folds, so an item's basis never uses its own cluster.

  python stageE_pca.py   (reads outputs/.../stageE/pca/clean_acts_shard*.pt)
      -> outputs/.../stageE/pca/pca_bases.pt   {bases: {(fold, L): [D, r] float16}, fold_of_cluster, ...}
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stageE_common as E                                                   # noqa: E402

BAND = list(range(36, 48))
NF = 4


def main():
    acts = {}
    for f in sorted((E.OUT / "pca").glob("clean_acts_shard*.pt")):
        acts.update(torch.load(f, weights_only=False)["caps"])
    cohort = {r["name"]: r for r in E.cohort()}
    assert set(acts) == set(cohort), (len(acts), len(cohort))
    clusters = sorted({E.rel(n) for n in cohort})
    fold_of = {c: i % NF for i, c in enumerate(clusters)}
    bases, meta = {}, {}
    torch.set_num_threads(64)
    for f in range(NF):
        train = [n for n in cohort if fold_of[E.rel(n)] != f]
        for li, L in enumerate(BAND):
            rows = []
            for n in train:
                X = acts[n][li].float()                                   # [P, D]
                rows.append((X - X.mean(0)) / X.shape[0] ** 0.5)
            M = torch.cat(rows)
            _, S, Vh = torch.linalg.svd(M, full_matrices=False)
            r = int((S > S[0] * 1e-6).sum())
            bases[(f, L)] = Vh[:r].T.contiguous().half()                  # [D, r], columns sorted by variance
            meta[f"{f}|{L}"] = {"n_items": len(train), "n_rows": int(M.shape[0]), "rank": r}
        print(f"fold {f}: {meta[f'{f}|{BAND[0]}']}", flush=True)
    torch.save({"bases": bases, "fold_of_cluster": fold_of, "meta": meta, "n_folds": NF,
                "construction": "007-style within-item-centred PCA, rows / sqrt(n_item), 4 relation-cluster folds"},
               E.OUT / "pca" / "pca_bases.pt")
    print("saved", len(bases), "bases", flush=True)


if __name__ == "__main__":
    main()
