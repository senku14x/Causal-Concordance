"""CPU-only geometry check of saved 003 coefficients; no model forwards.

Run: OPENBLAS_NUM_THREADS=1 /venv/main/bin/python /workspace/codex_reports/check_004_span_geometry.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/Causal-Concordance')
OUT = Path('/workspace/codex_reports/2026-10-03_004_span_geometry.json')
SOURCES = [ROOT / 'outputs/003_labels/directions.npz',
           ROOT / 'outputs/003_labels/directions_info.json',
           ROOT / 'docs/design_specs/004_span_assay.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = {str(p): sha(p) for p in SOURCES}
    arrays = np.load(SOURCES[0])
    info = json.loads(SOURCES[1].read_text())['info']
    rows = []

    def basis(fold, fit, cell, layer, role):
        key = f'f{fold}|{fit}|{cell}|L{layer}'
        colours = sorted(c for c, n in info[key]['counts'][role].items() if n >= 2)
        coefs = np.stack([arrays[f'{key}|{role}|{c}'].astype(np.float64) for c in colours])
        coefs -= coefs.mean(0)
        _, singular, vt = np.linalg.svd(coefs, full_matrices=False)
        rank = int(np.sum(singular > singular[0] * 1e-6))
        q = vt[:rank].T
        assert np.allclose(q.T @ q, np.eye(rank), atol=1e-10)
        rows.append({'cell': cell, 'layer': layer, 'fold': fold, 'fit': fit,
                     'role': role, 'eligible_colours': colours, 'rank': rank,
                     'singular_values': singular.tolist(), 'kind': 'basis'})
        return q

    for cell, layer in [('emit_colon', 42), ('start_tok', 40)]:
        for fold in range(3):
            bases = {fit: {role: basis(fold, fit, cell, layer, role)
                           for role in ['start', 'z', 'answer']}
                     for fit in ['both', 'r1', 'r2']}
            for role in ['start', 'z', 'answer']:
                cos = np.linalg.svd(bases['r1'][role].T @ bases['r2'][role], compute_uv=False)
                rows.append({'cell': cell, 'layer': layer, 'fold': fold, 'role': role,
                             'kind': 'cross_fit', 'principal_cosines': cos.tolist(),
                             'mean_squared_cosine': float(np.mean(cos ** 2)),
                             'ranks': [bases[f][role].shape[1] for f in ['r1', 'r2']]})
            for role1, role2 in [('start', 'z'), ('z', 'answer'), ('start', 'answer')]:
                cos = np.linalg.svd(bases['both'][role1].T @ bases['both'][role2], compute_uv=False)
                rows.append({'cell': cell, 'layer': layer, 'fold': fold,
                             'kind': 'cross_role', 'roles': [role1, role2],
                             'principal_cosines': cos.tolist(),
                             'minimum_angle_degrees': float(np.degrees(np.arccos(np.clip(cos[0], -1, 1))))})
    after = {str(p): sha(p) for p in SOURCES}
    assert before == after, 'Source changed during audit; rerun for a consistent snapshot.'
    result = {'method': 'Float64 SVD of centered eligible coefficient vectors; rank cutoff s_max * 1e-6.',
              'scope': 'Saved 003 coefficients only; geometry is not behavioral reliability or causal specificity.',
              'source_sha256': before, 'rows': rows}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {OUT}; {len(rows)} records; source hashes unchanged.')
    for cell, role in [('emit_colon', 'z'), ('start_tok', 'start')]:
        vals = [r['mean_squared_cosine'] for r in rows
                if r['kind'] == 'cross_fit' and r['cell'] == cell and r['role'] == role]
        print(cell, role, 'cross-fit mean squared principal cosine', vals)
    for roles in [['start', 'z'], ['z', 'answer']]:
        vals = [r['minimum_angle_degrees'] for r in rows
                if r['kind'] == 'cross_role' and r['cell'] == 'emit_colon' and r['roles'] == roles]
        print('emit_colon', roles, 'smallest principal angles in degrees', vals)


if __name__ == '__main__':
    main()
