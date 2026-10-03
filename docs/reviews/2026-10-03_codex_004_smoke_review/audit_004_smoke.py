"""Independently audit saved 004 smoke arithmetic. No model imports or forwards."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/Causal-Concordance')
DEST = Path('/workspace/codex_reports/2026-10-03_004_smoke_audit.json')
COLOURS = ['black', 'blue', 'brown', 'gold', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'teal', 'white']
SOURCES = [ROOT / p for p in [
    'outputs/004_span_assay/smoke/rows.jsonl',
    'results/004_span_assay/smoke_checks.json',
    'results/004_span_assay/analysis_smoke/summary.json',
    'results/004_span_assay/arm_manifest.json',
    'results/004_span_assay/preflight.json',
    'scripts/004_span_assay/span_assay.py',
]]


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}


def main():
    before = hashes()
    rows = [json.loads(s) for s in SOURCES[0].read_text().splitlines()]
    checks = json.loads(SOURCES[1].read_text())
    summary = json.loads(SOURCES[2].read_text())
    manifest = json.loads(SOURCES[3].read_text())
    errors = Counter()
    maximum_error = Counter()

    def close(name, actual, expected):
        err = abs(actual - expected)
        maximum_error[name] = max(maximum_error[name], err)
        errors[name] += int(not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10))

    primary = [r for r in rows if 'check' not in r]
    records = []
    for pid in dict.fromkeys(r['pair_id'] for r in primary):
        rr = [r for r in primary if r['pair_id'] == pid]
        clean = next(r for r in rr if r['kind'] == 'clean')
        q0 = np.exp(clean['member']['logq'])
        G = clean['partner']['m'] - clean['member']['m']
        lookup = {(r['site'], r['fit'], r['arm']): r for r in rr if r['kind'] in ['swap', 'pca']}
        for r in rr:
            close('G', r['G'], G)
            scores = [r[k] for k in ['member', 'partner', 'edit', 'member_run', 'partner_run'] if k in r]
            for sc in scores:
                q = np.exp(sc['logq'])
                close('mass', sc['M'], float(q.sum()))
                close('outside', sc['outside'], 1 - float(q.sum()))
                close('margin', sc['m'], sc['lse_b'] - sc['lse_a'])
                close('answer_a', sc['lse_a'], sc['logq'][COLOURS.index(clean['a'])])
                close('answer_b', sc['lse_b'], sc['logq'][COLOURS.index(clean['a2'])])
            if 'S' in r:
                close('dm', r['dm'], r['edit']['m'] - clean['member']['m'])
                close('share', r['S'], (r['edit']['m'] - clean['member']['m']) / G)
            if 'N' in r:
                close('G_abl', r['G_abl'], r['partner_run']['m'] - r['member_run']['m'])
                close('necessity', r['N'], 1 - (r['partner_run']['m'] - r['member_run']['m']) / G)
        full = lookup['colon', None, 'full']['S']
        arm_data = {}
        for arm in ['z1d', 'Z', 'J', 'Zc', 'Jc', 'S', 'A', 'pca_kZ', 'pca_kZ_dose', 'pca_kJ', 'pca_kJ_dose']:
            r = lookup['colon', 'both', arm]
            q = np.exp(r['edit']['logq'])
            gain = q - q0
            tp = float(np.maximum(gain, 0).sum())
            ft = float(max(gain[COLOURS.index(clean['a2'])], 0) / tp) if tp >= .01 else None
            arm_data[arm] = {'S': r['S'], 'recovery': r['S'] / full,
                             'F_target': ft, 'q_before': dict(zip(COLOURS, q0.tolist())),
                             'q_after': dict(zip(COLOURS, q.tolist())),
                             'gain_by_colour': dict(zip(COLOURS, gain.tolist()))}
        nec = {}
        for arm in ['z1d', 'Z', 'J']:
            real = next(r for r in rr if r['kind'] == 'nec' and r['site'] == 'colon' and r['fit'] == 'both' and r['arm'] == arm)
            ctr = [r['N'] for r in rr if r['kind'] == 'nec_ctrl' and r['site'] == 'colon' and r['arm'] == 'rot:' + arm]
            nec[arm] = {'N': real['N'], 'rotation_median': float(np.median(ctr)),
                        'rotation_q95': float(np.quantile(ctr, .95)), 'draws': len(ctr)}
        records.append({'pair_id': pid, 'table': clean['item'], 'z': clean['z'], 'z2': clean['z2'],
                        'a': clean['a'], 'a2': clean['a2'], 'G': G, 'S_full': full,
                        'arms': arm_data, 'necessity': nec,
                        'interaction_Z': arm_data['Z']['S'] + arm_data['Zc']['S'] - full,
                        'interaction_J': arm_data['J']['S'] + arm_data['Jc']['S'] - full,
                        'stress_Z': {fit: lookup['colon', fit, 'Z']['S'] for fit in ['r1', 'r2']}})
    negligible = []
    for r in primary:
        for label, value in [('single', r)] + [(k, r[k]) for k in ['member_run', 'partner_run'] if k in r]:
            if value.get('fid_negligible', 0):
                negligible.append({'pair_id': r['pair_id'], 'kind': r['kind'], 'arm': r['arm'],
                                   'draw': r.get('draw'), 'run': label, 'count': value['fid_negligible'],
                                   'write_norm_first_continuation': value['write_norm'],
                                   'h_norm_first_continuation': value['h_norm'],
                                   'fid_rho': value['fid_rho'], 'fid_kappa_min': value['fid_kappa_min']})
    absdiff = float(np.median([abs(r['stress_Z']['r1'] - r['stress_Z']['r2']) for r in records]))
    medrec = {fit: float(np.median([r['stress_Z'][fit] / r['S_full'] for r in records])) for fit in ['r1', 'r2']}
    corrected_s5 = all(x >= .5 for x in medrec.values()) and absdiff <= .05
    assert not any(errors.values()), dict(errors)
    assert before == hashes(), 'Sources changed during audit.'
    result = {'scope': 'Two smoke pairs only; no fresh subject-model forward.',
              'source_sha256': before, 'rows_including_checks': len(rows),
              'row_kinds': dict(Counter(r['kind'] for r in primary)),
              'arithmetic_failure_counts': dict(errors), 'max_arithmetic_errors': dict(maximum_error),
              'records': records, 'negligible_write_records': negligible,
              'fidelity_false_rows': [r for r in primary if r.get('fid_ok') is False],
              'S5': {'saved_definition': summary['predictions']['S5_Z_reproducible_under_stress'],
                     'median_abs_share_diff': absdiff, 'median_recovery': medrec,
                     'agreed_definition_holds': corrected_s5},
              'gate_results': checks['gates'], 'projected_run_minutes': checks['projected_run_minutes'],
              'arm_totals': manifest['totals']}
    DEST.write_text(json.dumps(result, indent=2) + '\n')
    print('Saved', DEST)
    print('Arithmetic checks:', dict(errors), 'max error:', max(maximum_error.values()))
    for r in records:
        print(r['pair_id'], 'S_full', round(r['S_full'], 6),
              'effects', {a: round(v['S'], 6) for a, v in r['arms'].items()},
              'I_Z/I_J', round(r['interaction_Z'], 6), round(r['interaction_J'], 6))
    print('Exempted write count', sum(r['count'] for r in negligible))
    print('No source hashes changed; no model forwards run.')


if __name__ == '__main__':
    main()
