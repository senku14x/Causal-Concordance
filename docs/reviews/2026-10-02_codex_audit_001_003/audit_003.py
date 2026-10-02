"""Independent audit of saved 001–003 results; no model forwards or API calls."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/Causal-Concordance')
DEST = Path('/workspace/codex_reports/2026-10-02_003_audit_results.json')


def read(path):
    return json.loads((ROOT / path).read_text())


def jl(path):
    return [json.loads(s) for s in (ROOT / path).read_text().splitlines() if s.strip()]


def dist(x):
    x = np.asarray(x, float)
    return {'n': len(x), 'median': float(np.median(x)),
            'q10': float(np.quantile(x, .1)), 'q90': float(np.quantile(x, .9)),
            'min': float(x.min()), 'max': float(x.max())}


def main():
    rows = jl('outputs/003_labels/run/edits.jsonl')
    labels = read('results/003_labels/analysis/labels.json')
    summary = read('results/003_labels/analysis/summary.json')
    meta = read('outputs/001_leverage_profile/run/captures_brew_meta.json')
    info = read('outputs/003_labels/directions_info.json')
    data = np.load(ROOT / 'outputs/001_leverage_profile/run/captures_brew.npz')['X']
    with np.load(ROOT / 'outputs/003_labels/directions.npz') as z:
        arrays = {k: z[k] for k in z.files}
    natural = [(i, r) for i, r in enumerate(meta['renders']) if r['role'] in ('member', 'partner')]
    member = {r['item']: i for i, r in natural if r['role'] == 'member'}
    partners = {(r['item'], r['start']): i for i, r in natural if r['role'] == 'partner'}
    pids = sorted({r['pair_id'] for r in rows})
    groups = defaultdict(list)
    for r in rows:
        groups[(r['kind'], r['cell'], r['layer'], r['content'])].append(r)

    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'n_pairs': len(pids),
              'n_tables': len({r['item'] for r in rows}), 'row_counts': dict(Counter(r['kind'] for r in rows)),
              'status_counts': dict(Counter(r.get('status', 'ok') for r in rows)),
              'score_evaluations': sum('S' in r for r in rows) + 2 * sum('N' in r for r in rows),
              'necessity': {}, 'effects': {}, 'reliability': {}, 'direction_geometry': {},
              'ablation_dose': {}, 'pooled_null_diagnostic': {}}

    for cell in ('emit_asst', 'emit_think', 'emit_colon'):
        for layer in (40, 42, 44):
            key = f'{cell}|L{layer}'
            sw = [r for r in groups['swap', cell, layer, 'z'] if 'S' in r]
            nn = groups['necessity_null', cell, layer, 'z']
            nec = [r for r in groups['necessity', cell, layer, 'z'] if 'N' in r]
            q95 = float(np.quantile([r['N'] for r in nn], .95))
            result['necessity'][key] = {'above_null': sum(r['N'] > q95 for r in nec), 'n': len(nec),
                'median_N': float(np.median([r['N'] for r in nec])), 'null_q95': q95,
                'reported_used_and_necessary': summary['summary'][key + '|z']['frac_used_necessary']}
            result['effects'][key] = {'S': dist([r['S'] for r in sw]),
                'positive_S': sum(r['S'] > 0 for r in sw),
                'donor_logprob_increases': sum(r['d_logP_b'] > 0 for r in sw),
                'recipient_logprob_decreases': sum(r['d_logP_a'] < 0 for r in sw),
                'table_means_positive': sum(np.mean([r['S'] for r in sw if r['item'] == t]) > 0
                                            for t in {r['item'] for r in sw})}
            null_by_pair = defaultdict(list)
            for r in groups['null', cell, layer, 'z']:
                null_by_pair[r['pair_id']].append(r['S'])
            pool_q95 = np.quantile([v for a in null_by_pair.values() for v in a], .95)
            result['pooled_null_diagnostic'][key] = {
                'per_pair_q95_distribution': dist([np.quantile(a, .95) for a in null_by_pair.values()]),
                'above_pooled_q95': sum(r['S'] > pool_q95 for r in sw),
                'above_own_eight_q95': sum(r['S'] > np.quantile(null_by_pair[r['pair_id']], .95) for r in sw)}

    for layer in (40, 42, 44):
        a = {r['pair_id']: r for r in groups['swap_r1', 'emit_colon', layer, 'z'] if 'S' in r}
        b = {r['pair_id']: r for r in groups['swap_r2', 'emit_colon', layer, 'z'] if 'S' in r}
        ids = sorted(a.keys() & b.keys())
        s1 = np.array([a[k]['S'] for k in ids]); s2 = np.array([b[k]['S'] for k in ids])
        u1, u2 = s1 >= .1, s2 >= .1
        result['reliability'][f'colon_L{layer}_threshold_only'] = {
            'both_positive': int(np.sum(u1 & u2)), 'r1_only': int(np.sum(u1 & ~u2)),
            'r2_only': int(np.sum(~u1 & u2)), 'both_negative': int(np.sum(~u1 & ~u2)),
            'pearson_S': float(np.corrcoef(s1, s2)[0, 1]),
            'median_absolute_S_difference': float(np.median(abs(s1-s2)))}

    def direction(fold, fit, cell, layer, role, c0, c1):
        prefix = f'f{fold}|{fit}|{cell}|L{layer}|{role}|'
        if prefix+c0 not in arrays or prefix+c1 not in arrays:
            return None
        u = arrays[prefix+c1].astype(float) - arrays[prefix+c0].astype(float)
        n = np.linalg.norm(u)
        return (u/n).astype(np.float32) if n else None

    for layer in (40, 42, 44, 60):
        li = meta['layers'].index(layer); ci = meta['cells'].index('emit_colon')
        geometry = []
        for pid in pids:
            item, start_p = pid.split('__')
            m, p = member[item], partners[item, start_p]
            mr, pr = meta['renders'][m], meta['renders'][p]
            fold = info['folds'][item]
            u = direction(fold, 'both', 'emit_colon', layer, 'z', mr['z'], pr['z'])
            a = direction(fold, 'r1', 'emit_colon', layer, 'z', mr['z'], pr['z'])
            b = direction(fold, 'r2', 'emit_colon', layer, 'z', mr['z'], pr['z'])
            delta = data[p, ci, li].astype(float)-data[m, ci, li].astype(float)
            geometry.append({'pair_id': pid, 'cos_r1_r2': float(a@b) if a is not None and b is not None else None,
                'z_coordinate_delta': float(delta@u), 'full_delta_norm': float(np.linalg.norm(delta)),
                'z_edit_fraction_of_full_norm': float(abs(delta@u)/np.linalg.norm(delta))})
        result['direction_geometry'][f'colon_L{layer}'] = {
            'cos_r1_r2': dist([r['cos_r1_r2'] for r in geometry if r['cos_r1_r2'] is not None]),
            'z_edit_fraction_of_full_norm': dist([r['z_edit_fraction_of_full_norm'] for r in geometry]),
            'donor_coordinate_higher': sum(r['z_coordinate_delta'] > 0 for r in geometry), 'pairs': geometry}

    # Replay only NumPy random draws to reconstruct actual ablation edit norms from saved prompt activations.
    # No forward pass; this uses the prompt-only (empty-continuation) state, not appended-answer states.
    grid = [(c, L) for c in ('emit_asst', 'emit_think', 'emit_colon') for L in (40,42,44)]
    doses = defaultdict(list)
    mu_errors = []
    for pid in pids:
        item, start_p = pid.split('__'); m, p = member[item], partners[item,start_p]
        mr, pr = meta['renders'][m], meta['renders'][p]; fold = info['folds'][item]
        rng = np.random.default_rng(20261006 + sum(map(ord, pid)))
        for _ in range(11*3*8):
            rng.standard_normal(5120)  # every primary direction exists for these 82 pairs
        for cell, layer in grid:
            ci, li = meta['cells'].index(cell), meta['layers'].index(layer)
            fm = arrays[f'f{fold}|both|{cell}|L{layer}|fitmean']
            hm, hp = data[m,ci,li], data[p,ci,li]
            u = direction(fold,'both',cell,layer,'z',mr['z'],pr['z'])
            real_rms = np.sqrt(((float(fm@u)-float(hm@u))**2+(float(fm@u)-float(hp@u))**2)/2)
            random_rms = []
            logged_random = [r for r in groups['necessity_null',cell,layer,'z'] if r['pair_id']==pid]
            for draw in range(8):
                r = rng.standard_normal(5120).astype(np.float32); r /= np.linalg.norm(r)
                mu_errors.append(abs(float(fm@r)-logged_random[draw]['mu']))
                random_rms.append(np.sqrt(((float(fm@r)-float(hm@r))**2+(float(fm@r)-float(hp@r))**2)/2))
            doses[f'{cell}|L{layer}'].append({'real_rms':float(real_rms),
                'random_rms':float(np.sqrt(np.mean(np.square(random_rms)))),
                'ratio':float(real_rms/np.sqrt(np.mean(np.square(random_rms))))})
    for key, values in doses.items():
        result['ablation_dose'][key] = {name:dist([r[name] for r in values]) for name in ('real_rms','random_rms','ratio')}
    result['random_ablation_replay_check'] = {'n_mu_compared':len(mu_errors),
        'max_absolute_mu_difference':float(max(mu_errors))}

    # Check whether 002's positive difference-in-differences means an actual pairwise ranking reversal.
    ls=read('results/002_claim_scoring/lens/colour_scores.json'); cols=ls['colours']; render=ls['renders']
    by_item=defaultdict(list)
    for i,r in enumerate(render):
        if r['role'] in ('member','partner'): by_item[r['item']].append(i)
    tracking={}
    for rd in ('J_NP_cosine','J_CB_standard','logit'):
        for layer in (40,42,44):
            S=np.array(ls['scores'][f'{rd}|2|{ls["layers"].index(layer)}'])
            T=S if rd=='J_NP_cosine' else np.log(np.maximum(S,1e-30))
            table_rows=[]
            for item,idx in by_item.items():
                mi=[i for i in idx if render[i]['role']=='member'][0]
                z=cols.index(render[mi]['z']); pairs=[]
                for pi in idx:
                    if render[pi]['role']!='partner' or render[pi]['z']==render[mi]['z']: continue
                    zp=cols.index(render[pi]['z']); dm=float(T[mi,z]-T[mi,zp]); dp=float(T[pi,z]-T[pi,zp])
                    pairs.append({'d_member':dm,'d_partner':dp,'tracking':dm-dp,
                                  'correct_reversal':dm>0 and dp<0})
                if pairs: table_rows.append({'item':item,'positive_mean_tracking':np.mean([p['tracking'] for p in pairs])>0,
                    'n_pairs':len(pairs),'n_correct_reversals':sum(p['correct_reversal'] for p in pairs)})
            tracking[f'{rd}|colon_L{layer}']={'n_tables':len(table_rows),
                'tables_positive_mean_tracking':sum(r['positive_mean_tracking'] for r in table_rows),
                'tables_any_correct_reversal':sum(r['n_correct_reversals']>0 for r in table_rows),
                'tables_all_pairs_correct_reversals':sum(r['n_correct_reversals']==r['n_pairs'] for r in table_rows),
                'n_pairs':sum(r['n_pairs'] for r in table_rows),'correct_pairwise_reversals':sum(r['n_correct_reversals'] for r in table_rows)}
    result['002_tracking_vs_reversal']=tracking
    sources=['scripts/003_labels/labels.py','docs/design_specs/003_labels.md','docs/reports/003_labels.md',
        'outputs/003_labels/run/edits.jsonl','outputs/003_labels/directions.npz','outputs/003_labels/directions_info.json',
        'results/003_labels/analysis/summary.json','results/003_labels/analysis/labels.json',
        'outputs/001_leverage_profile/run/captures_brew.npz','outputs/001_leverage_profile/run/captures_brew_meta.json',
        'results/002_claim_scoring/lens/colour_scores.json','scripts/002_claim_scoring/claim_scoring.py']
    result['sources']=[{'path':str(ROOT/p),'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sources]
    DEST.write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    print('Saved',DEST)
    for k in ['necessity','reliability','ablation_dose','002_tracking_vs_reversal']:
        print(k,json.dumps(result[k],default=lambda x:x.item()))
    print('DIRECTION_COSINES', {k:v['cos_r1_r2'] for k,v in result['direction_geometry'].items()})


if __name__=='__main__': main()
