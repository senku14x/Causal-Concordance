"""Read-only checks for the 2026-10-02 design-002 review. No model/API calls."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/Causal-Concordance')
OUT = Path('/workspace/codex_reports/2026-10-02_design_002_checks.json')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    snapshot = ROOT / 'results/001_leverage_profile/analysis/snapshot.json'
    tables = json.loads(snapshot.read_text())['tables']
    by_id = {t['item']: t for t in tables}
    checks = []
    for table in tables:
        checks.append({
            'item': table['item'],
            'original_candidates': table['candidates'],
            'member': {k: table[k] for k in ('start', 'z', 'answer')},
            'partner': {k: table['partner'][k] for k in ('start', 'z', 'answer')},
            'missing_partner_roles': [k for k in ('start', 'z', 'answer')
                                      if table['partner'][k] not in table['candidates']],
        })
    folder = ROOT / 'results/002_claim_scoring/validation'
    key = read_jsonl(folder / 'key.jsonl')
    sets = {name: read_jsonl(folder / f'{name}.jsonl') for name in ('pilot', 'validation')}
    set_of = {r['id']: name for name, rows in sets.items() for r in rows}
    bad_key_rows = []
    for row in key:
        item, variant, *_ = row['id'].split('|')
        if variant == 'partner':
            truth = by_id[item]['partner']
            wrong = [k for k in ('start', 'z', 'answer') if row[k] != truth[k]]
            if wrong:
                bad_key_rows.append({'id': row['id'], 'set': set_of.get(row['id']),
                                     'wrong_roles': wrong})
    paths = [snapshot, ROOT / 'docs/design_specs/002_claim_scoring.md',
             ROOT / 'docs/design_specs/001_leverage_profile.md',
             ROOT / 'scripts/002_claim_scoring/make_sets.py',
             ROOT / 'scripts/002_claim_scoring/claim_scoring.py',
             ROOT / 'scripts/001_leverage_profile/leverage_profile.py',
             folder / 'pilot.jsonl', folder / 'validation.jsonl', folder / 'key.jsonl',
             folder / 'author_codes.jsonl']
    out = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'n_tables': len(tables),
        'missing_partner_role_counts': dict(Counter(k for r in checks for k in r['missing_partner_roles'])),
        'candidate_checks': checks,
        'wrong_partner_key_rows': bad_key_rows,
        'wrong_partner_key_counts_by_set': dict(Counter(r['set'] for r in bad_key_rows)),
        'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in paths if p.exists()],
    }
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('checked_at_utc', 'n_tables',
          'missing_partner_role_counts', 'wrong_partner_key_counts_by_set')}, indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
