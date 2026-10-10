"""015: markdown tables for the smoke / full report, from `results/015_oracle_hallucination/analysis/oh_{which}.json`
only, rounded half-up (Decimal). EXPLORATORY.

  python smoke_tables.py [smoke|all]
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WHICH = sys.argv[1] if len(sys.argv) > 1 else "smoke"
R = json.loads((ROOT / f"results/015_oracle_hallucination/analysis/oh_{WHICH}.json").read_text())


def h(x, nd=1, sign=True):
    if x is None:
        return "—"
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    q = abs(q) if q == 0 else q
    s = f"{q:.{nd}f}"
    return ("+" + s if sign and q > 0 else s).replace("-", "−")


def d(x, nd=1):
    if not x:
        return "—"
    return f"{h(x['mean'], nd)} [{h(x['ci95'][0], nd)}, {h(x['ci95'][1], nd)}] (n {x['n']})"


def ps(x):
    if not x:
        return "—"
    return f"{h(x['points_per_unit_share'])} [{h(x['ci95'][0])}, {h(x['ci95'][1])}]"


CELLS = [("oracle_texts|oracle_ar", "Oracle", "Oracle AR"), ("oracle_texts|nla_ar", "Oracle", "NLA AR"),
         ("nla_texts|nla_ar", "NLA", "NLA AR"), ("nla_texts|oracle_ar", "NLA", "Oracle AR")]

print("### Pre-named contrasts, raw (points, scorer's own; Δ = FVE(control) − FVE(cleaned))\n")
print("| Texts | Scorer | Primary Δ = R1 − C1 | Δ_A = RA − CA | Δ_AP = PA − CA | mean cos O |")
print("|---|---|---|---|---|---|")
for k, t, s in CELLS:
    c = R["cells"][k]
    print(f"| {t} | {s} | {d(c['primary_R1_minus_C1'])} | {d(c['dA_RA_minus_CA'])} | {d(c['dAP_PA_minus_CA'])} | {h(c['cos_mean']['O'], 3, False)} |")

print("\n### Per-share (Σ Δ / Σ deleted token share) and English-only\n")
print("| Texts | Scorer | Primary per share | Δ_A per share | Δ_AP per share | Primary, English-only | Δ_A, English-only |")
print("|---|---|---|---|---|---|---|")
for k, t, s in CELLS:
    c = R["cells"][k]
    print(f"| {t} | {s} | {ps(c['primary_per_share'])} | {ps(c['dA_per_share'])} | {ps(c['dAP_per_share'])} | "
          f"{d(c['primary_english_only'])} | {d(c['dA_english_only'])} |")

print("\n### Rewrites and other contrasts\n")
print("| Texts | Scorer | F − P | F − P (P within 20%) | F − O | P − O | O − C1 | O − CA | K1 − C1 | RS1 − C1 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, t, s in CELLS:
    c = R["cells"][k]
    w = c.get("F_minus_P_within20pct") or c.get("F_minus_Pstar_within20pct")
    print(f"| {t} | {s} | {d(c['F_minus_P'])} | {d(w)} | {d(c['F_minus_O'])} | {d(c['P_minus_O'])} | {d(c['O_minus_C1'])} | "
          f"{d(c['O_minus_CA'])} | {d(c['K1_minus_C1'])} | {d(c['RS1_minus_C1'])} |")
for k, t, s in CELLS[2:]:
    c = R["cells"][k]
    print(f"\nNLA P* ({s}): F − P* {d(c['F_minus_Pstar'])}; P* − O {d(c['Pstar_minus_O'])}")

print("\n### Per-kind, clause control (Rk − C1k) and sentence control (RSk − C1k)\n")
print("| Texts | Scorer | Kind | n | Clause | Sentence | Deleted token share |")
print("|---|---|---|---|---|---|---|")
for k, t, s in CELLS:
    c = R["cells"][k]
    for kk in ("misquote", "invented_context", "wrong_entity", "false_fact"):
        x = c[f"kind_{kk}"]
        if x["n"]:
            print(f"| {t} | {s} | {kk} | {x['n']} | {d(x['delta_clause'])} | {d(x['delta_sentence'])} | {h(x['token_share_deleted_mean'], 2, False)} |")

print("\n### Levels (mean FVE, mean cos) and J share\n")
print("| Texts | Scorer | O | C1 | R1 | CA | RA | PA | F | P | J share gold / O / F / P |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k, t, s in CELLS:
    c = R["cells"][k]
    cell = lambda n: f"{h(c['fve_mean'].get(n))} ({h(c['cos_mean'].get(n), 3, False)})"   # noqa: E731
    j = c["share_J_mean"]
    print(f"| {t} | {s} | " + " | ".join(cell(n) for n in ("O", "C1", "R1", "CA", "RA", "PA", "F", "P"))
          + f" | {h(j['gold'], 3, False)} / {h(j['O'], 3, False)} / {h(j.get('F'), 3, False)} / {h(j.get('P'), 3, False)} |")

print("\n### Reader contrasts, NLA − Oracle (paired on (item, state), averaged over samples)\n")
print("| Contrast | Scorer | Raw [95% CI] | Per share [95% CI] | English-only raw | (item, state) n | Claim rule met |")
print("|---|---|---|---|---|---|---|")
for cn, lab in (("primary", "Primary Δ"), ("dA", "Δ_A"), ("dAP", "Δ_AP")):
    for s, sl in (("oracle_ar", "Oracle AR"), ("nla_ar", "NLA AR")):
        x = R["reader_contrasts_nla_minus_oracle"][cn][s]
        a, e = x["all"], x["english_only"]
        print(f"| {lab} | {sl} | {h(a['raw']['mean'])} [{h(a['raw']['ci95'][0])}, {h(a['raw']['ci95'][1])}] | "
              f"{h(a['per_share']['value'])} [{h(a['per_share']['ci95'][0])}, {h(a['per_share']['ci95'][1])}] | "
              f"{h(e['raw']['mean']) + ' (n ' + str(e['n_item_states']) + ')' if e else '—'} | {a['n_item_states']} | {x['claim_holds']} |")

g, rw, cov = R["gates"], R["rewrites"], R["answer_arm_coverage"]
print("\n### Gates and checks\n")
print(f"- G1 (A parse, Oracle): {h(g['G1_A_ok_rate_oracle'], 2, False)}; K parse {h(g['K_ok_rate_oracle'], 2, False)}; "
      f"answer-span parse Oracle {h(g['ANS_ok_rate']['oracle'], 2, False)}, NLA {h(g['ANS_ok_rate']['nla'], 2, False)}")
print(f"- G3 κ (Oracle, A vs B per sentence): {h(g['G3_kappa_oracle'], 2, False)} over {g['G3_n_sentences_oracle']} sentences")
for rd, key in (("oracle", "P"), ("nla", "Pstar")):
    r_ = g[f"B_contradicted_sentence_rate_{rd}"]
    print(f"- B contradicted-sentence rate, {rd}: " + ", ".join(f"{k} {h(v, 3, False)}" for k, v in r_.items())
          + f"; F ≤ 0.10: {g[f'rewrite_gate_F_le_0.10_{rd}']}; {key} ≥ 0.7 × O: {g[f'rewrite_gate_{key}_ge_0.7_O_{rd}']}")
print(f"- G-repro (NLA AR on NLA texts vs 013): versions identical {R['G_repro_nla']['versions_same_as_013']}/{R['G_repro_nla']['n']}; "
      f"max |ΔMSE| {R['G_repro_nla']['max_abs_mse_013_versions']} over {R['G_repro_nla']['n_values']} values; rewrites "
      f"{R['G_repro_nla']['max_abs_mse_013_rewrites']}; pass {R['G_repro_nla']['pass_1e-6']}")
print(f"- Oracle O vs step 2 (batch layout): max {h(R['oracle_O_vs_r1_layout_points']['max'], 2, False)} points (n {R['oracle_O_vs_r1_layout_points']['n']})")
print(f"- Answer arm coverage: Oracle {cov['oracle']['ans_eligible']}/{R['n_eids']} eligible, PA {cov['oracle']['PA_computable']}; "
      f"NLA {cov['nla']['ans_eligible']}, PA {cov['nla']['PA_computable']}; answer token share Oracle "
      f"{h(cov['oracle']['answer_token_share_mean'], 3, False)}, NLA {h(cov['nla']['answer_token_share_mean'], 3, False)}")
print(f"- Flagged (contradicted) token share: Oracle {h(R['flagged_token_share_C1']['oracle'], 3, False)}, NLA {h(R['flagged_token_share_C1']['nla'], 3, False)}")
print(f"- English-only eids: {R['n_english_only_eids']}/{R['n_eids']}")
fo, fn = rw["oracle"]["F_composition_sum"], rw["nla"]["F_composition_sum"]
print(f"- F composition (words), Oracle: deleted {fo['deleted']}, replaced {fo['replaced_O']} → {fo['replaced_F']}, inserted {fo['inserted']}, "
      f"of {fo['words_O']} words in O; NLA: deleted {fn['deleted']}, replaced {fn['replaced_O']} → {fn['replaced_F']}, inserted {fn['inserted']}, of {fn['words_O']}")
print(f"- P match: Oracle P had delimiters {rw['oracle']['P_had_delimiters']}/20; words changed F {h(rw['oracle']['words_changed_F_mean'], 1, False)}, "
      f"P {h(rw['oracle']['words_changed_P_mean'], 1, False)}; median ratio {h(rw['oracle']['P_match_ratio_median'], 2, False)}; within 20% "
      f"{rw['oracle']['P_within_20pct']}/20. NLA: F {h(rw['nla']['words_changed_F_mean'], 1, False)}, P(013) {h(rw['nla']['words_changed_P013_mean'], 1, False)}, "
      f"P* {h(rw['nla']['words_changed_Pstar_mean'], 1, False)}; P* median ratio {h(rw['nla']['Pstar_match_ratio_median'], 2, False)}; within 20% "
      f"{rw['nla']['Pstar_within_20pct']}/20")
cw = R["cw20_metric_oracle_texts"]
print(f"- CW-20 metric (Oracle texts × Oracle AR): global scale {h(cw['global_scale'], 3, False)}; scaled FVE "
      + ", ".join(f"{k} {h(v, 3, False)}" for k, v in cw["scaled_fve_mean"].items())
      + f"; RA − CA {h(cw['scaled_fve_mean']['RA'] - cw['scaled_fve_mean']['CA'], 3)} (CW-20: removing the answer −7.7 points, i.e. +0.077 in this sign)")
print(f"- Degenerate entries: " + ", ".join(f"{k} {c['n_degenerate_entries']}" for k, c in R["cells"].items()))
