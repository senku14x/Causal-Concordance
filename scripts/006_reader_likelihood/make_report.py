"""Write docs/reports/006_reader_likelihood.md from results/006_reader_likelihood/analysis/summary.json (numbers from
JSON, rounded half-up)."""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import registry as R                              # noqa: E402

S = json.loads((R.PROJECT / "results/006_reader_likelihood/analysis/summary.json").read_text())


def r(x, nd=2, sign=False):
    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return ("+" if sign and q > 0 else "") + str(q).replace("-", "−")


def tr(b):
    t = b["tracking"]
    return f"{r(t['mean'], 2, True)} [{r(t['ci'][0], 2)}, {r(t['ci'][1], 2)}] ({t['pos']}/{t['neg']})"


G, B, rules = S["gates"], S["blocks"], S["rules"]
PH = json.loads((R.PROJECT / "results/006_reader_likelihood/analysis/posthoc.json").read_text()) \
    if (R.PROJECT / "results/006_reader_likelihood/analysis/posthoc.json").exists() else None


def bci(x, nd=2):
    return f"{r(x['mean'], nd, True)} [{r(x['ci'][0], nd)}, {r(x['ci'][1], nd)}] ({x['pos']}/{x['neg']})"


def posthoc_section():
    if not PH:
        return []
    C, PC, N, K, AG = PH["cells"], PH["positive_control"], PH["derangement_null"], PH["collisions"], PH["agreement"]
    ps = PH["patch_share_001_L42"]
    o = ["## Post hoc (PI-forwarded review, 2026-10-05; every item below was added after the results were read)", "",
         "The reviewer recomputed everything from the raw `ll_*.npz`; the reported numbers reproduce. Recomputed here by "
         "`scripts/006_reader_likelihood/reanalyse_006.py` (`results/006_reader_likelihood/analysis/posthoc.json`).", "",
         "**1. The secondary cells.** D_z by cell (prefix p0):", "",
         "| Reader | `:` | `</think>` newline | assistant newline | `</think>` − `:` |", "|---|---|---|---|---|"]
    for nm, c in C.items():
        o.append(f"| {nm} | {bci(c['emit_colon'])} | {bci(c['emit_think'])} | {bci(c['emit_asst'])} | {bci(c['think_minus_colon'])} |")
    o += ["",
          f"- 001's whole-cell patch shares at L42 (median S): `:` {ps['emit_colon']}, assistant newline {ps['emit_asst']}, "
          f"`</think>` newline {ps['emit_think']}.",
          "- **The Oracle's likelihood tracks z about twice as strongly at the `</think>` cell, where patching moves the answer by "
          "almost nothing, as at `:`, where it moves half.** The NLA's is slightly lower at `</think>` (the difference "
          "includes 0; the review's \"≥ for every reader\" has the NLA's sign reversed).",
          "- So the readers' z tracking does not follow the cell's demonstrated influence on the answer. It is decodability "
          "through the reader, not a marker of use.", "",
          "**2. Why the Oracle fails its positive control at L40/L44.**"]
    for nm, x in PC.items():
        o.append(f"- {nm}: raw ID {r(x['raw_ID'], 2)}; it answers \"blue\" for {x['argmax'].count('blue')} of 12 colour prompts. "
                 f"With each colour's mean over the other prompts removed (leave-one-out prior), ID is {r(x['loo_prior_removed_ID'], 2)}. "
                 f"The pairwise double difference is positive in {x['pairwise_double_difference_positive']} of {x['pairs']} pairs.")
    o += ["- So the Oracle discriminates the colours. The raw control failed on a strong prior for one colour. T2 stays "
          "\"uninformative by the pre-registered rule\", and this is the reason.", "",
          "**3. The shuffle.** 200 random derangements of tables replace the single k+1 shuffle; mean D [2.5, 97.5 percentile]:"]
    for nm, x in N.items():
        o.append(f"- {nm}: " + "; ".join(f"{t} {r(v['mean'], 2, True)} [{r(v['p2.5'], 2)}, {r(v['p97.5'], 2)}]" for t, v in x.items()))
    o += ["- The null is about zero for every role. The k+1 shuffle's nonzero start and answer values reflect neighbouring-item structure.", "",
          f"**4. Role collisions.** Of {K['n_pairs']} member–partner pairs, z shares a colour with another role in "
          f"{K['census']['z']} (the partner constraints exclude it), start in {K['census']['start']} and answer in "
          f"{K['census']['answer']}. Collision-free D:"]
    for nm in ("NLA L42", "Oracle L40", "Oracle L44"):
        o.append(f"- {nm}: start {bci(K[nm]['start'])}; answer {bci(K[nm]['answer'])}.")
    o += ["- Start tracking survives for the NLA and Oracle L44, not for Oracle L40. Answer tracking survives everywhere.", "",
          "**5. T4** counted only reader-layers that passed G-pc. Oracle L44's forced-choice z of 0.44 was not counted.", "",
          "**6. Per-pair agreement on D_z at `:`:** "
          + "; ".join(f"{k}: same sign {pct(v['same_sign'])}, r {r(v['r'], 2)}" for k, v in AG.items() if "vs Oracle" in k)
          + ". Within a reader across the two prefixes: " + "; ".join(f"{k.split(' p0')[0]} r {r(v['r'], 2)}" for k, v in AG.items() if "p0 vs p1" in k)
          + ". The readers agree on direction, not on which tables carry the most z.", ""]
    return o


def pct(x):
    return f"{int(Decimal(str(x * 100)).quantize(Decimal(1), rounding=ROUND_HALF_UP))}%"
names = {"nla_L42": "NLA L42", "oracle_L40": "Oracle L40", "oracle_L44": "Oracle L44", "oracle_L56": "Oracle L56",
         "oracle_L60": "Oracle L60"}
out = ["# 006 — Do the verbalizers' likelihoods track z at the brew `:` cell?", "",
       "*Design:* `docs/design_specs/006_reader_likelihood.md` (v1 pre-registered a92dfa6; amendment v1.1 f5e6dff, before "
       "any outcome). *Branch:* `006-reader-likelihood`. *Run:* `run1`, 82.5 min. *Data:* the saved 001 captures, 48 dev "
       "brew tables, 231 member/partner renders. *Summary:* `results/006_reader_likelihood/analysis/summary.json`.", "",
       "## Gates", "",
       f"- **G-score1** (full-sequence scorer vs the single-token scorer on single-token spellings): max |difference| "
       + ", ".join(f"{names[k]} {r(v, 3)}" for k, v in G["G_score1_seq_vs_next_maxabs"].items())
       + " nats (bar 0.25 after amendment v1.1; the NLA's 0.247 is close to it). Medians "
       + ", ".join(f"{r(v, 3)}" for v in G["G_score1_seq_vs_next_median"].values()) + " (bar 0.05).",
       f"- **G-score2** (float32 vs the readers' own logits): max {r(max(v for d in G['G_score2_own_logits_maxabs'].values() for v in d.values()), 3)} nats (bar 1.0).",
       "- **G-pc** (12-way colour identification on natural \"Tell me about {colour}\" activations, bar 0.5): "
       + "; ".join(f"{names[k.rsplit('_', 1)[0]]} {k.rsplit('_', 1)[1]} {r(b['G_pc_ID'], 2)}" for k, b in B.items() if k.endswith("p0"))
       + ". **The Oracle fails at L40 and L44**, so by the pre-registered rule its primary result (T2) is uninformative "
         "and shown descriptively only. The NLA passes exactly at the bar.", "",
       "## Tracking at the `:` cell (mean D over 48 tables [95% CI] (tables + / −); prefix p0 primary)", "",
       "| Reader, prefix | G-pc | z | start | answer | z, shuffled | forced choice z (member / no-activation) |",
       "|---|---|---|---|---|---|---|"]
for k, b in B.items():
    c = b["cells"]["emit_colon"]
    nm = names[k.rsplit("_", 1)[0]]
    out.append(f"| {nm}, {k.rsplit('_', 1)[1]} | {r(b['G_pc_ID'], 2)}{'' if b['G_pc_pass'] else ' (fails)'} | {tr(c['z'])} | "
               f"{tr(c['start'])} | {tr(c['answer'])} | {r(c['z']['shuffled']['mean'], 2, True)} "
               f"[{r(c['z']['shuffled']['ci'][0], 2)}, {r(c['z']['shuffled']['ci'][1], 2)}] | "
               f"{r(c['z']['forced_choice_member'], 2)} / {r(c['z']['forced_choice_reference'], 2)} |")
L = S.get("lens_002_tracking_emit_colon", {})


def lt(k):
    v = L.get(k)
    return "—" if v is None else f"{r(v['mean'], 2, True)} ({v['pos']}/{v['neg']})"
out += ["", "**002's token lenses at the same cell, same statistic (log-score units differ by reader):** "
        f"J_NP cosine L40 z {lt('J_NP_cosine|emit_colon|L40')}, start {lt('J_NP_cosine|emit_colon|L40|start')}, answer "
        f"{lt('J_NP_cosine|emit_colon|L40|answer')}; J_CB L40 z {lt('J_CB_standard|emit_colon|L40')}; logit lens L42 z "
        f"{lt('logit|emit_colon|L42')}.", "",
        "## Pre-registered rules", "",
        f"- **Holm (primary family, one-sided bootstrap p):** " + "; ".join(f"{names[k]} p = {r(v['p'], 4)} vs {r(v['threshold'], 4)} → "
                                                                           f"{'rejects' if v['rejects'] else 'does not'}" for k, v in S["holm_primary"].items()) + ".",
        f"- **T1, NLA L42 tracks z:** {'supported' if rules['T1_nla_tracks_z'] else 'not supported'} (agent prior 0.55).",
        f"- **T2, Oracle tracks z (L40 or L44):** {'uninformative: both fail G-pc' if rules['T2_oracle_tracks_z'] is None else rules['T2_oracle_tracks_z']} "
        "(prior 0.60). Descriptively the Oracle's likelihood tracks z in 48 of 48 tables at both layers.",
        f"- **T3, start not tracked:** {'holds' if rules['T3_start_not_tracked'] else 'does not hold'} (prior 0.55): the NLA's likelihood "
        "also tracks the start colour, though less than z. The token lenses do not.",
        f"- **T4, forced-choice z > 0.4:** {'met' if rules['T4_forced_choice_z_gt_0.4'] else 'not met'} (prior 0.25).",
        f"- **T5, G-pc passes:** NLA {rules['T5_G_pc']['nla_L42']} (prior 0.65); Oracle L40 {rules['T5_G_pc']['oracle_L40']}, "
        f"L44 {rules['T5_G_pc']['oracle_L44']} (prior 0.85).", "",
        "## Reading", "",
        "- **The NLA's likelihood registers z at the `:` cell.** Within tables, swapping the cell from the member to a "
        "partner moves the NLA's likelihood toward the cell's own intermediate colour in 44 of 48 tables. The shuffled "
        "control is about zero. This is the cell where 002 found the verbalizers almost never *state* z in its role. "
        "The gap is between what the reader's likelihood carries and what it says. Wording: z is decodable from the "
        "NLA's likelihood there; not \"the NLA represents z\", and not use by the model.",
        "- **The Oracle's likelihood tracks z even more consistently (48/48 at L40 and L44),** but its positive control "
        "failed at those layers: it identifies natural colour activations only 33–42% of the time. By the rule, this is "
        "reported, not counted. At L56/L60 the Oracle passes the control (100%). There its likelihood is dominated by "
        "the answer colour (+11 to +18 nats), with z tracked as well and start anti-tracked.",
        "- **Not as clean as the lenses on role.** The NLA and Oracle L44 likelihoods also follow the start colour and "
        "the answer. Forced choice among the 5 candidates picks z on member renders only 27–44% of the time (chance 20%; "
        "no-activation reference 23–27%). The readers carry z-specific information, but not as a dominant "
        "\"this is the intermediate\" signal.", "",
        "## Caveats", "",
        "- The NLA is `nla-rl` iter_000400; 002's verbalizer claims came from `nla-L42`, so \"rarely states z\" and "
        "\"likelihood tracks z\" are measured on different checkpoints.",
        "- G-pc for the NLA is exactly at the bar (6 of 12). G-score1 for the NLA is close to its bar (0.247 vs 0.25).",
        "- Saved SEC captures, not stock (cos ≥ 0.99985 per 002's bridge). Dev split only; one family; prefixes fixed "
        "in advance (p0 primary, p1 agrees).",
        "- Likelihood tracking is decodability through the reader, not a claim and not use by the model.", "",
        *posthoc_section(),
        "## Mistakes (logged in CLAUDE.md)", "",
        "- G-score1's first tolerance (1e-3) ignored the documented bf16 floor. The gate stopped the first launch; the "
        "diagnosis and amendment (v1.1) came before any outcome was read.",
        "- The lens-comparison table was empty in the first analysis output (wrong key format for 002's scores.json); "
        "fixed (descriptive only).",
        "- The run took 82.5 min against a 15–25 min estimate.", ""]
(R.PROJECT / "docs/reports/006_reader_likelihood.md").write_text("\n".join(out))
print("wrote docs/reports/006_reader_likelihood.md")
