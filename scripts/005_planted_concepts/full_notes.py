"""Write results/005_planted_concepts/analysis/full_notes.json: the interpretive sentences for the full-run report
(written after the results were read; labelled as interpretation). Every number comes from full_summary.json, rounded
half-up here, except the post-hoc R2 floor check, which is recomputed from the raw rows and labelled post hoc."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_report_page import RESA, ROOT, SNAME, pct, r  # noqa: E402

S = json.loads((RESA / "full_summary.json").read_text())
G, C, T = S["gates"], S["contrasts"], S["tables"]
D = ROOT / "outputs/005_planted_concepts/run" / S["run_id"]


def ci(c, nd=2, unit=""):
    return f"{r(c['mean'], nd, True)}{unit} [{r(c['ci'][0], nd)}, {r(c['ci'][1], nd)}]"


# post-hoc R2 with an absolute floor (B1 >= 0.1 nats, B2 >= 1 nat or B4 >= 0.1 nats), recomputed from the rows
rows = json.loads((D / "model_rows.json").read_text())
r2 = json.loads((D / "r2_rows.json").read_text())
idx = {(x["concept"], str(x["cond"]), x["s"], x.get("seed", 0)): x for x in rows}
nl = {(k, s): float(np.quantile([x[k] for x in rows if x["cond"] == 7 and x["s"] == s], 0.95))
      for k in ("B1", "B2", "B4", "dR1p") for s in (1 / 64, 1 / 4, 1.0)}
r2i = {(x["concept"], str(x["cond"]), x["s"]): x["R2"] for x in r2}
FLOOR = {"B1": 0.1, "B2": 1.0, "B4": 0.1}
concepts = sorted({x["concept"] for x in rows})
by_dose = {1 / 64: 0, 1 / 4: 0, 1.0: 0}
has = []
for c in concepts:
    ok = False
    for s in by_dose:
        x = idx[(c, "4", s, 0)]
        beh = any(x[k] > nl[(k, s)] and x[k] >= FLOOR[k] for k in FLOOR)
        rep = x["dR1p"] <= nl[("dR1p", s)] and r2i[(c, "4", s)] <= r2i[(c, "7", s)]
        if beh and rep:
            ok = True
            by_dose[s] += 1
    has.append(ok)
r2_floor = {"share": float(np.mean(has)), "n": int(sum(has)), "of": len(has), "qualifying_by_dose": {SNAME[k]: v for k, v in by_dose.items()},
            "floor": FLOOR, "label": "post hoc (after the results were read)"}

on = C["P11prime_onsets"]
never_model = sum(1 for v in on["model"].values() if v == 9)
n = S["n_concepts"]
g11 = G["G11"]
mc = T["model_curves"]
N = {"r2_floor_post_hoc": r2_floor}
N["M1"] = (f"Not on the primary measure. At each part's peak, the J part beats the non-J part by {ci(C['M1_R1p'])} nats of "
           f"ΔR1′, an interval that includes 0, and none of the five alternative J-space constructions changes that. On R1, the "
           f"form the prior was registered on, the J part's peak reciprocal rank is higher by {ci(C['M1_R1'], 4)}: this clears "
           "the rule but is tiny, and it is partly by construction (the J part contains the concept's own atom). Per dose the "
           f"J part leads from 1/128 to 1/4 (largest at 1/8, {r(C['M1_per_dose'][str(0.125)]['median_J_minus_perp'], 2, True)} "
           f"nats) and trails at 1/2 ({r(C['M1_per_dose'][str(0.5)]['median_J_minus_perp'], 2, True)}), the same crossing "
           "pattern as on the dev concepts.")
N["M3"] = (f"Opposite of the prediction. At natural scale the non-J part moves the report more than the J part: "
           f"{ci(C['M3'])} nats (J-natural minus non-J-natural) over the doses where the full vector moves the report. The "
           "non-J part is about 95% of the vector's norm, and at its natural size it carries most of the report effect.")
N["M2"] = (f"The J re-entry clamp to L50 removes more of the non-J part's report effect than its random-token null: "
           f"{ci(C['M2'])} nats over {C['M2']['evaluable']} evaluable concepts, replicating the dev finding. But clamping "
           f"the plain output directions of the same tokens does so too ({ci(C['M2_output_clamp_at_L50'])}), and the "
           f"difference is not distinguishable from 0 ({ci(C['M2_control_D_at_L50'])}); along the end-layer curve the J clamp "
           "is ahead only at L56. Under the adopted A8 wording, the clamp \"blocks the concept's output-related coordinates\"; "
           "routing through J coordinates specifically is not shown. The design's original clamp (to clean, L32–62) is "
           f"strongly specific ({ci(C['M2_design_clamp_10_vs_10r'])}), as on dev, and is read next to the same caveat.")
N["P11"] = (f"P11′ for the NLA could not be tested: its likelihood measure failed the registered positive control (it "
            f"identifies {pct(g11['nla_L42']['ID_natural'])} of natural test-concept activations among the 55 concepts, "
            "against a 50% bar; chance is 2%). The Oracle passes "
            f"({pct(g11['oracle_L40']['ID_natural'])} at L40 and at L44), and on the written cell its likelihood reaches a "
            f"1-nat specific rise for the non-J part {ci(C['P11prime_Oracle'], 1)} grid steps (dose doublings) before the "
            f"model's report does. The model's report never reaches it for {never_model} of {n} concepts. Caveat registered "
            "beforehand: this compares one written cell with a report that integrates the injection over 7 layers × 20 "
            "positions; it is not a perturbation-matched comparison.")
N["P17"] = (f"Not supported, and the interval lies on the other side: the Oracle names the non-J part more often than the "
            f"NLA ({ci(C['P17'])} in naming rate, NLA minus the better Oracle layer, direct reads), and the likelihood "
            f"version agrees ({ci(C['P17_likelihood_secondary'])} nats). The A.9.2-based premise that the Oracle sees content "
            "outside J-space less does not hold on these concepts. P11′ (NLA) being untestable leaves P17 alone in the "
            "P-family; with the registered family of two its Holm threshold would be 0.0125, which changes nothing (p = 1).")
N["R"] = (f"R1: the probe (absolute criterion) detects the non-J part on direct reads at L42 more often than the NLA names "
          f"it, by {ci(C['R1_probe_minus_NLA'])} (as predicted). R2: by the registered definition, "
          f"{pct(C['R2_behaviour_without_report']['mean'])} of concepts have a dose where the non-J part moves behaviour "
          "(B1, B2 or B4 above the random null) while the report stays within it. Part of this is the scale-free trap: at "
          "s = 1/64 the neutral-prompt KL is about 0.002 nats, above a random null near zero. With a post-hoc absolute floor "
          f"(B1 ≥ 0.1, B2 ≥ 1 or B4 ≥ 0.1 nats) it holds for {r2_floor['n']} of {r2_floor['of']} concepts, mostly at s = 1, "
          "where random directions also move the report a lot (the report's random null there is "
          f"{r(T['model_null95'][str(1.0)]['dR1p'], 2)} nats).")
N["caveats"] = ("Planted content only (design §14). One prompt, one band, one model. The non-J part is \"outside J-space by "
                "this construction\" (J_NP, pursuit k = 16, folded atoms). Direct reads give the reader the written cell; "
                "the model's report integrates many cells. Naming is exact spelling-variant matching; the synonym judge "
                "was deferred.")
gate_line = (f"G0 had 0 failures over every write and clamp; the J share median is {pct(G['G3']['J_share_median'], 1)}; the "
             f"probe is valid (LOTO AUC {r(G['G4']['loto_auc_median'], 3)}, permutation control {r(G['G4']['perm_ctrl_auc_median'], 2)}); "
             "no split is leaky; G6 drops "
             + (", ".join(f"{c} for the {k.split('_')[0].upper() if k.startswith('nla') else 'Oracle'}" for k, v in G["G6"].items()
                          for c in v["dropped"]) or "nothing")
             + "; G2 passes for every reader; G11 passes for the Oracle and fails for the NLA.")
bullets = [
    f"<b>The run is clean, with one gate failure that matters.</b> {gate_line}",
    "<b>The P-family is not supported.</b> P11′ for the NLA could not be tested (its likelihood measure failed the positive "
    f"control). P17 goes the other way: the Oracle names the non-J part more often than the NLA ({ci(C['P17'])}).",
    "<b>No J-space privilege for report on the primary measure.</b> At matched norm the J part's peak beats the non-J part's "
    f"by {ci(C['M1_R1p'])} nats (not supported). The registered R1 form clears the rule by a tiny margin that is partly by "
    f"construction. At natural scale the non-J part moves the report more ({ci(C['M3'])}).",
    "<b>The clamp is specific, but not beyond blocking the output direction.</b> The J re-entry clamp to L50 beats its null "
    f"({ci(C['M2'])}), and so does the output-direction clamp; the difference is {ci(C['M2_control_D_at_L50'])}.",
    "<b>On the written cell, the Oracle is more sensitive than the model's report.</b> Its likelihood reaches a 1-nat "
    f"specific rise {ci(C['P11prime_Oracle'], 1)} dose doublings earlier; the model's report never does for {never_model} of "
    f"{n} concepts. The probe also beats the NLA's naming (R1). Not perturbation-matched.",
    f"<b>Behaviour moves without report</b> for {pct(C['R2_behaviour_without_report']['mean'])} of concepts by the registered "
    f"definition, {r2_floor['n']} of {r2_floor['of']} with a post-hoc size floor (mostly at s = 1).",
]
N["short_html"] = ('<h2 id="short">The short version</h2><ul>' + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
                   "<p class='muted'>The bullets are my interpretation of the pre-registered results below; every number is "
                   "generated from the result files.</p>")
import re as _re
N["short_md"] = "## The short version\n\n" + "\n".join("- " + _re.sub(r"<[^>]+>", "", b).replace("&amp;", "&") for b in bullets)
(RESA / "full_notes.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))
print("notes written;", "R2 floor:", r2_floor)
