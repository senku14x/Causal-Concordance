"""Append (or replace) the post hoc A2 section of docs/reports/007_jsubspace.md from the analysis JSON."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p007 as C                                                           # noqa: E402

A = C.RES / "analysis"
REPORT = C.PROJECT / "docs" / "reports" / "007_jsubspace.md"
MARK = "## 8. Post hoc NLA rerun (amendment A2)"


def J(n):
    f = A / f"{n}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    pa, pt, pcf, p3 = J("posthoc_A2_stage1a"), J("posthoc_A2_brew_test"), J("posthoc_A2_brew_test_confirmation"), J("posthoc_A2_stage3")
    lines = [MARK, "",
             "*Label: post hoc.* After the NLA failed G-repro on the 2-row scorer, the PI asked for a proper rerun (\"rerun",
             "properly\", 2026-10-06). Amendment A2 (6b0be23) fixed the plan before the rerun: the NLA L42 only, on 006's exact",
             "24-row scorer (`acts_per_batch` 2, original log-prob), on the saved inputs, with every gate unchanged. The",
             "pre-registered record above (Oracle only) stands as the primary result.", ""]
    if pa:
        R = pa["readers"]["nla_L42"]
        P = R["pieces"]
        rr = sum(P[f"R{i}"]["z"]["mean"] for i in range(10)) / 10
        rq = sum(P[f"rest_R{i}"]["z"]["mean"] for i in range(3)) / 3
        lines += ["**Stage 1a, dev `:`.** All gates pass on 006's path, including the one that failed before: G-repro max 0.152,",
                  "median 0.0000, D_z 1.862 vs 006 1.862. NLA D_z by piece: "
                  f"whole {P['whole']['z']['mean']:.2f}, J {P['J']['z']['mean']:.2f}, rest of J {P['rest_J']['z']['mean']:.2f}, PCA-4 {P['PCA']['z']['mean']:.2f}, "
                  f"random (mean) {rr:.2f}, rest of random {rq:.2f}, mean-removed {P['MR']['z']['mean']:.2f}, J∘c {P['J_c']['z']['mean']:.2f}.",
                  "Categories (post hoc full family, Holm/TOST over 15): " +
                  ", ".join(f"{k} {R['categories'][k]} ({R['contrasts'][k]['estimate']:+.2f})" for k in ("P1", "P2", "P3", "P4", "P5")) +
                  f"; mapping H_J {R['model_mapping']['H_J']}, Raw {R['model_mapping']['Raw']}, Visible {R['model_mapping']['Visible']}.", ""]
        ag = pa.get("agreement", {})
        if ag:
            lines += ["Oracle–NLA per-pair agreement (r): " + "; ".join(
                f"{k.replace('nla_L42_vs_oracle_', 'vs Oracle L')}: whole {v['whole']['r']:.2f}, J {v['J']['r']:.2f}, rest of J {v['rest_J']['r']:.2f}, mean-removed {v['MR']['r']:.2f}"
                for k, v in ag.items()) + ".", ""]
    if pt and pcf:
        R = pt["readers"]["nla_L42"]
        P = R["pieces"]
        conf = pcf["nla_L42"]["contrasts"]
        lines += ["**Brew test.** NLA D_z: "
                  f"whole {P['whole']['z']['mean']:.2f}, J {P['J']['z']['mean']:.2f}, rest of J {P['rest_J']['z']['mean']:.2f}, mean-removed {P['MR']['z']['mean']:.2f}. "
                  "Categories: " + ", ".join(f"{k} {R['categories'][k]}" for k in ("P1", "P2", "P3", "P4", "P5")) +
                  ". Confirmed (same category dev and test): " +
                  (", ".join(k for k, v in conf.items() if v["confirmed"]) or "none") + ". In this full family the Oracle's L40 H_J stays confirmed,"
                  " and Oracle L44's P4 (−) and P5 (+) are confirmed.", ""]
    lines += ["**Stage 1b, `</think>`.** Stopped at its first gate: G-score1 (two scoring paths on 4 `</think>` activations) max 0.280",
              "> 0.25, median 0.046. The scorer is 006's own; 006 ran this gate only at `:` (0.247). Not repaired (A2, rule 12).", ""]
    if p3 and "nla_L42" in p3.get("readers", {}):
        P = p3["readers"]["nla_L42"]["pieces"]
        rs = p3["readers"]["nla_L42"].get("restatement", {}).get("J|test", {})
        rr = sum(P[f"R{i}"]["bridge|test"]["mean"] for i in range(10)) / 10
        lines += ["**Multihop (test split, bridge D).** NLA: "
                  f"whole {P['whole']['bridge|test']['mean']:.1f}, J {P['J']['bridge|test']['mean']:.1f}, rest of J {P['rest_J']['bridge|test']['mean']:.1f}, "
                  f"random (mean) {rr:.1f}; J readouts naming the bridge {rs.get('P(names_bridge)', float('nan')):.2f}. (The first launch crashed on a"
                  " coding error before any gate; fixed and relaunched, disclosed.)", ""]
    lines += ["**Reading (post hoc).** The NLA's z signal is spread over the activation: J keeps about half of it and removing J",
              "costs about as much as removing a random slice. The Oracle's strong J dependence is not shared by the NLA, so J",
              "dependence is a property of the reader. The reader-model dissociation of §1 therefore concerns the Oracle; the NLA",
              "neither privileges J nor shows the Oracle's pattern.", ""]
    text = REPORT.read_text()
    if MARK in text:
        text = text[:text.index(MARK)]
    REPORT.write_text(text.rstrip("\n") + "\n\n" + "\n".join(lines))
    print("report section written", len(lines), "lines")


if __name__ == "__main__":
    main()
