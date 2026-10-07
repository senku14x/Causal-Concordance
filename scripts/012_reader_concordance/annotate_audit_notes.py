"""012: add audit notes to analysis JSON written by GPU scripts (CPU only; idempotent; 2026-10-07, after report 34).

stageH.py and stageE0_ar_profile.py are GPU scripts, so their JSON is annotated here instead of rerunning them.
Existing keys and values are never changed: the rule's label and the original key names stay as recorded.
"""
import json
from pathlib import Path

A = Path(__file__).resolve().parents[2] / "results/012_reader_concordance/analysis"

NOTES = {
    "stageH_summary.json": (
        "Codex audit (2026-10-07) and report 34: 'family explains it' is the literal label of the §15 rule "
        "(answer-atom control Aa_L40-48 below 0.15). It is not a supported explanation: the late-band control fails the "
        "power bar, and Stage H also changed the numerical regime and the atom convention relative to 008. Which "
        "difference explains 008's brew null is not established."),
    "stageE0_ar_profile.json": (
        "Codex audit (2026-10-07) and report 34: keys named 'global_scale' / 'validated_up_to_global_scale' hold one "
        "least-squares scale fitted per layer on the same samples it scores (scaled FVE = squared cosine), not one "
        "global scale. The 1,024 spans are reused calibration contexts (crops of 011's 12 responses, 792 distinct "
        "targets), not held-out validation. Key names kept for continuity."),
}


def main():
    for f, note in NOTES.items():
        p = A / f
        d = json.loads(p.read_text())
        if d.get("audit_note") == note:
            print(f, "already annotated")
            continue
        d["audit_note"] = note
        p.write_text(json.dumps(d, indent=1))
        print(f, "annotated")


if __name__ == "__main__":
    main()
