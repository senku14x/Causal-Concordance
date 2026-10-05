"""Write docs/reports/005_pilot_note.md from pilot_summary.json and report_notes.json (numbers from JSON only)."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_report_page import RESA, ROOT, SNAME, load, pct, r  # noqa: E402

S = load(RESA / "pilot_summary.json") or {}
N = load(RESA / "report_notes.json") or {}


def plain(h):
    h = re.sub(r"</ul>|</p>|</div>", "\n\n", h or "")
    h = re.sub(r"<li>", "\n- ", h)
    h = re.sub(r"<h[23][^>]*>(.*?)</h[23]>", r"\n**\1**\n", h)
    return re.sub(r"<[^>]+>", "", h).replace("&amp;", "&").strip()


out = ["# 005 dev-only pilot note (run `pilot1`: pilot 20:17–22:59 UTC, step 7b 22:59–23:03, step 8 23:04–23:39, "
       "step 9 23:44–23:47, step 10 23:58–00:04, step 11 00:49–00:55)", "",
       "*Design:* `docs/design_specs/005_planted_concepts.md` §16–§16.6 (questions, rules and agent priors committed",
       "before each run: 0b0ca0b, ba31daa, c7cde05, bf1da37, 94be4ec, 3bfd2f0, 6058e38). *Data:* `outputs/005_planted_concepts/pilot/pilot1/`;",
       "summary `results/005_planted_concepts/analysis/pilot_summary.json`. **Dev concepts only (10); descriptive; no",
       "test-concept outcome exists.** The full report with charts and raw readouts is the published report page.", ""]
out.append(plain(N.get("short_html", "")))
out.append("")
for q, title in (("Q1", "Report measure"), ("Q2", "Prefill"), ("Q3", "Band"), ("Q4", "Onset"), ("Q5", "Readers"),
                 ("Q6", "Clamp (design's)"), ("Q7", "Leaky splits"), ("Q8", "Concept-vector baseline"),
                 ("Q10", "Re-entry clamp"), ("Q11", "Layer sweep at the report position"),
                 ("Q12", "Arrival at the report position (step 8)"), ("Q13", "Oracle at late layers (step 8)"),
                 ("Q14", "Where the re-entry clamp starts to act (step 10)"),
                 ("Q15", "Is the report measure specific to the planted concept? (step 10)"),
                 ("Q16", "Does blocking the plain output direction do what the J clamp does? (step 11)"),
                 ("Q17", "Are the readers less sensitive than the model under a likelihood measure? (step 11)")):
    if N.get(q):
        out += [f"## {q}. {title}", "", plain(N[q]), ""]
for k, title in (("step9", "Step 9: robustness"), ("M1_preview", "Preview: J vs non-J at matched norm"), ("P17_preview", "Preview: Oracle vs NLA on the non-J part"),
                 ("rlens", "R-lens"), ("budget", "Regime and budget"), ("power", "Power")):
    if N.get(k):
        out += [f"## {title}", "", plain(N[k]), ""]
g10 = (S.get("step10") or {}).get("G10")
if g10:
    ch = g10["checks"].values()
    out += ["## Step 10 gate G10 (consistency)", "",
            f"Rows repeated from the pilot and step 7b reproduce their stored ΔR1′: median |difference| "
            f"{r(max(v['median_abs'] for v in ch), 3)} nats, 95th percentile {r(max(v['p95_abs'] for v in ch), 3)}, maximum "
            f"{r(max(v['max_abs'] for v in ch), 3)}, over {sum(v['n'] for v in ch)} rows (rule: {g10['rule']}). "
            + ("Passes." if g10["pass"] else "**Fails: step 10 is not used for any amendment item.**"), ""]
c11 = (S.get("step11") or {}).get("consistency")
if c11:
    ch = c11["checks"].values()
    out += ["## Step 11 consistency", "",
            f"Unclamped rows repeat step 10's: median |difference| {r(max(v['median_abs'] for v in ch), 3)} nats, maximum "
            f"{r(max(v['max_abs'] for v in ch), 3)}, over {sum(v['n'] for v in ch)} rows ({c11['rule']}). "
            + ("Passes." if c11["pass"] else "**Fails: step 11a is not used for any amendment item.**"), ""]
out += ["## Amendment", "", "Proposals filled from these results: `docs/design_specs/005_amendment_01.md`.", ""]
p = ROOT / "docs/reports/005_pilot_note.md"
p.write_text("\n".join(out))
print("wrote", p, len(out), "lines")
