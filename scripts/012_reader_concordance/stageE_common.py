"""012 Stage E shared definitions (design `docs/design_specs/012E_ar_concept_swaps.md` §3-§4). CPU only, no model.

Concept inventory, forbidden-word lists, and the code checks every generated phrase and prompt must pass. Everything
here is derived mechanically from the committed banks and Stage B's cohort, before any generation call.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
import wsb                                                                  # noqa: E402

STAGEB_ITEMS = ROOT / "outputs/012_reader_concordance/stageB/items.jsonl"
OUT = ROOT / "outputs/012_reader_concordance/stageE"
RES = ROOT / "results/012_reader_concordance/analysis"
COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
TEMPLATES = ("{c}", "{c} is", "the {c}", "about {c}", "{c} was", "and {c}", "like {c}", "{c} has")
N_KEEP, N_MIN = 32, 16                 # phrases kept per (concept, set); fewer than N_MIN -> dropped (design §3)
P_KEEP, P_MIN = 16, 10                 # gated cue-varied prompts kept per concept; fewer than P_MIN -> dropped (§4)
MAX_PER_REL, MAX_ANSWER_FRAC = 4, 1 / 3

# Function words, plus the bank's generic frame words (they occur in many prompts and say nothing about the clue).
STOP = set("""a an the and or but of in on at to for from by with as is are was were be been being it its this that
these those there their they them then than which who whom whose what when where why how not no nor so such also only
each both other some many much more most very about after before into over under between through during without within
upon onto your yours our ours his her hers him she he we you i me my mine will would could should shall may might must can
does did done have has had having just like same into fact country countries city cities state states number numbers
letter letters word words name names named first second third last main world home written spoken stand stands located
official called known commonly usually often largest biggest smallest highest tallest longest one two three
famous used great features originated containing immediately celebrated""".split())


def content_words(text):
    return {w.lower() for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 4 and w.lower() not in STOP}


def rel(name):
    return name.split("-")[-1]                                              # as stageD_analysis.rel


def cohort():
    """Stage A/D's 82-item cohort (as stageA_splits.py)."""
    rows = []
    for line in open(STAGEB_ITEMS):
        r = json.loads(line)
        if r.get("status") == "evaluated" and r["clean_correct"] and not r["clean_contains_y_b"]:
            rows.append(r)
    assert len(rows) == 82, len(rows)
    return rows


def word_rx(c):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(c) + r"(?![A-Za-z0-9])", re.I)


def eval_prompts():
    mh = [x for x in wsb.load_bank("multihop") if isinstance(x, list)][0]
    br = wsb.load_bank("brew")
    br = [x for x in br if isinstance(x, list)][0] if any(isinstance(x, list) for x in br) else list(br)
    return [it["prompt"] for it in mh] + [it["prompt"] for it in br]


def ngrams(text, n=5):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def inventory():
    """Two families: inv["multihop"] (kind 'mh' = src/tgt concept; kind 'answer' = a y_a / y_b that is not an mh
    concept) and inv["brew"] (kind 'colour'). A word can appear in both families (e.g. a colour that is also a
    multihop bridge); the entries are separate. Forbidden lists never contain the concept itself."""
    items = cohort()
    mh = sorted({r["src"] for r in items} | {r["tgt"] for r in items})
    answers = sorted({r["y_a"] for r in items} | {r["y_b"] for r in items})
    inv = {}
    for c in mh:
        f = set()
        for r in items:
            if r["src"] == c:
                f |= content_words(r["prompt"])
                f |= {x for x in [r["y_a"], *r.get("y_a_alts", [])]}
            if r["tgt"] == c:
                f |= {x for x in [r["y_b"], *r.get("y_b_alts", [])]}
        inv[c] = {"kind": "mh", "forbidden": f}
    for c in answers:
        if c in inv:
            inv[c]["also_answer"] = True
            continue
        f = set()
        for r in items:
            if c in (r["y_a"], r["y_b"]):
                f |= {r["src"], r["tgt"]} | content_words(r["prompt"])
        inv[c] = {"kind": "answer", "forbidden": f}
    brew_frame = content_words("Here is a colour-mixing problem. Answer with a single colour word and nothing else. "
                               "Problem: A potion changes colour each time it is stirred, according to these rules: "
                               "A potion turns when stirred. stir")
    brew = {c: {"kind": "colour", "forbidden": (set(COLOURS) - {c}) | brew_frame} for c in COLOURS}
    fam = {"multihop": inv, "brew": brew}
    for d in fam.values():
        for c, v in d.items():
            v["forbidden"] = sorted(w for w in v["forbidden"] if w.lower() != c.lower() and not word_rx(c).fullmatch(w))
            v["weak_input"] = bool(re.fullmatch(r"[A-Za-z]{1,2}|\d+", c))    # single letters, symbols, numerals
    return fam, items


def check_phrase(p, c, mode, forbidden, grams):
    """Design §3 code checks. Returns None if the phrase passes, else the first failing check's name."""
    p = p.strip().strip('"“”').strip()
    words = p.split()
    if not 3 <= len(words) <= 12:
        return "length"
    rx = word_rx(c)
    if mode == "starts":
        if not re.match(r"^\W*" + re.escape(c) + r"(?![A-Za-z0-9])", p, re.I):
            return "position"
    elif mode == "mentions":
        if re.match(r"^\W*" + re.escape(c) + r"(?![A-Za-z0-9])", p, re.I) or not rx.search(p):
            return "position"
    for w in forbidden:
        if word_rx(w).search(p):
            return "forbidden"
    if ngrams(p) & grams:
        return "ngram5"
    return None


def norm_text(p):
    return " ".join(re.findall(r"[a-z0-9]+", p.lower()))


def check_prompt(rec, c, forbidden, grams):
    """Design §4 code checks on one generated cue-varied prompt record. Returns None or the failing check's name."""
    p, ans = (rec.get("prompt") or "").strip(), (rec.get("answer") or "").strip()
    if not p.startswith("Fact: The ") or not p.endswith(" is"):
        return "format"
    if not ans or len(ans.split()) > 3:
        return "answer"
    if word_rx(c).search(p):
        return "names_concept"
    for w in forbidden:
        if word_rx(w).search(p):
            return "forbidden"
    if ngrams(p) & grams:
        return "ngram5"
    if not rec.get("clue") or rec["clue"].strip().lower() not in p.lower():
        return "clue_not_in_prompt"
    return None
