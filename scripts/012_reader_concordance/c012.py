"""012 Stage C shared helpers (reader side only; Stage B never imports this): paths, LSE per concept, d_r, naming."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WSB = Path("/workspace/sources/repos/workspace-bench/evals/multihop/items.json")
_WSB = {it["name"]: it for it in json.loads(WSB.read_text())["items"]}
NUM = {str(i): w for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def paths(tag):
    sub = "" if tag == "full" else tag
    b, c = ROOT / "outputs/012_reader_concordance/stageB" / sub, ROOT / "outputs/012_reader_concordance/stageC" / sub
    c.mkdir(parents=True, exist_ok=True)
    return {"captures": b / "captures.pt", "items": b / "items.jsonl",
            "readouts": lambda r: c / f"readouts_{r}.jsonl", "ll": lambda r: c / f"ll_{r}.jsonl",
            "summary": lambda r: c / f"summary_{r}.json"}


def lse_owner(arr, owner, it):
    """Per-spelling full-sequence log-probs -> LSE over each concept's spellings (as 007 dstat.colour_ll)."""
    arr, owner = np.asarray(arr, dtype=np.float64), np.asarray(owner)
    out = {}
    for c in ("src", "tgt"):
        cols = arr[owner == it[c]]
        assert cols.size >= 1, it[c]
        m = cols.max()
        out[c] = float(m + np.log(np.exp(cols - m).sum()))
    return out


def d_r(ed, cl):
    if None in (ed["src"], ed["tgt"], cl["src"], cl["tgt"]):
        return None
    return (ed["tgt"] - ed["src"]) - (cl["tgt"] - cl["src"])


def concept_names(it):
    """Names matched in readouts: the concept, WSB's intermediates when the src is one of them, the number word."""
    src, tgt = [it["src"]], [it["tgt"]]
    w = _WSB.get(it["item"] if "item" in it else it["name"])
    if w and it["src"].lower() in [x.lower() for x in w["intermediates"]]:
        src += [x for x in w["intermediates"] if x.lower() != it["src"].lower()]
    for lst in (src, tgt):
        lst += [NUM[x] for x in list(lst) if x in NUM]
    return src, tgt


def _hit(text, names):
    return any(re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.I) for n in names)


def category(text, src_names, tgt_names, labels=("src", "tgt")):
    s, t = _hit(text, src_names), _hit(text, tgt_names)
    return "both" if s and t else f"{labels[0]}-only" if s else f"{labels[1]}-only" if t else "neither"


def answer_names(it):
    """Answer naming: y_a (+alts) vs y_b (+alts), same case-insensitive whole-word match."""
    return [it["y_a"], *it.get("y_a_alts", [])], [it["y_b"], *it.get("y_b_alts", [])]


def answer_category(text, it):
    return category(text, *answer_names(it), labels=("ya", "yb"))
