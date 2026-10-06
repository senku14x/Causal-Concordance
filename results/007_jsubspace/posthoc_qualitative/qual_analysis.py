"""EXPLORATORY, CPU-only word-match analysis of saved 007 reader readouts (no model, no GPU, no judge).

Brew (stage1a qualitative samples, ':' cell) and multihop (stage3 restatement samples). Reads the repo read-only and
writes only to this script's directory:
  qual_summary.json      all tables + selections (hand codes merged from hand_codes.json if present)
  qual_tables.md         auto-generated tables
  qual_appendix.md       auto-generated quotes + hand codes, multihop quotes, matcher sanity check
  handcode_selection.md  the fixed-seed samples to hand-code (full text)
Hand codes (mine, unvalidated) are read from hand_codes.json.

  /venv/main/bin/python qual_analysis.py
"""
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/workspace/causal-concordance")
O7 = REPO / "outputs" / "007_jsubspace"
BREW_FILES = [O7 / "stage1a/r2/qualitative_oracle_L40.jsonl", O7 / "stage1a/r2/qualitative_oracle_L44.jsonl",
              O7 / "stage1a/r2_nla_posthoc/qualitative_nla_L42.jsonl"]
MH_FILES = [O7 / "stage3/m1/restatement_oracle_L40.jsonl", O7 / "stage3/m1/restatement_oracle_L44.jsonl",
            O7 / "stage3/m1_nla_posthoc/restatement_nla_L42.jsonl"]
MH_DESIGN = O7 / "stage3/m1/design_material.json"        # byte-identical to m1_nla_posthoc/design_material.json
MH_ITEMS = Path("/workspace/sources/repos/workspace-bench/evals/multihop/items.json")
BREW_ITEMS = Path("/workspace/sources/repos/workspace-bench/evals/brew_intermediates/items.json")
SEED = 20261006
PIECE_ORDER = ["whole", "J", "rest_J", "PCA", "MR", "GMR", "R0"]

# ------------------------------------------------------------------------------------------- colour lexicon
BREW = ["black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white"]
EN = {"black": r"blacks?", "blue": r"blues?", "brown": r"browns?", "gold": r"gold(?:en|s)?", "gray": r"gr[ae]ys?",
      "green": r"greens?", "orange": r"oranges?", "pink": r"pinks?", "purple": r"purples?", "red": r"reds?",
      "teal": r"teals?", "white": r"whites?"}
EN_EXTRA = ["yellow", "silver", "violet", "cyan", "amber", "olive", "coral", "beige", "ivory", "crimson", "navy",
            "magenta", "indigo", "maroon", "turquoise", "lavender", "scarlet"]
# CJK: compounds first (their span is masked so the single characters inside are not counted again), then singles.
CJK_COMPOUND = [("粉红", "pink"), ("粉紅", "pink"), ("蓝绿", "teal"), ("藍綠", "teal"), ("青绿", "teal"),
                ("金黄", "gold"), ("金黃", "gold"), ("黄金", "gold"), ("黃金", "gold"), ("咖啡色", "brown")]
CJK_SINGLE = {"黑": "black", "蓝": "blue", "藍": "blue", "棕": "brown", "褐": "brown", "金": "gold", "灰": "gray",
              "绿": "green", "綠": "green", "橙": "orange", "橘": "orange", "粉": "pink", "紫": "purple", "红": "red",
              "紅": "red", "青": "teal", "白": "white", "黄": "yellow", "黃": "yellow", "银": "silver", "銀": "silver"}
EN_RE = [(c, re.compile(r"\b" + p + r"\b", re.I | re.A)) for c, p in EN.items()] + \
        [(c, re.compile(r"\b" + c + r"s?\b", re.I | re.A)) for c in EN_EXTRA]


def is_han(ch):
    return "一" <= ch <= "鿿"


def colour_hits(text, strict_cjk=False):
    """[(position, colour, matched string)] sorted by position. English: ASCII word boundary, case-insensitive.
    CJK: plain substring (compounds first); strict_cjk additionally requires the match to end a word (next char is
    色 or not a Han character), which drops e.g. 黑屏 'black screen', 蓝莓 'blueberry', 红队 'red team'."""
    hits = []
    for c, rx in EN_RE:
        hits += [(m.start(), c, m.group(0)) for m in rx.finditer(text)]
    masked = list(text)
    for comp, c in CJK_COMPOUND:
        for m in re.finditer(re.escape(comp), text):
            nxt = text[m.end()] if m.end() < len(text) else ""
            if not strict_cjk or nxt == "色" or not is_han(nxt):
                hits.append((m.start(), c, comp))
            for i in range(m.start(), m.end()):
                masked[i] = " "
    for i, ch in enumerate(masked):
        if ch in CJK_SINGLE:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not strict_cjk or nxt == "色" or not is_han(nxt):
                hits.append((i, CJK_SINGLE[ch], ch))
    return sorted(hits)


# ------------------------------------------------------------------------------------------- task markers
# Generic task-structure markers (the caller's list, with inflections; English word-boundary, CJK substring).
TASK_EN = {"step": r"steps?", "twice": r"twice", "two": r"two", "apply": r"appl(?:y|ies|ied|ying|ication|ications)",
           "rule": r"rules?", "map": r"(?:map|maps|mapping|mapped)", "mix": r"(?:mix|mixes|mixed|mixing|mixture)",
           "transform": r"transform(?:s|ed|ing|ation|ations)?", "sequence": r"sequences?", "chain": r"chains?",
           "cycle": r"cycles?", "table": r"tables?"}
TASK_SYM = {"arrow_unicode": "→", "arrow_ascii": "->"}
TASK_CJK = {"步": "step", "两次": "twice", "二次": "twice", "规则": "rule", "循环": "cycle", "变换": "transform",
            "转换": "transform", "序列": "sequence", "链": "chain", "混合": "mix"}
# Prompt-lexical markers: words of the brew prompt itself ("A potion changes colour each time it is stirred ...").
PROMPT_EN = {"potion": r"potions?", "stir": r"stir(?:s|red|ring)?", "brew": r"brew(?:s|ed|ing)?",
             "colour_word": r"colou?rs?"}
TASK_EN_RE = {k: re.compile(r"\b" + p + r"\b", re.I | re.A) for k, p in TASK_EN.items()}
PROMPT_EN_RE = {k: re.compile(r"\b" + p + r"\b", re.I | re.A) for k, p in PROMPT_EN.items()}


def task_markers(text):
    found = {k for k, rx in TASK_EN_RE.items() if rx.search(text)}
    found |= {k for k, s in TASK_SYM.items() if s in text}
    found |= {"cjk_" + v for k, v in TASK_CJK.items() if k in text}
    return found


def prompt_markers(text):
    return {k for k, rx in PROMPT_EN_RE.items() if rx.search(text)}


# ------------------------------------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def frac(k, n):
    return {"k": k, "n": n, "share": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def clean_len(s):
    return len(s.replace("<explanation>", "").replace("</explanation>", "").strip())


# ------------------------------------------------------------------------------------------- brew
def brew_tables():
    bank = {it["name"]: it for it in json.loads(BREW_ITEMS.read_text())["items"]}
    rows = []
    for f in BREW_FILES:
        rows += [json.loads(l) for l in open(f)]
    # pair-mate lookup (same reader/layer/cell/piece/table, other role)
    mate = {}
    for r in rows:
        mate[(r["reader"], r["layer"], r["cell"], r["piece"], r["table"], r["role"])] = r
    feats = []
    for ri, r in enumerate(rows):
        it = bank[r["item"]]
        tab = it["table"]
        prompt_cols = set(tab) | {v["stir"] for v in tab.values()}
        start, z, ans = r["path"]
        assert tab[start]["stir"] == z and tab[z]["stir"] == ans, (r["item"], r["path"])
        over = tab.get(ans, {}).get("stir")              # one stir past the answer (overshoot), if in the table
        m = mate[(r["reader"], r["layer"], r["cell"], r["piece"], r["table"], "partner" if r["role"] == "member" else "member")]
        mz, ma = m["path"][1], m["path"][2]
        for si, s in enumerate(r["samples"]):
            f = {"row": ri, "sample": si, "reader": r["reader"], "layer": r["layer"], "cell": r["cell"],
                 "piece": r["piece"], "table": r["table"], "render": r["render"], "role": r["role"], "path": r["path"],
                 "len": clean_len(s), "over": over, "mate_z": mz}
            for mode in ("substr", "strict"):
                hits = colour_hits(s, strict_cjk=(mode == "strict"))
                first = {}
                for pos, c, _ in hits:
                    first.setdefault(c, pos)
                named = set(first)
                brew_named = named & set(BREW)
                off_brew = brew_named - set(r["path"])
                nonbrew = named - set(BREW)
                g = {"z": z in named, "answer": ans in named, "start": start in named,
                     "offpath_brew": bool(off_brew), "offpath_any": bool(off_brew or nonbrew), "nonbrew": bool(nonbrew),
                     "offpath_not_in_prompt": bool(off_brew - prompt_cols),
                     "no_colour": not named, "no_brew_colour": not brew_named,
                     "z_not_answer": (z in named) and (ans not in named),
                     "answer_not_z": (ans in named) and (z not in named),
                     "both": (z in named) and (ans in named),
                     "z_first": (z in named) and (ans in named) and first[z] < first[ans],
                     "answer_first": (z in named) and (ans in named) and first[ans] < first[z],
                     "mate_z": (mz in named) if mz not in r["path"] else None,
                     "mate_answer": (ma in named) if ma not in r["path"] else None,
                     "overshoot": (over in named) if over and over not in r["path"] else None,
                     "n_brew_named": len(brew_named), "n_offpath_brew": len(off_brew)}
                if hits:
                    c0 = hits[0][1]
                    g["first_colour"] = ("start" if c0 == start else "z" if c0 == z else "answer" if c0 == ans
                                         else "offpath_brew" if c0 in BREW else "nonbrew")
                else:
                    g["first_colour"] = "none"
                f[mode] = g
                f[mode + "_named"] = named
                if mode == "substr":
                    f["hits"] = hits
            tm, pm = task_markers(s), prompt_markers(s)
            f["task_markers"] = sorted(tm)
            f["prompt_markers"] = sorted(pm)
            f["task_any"] = bool(tm)
            f["task_core"] = bool(tm - {"two", "arrow_unicode", "arrow_ascii"})   # excluding the most generic three
            f["prompt_any"] = bool(pm)
            f["prompt_potion_stir"] = bool(pm & {"potion", "stir", "brew"})
            feats.append(f)
    groups = defaultdict(list)
    for f in feats:
        groups[(f["reader"], f["layer"], f["cell"], f["piece"])].append(f)
    out = {}
    for key, fs in groups.items():
        n = len(fs)
        G = {"n_samples": n, "n_renders": len({f["render"] for f in fs}),
             "mean_len_chars": round(sum(f["len"] for f in fs) / n, 1)}
        for mode in ("substr", "strict"):
            M = {}
            for k in ("z", "answer", "start", "offpath_brew", "offpath_any", "nonbrew", "offpath_not_in_prompt",
                      "no_colour", "no_brew_colour", "z_not_answer", "answer_not_z", "both"):
                M[k] = frac(sum(f[mode][k] for f in fs), n)
            nb = sum(f[mode]["both"] for f in fs)
            M["order_when_both"] = {"n_both": nb, "z_first": sum(f[mode]["z_first"] for f in fs),
                                    "answer_first": sum(f[mode]["answer_first"] for f in fs)}
            for k in ("mate_z", "mate_answer", "overshoot"):
                v = [f[mode][k] for f in fs if f[mode][k] is not None]
                M[k] = frac(sum(v), len(v))
            # colour-prior-matched expectation: base_c = P(sample names c | c is off-path for its render and not its
            # overshoot colour), pooled over this group's samples; expected(role) = sum over samples of base_{role colour}
            base = {}
            for c in BREW:
                elig = [f for f in fs if c not in f["path"] and c != f["over"]]
                base[c] = (sum(c in f[mode + "_named"] for f in elig) / len(elig)) if elig else None
            M["prior_matched"] = {}
            for role, getc in (("start", lambda f: f["path"][0]), ("z", lambda f: f["path"][1]),
                               ("answer", lambda f: f["path"][2]), ("overshoot", lambda f: f["over"]),
                               ("mate_z", lambda f: f["mate_z"] if f["mate_z"] not in f["path"] else None)):
                ok = [f for f in fs if getc(f) is not None and getc(f) not in (f["path"] if role in ("overshoot", "mate_z") else ())
                      and base.get(getc(f)) is not None]
                obs = sum(getc(f) in f[mode + "_named"] for f in ok)
                exp = sum(base[getc(f)] for f in ok)
                M["prior_matched"][role] = {"n": len(ok), "observed": obs, "expected": round(exp, 1),
                                            "obs_over_exp": round(obs / exp, 2) if exp else None}
            M["offpath_base_rates"] = {c: (round(v, 3) if v is not None else None) for c, v in base.items()}
            M["first_colour"] = dict(Counter(f[mode]["first_colour"] for f in fs))
            M["mean_n_brew_colours_named"] = round(sum(f[mode]["n_brew_named"] for f in fs) / n, 2)
            M["mean_n_offpath_brew_named"] = round(sum(f[mode]["n_offpath_brew"] for f in fs) / n, 2)
            # render-level (clustering check): renders with >=1 of 2 samples naming z / answer
            byr = defaultdict(list)
            for f in fs:
                byr[f["render"]].append(f)
            M["renders_any_z"] = frac(sum(any(x[mode]["z"] for x in v) for v in byr.values()), len(byr))
            M["renders_any_answer"] = frac(sum(any(x[mode]["answer"] for x in v) for v in byr.values()), len(byr))
            G[mode] = M
        G["task_any"] = frac(sum(f["task_any"] for f in fs), n)
        G["task_core"] = frac(sum(f["task_core"] for f in fs), n)
        G["prompt_any"] = frac(sum(f["prompt_any"] for f in fs), n)
        G["prompt_potion_stir"] = frac(sum(f["prompt_potion_stir"] for f in fs), n)
        G["task_marker_counts"] = dict(Counter(m for f in fs for m in f["task_markers"]).most_common())
        G["prompt_marker_counts"] = dict(Counter(m for f in fs for m in f["prompt_markers"]).most_common())
        out["|".join(map(str, key))] = G
    return rows, feats, out


def handcode_selection(rows, feats):
    """random.seed(SEED); for (reader, layer) in file order, for piece in PIECE_ORDER: random.sample(group, 3), where
    group = that piece's (row, sample) pairs in file order."""
    random.seed(SEED)
    by = defaultdict(list)
    for f in feats:
        by[(f["reader"], f["layer"], f["piece"])].append(f)
    sel = []
    for rd, L in [("oracle", 40), ("oracle", 44), ("nla", 42)]:
        for pc in PIECE_ORDER:
            for f in random.sample(by[(rd, L, pc)], 3):
                s = rows[f["row"]]["samples"][f["sample"]]
                sel.append({"id": f"{rd}{L}-{pc}-r{f['render']}-s{f['sample']}", "reader": rd, "layer": L, "piece": pc,
                            "render": f["render"], "path": f["path"], "sample": f["sample"],
                            "colours_matched": sorted({c for _, c, _ in f["hits"]}),
                            "quote": s[:250] + ("…" if len(s) > 250 else ""), "full": s})
    return sel


def handcode_tally(sel):
    t = defaultdict(Counter)
    for s in sel:
        if s.get("hand_code"):
            t[f"{s['reader']}{s['layer']}|{s['piece']}"][s["hand_code"]["primary"]] += 1
    return {k: dict(v) for k, v in t.items()}


def sanity(rows, feats):
    rng = random.Random(SEED + 5)
    out = []
    for f in rng.sample(feats, 10):
        s = rows[f["row"]]["samples"][f["sample"]]
        ctx = [f"{c}<={m!r} @{p}: …{s[max(0, p - 18):p + len(m) + 12]!r}…" for p, c, m in f["hits"]]
        out.append({"reader": f["reader"], "layer": f["layer"], "piece": f["piece"], "render": f["render"],
                    "path": f["path"], "text_first_500": s[:500], "len": len(s), "matches": ctx,
                    "strict_named": sorted(k for k in ("z", "answer", "start") if f["strict"][k]),
                    "flags": {k: f["substr"][k] for k in ("z", "answer", "start", "offpath_brew", "nonbrew", "no_colour")}})
    return out


def _toks(s):
    s = s.replace("<explanation>", "").replace("</explanation>", "")
    han = "".join(ch for ch in s if is_han(ch))
    return set(re.findall(r"[a-z]{3,}", s.lower())) | {han[i:i + 2] for i in range(len(han) - 1)}


def _jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def render_specificity(rows, feats):
    """Post hoc, crude: word-set Jaccard (ASCII words >= 3 letters + Han bigrams) between samples of the SAME render vs
    DIFFERENT renders, within a (reader, layer, piece); ratio >> 1 = the text depends on the render. Also the same
    comparison ACROSS readers (Oracle L44 vs NLA L42; Oracle L40 vs L44) for the same piece."""
    g = defaultdict(list)
    for f in feats:
        g[(f["reader"], f["layer"], f["piece"])].append((f["render"], _toks(rows[f["row"]]["samples"][f["sample"]])))
    within, across = {}, {}
    for k, v in g.items():
        same = [_jac(a[1], b[1]) for i, a in enumerate(v) for b in v[i + 1:] if a[0] == b[0]]
        diff = [_jac(a[1], b[1]) for i, a in enumerate(v) for b in v[i + 1:] if a[0] != b[0]]
        within["|".join(map(str, k))] = {"same_render": round(sum(same) / len(same), 3),
                                         "cross_render": round(sum(diff) / len(diff), 3),
                                         "ratio": round((sum(same) / len(same)) / (sum(diff) / len(diff)), 2)}
    for (ra, la), (rb, lb) in [(("oracle", 44), ("nla", 42)), (("oracle", 40), ("oracle", 44)), (("oracle", 40), ("nla", 42))]:
        for pc in PIECE_ORDER:
            A, B = g[(ra, la, pc)], g[(rb, lb, pc)]
            same = [_jac(a[1], b[1]) for a in A for b in B if a[0] == b[0]]
            diff = [_jac(a[1], b[1]) for a in A for b in B if a[0] != b[0]]
            across[f"{ra}{la}~{rb}{lb}|{pc}"] = {"same_render": round(sum(same) / len(same), 3),
                                                 "cross_render": round(sum(diff) / len(diff), 3),
                                                 "ratio": round((sum(same) / len(same)) / (sum(diff) / len(diff)), 2)}
    # cyan as a near-synonym of teal (not counted as teal by the matcher)
    cy = Counter()
    tot = Counter()
    for f in feats:
        role = "z" if f["path"][1] == "teal" else "answer" if f["path"][2] == "teal" else "start" if f["path"][0] == "teal" else "absent"
        tot[role] += 1
        cy[role] += any(h[1] == "cyan" for h in f["hits"])
    cyan = {r: f"{cy[r]}/{tot[r]}" for r in tot}
    return within, across, cyan


# ------------------------------------------------------------------------------------------- multihop
def multihop():
    items = json.loads(MH_ITEMS.read_text())["items"]
    dm = json.loads(MH_DESIGN.read_text())
    cues = {int(k): v for k, v in dm["cue_words"].items()}
    res, flat = {}, []
    for fpath in MH_FILES:
        for l in open(fpath):
            r = json.loads(l)
            it = items[r["idx"]]
            b = it["intermediates"][0]
            bw = [b] + ([b.split()[-1]] if len(b.split()) > 1 else [])
            cw = cues.get(r["idx"], [])
            for si, s in enumerate(r["samples"]):
                t = s.lower()                                               # analysis_secondary.py convention, verbatim
                nb = any(re.search(r"\b" + re.escape(w.lower()) + r"\b", t) for w in bw)
                nt = bool(re.search(r"\b" + re.escape(it["target"].lower()) + r"\b", t))
                rc = any(re.search(r"\b" + re.escape(w.lower()) + r"\b", t) for w in cw) if cw else None
                flat.append({"reader": r["reader"], "layer": r["layer"], "piece": r["piece"], "split": r["split"],
                             "idx": r["idx"], "item": r["item"], "sample": si, "nb": nb, "nt": nt, "rc": rc, "text": s,
                             "bridge": b, "target": it["target"], "cues": cw, "prompt": it["prompt"]})
    g = defaultdict(list)
    for x in flat:
        g[(x["reader"], x["layer"], x["piece"], x["split"])].append(x)
    for k, xs in g.items():
        cue = [x for x in xs if x["rc"] is not None]
        nocue = [x for x in xs if x["rc"] is None]
        bc = [x for x in cue if x["nb"]]
        res["|".join(map(str, k))] = {
            "n": len(xs), "n_items": len({x["idx"] for x in xs}), "n_cue_item_samples": len(cue),
            "n_nocue_item_samples": len(nocue), "n_cue_items": len({x["idx"] for x in cue}),
            "names_bridge": frac(sum(x["nb"] for x in xs), len(xs)),
            "names_target": frac(sum(x["nt"] for x in xs), len(xs)),
            "bridge_on_cue_items": frac(len(bc), len(cue)),
            "bridge_and_cue__over__bridge_on_cue_items": frac(sum(x["rc"] for x in bc), len(bc)),
            "bridge_without_cue": len(bc) - sum(x["rc"] for x in bc),
            "bridge_on_nocue_items": frac(sum(x["nb"] for x in nocue), len(nocue)),
            "restates_cue__over__cue_item_samples": frac(sum(x["rc"] for x in cue), len(cue)),
            "restates_cue_not_bridge": sum(x["rc"] and not x["nb"] for x in cue),
            "items_with_any_bridge": len({x["idx"] for x in xs if x["nb"]}),
            "target_among_bridge_naming": frac(sum(x["nt"] for x in xs if x["nb"]), sum(x["nb"] for x in xs)),
            "bridge_not_target": sum(x["nb"] and not x["nt"] for x in xs),
            "short_or_numeric_target_items": len({x["idx"] for x in xs if len(x["target"]) <= 2 or x["target"].isdigit()}),
        }
    rng = random.Random(SEED)
    quotes = []
    for rd, L in [("oracle", 40), ("oracle", 44), ("nla", 42)]:
        pool = [x for x in g[(rd, L, "J", "test")] if x["nb"] and x["rc"] is False]
        for x in rng.sample(pool, 2):
            quotes.append({k: x[k] for k in ("reader", "layer", "idx", "item", "sample", "prompt", "bridge", "target", "cues")}
                          | {"quote": x["text"][:400] + ("…" if len(x["text"]) > 400 else ""), "names_target": x["nt"]})
    return res, quotes


# ------------------------------------------------------------------------------------------- tables (markdown)
def md_tables(brew, mh):
    L = []
    for rd, Ly in [("oracle", 40), ("oracle", 44), ("nla", 42)]:
        L += [f"### Brew `:` cell — {rd} L{Ly} (n = 32 samples per piece: 16 renders × 2; CJK substring)", "",
              "| piece | z | answer | start | off-path brew | any non-path colour | no colour | z not answer | "
              "both: z first / ans first | mate's z | task marker (core) | prompt words (potion/stir/brew) | mean chars |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for pc in PIECE_ORDER:
            G = brew.get(f"{rd}|{Ly}|emit_colon|{pc}")
            if not G:
                continue
            M = G["substr"]
            c = lambda d: f"{d['k']}/{d['n']}"
            o = M["order_when_both"]
            L.append(f"| {pc} | {c(M['z'])} | {c(M['answer'])} | {c(M['start'])} | {c(M['offpath_brew'])} | "
                     f"{c(M['offpath_any'])} | {c(M['no_colour'])} | {c(M['z_not_answer'])} | {o['z_first']}/{o['answer_first']} "
                     f"(of {o['n_both']}) | {c(M['mate_z'])} | {c(G['task_core'])} | {c(G['prompt_potion_stir'])} | {G['mean_len_chars']:.0f} |")
        L += ["", f"Strict-CJK variant (z / answer / no colour): " + "; ".join(
            f"{pc} {brew[f'{rd}|{Ly}|emit_colon|{pc}']['strict']['z']['k']}/"
            f"{brew[f'{rd}|{Ly}|emit_colon|{pc}']['strict']['answer']['k']}/"
            f"{brew[f'{rd}|{Ly}|emit_colon|{pc}']['strict']['no_colour']['k']}" for pc in PIECE_ORDER), ""]
        L += [f"Renders (of 16) with ≥1 sample naming z: " + ", ".join(
            f"{pc} {brew[f'{rd}|{Ly}|emit_colon|{pc}']['substr']['renders_any_z']['k']}" for pc in PIECE_ORDER), ""]
        L += ["Colour-prior-matched: observed / expected samples naming the role colour (expected = that colour's naming "
              "rate when it is off-path, same reader/layer/piece)", "",
              "| piece | z | answer | start | overshoot (answer's successor) | mate's z |", "|---|---|---|---|---|---|"]
        for pc in PIECE_ORDER:
            P = brew[f"{rd}|{Ly}|emit_colon|{pc}"]["substr"]["prior_matched"]
            L.append(f"| {pc} | " + " | ".join(f"{P[r]['observed']} / {P[r]['expected']} (n={P[r]['n']})"
                                                 for r in ("z", "answer", "start", "overshoot", "mate_z")) + " |")
        L.append("")
    L += ["### Multihop restatement, test split (n = 80 samples per piece: 40 items × 2; 28 cue items = 56 samples, "
          "12 no-cue items = 24 samples)", "",
          "| reader | piece | names bridge | names target | bridge on cue items | of those, restate cue | bridge on no-cue items |",
          "|---|---|---|---|---|---|---|"]
    for rd, Ly in [("oracle", 40), ("oracle", 44), ("nla", 42)]:
        for pc in ["whole", "J", "rest_J", "PCA", "MR", "R0"]:
            R = mh[f"{rd}|{Ly}|{pc}|test"]
            c = lambda d: f"{d['k']}/{d['n']}"
            L.append(f"| {rd} L{Ly} | {pc} | {c(R['names_bridge'])} | {c(R['names_target'])} | {c(R['bridge_on_cue_items'])} | "
                     f"{c(R['bridge_and_cue__over__bridge_on_cue_items'])} | {c(R['bridge_on_nocue_items'])} |")
    return "\n".join(L) + "\n"


def md_tables_extra(summary):
    L = ["### Render specificity of the text (post hoc, crude word-overlap)", "",
         "Mean Jaccard of word sets (ASCII words ≥ 3 letters + Han bigrams): same-render pairs ÷ different-render pairs. "
         "≈ 1 = the text does not depend on the render. Last column: Oracle L44 vs NLA L42 samples (across readers).", "",
         "| piece | Oracle L40 | Oracle L44 | NLA L42 | across: Oracle L44 ~ NLA |", "|---|---|---|---|---|"]
    W, X = summary["brew_render_specificity_posthoc"]["within_reader"], summary["brew_render_specificity_posthoc"]["across_readers"]
    for pc in PIECE_ORDER:
        L.append(f"| {pc} | {W[f'oracle|40|{pc}']['ratio']} | {W[f'oracle|44|{pc}']['ratio']} | {W[f'nla|42|{pc}']['ratio']} | "
                 f"{X[f'oracle44~nla42|{pc}']['ratio']} |")
    L += ["", "### Hand-code tally (3 samples per cell; my codes, unvalidated)", "",
          "| piece | Oracle L40 | Oracle L44 | NLA L42 |", "|---|---|---|---|"]
    T = summary["brew_handcode_tally"]
    ab = {"task_restatement": "task", "offtopic_invented": "off-topic", "z_intermediate": "z-step", "z_answer": "z-as-answer",
          "answer": "answer", "generic_meta": "meta"}
    for pc in PIECE_ORDER:
        L.append(f"| {pc} | " + " | ".join(", ".join(f"{ab[k]} {v}" for k, v in sorted(T.get(f"{rl}|{pc}", {}).items()))
                                          for rl in ("oracle40", "oracle44", "nla42")) + " |")
    L += ["", "### Multihop (test): does the bridge come with the target?", "",
          "| reader | piece | bridge-naming samples that also name the target | bridge named, target not |", "|---|---|---|---|"]
    for rd, Ly in [("oracle", 40), ("oracle", 44), ("nla", 42)]:
        for pc in ["whole", "J", "rest_J", "MR", "R0"]:
            R = summary["multihop"][f"{rd}|{Ly}|{pc}|test"]
            t = R["target_among_bridge_naming"]
            L.append(f"| {rd} L{Ly} | {pc} | {t['k']}/{t['n']} | {R['bridge_not_target']} |")
    return "\n".join(L) + "\n"


def _flat(s):
    return s.replace("\r", "").replace("\n", " / ")


def md_appendix(summary):
    A = ["## Appendix A — fixed-seed samples, 3 per piece, with hand codes", "",
         "Selection: `random.seed(20261006)`, then for (oracle L40, oracle L44, nla L42) × pieces in the order "
         "whole, J, rest_J, PCA, MR, GMR, R0: `random.sample(group, 3)`. Quotes are verbatim, first 250 characters, "
         "newlines shown as ` / `. **Hand codes are mine (single coder, not blind to the path), unvalidated**, and "
         "were made on the full sample text (so a code can rest on text past the quote). Full texts: "
         "`handcode_selection.md`.", ""]
    order = [("nla", 42), ("oracle", 44), ("oracle", 40)]
    sel = summary["brew_handcode_selection"]
    for rd, L in order:
        A += [f"### A.{order.index((rd, L)) + 1} {rd} L{L}", ""]
        for s in [x for x in sel if x["reader"] == rd and x["layer"] == L]:
            hc = s.get("hand_code") or {}
            sec = "; ".join(hc.get("secondary", []))
            A += [f"- **{s['id']}** — path {' → '.join(s['path'])} — matched: {', '.join(s['colours_matched']) or 'none'} — "
                  f"**code: {hc.get('primary', '?')}**" + (f" ({sec})" if sec else "") + (f". {hc['note']}" if hc.get("note") else ""),
                  f"  > {_flat(s['quote'])}", ""]
    A += ["## Appendix B — multihop: bridge-naming J samples (test) that do not restate the cue", "",
          "`random.Random(20261006).sample(pool, 2)` per reader; first 400 characters, newlines as ` / `.", ""]
    for q in summary["multihop_quotes_J_bridge_no_cue"]:
        A += [f"- **{q['reader']} L{q['layer']}**, item {q['idx']} `{q['item']}` — prompt: “{q['prompt']}” — bridge **{q['bridge']}**, "
              f"target {q['target']}, cue words {q['cues']}; also names target: {q['names_target']}",
              f"  > {_flat(q['quote'])}", ""]
    A += ["## Appendix C — colour-matcher sanity check (10 random brew samples)", "",
          "`random.Random(20261011).sample(all 672 samples, 10)`. Each match is shown as colour ⇐ matched string with "
          "context; first 300 characters of the text. Full match lists in `qual_summary.json` → `brew_sanity_10`.", ""]
    for i, s in enumerate(summary["brew_sanity_10"]):
        fl = ", ".join(k for k, v in s["flags"].items() if v)
        ms = sorted({m.split(" @")[0] for m in s["matches"]})
        A += [f"{i + 1}. **{s['reader']} L{s['layer']} {s['piece']} r{s['render']}** — path {' → '.join(s['path'])} — "
              f"flags: {fl or '—'} — distinct matches: {', '.join(ms) or 'none'}",
              f"   > {_flat(s['text_first_500'][:300])}", ""]
    return "\n".join(A) + "\n"


def main():
    rows, feats, brew = brew_tables()
    sel = handcode_selection(rows, feats)
    hc_path = HERE / "hand_codes.json"
    if hc_path.exists():
        hc = json.loads(hc_path.read_text())
        for s in sel:
            s["hand_code"] = hc.get(s["id"])
    mh, mh_quotes = multihop()
    rs_within, rs_across, cyan = render_specificity(rows, feats)
    repro = {k: mh[k]["bridge_and_cue__over__bridge_on_cue_items"] for k in
             ("oracle|40|J|test", "oracle|44|J|test", "nla|42|J|test")}
    summary = {
        "label": "EXPLORATORY — CPU-only analysis of saved readouts; word-match, no judge",
        "inputs": {str(p.relative_to(REPO) if str(p).startswith(str(REPO)) else p): sha(p)
                   for p in BREW_FILES + MH_FILES + [MH_DESIGN, MH_ITEMS, BREW_ITEMS]},
        "notes": ["Only the ':' cell (emit_colon) has qualitative brew samples; no qualitative_*.jsonl exists under "
                  "stage1b/ or brew_test/ (they hold likelihood matrices only), so the '</think>' cell is not covered.",
                  "Brew: 8 tables (fixed seed in reader_split.py), one member-partner pair each = 16 renders, 2 samples "
                  "per render per piece = 32 samples per (reader, layer, piece). Samples are clustered (2 per render).",
                  "Oracle samples are multi-bullet lists (the carrier asks for distinct concepts as '- ' bullets), so a "
                  "single sample often names several colours."],
        "lexicon": {"brew_colours": BREW, "english_patterns": EN, "english_nonbrew": EN_EXTRA,
                    "cjk_compounds": CJK_COMPOUND, "cjk_single": CJK_SINGLE,
                    "english_rule": "ASCII word boundary, case-insensitive", "cjk_rule": "substring (primary); strict = "
                    "match must end a word (next char 色 or non-Han)", "emoji": "not counted (e.g. 💙 🟢 ⚫)"},
        "markers": {"task_generic_en": TASK_EN, "task_symbols": TASK_SYM, "task_cjk": TASK_CJK,
                    "task_core": "task_any minus {two, →, ->}", "prompt_lexical_en": PROMPT_EN,
                    "prompt_potion_stir": "any of potion/stir/brew"},
        "brew": brew,
        "brew_render_specificity_posthoc": {"within_reader": rs_within, "across_readers": rs_across},
        "brew_cyan_by_teal_role": cyan,
        "brew_handcode_selection": [{k: v for k, v in s.items() if k != "full"} for s in sel],
        "brew_handcode_tally": handcode_tally(sel),
        "brew_sanity_10": sanity(rows, feats),
        "multihop": mh,
        "multihop_reproduction_J_test": {"expected": {"oracle|40": "15/53", "oracle|44": "13/54", "nla|42": "8/53"},
                                         "ours": {k: f"{v['k']}/{v['n']}" for k, v in repro.items()}},
        "multihop_quotes_J_bridge_no_cue": mh_quotes,
    }
    (HERE / "qual_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (HERE / "qual_tables.md").write_text(md_tables(brew, mh) + "\n" + md_tables_extra(summary))
    (HERE / "qual_appendix.md").write_text(md_appendix(summary))
    hs = ["# Hand-code selection (random.seed(20261006); 3 per piece; full text)", ""]
    for s in sel:
        hs += [f"## {s['id']}  path={s['path']}  colours={s['colours_matched']}", "", "```", s["full"], "```", ""]
    (HERE / "handcode_selection.md").write_text("\n".join(hs))
    print(json.dumps(summary["multihop_reproduction_J_test"]))
    print((HERE / "qual_tables.md").read_text())


if __name__ == "__main__":
    main()
