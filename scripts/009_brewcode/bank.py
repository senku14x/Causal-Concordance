"""009 brew-code bank (design docs/design_specs/008_009_trackb_brewcode.md §2.2). CPU only; reader-free.

Fresh random brew tables in WSB's sentence format plus a colour -> two-digit code key; the answer is a code, so no colour
is ever a candidate output. Two tasks with identical prompts except one question token:
  final-code  "What is the code of the potion's colour at the end?"    (needs z at depth 2)
  start-code  "What is the code of the potion's colour at the start?"  (does not)

  python bank.py                 # build dev (300) + test (150) tables at depth 2 and depth 1, check alignment, write files

Test-bank items are rendered here only for tokenisation / alignment checks; nothing else may read them until a
confirmatory design note is committed.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import c009 as C                                    # noqa: E402

REGEN_MAX = 200                                       # draws per table slot before giving up


# --------------------------------------------------------------------------------------------- generation
def trajectory(table, s, depth):
    """[z, f] at depth 2, [f] at depth 1 (None if a colour leaves the table)."""
    out, c = [], s
    for _ in range(depth):
        if c not in table:
            return None
        c = table[c]
        out.append(c)
    return out


def roles(table, s, depth):
    """(z, f) with z := f at depth 1; None unless s is a valid start (s, z, f pairwise distinct)."""
    t = trajectory(table, s, depth)
    if t is None:
        return None
    z, f = (t[0], t[1]) if depth == 2 else (t[0], t[0])
    if depth == 2 and len({s, z, f}) != 3:
        return None
    if depth == 1 and s == f:
        return None
    return z, f


def partner_ok(m, p, depth):
    """Design §2.2 constraints for partner p of member m (each a dict with s, z, f)."""
    if p["s"] == m["s"]:
        return False
    if depth == 2:
        return (p["z"] != m["z"] and p["f"] != m["f"] and p["z"] not in (m["f"], p["f"]) and m["z"] != p["f"]
                and p["s"] not in (m["z"], m["f"]))
    return p["f"] != m["f"] and p["s"] != m["f"]                                     # depth 1: z := f


def draw_table(rng):
    """10 rules over a random 10 of the 12 colours; destinations within the table's colours; no fixed points; a code
    key over all 12 colours (distinct codes in 10-99), listed in random order."""
    src = rng.sample(list(C.COLOURS), C.N_RULES)
    table = {c: rng.choice([d for d in src if d != c]) for c in src}
    rule_order = list(src)
    rng.shuffle(rule_order)
    codes = dict(zip(C.COLOURS, rng.sample(range(C.CODE_RANGE[0], C.CODE_RANGE[1] + 1), len(C.COLOURS))))
    code_order = list(C.COLOURS)
    rng.shuffle(code_order)
    return table, rule_order, codes, code_order


def rule_sentence(c, d):
    art = "An" if c[0] in "aeiou" else "A"
    return f"{art} {c} potion turns {d} when stirred."


def prompt_text(item, start, task):
    lines = [C.INSTRUCTION, C.PROBLEM]
    lines += [rule_sentence(c, item["table"][c]) for c in item["rule_order"]]
    lines.append("Each colour has a code: " + ", ".join(f"{c} {item['codes'][c]}" for c in item["code_order"]) + ".")
    lines += [C.STIR[item["depth"]], C.START_FMT.format(start), C.QUESTION[task]]
    return "\n".join(lines)


def member_dict(item, s):
    z, f = roles(item["table"], s, item["depth"])
    return {"start": s, "z": z, "f": f, "answer": {"final": item["codes"][f], "start": item["codes"][s]}}


def draw_item(rng, name, split, depth):
    """Draw tables until one has a valid member with >= 1 valid partner; returns (item, n_rejected)."""
    for rej in range(REGEN_MAX):
        table, rule_order, codes, code_order = draw_table(rng)
        valid = {s: roles(table, s, depth) for s in rule_order}
        valid = {s: {"s": s, "z": v[0], "f": v[1]} for s, v in valid.items() if v is not None}
        cand = {s: [p for p in sorted(valid) if partner_ok(valid[s], valid[p], depth)] for s in sorted(valid)}
        cand = {s: v for s, v in cand.items() if v}
        if not cand:
            continue
        s = rng.choice(sorted(cand))
        item = {"name": name, "split": split, "depth": depth, "table": table, "rule_order": rule_order,
                "codes": codes, "code_order": code_order}
        item["member"] = member_dict(item, s)
        item["partners"] = [member_dict(item, p) for p in cand[s]]
        item["prompts"] = {t: {"member": prompt_text(item, s, t), "partners": [prompt_text(item, p, t) for p in cand[s]]}
                           for t in C.TASKS}
        return item, rej
    raise RuntimeError(f"{name}: no valid table in {REGEN_MAX} draws")


def build(split, depth, n, seed):
    rng = random.Random(f"{seed}|{split}|depth{depth}")
    items, rejected = [], 0
    for i in range(n):
        it, rej = draw_item(rng, f"brewcode-{split}-d{depth}-{i:04d}", split, depth)
        items.append(it)
        rejected += rej
    if split == "dev":                                                        # 5 folds by table (seeded)
        order = list(range(n))
        random.Random(f"{seed}|folds|depth{depth}").shuffle(order)
        for k, i in enumerate(order):
            items[i]["fold"] = k % C.N_FOLDS
    return items, rejected


# --------------------------------------------------------------------------------------------- rendering
def render(tok, text, name="brewcode"):
    import wsb
    ids = wsb.render_text(tok, "brew", text)
    return wsb.Rendered(family="brew", item=name, text=text, render="chat_prefill", ids=ids,
                        tokens=wsb._display(tok, ids), decoded=[tok.decode([i]) for i in ids])


def _find(dec, seq, start=0):
    hits = [i for i in range(start, len(dec) - len(seq) + 1) if dec[i:i + len(seq)] == seq]
    return hits


def cells(r, start_colour=None):
    """Positions of the start token, the 13-token question span and the three emit cells in a render (asserted)."""
    dec = r.decoded
    T = len(dec)
    assert dec[T - 1] == ":" and dec[T - 2] == "Answer", f"{r.item}: last tokens {dec[-2:]}"
    think = [j for j in range(1, T) if dec[j - 1] == "</think>" and dec[j] == "\n\n"]
    assert len(think) == 1, (r.item, think)
    asst = [j for j in range(2, T) if dec[j - 1] == "assistant" and dec[j] == "\n" and dec[j - 2] == "<|im_start|>"]
    assert len(asst) == 1, (r.item, asst)
    q0 = [i for i in range(T - C.QUESTION_TOKENS) if dec[i] == "What" and dec[i - 1] == "\n"
          and "".join(dec[i:i + C.QUESTION_TOKENS]) in C.QUESTION.values()]
    assert len(q0) == 1, (r.item, q0)
    q = list(range(q0[0], q0[0] + C.QUESTION_TOKENS))
    assert dec[q[-1]] == "?" and dec[q[-1] + 1] == "<|im_end|>", (r.item, dec[q[-1]:q[-1] + 2])
    st = [j for j in range(2, T) if dec[j - 2] == " starts" and dec[j - 1] == " out"]
    assert len(st) == 1 and dec[st[0] + 1] == ".", (r.item, st)
    if start_colour is not None:
        assert dec[st[0]] == " " + start_colour, (r.item, dec[st[0]], start_colour)
    out = {"start": st[0], "question": q, "emit_asst": asst[0], "emit_think": think[0], "emit_colon": T - 1}
    assert out["start"] < q[0] < out["emit_asst"] < out["emit_think"] < out["emit_colon"]
    return out


def check_item(tok, it):
    """Render every start under both tasks; assert member/partner alignment (only the start token differs) and task
    alignment (only the question's end/start token differs); return the cell positions (shared by every render)."""
    import wsb
    rs = {t: [render(tok, it["prompts"][t]["member"], it["name"])] +
          [render(tok, x, it["name"]) for x in it["prompts"][t]["partners"]] for t in C.TASKS}
    starts = [it["member"]["start"]] + [p["start"] for p in it["partners"]]
    pos = None
    for t in C.TASKS:
        for r, s in zip(rs[t], starts):
            c = cells(r, s)
            pos = pos or c
            assert c == pos, (it["name"], t, c, pos)
        for r in rs[t][1:]:
            diff = wsb.assert_aligned(rs[t][0], r, [pos["start"]])
            assert diff == [pos["start"]], (it["name"], t, diff)
    for rf, rst in zip(rs["final"], rs["start"]):
        diff = wsb.assert_aligned(rf, rst, [pos["question"][11]])
        assert diff == [pos["question"][11]], (it["name"], diff)
        assert rf.decoded[pos["question"][11]] == " end" and rst.decoded[pos["question"][11]] == " start"
    return pos, len(rs["final"][0])


# --------------------------------------------------------------------------------------------- answer parsing
_NUM = re.compile(r"\d+")


def parse_code(text):
    """The first number in the decode, if it has exactly two digits; else None (a 3+ digit or 1-digit number is wrong)."""
    m = _NUM.search(text)
    if m is None or len(m.group(0)) != 2:
        return None
    return int(m.group(0))


def code_correct(text, answer):
    return parse_code(text) == int(answer)


COLOUR_RE = re.compile(r"\b(" + "|".join(C.COLOURS) + r"|grey)\b", re.I)


def names_colour(text):
    return bool(COLOUR_RE.search(text))


# --------------------------------------------------------------------------------------------- io
def bank_path(split, depth):
    return C.BANK_DIR / f"{split}_depth{depth}.json"


def load(split, depth):
    d = json.loads(bank_path(split, depth).read_text())
    return d["items"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="2,1")
    a = ap.parse_args()
    import registry as R
    tok = R.make_tokenizer()
    man = {"design": "docs/design_specs/008_009_trackb_brewcode.md (Part 2)", "seed": C.BANK_SEED, "files": {},
           "counts": {}}
    for depth in [int(x) for x in a.depths.split(",")]:
        for split, n, seed in (("dev", C.N_DEV, C.BANK_SEED), ("test", C.N_TEST, C.BANK_SEED + 1)):
            items, rej = build(split, depth, n, seed)
            lens = []
            for it in items:
                pos, T = check_item(tok, it)
                it["cells"] = pos
                it["n_tokens"] = T
                lens.append(T)
            path = bank_path(split, depth)
            C.dump(path, {"split": split, "depth": depth, "seed": seed, "n_items": len(items), "items": items})
            npairs = sum(len(it["partners"]) for it in items)
            man["files"][path.name] = sha256(path)
            man["counts"][f"{split}_depth{depth}"] = {
                "tables": len(items), "candidate_pairs": npairs, "renders_per_task": npairs + len(items),
                "rejected_table_draws": rej, "partners_per_table": {str(k): sum(len(it["partners"]) == k for it in items)
                                                                    for k in sorted({len(it["partners"]) for it in items})},
                "n_tokens_min_max": [min(lens), max(lens)]}
            print(split, depth, man["counts"][f"{split}_depth{depth}"], flush=True)
    C.dump(C.RES / "analysis" / "bank_manifest.json", man)
    print("wrote", C.RES / "analysis" / "bank_manifest.json")


if __name__ == "__main__":
    main()
