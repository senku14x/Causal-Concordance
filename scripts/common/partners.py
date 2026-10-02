"""Counterfactual partner constructors (brew, chain, relational) and the competence gate.

Constraints (AMENDMENT_01 §1): z' != z, a' != a, z' not in {a, a'}, z not in {a'}; the changed input is a
single token in context (brew colours) or keeps its digit count (chain); the two renders have identical
token counts and differ only at the changed input tokens. The gate: greedy correct and >= 9/10 correct at
T = 0.8 (top_p 0.95, max 16 new tokens, EOS suppressed as in WSB's gate config), for BOTH pair members.
"""
import math
import re
from dataclasses import asdict, dataclass, field

import wsb

# ---------------------------------------------------------------------------------------------- brew
BREW_START_FMT = "The potion starts out {}."


def brew_trajectory(table, start, depth=2):
    """Colours after each stir: [z (after stir 1), ..., answer]. None if a colour leaves the table."""
    out, c = [], start
    for _ in range(depth):
        if c not in table:
            return None
        c = table[c]["stir"]
        out.append(c)
    return out


@dataclass
class Member:
    item: str               # bank item id this member belongs to (partner shares the item's table/steps)
    text: str               # prompt text (user turn, or bare text for relational)
    inp: object             # the changed input: start colour / start number / (hop1, hop2)
    z: object               # latent: brew intermediate colour; chain tuple of intermediates; relational hop order
    a: object               # answer: colour / int / "hop1's hop2"


@dataclass
class Pair:
    pair_id: str
    family: str
    item: str
    m: Member               # the bank member (original item)
    p: Member               # the constructed partner
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return {"pair_id": self.pair_id, "family": self.family, "item": self.item,
                "m": asdict(self.m), "p": asdict(self.p), "meta": self.meta}


def brew_candidates(it):
    """All same-table partners satisfying the AMENDMENT_01 constraints (before tokenisation and gating)."""
    table, s = it["table"], it["start"]
    traj = brew_trajectory(table, s, it.get("depth", 2))
    assert traj is not None and traj[0] == it["intermediates"][0] and traj[-1] == it["answer"], it["name"]
    z, a = traj[0], traj[-1]
    base = BREW_START_FMT.format(s)
    assert it["prompt"].count(base) == 1, f"{it['name']}: start sentence not unique"
    out = []
    for s2 in sorted(table):
        if s2 == s:
            continue
        t2 = brew_trajectory(table, s2, it.get("depth", 2))
        if t2 is None:
            continue
        z2, a2 = t2[0], t2[-1]
        if z2 == z or a2 == a or z2 in (a, a2) or z == a2:
            continue
        text2 = it["prompt"].replace(base, BREW_START_FMT.format(s2))
        out.append(Pair(pair_id=f"{it['name']}__{s2}", family="brew", item=it["name"],
                        m=Member(it["name"], it["prompt"], s, z, a), p=Member(it["name"], text2, s2, z2, a2),
                        meta={"start_is_z_or_a_of_member": s2 in (z, a), "partner_start_in_member_traj": s2 in traj,
                              "off_trajectory_m": sorted(set(table) - {s, z, a}),
                              "off_trajectory_p": sorted(set(table) - {s2, z2, a2})}))
    return out


# --------------------------------------------------------------------------------------------- chain
_STEP_RE = [
    (re.compile(r"^Add (\d+)\.$"), lambda k: (lambda x: x + k)),
    (re.compile(r"^Subtract (\d+)\.$"), lambda k: (lambda x: x - k)),
    (re.compile(r"^Double it\.$"), lambda _: (lambda x: 2 * x)),
    (re.compile(r"^Multiply it by (\d+)\.$"), lambda k: (lambda x: k * x)),
    (re.compile(r"^Halve it, rounding down\.$"), lambda _: (lambda x: math.floor(x / 2))),
    (re.compile(r"^Halve it, rounding up\.$"), lambda _: (lambda x: math.ceil(x / 2))),
]


def chain_step_fn(step_text):
    for rx, mk in _STEP_RE:
        m = rx.match(step_text.strip())
        if m:
            return mk(int(m.group(1)) if m.groups() else None)
    raise ValueError(f"unknown chain step {step_text!r}")


def chain_values(steps, start):
    """[intermediate_1, ..., answer] for a start number."""
    vals, x = [], start
    for s in steps:
        x = chain_step_fn(s)(x)
        vals.append(x)
    return vals


CHAIN_START_FMT = "The starting number is {}."


def _numerals(text):
    return {int(x) for x in re.findall(r"\d+", text)}


def chain_single_token_item(it):
    return all(isinstance(v, int) and 0 <= v <= 9 for v in it["intermediates"])


def chain_candidates(it, start_range=range(0, 100)):
    """Same steps, different start with the same digit count; single-digit, non-negative intermediates;
    WSB's own chain constraints (no intermediate is a numeral of the prompt; start, intermediates and answer
    pairwise distinct) applied to the partner; AMENDMENT_01's distinctness constraints applied elementwise."""
    steps, s = it["steps"], it["start"]
    vals = chain_values(steps, s)
    assert vals[:-1] == it["intermediates"] and vals[-1] == it["answer"], it["name"]
    z, a = tuple(vals[:-1]), vals[-1]
    base = CHAIN_START_FMT.format(s)
    assert it["prompt"].count(base) == 1, f"{it['name']}: start sentence not unique"
    out = []
    for s2 in start_range:
        if s2 == s or len(str(s2)) != len(str(s)):
            continue
        v2 = chain_values(steps, s2)
        z2, a2 = tuple(v2[:-1]), v2[-1]
        if not all(0 <= v <= 9 for v in z2) or a2 < 0:
            continue
        if any(x == y for x, y in zip(z, z2)) or a2 == a or any(v in (a, a2) for v in z2) or a2 in z:
            continue
        text2 = it["prompt"].replace(base, CHAIN_START_FMT.format(s2))
        if any(v in _numerals(text2) for v in z2):      # WSB: no intermediate appears as a numeral in the prompt
            continue
        seq = [s2, *z2, a2]
        if len(set(seq)) != len(seq):                    # WSB: start, intermediates, answer pairwise distinct
            continue
        out.append(Pair(pair_id=f"{it['name']}__{s2}", family="chain", item=it["name"],
                        m=Member(it["name"], it["prompt"], s, z, a), p=Member(it["name"], text2, s2, z2, a2),
                        meta={"steps": steps}))
    return out


# ---------------------------------------------------------------------------------------- relational
def relational_pairs(items):
    """The bank's own a/b members (the two relation words swapped). Kept only if token-aligned (checked at
    render time by `token_align`)."""
    by = {}
    for it in items:
        by.setdefault(it["scene"], {})[it["id"][-1]] = it
    out = []
    for scene, d in sorted(by.items()):
        if set(d) != {"a", "b"}:
            continue
        A, B = d["a"], d["b"]
        out.append(Pair(pair_id=f"rel-{scene}", family="relational", item=A["id"],
                        m=Member(A["id"], A["stimulus"], (A["hop1"], A["hop2"]), (A["hop1"], A["hop2"]), A["gold"]),
                        p=Member(B["id"], B["stimulus"], (B["hop1"], B["hop2"]), (B["hop1"], B["hop2"]), B["gold"]),
                        meta={"names": A["names"]}))
    return out


# ------------------------------------------------------------------------------------ token alignment
def changed_positions(r1, r2):
    return [i for i, (a, b) in enumerate(zip(r1.ids, r2.ids)) if a != b]


def token_align(tok, pair, bank_item):
    """Render both members; return (r_m, r_p, diff) if token counts are equal, else None.
    For brew the only allowed difference is the start-colour token; for chain the start digits;
    for relational the two relation-word tokens."""
    fam = pair.family
    item_m = bank_item if fam != "relational" else bank_item
    r_m = wsb.render_item(tok, fam, item_m, text=pair.m.text)
    r_p = wsb.render_item(tok, fam, item_m, text=pair.p.text)
    if len(r_m) != len(r_p):
        return None
    diff = changed_positions(r_m, r_p)
    if fam == "brew":
        start_pos = sorted(p for p, g in r_m.regions.items() if g == "start")
        if len(diff) != 1 or diff[0] not in start_pos:
            return None
    elif fam == "chain":
        if not diff or len(diff) > len(str(pair.m.inp)):
            return None
    elif fam == "relational":
        if len(diff) != 2:
            return None
    return r_m, r_p, diff


# ----------------------------------------------------------------------------------------------- gate
GATE = {"greedy": True, "k": 10, "min_correct": 9, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 16,
        "suppress": ["<|im_end|>", "<|endoftext|>"]}


def parse_answer(family, text):
    t = text.strip()
    if family == "brew":
        m = re.match(r"[^A-Za-z]*([A-Za-z]+)", t)
        return m.group(1).lower() if m else None
    if family == "chain":
        m = re.search(r"-?\d+", t)
        return int(m.group(0)) if m else None
    if family == "relational":
        return re.sub(r"\s+", " ", t.replace("’", "'")).lower()
    raise ValueError(family)


def is_correct(family, text, answer):
    got = parse_answer(family, text)
    if family == "relational":
        return got is not None and got.startswith(str(answer).lower())
    return got == answer


def gate_member(model, tok, ids, family, answer, seed, gate=GATE):
    """Greedy + k samples on one rendered prompt. Returns a dict with the decodes and the pass flag."""
    import torch
    sup = [tok.convert_tokens_to_ids(t) for t in gate["suppress"]]
    sup = [i for i in sup if isinstance(i, int) and i >= 0]
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=gate["max_new_tokens"],
                           suppress_tokens=sup, pad_token_id=tok.eos_token_id)
        greedy = tok.decode(g[0, len(ids):], skip_special_tokens=True)
        torch.manual_seed(seed)
        s = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=gate["temperature"],
                           top_p=gate["top_p"], top_k=0, max_new_tokens=gate["max_new_tokens"],
                           num_return_sequences=gate["k"], suppress_tokens=sup, pad_token_id=tok.eos_token_id)
        samples = [tok.decode(row[len(ids):], skip_special_tokens=True) for row in s]
    ok_g = is_correct(family, greedy, answer)
    n_ok = sum(is_correct(family, t, answer) for t in samples)
    return {"greedy": greedy, "greedy_ok": ok_g, "samples": samples, "n_correct": n_ok, "k": gate["k"],
            "pass": bool(ok_g and n_ok >= gate["min_correct"])}


def gate_pair(model, tok, pair, r_m, r_p, seed=20261002):
    gm = gate_member(model, tok, r_m.ids, pair.family, pair.m.a, seed)
    gp = gate_member(model, tok, r_p.ids, pair.family, pair.p.a, seed + 1)
    return {"pair_id": pair.pair_id, "m": gm, "p": gp, "pass": gm["pass"] and gp["pass"]}
