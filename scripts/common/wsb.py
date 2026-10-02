"""WorkspaceBench banks: loading (sha-checked) and rendering with the pinned tokenizer, plus position assertions.

Renders reproduce WSB's `wsbench.produce.render` exactly (commit 92d763e):
  brew        chat_prefill  user turn = item prompt, assistant prefill "Answer:", thinking off
  chain       chat          user turn = item prompt, thinking off (render ends at the assistant onset)
  relational  bare          the cloze text, no chat template, no special tokens
  multihop    plain         the prompt text, no chat template
  role_bound  chat_summarize  stimulus + "\n\nSummarize the preceding story in one sentence."
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from registry import REGISTRY

WSB_DIR = Path(REGISTRY["paths"]["wsb_clone"])
BANK_KEYS = {"brew": "brew_intermediates", "chain": "chain_intermediates", "relational": "relational_multihop",
             "multihop": "multihop", "role_bound": "role_bound_association"}
RENDER_KIND = {"brew": "chat_prefill", "chain": "chat", "relational": "bare", "multihop": "plain",
               "role_bound": "chat_summarize"}
BREW_PREFILL = "Answer:"
SUMMARIZE_SUFFIX = "\n\nSummarize the preceding story in one sentence."
BREW_REGION_ORDER = ("stir", "start", "question", "emit_asst", "emit_think", "emit_colon")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bank(family):
    """Return (header dict, list of items). Asserts the bank file's sha256 against the registry."""
    key = BANK_KEYS[family]
    b = REGISTRY["wsb"]["banks"][key]
    path = WSB_DIR / b["file"]
    got = _sha256(path)
    assert got == b["sha256"], f"{key} bank sha256 mismatch: {got} (WSB clone not at {REGISTRY['wsb']['commit']}?)"
    d = json.loads(path.read_text())
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if k != "items"}, d["items"]
    return {}, d


def item_id(family, it):
    return it["name"] if "name" in it else it["id"]


@dataclass
class Rendered:
    family: str
    item: str
    text: str
    render: str
    ids: list
    tokens: list            # display spelling (byte-level BPE with Ġ/▁ shown as a space)
    decoded: list           # tokenizer.decode([id]) per token
    read_positions: list = field(default_factory=list)
    regions: dict = field(default_factory=dict)   # {position: region} where the bank defines regions

    def __len__(self):
        return len(self.ids)


def _display(tok, ids):
    return [t.replace("Ġ", " ").replace("▁", " ") if isinstance(t, str) else "" for t in tok.convert_ids_to_tokens(ids)]


def _chat_ids(tok, messages, prefill=""):
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = list(out["input_ids"] if hasattr(out, "keys") else out)
    if prefill:
        ids += tok(prefill, add_special_tokens=False)["input_ids"]
    return ids


def render_text(tok, family, text):
    """Token ids for one prompt text under the family's WSB render."""
    kind = RENDER_KIND[family]
    if kind == "chat_prefill":
        ids = _chat_ids(tok, [{"role": "user", "content": text}], BREW_PREFILL)
    elif kind == "chat":
        ids = _chat_ids(tok, [{"role": "user", "content": text}])
    elif kind == "chat_summarize":
        ids = _chat_ids(tok, [{"role": "user", "content": text + SUMMARIZE_SUFFIX}])
    elif kind in ("bare", "plain"):
        ids = tok(text, add_special_tokens=False)["input_ids"]
    else:
        raise ValueError(kind)
    return list(ids)


def item_text(family, it):
    if family in ("brew", "chain", "multihop"):
        return it["prompt"]
    if family in ("relational", "role_bound"):
        return it["stimulus"]
    raise ValueError(family)


def render_item(tok, family, it, text=None):
    """Render a bank item (or an alternative `text` for the same item, e.g. a partner prompt)."""
    text = item_text(family, it) if text is None else text
    ids = render_text(tok, family, text)
    r = Rendered(family=family, item=item_id(family, it), text=text, render=RENDER_KIND[family], ids=ids,
                 tokens=_display(tok, ids), decoded=[tok.decode([i]) for i in ids])
    if family == "brew":
        r.read_positions = [int(p) for p in it["eval_positions"]]
        r.regions = {int(p): reg for p, reg in it["regions"].items()}
    elif family == "chain":
        what = [i for i, t in enumerate(r.tokens) if t == " What"]
        r.read_positions = list(range(what[0], len(ids))) if what else []
        r.regions = {p: ("final" if p == len(ids) - 1 else "question") for p in r.read_positions}
    elif family == "relational":
        r.read_positions = [len(ids) - 1]
        r.regions = {len(ids) - 1: "blank"}
    return r


# ----------------------------------------------------------------------------------------- assertions
def brew_region_text(r, region):
    pos = sorted(p for p, g in r.regions.items() if g == region)
    return "".join(r.decoded[p] for p in pos)


def assert_brew_positions(r, it):
    """WSB's 24 pinned brew cells must land on the intended tokens under our pinned render."""
    assert len(r.read_positions) == 24, f"{r.item}: {len(r.read_positions)} pinned positions"
    assert max(r.read_positions) == len(r) - 1, f"{r.item}: last pinned cell is not the final token"
    want = {"stir": "You stir the potion twice.", "start": f"The potion starts out {it['start']}.",
            "question": "What colour is the potion at the end?"}
    for reg, txt in want.items():
        got = brew_region_text(r, reg)
        assert got == txt, f"{r.item}: region {reg} decodes {got!r}, expected {txt!r}"
    by = {g: p for p, g in r.regions.items() if g.startswith("emit_")}
    assert r.decoded[by["emit_asst"]] == "\n" and r.decoded[by["emit_asst"] - 1] == "assistant", \
        f"{r.item}: emit_asst mislocated"
    assert r.decoded[by["emit_think"]] == "\n\n" and r.decoded[by["emit_think"] - 1] == "</think>", \
        f"{r.item}: emit_think mislocated"
    assert r.decoded[by["emit_colon"]] == ":" and r.decoded[by["emit_colon"] - 1] == "Answer", \
        f"{r.item}: emit_colon mislocated"
    counts = {g: sum(1 for x in r.regions.values() if x == g) for g in BREW_REGION_ORDER}
    assert counts == {"stir": 6, "start": 6, "question": 9, "emit_asst": 1, "emit_think": 1, "emit_colon": 1}, counts
    return True


def assert_chain_positions(r):
    assert r.read_positions, f"{r.item}: no ' What' token"
    last = len(r) - 1
    assert r.decoded[last] == "\n\n" and r.decoded[last - 1] == "</think>", \
        f"{r.item}: final token is {r.decoded[last]!r} after {r.decoded[last - 1]!r}"
    return True


def assert_relational_positions(r):
    assert r.decoded[-1] == "'s", f"{r.item}: blank token is {r.decoded[-1]!r}"
    return True


def assert_aligned(r1, r2, allowed_diff):
    """Token-aligned pair: equal length, and ids differ only at `allowed_diff` positions."""
    assert len(r1) == len(r2), f"{r1.item} vs {r2.item}: lengths {len(r1)} != {len(r2)}"
    diff = [i for i, (a, b) in enumerate(zip(r1.ids, r2.ids)) if a != b]
    assert set(diff) <= set(allowed_diff), f"{r1.item}: ids differ at {diff}, allowed {allowed_diff}"
    return diff
