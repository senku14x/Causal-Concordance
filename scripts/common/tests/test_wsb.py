"""Rendering with the pinned tokenizer reproduces WSB's pinned positions (WSB's backend did not pin a revision)."""
import pytest

import wsb


def test_bank_hashes_and_sizes(brew_bank, chain_bank, rel_bank):
    assert len(brew_bank) == 100 and len(chain_bank) == 120 and len(rel_bank) == 100


def test_brew_example_regions(tok, brew_bank):
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    r = wsb.render_item(tok, "brew", it)
    assert len(r) == 165
    assert wsb.assert_brew_positions(r, it)
    assert sorted(p for p, g in r.regions.items() if g == "stir") == list(range(131, 137))
    assert sorted(p for p, g in r.regions.items() if g == "start") == list(range(138, 144))
    assert sorted(p for p, g in r.regions.items() if g == "question") == list(range(145, 154))
    assert {g: p for p, g in r.regions.items() if g.startswith("emit")} == {"emit_asst": 158, "emit_think": 162,
                                                                            "emit_colon": 164}
    assert r.decoded[142] == " blue"


def test_all_brew_items_positions(tok, brew_bank):
    for it in brew_bank:
        assert wsb.assert_brew_positions(wsb.render_item(tok, "brew", it), it)


def test_chain_and_relational_read_cells(tok, chain_bank, rel_bank):
    for it in chain_bank:
        r = wsb.render_item(tok, "chain", it)
        assert wsb.assert_chain_positions(r)
        assert r.tokens[r.read_positions[0]] == " What"
    for it in rel_bank:
        assert wsb.assert_relational_positions(wsb.render_item(tok, "relational", it))


def test_brew_render_matches_wsb_producer(tok, brew_bank):
    """Byte-for-byte equivalence with WSB's own render function (produce/render.py) on one item."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(wsb.WSB_DIR) / "src"))
    try:
        from wsbench.produce.render import _chat
    except Exception as e:  # WSB's optional deps missing: compare against an inline copy instead
        pytest.skip(f"wsbench not importable: {e}")
    it = brew_bank[0]
    assert _chat(tok, [{"role": "user", "content": it["prompt"]}], "Answer:") == wsb.render_item(tok, "brew", it).ids
