import pytest

from cells import Cell, read_jsonl, write_jsonl


def test_roundtrip(tmp_path):
    c = Cell(item="brew-s2-r10d2-0000", pair_id="brew-s2-r10d2-0000__gray", position=164, layer=42,
             family="brew", half="A", split="dev")
    assert Cell.from_json(c.to_json()) == c
    assert Cell.from_dict(c.to_dict()) == c
    p = tmp_path / "cells.jsonl"
    write_jsonl([c, c], p)
    assert read_jsonl(p) == [c, c]
    assert c.key == "brew|brew-s2-r10d2-0000__gray|brew-s2-r10d2-0000|p164|L42"


@pytest.mark.parametrize("bad", [dict(layer=64), dict(half="C"), dict(split="train"), dict(family="arith"),
                                 dict(position=-1), dict(item="")])
def test_validation(bad):
    base = dict(item="x", pair_id="x__y", position=0, layer=42, family="brew", half="A", split="dev")
    base.update(bad)
    with pytest.raises(ValueError):
        Cell(**base)


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Cell.from_dict(dict(item="x", pair_id="x", position=0, layer=0, family="brew", half="A", split="dev", z="green"))
