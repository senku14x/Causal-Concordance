import pytest

import partners as P
import wsb


def test_brew_trajectory_matches_bank(brew_bank):
    for it in brew_bank:
        t = P.brew_trajectory(it["table"], it["start"], it["depth"])
        assert t[0] == it["intermediates"][0] and t[-1] == it["answer"]


def test_brew_hand_checked_partner(tok, brew_bank):
    """brew-s2-r10d2-0000: blue -> green -> black. Partner start gray: gray -> blue -> green.
    z' = blue != green, a' = green != black, z' not in {black, green}, z = green not in {a' = green}? -> excluded.
    Partner start pink: pink -> gray -> blue: z' = gray, a' = blue; all constraints hold."""
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    cands = {p.p.inp: p for p in P.brew_candidates(it)}
    assert "gray" not in cands            # z == a' (green) violates "z not in {a'}"
    p = cands["pink"]
    assert (p.m.z, p.m.a, p.p.z, p.p.a) == ("green", "black", "gray", "blue")
    r_m, r_p, diff = P.token_align(tok, p, it)
    assert len(r_m) == len(r_p) == 165 and diff == [142]
    assert r_m.decoded[142] == " blue" and r_p.decoded[142] == " pink"
    assert r_m.read_positions == r_p.read_positions


def test_brew_constraints_hold_for_all_candidates(brew_bank):
    for it in brew_bank:
        for p in P.brew_candidates(it):
            z, a, z2, a2 = p.m.z, p.m.a, p.p.z, p.p.a
            assert z2 != z and a2 != a and z2 not in (a, a2) and z != a2 and p.p.inp != p.m.inp


def test_chain_step_functions_reproduce_bank(chain_bank):
    for it in chain_bank:
        v = P.chain_values(it["steps"], it["start"])
        assert v[:-1] == it["intermediates"] and v[-1] == it["answer"], it["name"]


def test_chain_hand_checked_partner(tok, chain_bank):
    """chain-w2-d3p-0408: 19 -> 9 -> 4 -> 2 (halve down x3). Partner start 14: 14 -> 7 -> 3 -> 1."""
    it = next(x for x in chain_bank if x["name"] == "chain-w2-d3p-0408")
    cands = {p.p.inp: p for p in P.chain_candidates(it)}
    p = cands[14]
    assert (p.m.z, p.m.a, p.p.z, p.p.a) == ((9, 4), 2, (7, 3), 1)
    r_m, r_p, diff = P.token_align(tok, p, it)
    assert len(r_m) == len(r_p)
    assert [r_m.decoded[i] for i in diff] == ["9"] and [r_p.decoded[i] for i in diff] == ["4"]
    assert r_m.read_positions == r_p.read_positions


def test_relational_pairs_aligned(tok, rel_bank):
    by = {x["id"]: x for x in rel_bank}
    pairs = P.relational_pairs(rel_bank)
    assert len(pairs) == 50
    for p in pairs:
        r_m, r_p, diff = P.token_align(tok, p, by[p.item])
        assert len(diff) == 2


@pytest.mark.parametrize("family,text,answer,ok", [
    ("brew", " black", "black", True), ("brew", " Black.", "black", True), ("brew", " green", "black", False),
    ("chain", "2", 2, True), ("chain", "12", 2, False), ("chain", "The answer is 2", 2, True),
    ("relational", " landlord's sibling.", "landlord's sibling", True),
    ("relational", " sibling's landlord", "landlord's sibling", False)])
def test_answer_parsing(family, text, answer, ok):
    assert P.is_correct(family, text, answer) is ok


def test_brew_substitution_hand_checked(tok, brew_bank):
    """brew-s2-r10d2-0000 (blue -> green -> black): gray is in the table and off the trajectory (key of 'gray -> blue',
    output of 'pink -> gray'); gold and teal are absent. gray -> teal changes exactly those two tokens, keeps the
    trajectory and the answer, and sits inside the rule table."""
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    subs = {(p.meta["c"], p.meta["c2"]): p for p in P.brew_substitutions(tok, it)}
    assert set(c2 for _, c2 in subs) <= {"gold", "teal"}
    assert not any(c in ("blue", "green", "black", "orange") for c, _ in subs)
    p = subs[("gray", "teal")]
    r_m, r_p, diff = P.token_align(tok, p, it)
    assert len(diff) == 2 and [r_m.decoded[i] for i in diff] == [" gray", " gray"] and [r_p.decoded[i] for i in diff] == [" teal", " teal"]
    assert "A teal potion turns blue when stirred." in p.p.text and "A pink potion turns teal when stirred." in p.p.text
    assert (p.p.z, p.p.a) == ("green", "black") and max(diff) < 131


def test_every_brew_item_has_a_substitution(tok, brew_bank):
    for it in brew_bank:
        p = P.brew_substitution(tok, it)
        assert p is not None, it["name"]
        assert P.brew_substitution(tok, it).pair_id == p.pair_id          # deterministic by seed
