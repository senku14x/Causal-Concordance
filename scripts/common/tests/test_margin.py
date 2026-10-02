"""Margin scorer tests: sign implied by greedy; multi-token spellings scored at every token; writes clamped on every
forward of a multi-token spelling."""
import pytest
import torch

import engine as E
import margin as MG
import partners as P
import wsb

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def eng(model):
    return E.Engine(model, fp32_from=0)


def test_spellings(tok):
    assert MG.spellings(tok, "brew", "black") == [(3565,), (5068,)]
    assert MG.spellings(tok, "brew", "teal")[1] == (2568, 278)            # " Teal" = " Te" + "al"
    assert MG.spellings(tok, "chain", 27) == [(17, 22)]
    assert len(MG.spellings(tok, "relational", "landlord's sibling")[0]) == 3


def test_sign_matches_greedy(model, tok, eng, brew_bank, chain_bank):
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    pair = {p.p.inp: p for p in P.brew_candidates(it)}["pink"]
    r_m, r_p, _ = P.token_align(tok, pair, it)
    ci = next(x for x in chain_bank if x["name"] == "chain-w2-d3p-0408")
    cpair = {p.p.inp: p for p in P.chain_candidates(ci)}[14]
    c_m, c_p, _ = P.token_align(tok, cpair, ci)
    for fam, pr, rm, rp in (("brew", pair, r_m, r_p), ("chain", cpair, c_m, c_p)):
        Sa, Sb = MG.spellings(tok, fam, pr.m.a), MG.spellings(tok, fam, pr.p.a)
        for r, gold, want_neg in ((rm, pr.m.a, True), (rp, pr.p.a, False)):
            g = P.gate_member(model, tok, r.ids, fam, gold, seed=1, gate=dict(P.GATE, k=1))
            if not g["greedy_ok"]:
                pytest.skip(f"{fam} greedy wrong on {r.text[-60:]!r}: {g['greedy']!r}")
            m = MG.score(eng, r.ids, Sa, Sb)["m"]
            assert (m < 0) == want_neg, (fam, gold, m)


def test_multitoken_teacher_forcing_every_token(tok, eng, brew_bank):
    """lp(" Teal") from one teacher-forced forward equals lp(" Te" | prompt) + lp("al" | prompt + " Te") from two
    incremental forwards (to numerical noise), and the second token is not dropped."""
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    r = wsb.render_item(tok, "brew", it)
    s = MG.spellings(tok, "brew", "teal")[1]
    sc = MG.score(eng, r.ids, [s], [MG.spellings(tok, "brew", "teal")[0]])
    per = sc["spell"][s]["per_token"]
    assert len(per) == 2
    run1 = eng.forward(r.ids)
    run2 = eng.forward(r.ids + [s[0]])
    lp1 = float(eng.logprobs(run1, [len(r) - 1])[0, s[0]])
    lp2 = float(eng.logprobs(run2, [len(r)])[0, s[1]])
    # equal up to the sequence-length noise of the bf16-matmul regime (measured <= 0.25 nats, see STATUS.md)
    assert abs(per[0] - lp1) < 0.3 and abs(per[1] - lp2) < 0.3, (per, lp1, lp2)
    assert sc["spell"][s]["lp"] == pytest.approx(per[0] + per[1])
    # an off-by-one in the teacher-forcing index would read the other token's row: far outside that noise
    wrong = float(eng.logprobs(run2, [len(r) - 1])[0, s[1]])
    assert abs(wrong - per[1]) > 1.0, (wrong, per[1])


def test_shared_first_token_margin_comes_from_second_token(tok, eng, chain_bank):
    """Answers "21" vs "27" share their first token: a first-token-only scorer would return m = 0."""
    it = next(x for x in chain_bank if x["name"] == "chain-w2-d3p-0408")
    r = wsb.render_item(tok, "chain", it)
    sc = MG.score(eng, r.ids, MG.spellings(tok, "chain", 21), MG.spellings(tok, "chain", 27))
    a, b = sc["spell"][(17, 16)], sc["spell"][(17, 22)]
    assert a["per_token"][0] == pytest.approx(b["per_token"][0], abs=2e-3)
    assert sc["m"] == pytest.approx(b["per_token"][1] - a["per_token"][1], abs=4e-3)
    assert abs(sc["m"]) > 1e-3


def test_writes_applied_on_every_forward(tok, eng, brew_bank):
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    r = wsb.render_item(tok, "brew", it)
    conts = []

    def wf(cont):
        conts.append(cont)
        u = torch.zeros(5120, device=eng.device)
        u[0] = 1.0
        return [E.Swap1D(eng.blocks, 42, [164], u, torch.tensor([0.0]))]
    S_a = MG.spellings(tok, "brew", "black")
    S_b = MG.spellings(tok, "brew", "teal")                               # one single-token, one two-token spelling
    sc = MG.score(eng, r.ids, S_a, S_b, writes_fn=wf)
    assert conts == [(), (2568, 278)]
    assert [w.calls for w in sc["writes"]] == [1, 1]
