"""SecEngine (design 001 v3 regime): exact engine checks under SEC on the hand-checked brew pair
(brew-s2-r10d2-0000 start blue vs pink; the only differing token is position 142)."""
import gc

import pytest
import torch

import engine as E
import margin as MG
import partners as P
import regime as RG

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def sec(model):
    gc.collect()
    torch.cuda.empty_cache()
    eng = RG.SecEngine(model, l0=39)
    yield eng
    eng.restore()                     # the session model must be whole again for later tests
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def pair(tok, brew_bank):
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    p = {q.p.inp: q for q in P.brew_candidates(it)}["pink"]
    r_m, r_p, diff = P.token_align(tok, p, it)
    return p, r_m, r_p, diff


def test_self_patch_bitwise(sec, pair):
    _, r_m, _, _ = pair
    clean, acts = sec.forward(r_m.ids, record=[42])
    fh, _ = sec.forward(r_m.ids, upper={42: RG.patch_fn([164], acts[42][[164]])})
    assert torch.equal(fh, clean)


def test_stir_cell_patch_is_bitwise_noop(sec, pair):
    _, r_m, r_p, _ = pair
    clean, _ = sec.forward(r_m.ids)
    _, ap = sec.forward(r_p.ids, record=[42])
    fh, _ = sec.forward(r_m.ids, upper={42: RG.patch_fn([131], ap[42][[131]])})
    assert torch.equal(fh, clean)


def test_prefix_identical_before_first_difference(sec, pair):
    _, r_m, r_p, diff = pair
    L = [0, 20, 39, 42, 63]
    _, am = sec.forward(r_m.ids, record=L)
    _, ap = sec.forward(r_p.ids, record=L)
    assert all(torch.equal(am[l][:diff[0]], ap[l][:diff[0]]) for l in L)
    assert not torch.equal(am[63][diff[0]:], ap[63][diff[0]:])


def test_embedding_anchor_reproduces_partner(sec, pair):
    _, r_m, r_p, diff = pair
    fp, _ = sec.forward(r_p.ids)
    src = sec.embed(torch.tensor([r_p.ids], device=sec.model.device))[0, diff].detach()
    w = E.EmbedPatch(sec.embed, diff, src)
    fh, _ = sec.forward(r_m.ids, lower=[w])
    assert w.calls == 1 and torch.equal(fh, fp)


def test_last_layer_replacement_reproduces_partner_logits(sec, pair):
    _, r_m, r_p, _ = pair
    T = len(r_m)
    fp, ap = sec.forward(r_p.ids, record=[63])
    fh, _ = sec.forward(r_m.ids, upper={63: RG.patch_fn([T - 1], ap[63][[T - 1]])})
    assert torch.equal(sec.logits(fh, [T - 1]), sec.logits(fp, [T - 1]))


def test_lower_stack_write_lands_and_moves_the_answer(sec, pair):
    _, r_m, r_p, diff = pair
    _, ap = sec.forward(r_p.ids, record=[2])
    w = E.CellPatch(sec.blocks, 2, diff, ap[2][diff])
    clean, _ = sec.forward(r_m.ids)
    fh, got = sec.forward(r_m.ids, record=[2], lower=[w])
    assert w.calls == 1 and torch.equal(got[2][diff], ap[2][diff])
    assert not torch.equal(fh, clean)


def test_sec_score_matches_stock_and_scores_multitoken(sec, pair, tok):
    p, r_m, _, _ = pair
    Sa, Sb = MG.spellings(tok, "brew", p.m.a), MG.spellings(tok, "brew", p.p.a)
    sc = sec.score(r_m.ids, Sa, Sb)
    assert sc["m"] < 0                                               # member's greedy answer is a
    teal = MG.spellings(tok, "brew", "teal")
    sc2 = sec.score(r_m.ids, Sa, teal)
    assert len(sc2["spell"][teal[1]]["per_token"]) == 2               # " Teal" scored at both tokens


def test_sec_noise_floor_is_tiny(sec, pair, tok):
    """A 1e-4 relative perturbation at emit_colon L42 moves the margin by < 1e-3 nats under SEC (RHUT: up to ~0.05)."""
    p, r_m, _, _ = pair
    Sa, Sb = MG.spellings(tok, "brew", p.m.a), MG.spellings(tok, "brew", p.p.a)
    clean = sec.score(r_m.ids, Sa, Sb)
    _, am = sec.forward(r_m.ids, record=[42])
    h = am[42][164]
    v = torch.randn(h.shape[0], generator=torch.Generator().manual_seed(5)).to(h.device)
    src = (h + 1e-4 * h.norm() * v / v.norm())[None]
    ed = sec.score(r_m.ids, Sa, Sb, edit_fn=lambda c: ({42: RG.patch_fn([164], src)}, []))
    assert abs(ed["m"] - clean["m"]) < 1e-3
