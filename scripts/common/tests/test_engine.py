"""Engine/hook tests on the hand-checked brew pair (brew-s2-r10d2-0000, partner start pink), float32-residual regime."""
import pytest
import torch

import engine as E
import gates
import partners as P

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def eng(model):
    return E.Engine(model, fp32_from=0)


@pytest.fixture(scope="module")
def brew_pair(tok, brew_bank):
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    pair = {p.p.inp: p for p in P.brew_candidates(it)}["pink"]
    r_m, r_p, diff = P.token_align(tok, pair, it)
    return it, pair, r_m, r_p


@pytest.mark.parametrize("layer,pos", [(42, [164]), (40, [142]), (2, [142]), (63, [164]), (44, [158, 162, 164])])
def test_self_patch_bitwise(eng, brew_pair, layer, pos):
    g = gates.bitwise_gate(eng, brew_pair[2].ids, layer, pos)
    assert g["write_calls"] == 1 and g["pass"], g


def test_swap_fidelity_and_readback(eng, brew_pair):
    _, _, r_m, r_p = brew_pair
    cp = eng.forward(r_p.ids, record=[42])
    u = torch.randn(5120, generator=torch.Generator().manual_seed(0)).to(eng.device)
    u = u / u.norm()
    tgt = cp.acts[42][[164]] @ u
    w = E.Swap1D(eng.blocks, 42, [164], u, tgt)
    run = eng.forward(r_m.ids, record=[42], writes=[w])
    st = E.realized_write_stats(w)
    assert gates.fidelity_gate(st)["pass"], st
    assert st["dtype"] == "torch.float32"
    assert abs(float(run.acts[42][164] @ u) - float(tgt)) < 1e-4
    # the write changed exactly one row at layer 42
    clean = eng.forward(r_m.ids, record=[42])
    changed = (run.acts[42] != clean.acts[42]).any(-1).nonzero().flatten().tolist()
    assert changed == [164]


def test_swap_along_difference_equals_full_patch(eng, brew_pair):
    """A 1-D swap along unit(h_p - h_m) at the same cell is the full-cell patch (up to float32 rounding)."""
    _, _, r_m, r_p = brew_pair
    cm, cp = eng.forward(r_m.ids, record=[42]), eng.forward(r_p.ids, record=[42])
    d = cp.acts[42][164] - cm.acts[42][164]
    u = d / d.norm()
    sw = E.Swap1D(eng.blocks, 42, [164], u, cp.acts[42][[164]] @ u)
    a = eng.forward(r_m.ids, record=[42], writes=[sw])
    b = eng.forward(r_m.ids, record=[42], writes=[E.CellPatch(eng.blocks, 42, [164], cp.acts[42][[164]])])
    assert torch.equal(b.acts[42][164], cp.acts[42][164])
    rel = float((a.acts[42][164] - b.acts[42][164]).norm() / b.acts[42][164].norm())
    assert rel < 1e-5, rel
    # downstream, the two float32-identical-to-1e-6 residuals diverge at the bf16-matmul noise floor of the regime
    la, lb = eng.logprobs(a, [164]), eng.logprobs(b, [164])
    print(f"swap-vs-patch: residual rel diff {rel:.2e}; downstream max |dlogprob| {float((la - lb).abs().max()):.3f}")


def test_last_cell_replacement_reproduces_partner_logits(eng, brew_pair):
    _, _, r_m, r_p = brew_pair
    T = len(r_m)
    cp = eng.forward(r_p.ids, record=[63])
    w = E.CellPatch(eng.blocks, 63, [T - 1], cp.acts[63][[T - 1]])
    run = eng.forward(r_m.ids, writes=[w])
    assert torch.equal(eng.logits(run, [T - 1]), eng.logits(cp, [T - 1]))


def test_write_leaves_prefix_and_earlier_layers_untouched(eng, brew_pair):
    _, _, r_m, r_p = brew_pair
    L = list(range(64))
    cp = eng.forward(r_p.ids, record=[20])
    clean = eng.forward(r_m.ids, record=L)
    run = eng.forward(r_m.ids, record=L, writes=[E.CellPatch(eng.blocks, 20, [150], cp.acts[20][[150]])])
    assert all(torch.equal(run.acts[l], clean.acts[l]) for l in range(20))
    assert all(torch.equal(run.acts[l][:150], clean.acts[l][:150]) for l in L)
    assert not torch.equal(run.acts[21][150:], clean.acts[21][150:])


def test_stock_bf16_regime_loses_small_writes(model, brew_pair):
    """Why the float32 regime is the default: in the stock bf16 forward a small 1-D write at layer 42 is rounded away
    or distorted (reported, and only asserted to be worse than the float32 regime)."""
    _, _, r_m, r_p = brew_pair
    e16 = E.Engine(model, fp32_from=None)
    cp = e16.forward(r_p.ids, record=[42])
    u = torch.randn(5120, generator=torch.Generator().manual_seed(1)).to(e16.device)
    u = u / u.norm()
    w = E.Swap1D(e16.blocks, 42, [164], u, cp.acts[42][[164]] @ u)
    e16.forward(r_m.ids, writes=[w])
    st = E.realized_write_stats(w)
    print("bf16 regime 1-D write:", st)
    assert st["dtype"] == "torch.bfloat16"
    assert abs(st["rho"] - 1) > 1e-4 or st["kappa"] < 0.9999
