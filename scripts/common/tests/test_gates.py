import math

import pytest
import torch

import engine as E
import gates
import margin as MG
import partners as P
import wsb


def test_fidelity_thresholds():
    assert gates.fidelity_gate({"rho": 1.0, "kappa": 1.0})["pass"]
    assert not gates.fidelity_gate({"rho": 0.85, "kappa": 1.0})["pass"]
    assert not gates.fidelity_gate({"rho": 1.0, "kappa": 0.98})["pass"]
    assert gates.fidelity_gate({"rho": 1.1, "kappa": 0.99})["pass"]


def test_damage_gate_logic():
    d = {"off_pair_mass_clean": 0.05, "off_pair_mass_edit": 0.30, "top1_retention": 0.99, "n_retention_positions": 4}
    assert not gates.damage_gate(d, max_off_pair_increase=0.1, min_retention=0.9)["pass"]
    d["off_pair_mass_edit"] = 0.1
    assert gates.damage_gate(d, max_off_pair_increase=0.1, min_retention=0.9)["pass"]
    d["top1_retention"] = 0.5
    assert not gates.damage_gate(d, max_off_pair_increase=0.1, min_retention=0.9)["pass"]
    d.update(n_retention_positions=0, top1_retention=math.nan)
    assert gates.damage_gate(d, max_off_pair_increase=0.1, min_retention=0.9)["pass"]


def test_span_gate_hand_checked_pairs(tok, brew_bank, chain_bank):
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    pair = {p.p.inp: p for p in P.brew_candidates(it)}["pink"]
    g = gates.span_gate(tok, pair, it)
    assert g["pass"] and g["diff"] == [142] and g["len"] == 165
    ci = next(x for x in chain_bank if x["name"] == "chain-w2-d3p-0408")
    g = gates.span_gate(tok, {p.p.inp: p for p in P.chain_candidates(ci)}[14], ci)
    assert g["pass"] and len(g["diff"]) == 1


@pytest.mark.gpu
def test_damage_of_self_patch_is_zero(model, tok, brew_bank):
    eng = E.Engine(model, fp32_from=0)
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    r = wsb.render_item(tok, "brew", it)
    S_a, S_b = MG.spellings(tok, "brew", "black"), MG.spellings(tok, "brew", "blue")
    clean = MG.score(eng, r.ids, S_a, S_b)
    src = eng.forward(r.ids, record=[42]).acts[42][[150]]
    ed = MG.score(eng, r.ids, S_a, S_b, writes_fn=lambda c: [E.CellPatch(eng.blocks, 42, [150], src)])
    d = gates.damage(eng, clean, ed, edit_pos=150, T=len(r))
    assert d["d_logP_a"] == 0 and d["d_logP_b"] == 0 and d["top1_retention"] == 1.0
    assert d["n_retention_positions"] == len(r) - 1 - 151
    assert d["off_pair_mass_edit"] == d["off_pair_mass_clean"]


@pytest.mark.gpu
def test_seqlen_determinism_reports(model, tok, brew_bank):
    """Measured 2026-10-02: NOT bitwise (onset at block 3, the first full-attention block, ~6e-4 relative; growing to
    ~1% median by block 63). The test asserts the onset is at rounding level (no causal leak) and that the gate reports
    the failure; the consequence (fixed-length comparisons, matching-length sources) is enforced in margin.score."""
    eng = E.Engine(model, fp32_from=0)
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    r = wsb.render_item(tok, "brew", it)
    g = gates.seqlen_determinism(eng, r.ids, MG.spellings(tok, "brew", "teal")[1] + (13,))
    print("sequence-length determinism:", {k: v for k, v in g.items() if "per_layer" not in k})
    assert not g["leak_suspect"], g["onset_max_rel"]
    assert g["first_diff_layer"] is None or all(v == 0.0 for l, v in g["max_rel_diff_per_layer"].items()
                                                if l < g["first_diff_layer"])
