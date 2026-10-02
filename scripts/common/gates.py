"""Gates that run before any outcome is read (AMENDMENT_01 §4): span/rendering, same-source == clean bitwise,
write fidelity, damage, sequence-length determinism. Each returns a JSON-able dict with a boolean "pass".
Damage thresholds are fixed per experiment in its design spec; this module only measures."""
import math

import torch

import engine as E
import partners
import wsb

FIDELITY = {"rho_lo": 0.9, "rho_hi": 1.1, "kappa_min": 0.99}


# ------------------------------------------------------------------------------------------- span / rendering
def span_gate(tok, pair, bank_item):
    """Both members render with the family's pinned positions and are token-aligned with only the input changed."""
    fam = pair.family
    al = partners.token_align(tok, pair, bank_item)
    if al is None:
        return {"pass": False, "why": "not token-aligned"}
    r_m, r_p, diff = al
    try:
        for r in (r_m, r_p):
            if fam == "brew":
                it = dict(bank_item)
                it["start"] = pair.m.inp if r is r_m else pair.p.inp
                wsb.assert_brew_positions(r, it)
            elif fam == "chain":
                wsb.assert_chain_positions(r)
            elif fam == "relational":
                wsb.assert_relational_positions(r)
    except AssertionError as e:
        return {"pass": False, "why": str(e)}
    return {"pass": r_m.read_positions == r_p.read_positions, "len": len(r_m), "diff": diff}


# ------------------------------------------------------------------------------------------------- bitwise
@torch.no_grad()
def bitwise_gate(engine, ids, layer, positions):
    """Same-source patch (a cell overwritten with its own clean value) must reproduce the clean forward bitwise:
    final hidden states at every position and every recorded block output."""
    layers = list(range(len(engine.blocks)))
    clean = engine.forward(ids, record=layers)
    src = clean.acts[int(layer)][list(positions)].clone()
    w = E.CellPatch(engine.blocks, layer, positions, src)
    pat = engine.forward(ids, record=layers, writes=[w])
    fh = bool(torch.equal(clean.final_hidden, pat.final_hidden))
    acts = all(torch.equal(clean.acts[l], pat.acts[l]) for l in layers)
    return {"pass": fh and acts, "final_hidden_equal": fh, "acts_equal": acts, "layer": int(layer),
            "positions": list(map(int, positions)), "write_calls": w.calls}


# ------------------------------------------------------------------------------------------------ fidelity
def fidelity_gate(stats, thr=FIDELITY):
    """rho in [0.9, 1.1] and kappa >= 0.99 for a realised write (engine.realized_write_stats output)."""
    ok = thr["rho_lo"] <= stats["rho"] <= thr["rho_hi"] and stats["kappa"] >= thr["kappa_min"]
    return {"pass": bool(ok), **stats}


# -------------------------------------------------------------------------------------------------- damage
@torch.no_grad()
def damage(engine, clean_score, edit_score, edit_pos, T):
    """Damage readouts of an edit (inputs are margin.score outputs on the same prompt, clean and edited):
    dlogP(a), dlogP(a'), top-1 retention at the teacher-forced prompt positions after the edit (the answer position
    T-1 excluded), and the off-pair mass at the answer position (mass outside the first tokens of both answer sets)."""
    pos = list(range(int(edit_pos) + 1, T - 1))
    if pos:
        c = engine.logits(clean_score["run"], pos).argmax(-1)
        e = engine.logits(edit_score["run"], pos).argmax(-1)
        ret, n_ret = float((c == e).float().mean()), len(pos)
    else:
        ret, n_ret = math.nan, 0
    ft = clean_score["first_tokens"]

    def off(sc):
        return float(1.0 - sc["lp_last"][ft].exp().sum())
    return {"d_logP_a": edit_score["lse_a"] - clean_score["lse_a"], "d_logP_b": edit_score["lse_b"] - clean_score["lse_b"],
            "top1_retention": ret, "n_retention_positions": n_ret,
            "off_pair_mass_clean": off(clean_score), "off_pair_mass_edit": off(edit_score)}


def damage_gate(d, max_off_pair_increase, min_retention):
    """Broad damage => unresolved. Thresholds come from the experiment's design spec."""
    ok_off = (d["off_pair_mass_edit"] - d["off_pair_mass_clean"]) <= max_off_pair_increase
    ok_ret = d["n_retention_positions"] == 0 or d["top1_retention"] >= min_retention
    return {"pass": bool(ok_off and ok_ret), "off_pair_ok": bool(ok_off), "retention_ok": bool(ok_ret)}


# ------------------------------------------------------------------------------------ sequence-length determinism
@torch.no_grad()
def seqlen_determinism(engine, ids, cont, layers=None, leak_onset=1e-2):
    """Are block outputs at the prompt positions identical when a continuation is appended?

    Bitwise identity is the pass criterion; when it fails, every comparison must be made at one sequence length and
    sources captured at the matching length (margin.score does this). A causal leak is distinguished from rounding by
    the onset: rounding differences first appear at ~1e-4 relative and grow with depth; a leak appears at O(1) in the
    first block that sees the appended tokens. `leak_suspect` flags an onset above `leak_onset`."""
    layers = list(range(len(engine.blocks))) if layers is None else sorted(layers)
    T = len(ids)
    a = engine.forward(list(ids), record=layers)
    b = engine.forward(list(ids) + list(cont), record=layers)
    rel = {l: ((a.acts[l] - b.acts[l][:T]).norm(dim=-1) / a.acts[l].norm(dim=-1)) for l in layers}
    first = next((l for l in layers if float(rel[l].max()) > 0), None)
    fr = (a.final_hidden - b.final_hidden[:T]).norm(dim=-1) / a.final_hidden.norm(dim=-1)
    bit = first is None and float(fr.max()) == 0.0
    onset = float(rel[first].max()) if first is not None else 0.0
    return {"pass": bool(bit), "bitwise": bool(bit), "first_diff_layer": first, "onset_max_rel": onset,
            "n_positions_diff_at_onset": int((rel[first] > 0).sum()) if first is not None else 0,
            "leak_suspect": bool(onset > leak_onset),
            "max_rel_diff_per_layer": {l: float(v.max()) for l, v in rel.items()},
            "median_rel_diff_per_layer": {l: float(v.median()) for l, v in rel.items()},
            "final_hidden_max_rel_diff": float(fr.max()), "final_hidden_median_rel_diff": float(fr.median()),
            "final_hidden_last_pos_rel_diff": float(fr[T - 1]), "T": T, "cont_len": len(cont)}
