"""Design 005 (planted concepts) CPU tests: frozen material and renders, scoring, the injection / clamp hooks, the
cache-free sampler and the batched forward on a toy transformer, model-side metrics, and the G1 / onset rules."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/005_planted_concepts"))
import p005 as P          # noqa: E402


# ---------------------------------------------------------------------------------------------- material + renders
def test_material_matches_screen_and_protocol():
    _, prefill, concepts, base = P.protocol()
    scr = json.loads((ROOT / "results/005_planted_concepts/analysis/concept_screen.json").read_text())
    excluded = sorted(c for c, r in scr["concepts"].items() if r["excluded"])
    assert excluded == sorted(P.EXCLUDED)
    assert sorted(P.CONCEPTS + P.EXCLUDED) == sorted(scr["concepts"])          # the 58 eligible
    assert P.DEV[:6] == P.GATE and set(P.SMOKE) <= set(P.DEV)
    assert len(base) == 194 and not set(base) & set(concepts)


@pytest.fixture(scope="module")
def tok():
    import registry
    return registry.make_tokenizer()


def test_render_positions(tok):
    ids = P.render_intro(tok)
    assert len(ids) == 139
    assert tok.decode([ids[P.POS["cq"]]]) == "," and tok.decode([ids[P.POS["co"]]]) == ' "'
    assert tok.decode(ids[97:117]).startswith("Trial 1") and tok.decode(ids[126:139]).startswith("Yes, I detect")
    nid = P.render_neutral(tok)
    assert tok.decode(nid[3:10]) == P.NEUTRAL_TEXT


def test_spellings_all_concepts(tok):
    for c in P.CONCEPTS:
        sp = P.spellings(tok, c)
        assert len(sp["quote"]) in (1, 2) and sp["space_lower"] in sp["space"]
        assert set(sp["quote"]) <= set(sp["all"])
        assert tok.decode([sp["space_lower"]]) == " " + c


# --------------------------------------------------------------------------------------------------------- scoring
def test_names_and_quote_naming():
    assert P.names("It is about LOSS and grief", "loss") and P.names("several losses", "loss")
    assert not P.names("glossary", "loss") and not P.names("lost", "loss")
    assert P.names("two boxes", "box") and P.names("Dogs bark", "dog")
    assert P.named_in_quote('Loss". It feels', "loss") and P.named_in_quote('storms"', "storm")
    assert not P.named_in_quote('a storm"', "storm") and not P.named_in_quote("storm surge and", "storm")


def test_seeded_randomness_is_stable():
    a = P.gaussian(16, "rand-dir", "gel", 32, 0)
    b = P.gaussian(16, "rand-dir", "gel", 32, 0)
    c = P.gaussian(16, "rand-dir", "gel", 32, 1)
    assert torch.equal(a, b) and not torch.equal(a, c)


# ---------------------------------------------------------------------------------------------------- toy model
class _Block(torch.nn.Module):
    def __init__(self, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)

    def forward(self, h, **kw):
        mix = h.cumsum(1) / torch.arange(1, h.shape[1] + 1, dtype=h.dtype)[None, :, None]   # causal mean
        return (h + torch.tanh(mix @ self.W),)


class _Out:
    def __init__(self, h):
        self.last_hidden_state = h


class _Text(torch.nn.Module):
    def __init__(self, V=40, d=16, n=6):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.embed_tokens = torch.nn.Embedding(V, d)
        self.embed_tokens.weight.data = torch.randn(V, d, generator=g)
        self.layers = torch.nn.ModuleList([_Block(d, i + 1) for i in range(n)])

    def forward(self, input_ids, use_cache=False, **kw):
        h = self.embed_tokens(input_ids)
        for b in self.layers:
            h = b(h)[0]
        return _Out(h / h.norm(dim=-1, keepdim=True))


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Text()

    def get_input_embeddings(self):
        return self.model.embed_tokens


@pytest.fixture()
def toy():
    m = _Toy().eval()
    W = torch.randn(40, 16, generator=torch.Generator().manual_seed(9))
    return m, W


def test_band_inject_exact_rows_positions_and_fidelity(toy):
    m, W = toy
    blocks = m.model.layers
    ids = list(range(10))
    d = {2: torch.zeros(3, 16), 3: torch.zeros(3, 16)}
    d[2][0], d[3][1] = torch.randn(16), torch.randn(16)
    rec_clean = P.RowRecorder(blocks, [2], list(range(10)))
    with rec_clean:
        m.model(torch.tensor([ids] * 3))
    inj = P.BandInject(blocks, d, [4, 5])
    rec = P.RowRecorder(blocks, [2], list(range(10)))
    with inj, rec:
        m.model(torch.tensor([ids] * 3))
    diff = rec.acts[2] - rec_clean.acts[2]
    assert torch.allclose(diff[0, 4], d[2][0]) and torch.allclose(diff[0, 5], d[2][0])
    assert diff[0, :4].abs().max() == 0 and diff[0, 6:].abs().max() == 0          # only the written positions
    assert diff[1:].abs().max() == 0                                              # row 1 is written at layer 3 only
    assert inj.fid[2]["rows"] == 1 and abs(inj.fid[2]["rho_min"] - 1) < 1e-6 and inj.fid[2]["kappa_min"] > 1 - 1e-6


def test_band_inject_refuses_bf16(toy):
    m, W = toy
    m = m.to(torch.bfloat16)
    inj = P.BandInject(m.model.layers, {1: torch.zeros(1, 16)}, [0])
    with inj, pytest.raises(AssertionError, match="float32"):
        m.model(torch.tensor([[1, 2, 3]]))


def test_band_clamp_restores_clamped_coordinates_only(toy):
    m, W = toy
    blocks = m.model.layers
    ids = list(range(8))
    rc = P.RowRecorder(blocks, [1, 2, 3], list(range(8)))
    with rc:
        m.model(torch.tensor([ids]))
    hc = {l: rc.acts[l][0] for l in (1, 2, 3)}
    Q, _ = torch.linalg.qr(torch.randn(16, 3, generator=torch.Generator().manual_seed(3)))
    delta = {1: torch.randn(2, 16, generator=torch.Generator().manual_seed(4))}
    inj = P.BandInject(blocks, delta, [2, 3])
    cl = P.BandClamp(blocks, {0: {1: Q, 2: Q, 3: Q}}, hc, [1, 2, 3])
    rec = P.RowRecorder(blocks, [1, 3], list(range(8)))
    with inj, cl, rec:
        m.model(torch.tensor([ids] * 2))
    for l in (1, 3):
        assert float(((rec.acts[l][0] - hc[l]) @ Q).abs().max()) < 1e-5           # row 0: clamped coordinates clean
    # at the write layer, row 0 keeps exactly the part of the write orthogonal to Q
    keep = delta[1][0] - (delta[1][0] @ Q) @ Q.T
    assert torch.allclose(rec.acts[1][0, 2] - hc[1][2], keep, atol=1e-5)
    assert float(((rec.acts[1][1] - hc[1])[2] @ Q).abs().max()) > 1e-3            # row 1 is not clamped
    assert cl.log[0][1]["residual"] < 1e-5 and 0 < cl.log[0][1]["removed_share"] < 1


def test_reentry_clamp_keeps_the_raw_injection_and_removes_only_the_model_response(toy):
    """h - h_clean = A + M: the re-entry clamp (ref = h_clean + A) removes Q Q^T M and keeps A, so at the write layer
    (M = 0 there, nothing upstream was written) the injection survives exactly, where the standard clamp strips Q Q^T A."""
    m, W = toy
    blocks = m.model.layers
    ids = list(range(8))
    rc = P.RowRecorder(blocks, [1, 2, 3], list(range(8)))
    with rc:
        m.model(torch.tensor([ids]))
    hc = {l: rc.acts[l][0] for l in (1, 2, 3)}
    Q, _ = torch.linalg.qr(torch.randn(16, 4, generator=torch.Generator().manual_seed(7)))
    dl = torch.randn(16, generator=torch.Generator().manual_seed(8))
    pos = [2, 3]
    A = {1: dl, 2: dl, 3: dl}                                     # one write at layer 1, carried unchanged
    inj = P.BandInject(blocks, {1: dl[None]}, pos)
    cl = P.BandClamp(blocks, {0: {1: Q, 2: Q, 3: Q}}, hc, [1, 2, 3], offsets={0: A}, offset_pos=pos)
    rec = P.RowRecorder(blocks, [1, 2, 3], list(range(8)))
    with inj, cl, rec:
        m.model(torch.tensor([ids]))
    assert torch.allclose(rec.acts[1][0, 2] - hc[1][2], dl, atol=1e-5)        # write layer: the injection is intact
    assert cl.log[0][1]["removed_share"] < 1e-5                                # nothing to remove there (M = 0)
    for l in (2, 3):                                                           # downstream: Q^T (h - hc - A) = 0
        ref = hc[l].clone()
        ref[pos] += dl
        assert float(((rec.acts[l][0] - ref) @ Q).abs().max()) < 1e-5
    # the standard clamp at the same write layer strips the injection's component in span(Q)
    inj2 = P.BandInject(blocks, {1: dl[None]}, pos)
    cl2 = P.BandClamp(blocks, {0: {1: Q}}, hc, [1])
    rec2 = P.RowRecorder(blocks, [1], list(range(8)))
    with inj2, cl2, rec2:
        m.model(torch.tensor([ids]))
    assert torch.allclose(rec2.acts[1][0, 2] - hc[1][2], dl - (dl @ Q) @ Q.T, atol=1e-5)


def test_forward_rows_and_sampler_on_toy(toy):
    m, W = toy
    ids = [1, 5, 7, 2, 9, 3]
    blocks = m.model.layers
    d = {2: torch.randn(2, 16, generator=torch.Generator().manual_seed(5))}
    lp, acts = P.forward_rows(m, ids, 2, W, [4, 5], writes=[P.BandInject(blocks, d, [1, 2])], record=[3],
                              record_pos=[2, 5], fp32=False)
    assert lp.shape == (2, 2, 40) and acts[3].shape == (2, 2, 16)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 2), atol=1e-5)
    a = P.sample_rows(m, ids, W, d, [1, 2], n_new=5, seed=11, fp32=False)
    b = P.sample_rows(m, ids, W, d, [1, 2], n_new=5, seed=11, fp32=False)
    assert a.shape == (2, 5) and torch.equal(a, b)
    # the injection never touches generated positions: re-score the sampled sequence with the same write
    seq = ids + a[0].tolist()
    lp2, _ = P.forward_rows(m, seq, 1, W, list(range(len(ids))), writes=[P.BandInject(blocks, {2: d[2][:1]}, [1, 2])],
                            fp32=False)
    lp3, _ = P.forward_rows(m, ids, 1, W, list(range(len(ids))), writes=[P.BandInject(blocks, {2: d[2][:1]}, [1, 2])],
                            fp32=False)
    assert torch.allclose(lp2, lp3, atol=1e-6)            # causal toy: appended tokens leave prompt positions unchanged
    lp4, _ = P.forward_rows(m, ids, 1, W, [len(ids) - 1], fp32=False)
    assert not torch.allclose(lp3[:, -1:], lp4, atol=1e-4)  # and the write does reach the last prompt position


def test_top_p_sampler():
    logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]] * 4)
    g = torch.Generator().manual_seed(0)
    picks = torch.stack([P.top_p_sample(logits, g, top_p=0.5) for _ in range(50)])
    assert set(picks.flatten().tolist()) == {0}                      # p(0) = 0.70 > 0.5: nucleus = {0}
    g = torch.Generator().manual_seed(0)
    picks = torch.stack([P.top_p_sample(logits, g, top_p=0.95) for _ in range(200)])
    assert set(picks.flatten().tolist()) <= {0, 1}                   # 0.70 + 0.26 crosses 0.95
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(P.top_p_sample(logits, g1), P.top_p_sample(logits, g2))


# --------------------------------------------------------------------------------------------- metrics + rules
def test_intro_and_neutral_metrics():
    V = 20
    lpc = torch.log_softmax(torch.zeros(14, V), -1)
    lp = lpc.clone()[None]
    lp[0, 13, 7] = 3.0
    lp = torch.log_softmax(lp, -1)
    sp = {"quote": [7], "all": [7, 8], "space": [8]}
    m = P.intro_metrics(lp, lpc, sp)[0]
    assert m["R1"] == 1.0 and m["dR1p"] > 0 and abs(m["B4"]) < 1e-6           # positions 126-137 unchanged
    m0 = P.intro_metrics(lpc[None], lpc, sp)[0]
    assert m0["R1"] == 1.0 / 1 and abs(m0["dR1p"]) < 1e-9                      # ties: rank 1 (none strictly above)
    n = P.neutral_metrics(lpc[:8][None], lpc[:8], sp)[0]
    assert abs(n["B1"]) < 1e-9 and abs(n["B2"]) < 1e-6
    assert P.ranks_many(torch.tensor([0.1, 0.5, 0.3, 0.9]), [[1], [0, 2], [3]]) == [2, 3, 1]


def _g1_rows(r2_full, peak_top=False):
    rows = []
    for c in ["a", "b"]:
        for s in P.GRID:
            top = s == (P.GRID[-1] if peak_top else 0.25)
            rows.append({"concept": c, "cond": 1, "s": s, "R2": 0.5 if top else 0.0, "B3": 0.01, "B1": 1.0, "R1": 0.5})
            rows.append({"concept": c, "cond": 2, "s": s, "R2": r2_full if s == 0.25 else 0.0, "B3": 0.01, "B1": 1.0,
                         "R1": 0.2})
            rows.append({"concept": c, "cond": 7, "s": s, "R2": 0.0, "B3": 0.0, "B1": 0.5, "R1": 0.0})
    return rows


def test_g1_rule_and_calibration():
    d = P.g1_decide(_g1_rows(0.0), P.GRID)
    assert d["pass"] and d["by_cond"][1]["best_s"] == 0.25 and d["calibration"] is None
    d = P.g1_decide(_g1_rows(0.0, peak_top=True), P.GRID)
    assert d["calibration"] == "x4"
    rows = [dict(r, R2=0.1) if r["cond"] in (1, 2) and r["R2"] > 0 else r for r in _g1_rows(0.6)]
    assert not P.g1_decide(rows, P.GRID)["pass"]                                # naming 0.1 < 0.25


def test_onset_and_text_strengths():
    rows = []
    for c in range(10):
        for s in P.GRID:
            rows.append({"concept": c, "cond": 4, "s": s, "dR1p": 1.0 if s >= 0.25 else 0.0})
            for k in range(5):
                rows.append({"concept": c, "cond": 7, "s": s, "dR1p": 0.1 * k})
    s_star, trace = P.onset(rows, P.GRID)
    assert s_star == 0.25 and trace[0]["above"] is False
    assert P.text_strengths(0.25, P.GRID) == [1 / 16, 0.25, 1.0]
    assert P.text_strengths(1 / 32, P.GRID) == [1 / 32, 1 / 16, 1 / 8]
    assert P.text_strengths(1.0, P.GRID) == [1 / 4, 1 / 2, 1.0]
    assert P.text_strengths(None, P.GRID) == [1 / 4, 1 / 2, 1.0]
