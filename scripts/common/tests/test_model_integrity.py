"""Reader setup must not silently change the subject model: in-place merges are marked and refused by every capture
and labelling entry point; attach/unload readers must leave the subject's logits bitwise unchanged."""
import pytest
import torch

import engine as E
import regime as RG
import registry as R


class _Dummy(torch.nn.Module):
    pass


def _marked():
    m = _Dummy()
    setattr(m, R.MODIFIED_ATTR, "test merge")
    return m


@pytest.mark.parametrize("entry", ["engine", "sec", "capture"])
def test_entry_points_refuse_modified_model(entry):
    m = _marked()
    with pytest.raises(RuntimeError, match="modified in place"):
        if entry == "engine":
            E.Engine(m)
        elif entry == "sec":
            RG.SecReplay(m, 39)
        else:
            import readers as RD
            RD.capture_stock(m, [1, 2], [0], [0])


@pytest.mark.gpu
def test_oracle_attach_unload_leaves_subject_intact(model, tok, brew_bank):
    import readers as RD
    import wsb
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    r = wsb.render_item(tok, "brew", it)
    x = torch.tensor([r.ids], device=model.device)
    with torch.no_grad():
        before = model(x).logits[0, -1].float().clone()
    acts_before = RD.capture_stock(model, r.ids, [40], [164])[40]
    ol = RD.OracleLens(model, tok)
    acts_during = RD.capture_stock(ol.model, r.ids, [40], [164])[40]    # adapter disabled while capturing
    base = ol.unload()
    with torch.no_grad():
        after = base(x).logits[0, -1].float()
    assert base is model
    assert torch.equal(acts_before, acts_during)
    assert torch.equal(before, after)
    R.assert_subject_model_intact(base)


@pytest.mark.gpu
def test_mutation_after_construction_is_refused_at_use(model, tok, brew_bank):
    """Lifecycle: an Engine / SecReplay built on an intact model must refuse it once a reader has modified it."""
    import wsb
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    ids = wsb.render_item(tok, "brew", it).ids
    eng = E.Engine(model, fp32_from=0)
    sec = RG.SecReplay(model, 62)
    eng.forward(ids)                                           # intact: works
    setattr(model, R.MODIFIED_ATTR, "test mutation after construction")
    try:
        with pytest.raises(RuntimeError, match="modified in place"):
            eng.forward(ids)
        with pytest.raises(RuntimeError, match="modified in place"):
            sec.capture(ids)
        with pytest.raises(RuntimeError, match="modified in place"):
            sec.build()
    finally:
        delattr(model, R.MODIFIED_ATTR)
    R.assert_subject_model_intact(model)


# ----------------------------------------------------------------------------------- resident (not SecReplay-parked)
class _Tiny(torch.nn.Module):
    """Decoder-shaped stand-in: .model.layers (blocks) and an input embedding, all on the CPU. A block moved to the
    `meta` device plays the part of a block SecReplay.build() parked on the other device."""
    def __init__(self, n=4, d=3):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(n)])
        self.model.embed_tokens = torch.nn.Embedding(5, d)

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_resident_check_passes_when_every_block_is_on_the_embedding_device():
    R.assert_subject_model_resident(_Tiny())


def test_resident_check_refuses_a_parked_block_and_honours_blocks_upto():
    m = _Tiny()
    m.model.layers[3].to("meta")                               # "parked" upper block
    with pytest.raises(RuntimeError, match=r"blocks \[3\] are not on"):
        R.assert_subject_model_resident(m)
    R.assert_subject_model_resident(m, blocks_upto=2)          # the lower stack alone is fine (SecReplay.capture)
    m.model.layers[2].to("meta")
    with pytest.raises(RuntimeError, match=r"blocks \[2, 3\]"):
        R.assert_subject_model_resident(m, blocks_upto=3)


@pytest.mark.gpu
def test_sec_parked_model_is_refused_by_stock_entry_points_and_restored(model, tok, brew_bank):
    """Between SecReplay.build() and restore() the bf16 upper blocks sit on the CPU: Engine.forward, the gate and
    reader capture must refuse the model, and the context manager must restore it, also after an exception."""
    import partners as P
    import readers as RD
    import wsb
    it = next(x for x in brew_bank if x["name"] == "brew-s2-r10d2-0000")
    ids = wsb.render_item(tok, "brew", it).ids
    eng = E.Engine(model, fp32_from=0)
    before = eng.forward(ids).final_hidden[-1].clone()
    with RG.SecReplay(model, 39) as sec:
        assert sec.built and sec.parked
        for call in (lambda: eng.forward(ids), lambda: RD.capture_stock(model, ids, [40], [164]),
                     lambda: P.gate_member(model, tok, ids, "brew", it["answer"], 0)):
            with pytest.raises(RuntimeError, match="not on"):
                call()
        with pytest.raises(RuntimeError, match="not on"):
            RG.SecReplay(model, 39)                            # a second replay on a parked model is refused too
        sec.capture(ids)                                       # the lower stack is on the GPU: allowed
    assert not sec.built and not sec.parked
    assert torch.equal(eng.forward(ids).final_hidden[-1], before)
    with pytest.raises(ValueError):
        with RG.SecEngine(model, l0=39) as se:
            assert se.parked
            raise ValueError("inside")
    assert not RG.SecReplay(model, 39).parked                  # restored on the exception path as well
    assert torch.equal(eng.forward(ids).final_hidden[-1], before)
