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
