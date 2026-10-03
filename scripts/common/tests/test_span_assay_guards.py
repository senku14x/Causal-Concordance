"""004 launch-path guards (CPU, no model): Fidelity's three write modes and its failure cases; run_all stops on the
first failed fidelity row and saves it, so a pair with a bad write is never marked done."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "004_span_assay"))
import span_assay as SA  # noqa: E402


def _h():
    return torch.randn(5120, generator=torch.Generator().manual_seed(0)) * 2.0


def _write(scale):
    x = _h()[None].clone()
    d = torch.randn(5120, generator=torch.Generator().manual_seed(1))
    d = d / d.norm() * scale * x[0].norm()
    return SA.apply_write(x, 0, d=d)


@pytest.mark.parametrize("scale,mode", [(1e-2, "checked"), (2e-6, "checked"), (1e-8, "tiny")])
def test_fidelity_modes_pass(scale, mode):
    _, req, real, h = _write(scale)
    f = SA.Fidelity()
    f.add(req, real, h)
    assert f.ok() and f.writes[0]["mode"] == mode


def test_zero_write_is_exact_noop():
    x = _h()[None].clone()
    y, req, real, h = SA.apply_write(x, 0, d=torch.zeros(5120))
    f = SA.Fidelity()
    f.add(req, real, h)
    assert torch.equal(x, y) and f.ok() and f.writes[0]["mode"] == "zero"


def test_fidelity_failures():
    _, req, real, h = _write(1e-2)
    f = SA.Fidelity()
    f.add(req, 0.5 * real, h)                                   # realized half the requested write: rho 0.5
    assert not f.ok()
    _, req, real, h = _write(1e-8)
    f = SA.Fidelity()
    f.add(req, real + 1e-3 * h / h.norm(), h)                   # tiny write with a read-back error far above the bound
    assert not f.ok()
    f = SA.Fidelity()
    f.add(torch.zeros(5120), torch.full((5120,), 1e-9), _h())   # a zero request that changed the residual
    assert not f.ok()
    assert not SA.Fidelity().ok()                               # no write recorded is a failure, not a pass


class _Fake(SA.PairRunner):
    def __init__(self):
        self.base, self.cs, self.ds = {"pair_id": "p"}, {}, {}

    def evaluate(self, c):
        return {"pair_id": "p", "kind": "swap", "site": "colon", "fit": "both", "arm": c["arm"], "draw": None,
                "fid_ok": c["arm"] != "bad"}


def test_run_all_stops_on_failed_fidelity(tmp_path):
    fail = tmp_path / "fidelity_failures.jsonl"
    with pytest.raises(SA.FidelityGateError):
        _Fake().run_all([{"arm": "ok"}, {"arm": "bad"}, {"arm": "never"}], fail_path=fail)
    saved = [json.loads(l) for l in open(fail)]
    assert [r["arm"] for r in saved] == ["bad"]
    assert len(_Fake().run_all([{"arm": "ok"}], fail_path=fail)) == 2
