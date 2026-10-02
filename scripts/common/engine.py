"""Intervention engine on block outputs (ported from RHUT common/scripts/hooks.py + hooks_fp32.py).

Conventions
  * Layer l = the output of decoder block l (0..63), i.e. the residual stream entering block l+1. Batch size 1.
  * Regime (AMENDMENT_01 §3, precision): RHUT's float32-residual regime. A hook on block `fp32_from` upcasts its output
    to float32 and the forward runs under bf16 autocast, so matmuls see the same bf16 operands as the stock model while
    residual additions and every write are carried in float32. `fp32_from=None` gives the stock bf16 forward.
  * Logits are computed here in float32 from the final-norm output (`final_hidden @ W_U^T`), never from the bf16
    lm_head output, so margins are not quantised by bf16 logits (at |logit| ~ 20 the bf16 spacing is 0.125).
  * Every write is computed with autocast disabled (true float32) and records its requested and realised delta, so
    fidelity (rho = |delta_real| / |delta_req|, kappa = cos) is measured after the write.
  * Hook order on a block: the regime upcast, then writes, then the recorder (which therefore sees post-write values).
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


# --------------------------------------------------------------------------------------------- plumbing
class Both:
    """Enter several context managers in order (None entries skipped); exit in reverse."""
    def __init__(self, *c):
        self.c = [x for x in c if x is not None]

    def __enter__(self):
        for x in self.c:
            x.__enter__()
        return self

    def __exit__(self, *a):
        for x in reversed(self.c):
            x.__exit__(*a)


def _out(o):
    return o if torch.is_tensor(o) else o[0]


def _pack(o, h):
    return h if torch.is_tensor(o) else (h,) + tuple(o[1:])


class _Hooked:
    """Base class: forward hooks on a set of blocks, removed on exit."""
    def __init__(self, blocks, layers):
        self.blocks, self.layers, self._h = blocks, sorted(int(l) for l in layers), []

    def _mk(self, l):
        raise NotImplementedError

    def __enter__(self):
        self._h = [self.blocks[l].register_forward_hook(self._mk(l)) for l in self.layers]
        return self

    def __exit__(self, *a):
        for x in self._h:
            x.remove()
        self._h = []


class Fp32Residual:
    """RHUT hooks_fp32.Fp32Residual: float32 residual from block `from_layer` onward, bf16 autocast for the matmuls.
    Must be entered before any writer so its hook on `from_layer` fires first."""
    def __init__(self, blocks, from_layer):
        self.blocks, self.layer, self._h, self._ac = blocks, int(from_layer), None, None

    @staticmethod
    def _f(m, i, o):
        h = _out(o)
        return _pack(o, h.float()) if h.dtype != torch.float32 else None

    def __enter__(self):
        self._h = self.blocks[self.layer].register_forward_hook(self._f)
        self._ac = torch.autocast("cuda", dtype=torch.bfloat16)
        self._ac.__enter__()
        return self

    def __exit__(self, *a):
        self._ac.__exit__(*a)
        self._h.remove()


class Recorder(_Hooked):
    """Record block outputs at `layers` (row 0, all positions) as detached float32 copies on the device."""
    def __init__(self, blocks, layers):
        super().__init__(blocks, layers)
        self.acts = {}

    def _mk(self, l):
        def f(m, i, o):
            self.acts[l] = _out(o)[0].detach().float().clone()
        return f


# ------------------------------------------------------------------------------------------------ writes
class CellWrite(_Hooked):
    """Base class for a write at one layer and a set of positions. Subclasses implement `delta(hp) -> [P, d]` in float32.
    The write is h[pos] <- cast(h[pos] + delta); the realised delta is measured after the cast."""
    def __init__(self, blocks, layer, positions):
        super().__init__(blocks, [layer])
        self.layer = int(layer)
        self.pos = torch.as_tensor(list(positions), dtype=torch.long)
        self.realized = None
        self.calls = 0

    def delta(self, hp):
        raise NotImplementedError

    def _mk(self, l):
        @torch.autocast("cuda", enabled=False)
        def f(m, i, o):
            h = _out(o).clone()
            pos = self.pos.to(h.device)
            hp = h[0, pos, :].float()
            d = self.delta(hp)
            new = (hp + d).to(h.dtype)
            self.realized = {"delta_req": d.detach(), "delta_real": (new.float() - hp).detach(), "dtype": str(h.dtype)}
            self.calls += 1
            h[0, pos, :] = new
            return _pack(o, h)
        return f


class CellPatch(CellWrite):
    """Full-cell replacement: h[pos] <- source ([P, d]). With a source captured in the same regime this is exact."""
    def __init__(self, blocks, layer, positions, source):
        super().__init__(blocks, layer, positions)
        self.src = source

    def delta(self, hp):
        return self.src.to(hp.device).float().reshape(hp.shape) - hp


class Swap1D(CellWrite):
    """One-dimensional swap along unit u: h <- h + (target - u.h) u, target = u.h_partner (per position, [P])."""
    def __init__(self, blocks, layer, positions, u, target):
        super().__init__(blocks, layer, positions)
        u = u.float()
        self.u = u / u.norm()
        self.target = torch.as_tensor(target, dtype=torch.float32).reshape(-1)

    def delta(self, hp):
        u = self.u.to(hp.device)
        return (self.target.to(hp.device) - hp @ u)[:, None] * u[None, :]


class AblateToMean1D(Swap1D):
    """Necessity ablation along unit u: h <- h + (mu - u.h) u with mu the pooled mean of u.h over the fitting half."""
    def __init__(self, blocks, layer, positions, u, mu):
        super().__init__(blocks, layer, positions, u, torch.full((len(list(positions)),), float(mu)))


class SpanWriter(_Hooked):
    """RHUT SpanWriter: sustained replacement of block outputs at `positions` <- source[l] ([P, d]) for every layer in
    `layers`. Span patches are context only (AMENDMENT_01 §3); the unit of a label is one cell."""
    def __init__(self, blocks, layers, positions, source):
        super().__init__(blocks, layers)
        self.pos = torch.as_tensor(list(positions), dtype=torch.long)
        self.src = source

    def _mk(self, l):
        @torch.autocast("cuda", enabled=False)
        def f(m, i, o):
            if len(self.pos) == 0:
                return None
            h = _out(o).clone()
            h[0, self.pos.to(h.device), :] = self.src[l].to(h.device).to(h.dtype)
            return _pack(o, h)
        return f


def realized_write_stats(hook):
    """rho and kappa of a CellWrite's realised write (cell-level, all positions flattened). RHUT definition."""
    r = hook.realized
    assert r is not None, "write never fired"
    dreq, dreal = r["delta_req"].flatten().float(), r["delta_real"].flatten().float()
    nreq, nreal = float(dreq.norm()), float(dreal.norm())
    rho = nreal / (nreq + 1e-12)
    kappa = float((dreq @ dreal) / (nreq * nreal + 1e-12)) if nreq > 0 and nreal > 0 else float("nan")
    return {"rho": rho, "kappa": kappa, "req_norm": nreq, "real_norm": nreal, "dtype": r["dtype"]}


# ------------------------------------------------------------------------------------------------ engine
class Run:
    """Result of one forward: recorded block outputs and the final-norm hidden states (both float32, on device)."""
    def __init__(self, acts, final_hidden, ids):
        self.acts, self.final_hidden, self.ids = acts, final_hidden, list(ids)

    def __len__(self):
        return len(self.ids)


class Engine:
    """Batch-1 forwards of the pinned model under one precision regime, with float32 logits.

    fp32_from: block index whose output starts the float32 residual (RHUT regime); None = stock bf16 forward.
    """
    def __init__(self, model, fp32_from=0):
        import registry
        self.model = model
        self.blocks = registry.blocks(model)
        self.text_model = (model.get_base_model() if hasattr(model, "get_base_model") else model).model
        self.fp32_from = fp32_from
        self.device = next(model.parameters()).device
        self.W_U = model.get_output_embeddings().weight.detach().float()      # [V, d] float32, ~5.1 GB

    @torch.no_grad()
    def forward(self, ids, record=(), writes=()):
        """One forward. `record`: layers whose block outputs to keep. `writes`: hook objects (CellWrite, SpanWriter,
        ...) entered after the regime hook and before the recorder."""
        x = torch.tensor([list(ids)], device=self.device)
        rec = Recorder(self.blocks, record) if record else None
        regime = Fp32Residual(self.blocks, self.fp32_from) if self.fp32_from is not None else None
        with Both(regime, *writes, rec):
            out = self.text_model(input_ids=x, use_cache=False)
            fh = out.last_hidden_state[0].detach().float()
        return Run(rec.acts if rec else {}, fh, ids)

    @torch.no_grad()
    def logits(self, run, positions=None):
        """float32 logits [P, V] at `positions` (default: all)."""
        h = run.final_hidden if positions is None else run.final_hidden[torch.as_tensor(list(positions))]
        with torch.autocast("cuda", enabled=False):
            return h.float() @ self.W_U.T

    @torch.no_grad()
    def logprobs(self, run, positions=None):
        return torch.log_softmax(self.logits(run, positions), -1)
