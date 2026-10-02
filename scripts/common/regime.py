"""SEC's true-float32 replay (port of jacobian-lens research_secant/scripts/lib/engine.py): the label regime of
design 001 v3 (§2, §9.1) for every layer >= 40; RHUT's float32-residual regime (`engine.Engine`) remains the
cross-check (AMENDMENT_01 §3 precision, errata E4).

Stock bf16 forward through block l0 (early exit: blocks above l0 never run in bf16), then float32 deep copies of
blocks l0+1..63, the final norm and W_U with TF32 off. Edits at layers >= l0 act on the float32 stream between blocks.

Differences from SEC's engine (none changes the arithmetic of the replayed blocks):
  * resident layout: the bf16 copies of blocks l0+1..63 are parked on the CPU while the float32 copies are on the GPU,
    so capture needs no 54 GB CPU<->GPU move per prompt (bf16 0..l0 + float32 l0+1..63 + float32 W_U ~ 72 GB at l0=39);
  * the per-block call kwargs are captured from blocks <= l0 by layer type (attention masks are identical across
    blocks of one type) because the upper bf16 blocks never run;
  * rotary cos/sin are recomputed in float32 rather than upcast from bf16 (`rotary_check` reports the difference);
  * logits use raw W_U after the float32 final norm (SEC folds (1+gamma) into W_U; equivalent).
"""
import copy

import torch

import engine as E

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


class _Stop(Exception):
    pass


class SecReplay:
    """Use as a context manager (`with SecReplay(model, 39) as sec:`) so the stock model is restored on exit, including
    on an exception; `parked` says whether the bf16 upper blocks are currently on the CPU."""

    def __init__(self, model, l0, device="cuda"):
        import registry
        registry.assert_subject_model_intact(model)
        registry.assert_subject_model_resident(model)
        self.model, self.l0, self.device = model, int(l0), device
        self.blocks = registry.blocks(model)
        self.text = model.model
        self.types = registry.layer_types(model)
        self.n = len(self.blocks)
        self.up, self.norm32, self.rot32, self.W_U, self.built = {}, None, None, None, False

    @property
    def parked(self):
        """True while the bf16 upper blocks (l0+1..63) are parked on the CPU, i.e. between build() and restore()."""
        return any(p.device.type == "cpu" for i in range(self.l0 + 1, self.n) for p in self.blocks[i].parameters())

    def __enter__(self):
        if not self.built:
            self.build()
        return self

    def __exit__(self, *a):
        self.restore()

    # ------------------------------------------------------------------------------------------- resident stack
    def build(self):
        """float32 copies of blocks l0+1..63 on the GPU; their bf16 originals parked on the CPU."""
        import registry
        registry.assert_subject_model_intact(self.model)
        if self.built:
            return
        registry.assert_subject_model_resident(self.model)     # build twice without restore would park a parked model
        for i in range(self.l0 + 1, self.n):
            self.up[i] = copy.deepcopy(self.blocks[i]).float().to(self.device).eval()
            self.blocks[i].to("cpu")
        torch.cuda.empty_cache()
        self.norm32 = copy.deepcopy(self.text.norm).float().to(self.device).eval()
        self.rot32 = copy.deepcopy(self.text.rotary_emb).float().to(self.device)
        self.W_U = self.model.get_output_embeddings().weight.detach().float().to(self.device)
        self.built = True

    def restore(self):
        """Free the float32 stack and move the bf16 upper blocks back to the GPU (stock model again). Idempotent."""
        self.up, self.norm32, self.rot32, self.W_U, self.built = {}, None, None, None, False
        torch.cuda.empty_cache()
        for i in range(self.l0 + 1, self.n):
            self.blocks[i].to(self.device)

    # ----------------------------------------------------------------------------------------- capture (bf16)
    @torch.no_grad()
    def capture(self, ids, layers=(), writes=()):
        """Stock bf16 forward with early exit after block l0. Returns ({l: [T, d] float32 on device} for l0 and any
        `layers` <= l0, {layer_type: kwargs} from the last block of each type <= l0). `writes` are hook objects
        (engine.CellPatch on lower blocks, engine.EmbedPatch) entered for this capture only, before the recorders."""
        import registry
        registry.assert_subject_model_intact(self.model)       # at use time, not only at construction
        registry.assert_subject_model_resident(self.model, blocks_upto=self.l0)   # the lower stack must be on the GPU
        want = sorted(set(int(l) for l in layers) | {self.l0})
        assert max(want) <= self.l0
        store, kw, hs = {}, {}, []

        def mk_out(l):
            def f(m, a, o):
                store[l] = E._out(o)[0].detach().float().clone()
                if l == self.l0:
                    raise _Stop
            return f

        def mk_pre(l):
            def f(m, args, kwargs):
                kw[self.types[l]] = {k: v for k, v in kwargs.items()
                                     if k in ("position_embeddings", "attention_mask", "position_ids")}
            return f
        for w in writes:
            w.__enter__()
        try:
            for l in want:
                hs.append(self.blocks[l].register_forward_hook(mk_out(l)))
            for l in range(self.l0 + 1):
                hs.append(self.blocks[l].register_forward_pre_hook(mk_pre(l), with_kwargs=True))
            try:
                self.text(input_ids=torch.tensor([list(ids)], device=self.device), use_cache=False)
            except _Stop:
                pass
        finally:
            for h in hs:
                h.remove()
            for w in reversed(list(writes)):
                w.__exit__(None, None, None)
        assert set(self.types[self.l0 + 1:]) <= set(kw), f"no kwargs captured for {set(self.types) - set(kw)}"
        return store, kw

    def _kw(self, kw, l, T):
        d = {k: v for k, v in kw[self.types[l]].items() if k != "position_embeddings"}
        x = torch.zeros(1, T, self.W_U.shape[1], device=self.device, dtype=torch.float32)
        pos = torch.arange(T, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        d["position_embeddings"] = self.rot32(x, pos)
        return d

    def rotary_check(self, kw, T):
        """max |cos32 - cos_bf16| and |sin32 - sin_bf16| (should be at the bf16 spacing, ~4e-3 at most)."""
        c32, s32 = self._kw(kw, self.l0 + 1, T)["position_embeddings"]
        cb, sb = kw["full_attention"]["position_embeddings"]
        return float((c32 - cb.float()).abs().max()), float((s32 - sb.float()).abs().max())

    # -------------------------------------------------------------------------------------------- float32 replay
    @torch.no_grad()
    def replay(self, h_l0, kw, edits=None, record=()):
        """Float32 blocks l0+1..63 on h_l0 [T, d]. `edits` {layer: fn(x [T, d] float32) -> x} act on that layer's output
        (layer l0's edit acts on h_l0). Returns (final-norm hidden [T, d] float32, {l: recorded [T, d]})."""
        assert self.built, "call build() first"
        edits = edits or {}
        T = h_l0.shape[0]
        x = h_l0.to(self.device).float()
        if self.l0 in edits:
            x = edits[self.l0](x)
        rec = {self.l0: x.clone()} if self.l0 in record else {}
        x = x[None]
        for i in range(self.l0 + 1, self.n):
            x = E._out(self.up[i](x, **self._kw(kw, i, T), use_cache=False))
            if i in edits:
                x = edits[i](x[0])[None]
            if i in record:
                rec[i] = x[0].clone()
        return self.norm32(x)[0], rec

    @torch.no_grad()
    def logprobs(self, final_hidden, positions):
        return torch.log_softmax(final_hidden[list(positions)] @ self.W_U.T, -1)


# ---------------------------------------------------------------------------------- edits as replay functions
def patch_fn(pos, src):
    """Full-cell replacement at positions `pos` with src [P, d] (float32)."""
    def f(x):
        x = x.clone()
        x[list(pos)] = src.to(x.device).float()
        return x
    return f


def swap_fn(pos, u, target):
    """1-D swap along unit u at positions `pos` to coordinates `target` [P]."""
    u = (u / u.norm()).float()

    def f(x):
        x = x.clone()
        hp = x[list(pos)]
        x[list(pos)] = hp + (target.to(x.device).float().reshape(-1) - hp @ u.to(x.device))[:, None] * u.to(x.device)[None]
        return x
    return f


# ------------------------------------------------------------------------------------------- scoring under SEC
class SecEngine:
    """Label measurements in the SEC regime (design 001 v3): stock bf16 blocks <= l0, float32 replay of blocks > l0,
    float32 logits. Lower-stack captures of unedited prompts are cached per token sequence (clear() between pairs).

    forward(ids, record, upper, lower) -> (final hidden [T, d] fp32, {layer: [T, d] fp32})
      upper: {layer > l0 (or = l0): fn(x [T, d]) -> x}   (regime.patch_fn / swap_fn)
      lower: hook objects for the bf16 lower stack (engine.CellPatch on blocks <= l0, engine.EmbedPatch)
    """

    def __init__(self, model, l0=39, build=True):
        self.sec = SecReplay(model, l0)
        self.l0, self.model = int(l0), model
        self.embed = model.get_input_embeddings()
        self.blocks = self.sec.blocks
        self._cache = {}
        if build:
            self.sec.build()

    def __enter__(self):
        self.sec.__enter__()
        return self

    def __exit__(self, *a):
        self.restore()

    @property
    def parked(self):
        return self.sec.parked

    def clear(self):
        self._cache = {}

    def restore(self):
        self.clear()
        self.sec.restore()

    @torch.no_grad()
    def lower(self, ids, record=(), writes=()):
        rec = sorted(set(int(l) for l in record if l <= self.l0))
        key = tuple(ids)
        if not writes and not rec and key in self._cache:
            return self._cache[key]
        store, kw = self.sec.capture(ids, layers=rec, writes=writes)
        if not writes and not rec:
            self._cache[key] = (store, kw)
        return store, kw

    @torch.no_grad()
    def forward(self, ids, record=(), upper=None, lower=()):
        store, kw = self.lower(ids, record=record, writes=lower)
        up_rec = [l for l in record if l > self.l0]
        fh, rec = self.sec.replay(store[self.l0], kw, edits=upper or {}, record=up_rec)
        acts = {l: store[l] for l in record if l <= self.l0}
        acts.update(rec)
        return fh, acts

    @torch.no_grad()
    def logits(self, fh, positions):
        return fh[list(positions)] @ self.sec.W_U.T

    @torch.no_grad()
    def logprobs(self, fh, positions):
        return torch.log_softmax(self.logits(fh, positions), -1)

    @torch.no_grad()
    def score(self, prompt_ids, S_a, S_b, edit_fn=None, extra_spellings=()):
        """margin.score under SEC. edit_fn(cont) -> (upper dict, lower list) for the forward on prompt + cont, so sources
        can be taken at the matching sequence length. extra_spellings: further spelling sets whose log-probs are also
        returned (e.g. all 12 colours for answer-distribution shifts)."""
        prompt_ids = list(prompt_ids)
        T = len(prompt_ids)

        def run(cont):
            up, lo = edit_fn(tuple(cont)) if edit_fn is not None else ({}, [])
            fh, _ = self.forward(prompt_ids + list(cont), upper=up, lower=lo)
            return fh
        base = run(())
        lp_last = self.logprobs(base, [T - 1])[0]
        spell = {}
        allsp = list(S_a) + list(S_b) + [s for group in extra_spellings for s in group]
        for s in dict.fromkeys(allsp):
            if len(s) == 1:
                per = [float(lp_last[s[0]])]
            else:
                fh = run(s)
                lp = self.logprobs(fh, range(T - 1, T - 1 + len(s)))
                per = [float(lp[i, t]) for i, t in enumerate(s)]
            spell[s] = {"lp": float(sum(per)), "per_token": per}

        def lse(group):
            return float(torch.logsumexp(torch.tensor([spell[s]["lp"] for s in group], dtype=torch.float64), 0))
        lse_a, lse_b = lse(S_a), lse(S_b)
        return {"m": lse_b - lse_a, "lse_a": lse_a, "lse_b": lse_b, "spell": spell, "fh": base, "lp_last": lp_last,
                "first_tokens": sorted({s[0] for s in list(S_a) + list(S_b)}),
                "extra_lse": [lse(g) for g in extra_spellings]}


@torch.no_grad()
def sec_damage(eng, clean, edited, edit_pos, T):
    """gates.damage for SecEngine scores: dlogP(a), dlogP(a'), top-1 retention at teacher-forced prompt positions after
    the edit (answer position excluded), off-pair mass at the answer position."""
    import math
    pos = list(range(int(edit_pos) + 1, T - 1))
    if pos:
        c = eng.logits(clean["fh"], pos).argmax(-1)
        e = eng.logits(edited["fh"], pos).argmax(-1)
        ret = float((c == e).float().mean())
    else:
        ret = math.nan
    ft = clean["first_tokens"]
    off = lambda sc: float(1.0 - sc["lp_last"][ft].exp().sum())
    return {"d_logP_a": edited["lse_a"] - clean["lse_a"], "d_logP_b": edited["lse_b"] - clean["lse_b"],
            "top1_retention": ret, "n_retention_positions": len(pos),
            "off_pair_mass_clean": off(clean), "off_pair_mass_edit": off(edited)}
