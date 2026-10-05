"""Design 005 (planted concepts) library: frozen material, renders, injection / clamp hooks, batched float32-residual
forwards, cache-free sampling, model-side metrics and reader scoring (docs/design_specs/005_planted_concepts.md v2).

Reader-side code (it builds J-space vectors and drives readers): label code must never import it
(tests/test_import_boundary.py). Pure functions are unit-tested in scripts/common/tests/test_planted.py.
"""
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/common"))
import engine as E          # noqa: E402
import registry as R        # noqa: E402

# ------------------------------------------------------------------------------------------------ frozen material
SEED = 20261004
PROTOCOL_DIR = Path(R.REGISTRY["paths"]["jlens_clone"]) / "data/experiments"
PROTOCOL_SHA = {"verbal-introspection.json": "c16e92d32b43e50027863d939445bb839301457f494363c00e2d00e5c6e73560",
                "verbal-report.json": "9a33b48074c4565413247bace11d37537a963774936740287f0fb7dff460652c"}
DEV = ["loss", "enemy", "union", "skin", "sin", "storm", "profit", "frame", "gel", "earth"]
TEST = ["corner", "thumb", "ear", "review", "custom", "dog", "symbol", "free", "ship", "role", "friend", "cook", "stand",
        "gray", "curve", "method", "weight", "summary", "photo", "bags", "desk", "pool", "ash", "steel", "tap", "case",
        "box", "trade", "bomb", "soap", "process", "picture", "diamond", "ease", "hat", "sheet", "information", "race",
        "hair", "plane", "judge", "board", "formula", "sock", "scene"]
EXCLUDED = ["time", "line", "path"]                          # concept screen (results/005_planted_concepts/analysis)
CONCEPTS = DEV + TEST
GATE = DEV[:6]
SMOKE = ["gel", "earth"]
BAND = list(range(32, 39))                                   # L32-L38
BAND2 = list(range(24, 39))                                  # G1 fallback
READ_LAYERS = [40, 42, 44]
GRID = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
K = 16
POS = {"T": 139, "span": list(range(97, 117)), "cq": 110, "prefill": list(range(126, 139)), "co": 138}
NEUTRAL_TEXT = "Write one sentence about the weather."
NEUTRAL = {"T": 19, "span": list(range(3, 10)), "n_cont": 8}
TEMPLATES = ["Here is a short note about {w}.", "My favourite thing to think about is {w}.", "Can you describe a {w}?",
             "I keep thinking about the {w}.", "The word of the day is {w}.", "Write a haiku about {w}."]
NAME_SAMPLES, NAME_NEW = 8, 8
TEXT_K, CLEAN_K = 4, 32
assert len(DEV) == 10 and len(TEST) == 45 and not set(DEV) & set(TEST) and not set(EXCLUDED) & set(CONCEPTS)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol():
    """(intro messages, default prefill, the 101 concept surfaces, the 194 baseline words), sha256-checked."""
    for f, h in PROTOCOL_SHA.items():
        assert _sha(PROTOCOL_DIR / f) == h, f"{f} sha256 mismatch (jacobian-lens clone not at 581d398?)"
    d = json.loads((PROTOCOL_DIR / "verbal-introspection.json").read_text())
    vr = json.loads((PROTOCOL_DIR / "verbal-report.json").read_text())
    base = sorted({w for v in vr["candidates"].values() for w in v})
    concepts = [c["surface"] for c in d["concepts"]]
    assert len(concepts) == 101 and len(set(concepts)) == 101 and len(base) == 194
    assert set(CONCEPTS) <= set(concepts) and not set(base) & set(concepts)
    return d["intro_prompt"], d["prefills"]["default"], concepts, base


# ------------------------------------------------------------------------------------------------------- renders
def _chat(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prefills():
    """The protocol's prefills: {'default': ..., 'word': ...} (sha256-checked file)."""
    protocol()
    return json.loads((PROTOCOL_DIR / "verbal-introspection.json").read_text())["prefills"]


def render_intro(tok, variant="default"):
    """ids of the introspection prompt + the stripped prefill `variant`, with every frozen position asserted
    (default: 139 tokens; the report position C_o is always the last token, the open quote)."""
    msgs, _, _, _ = protocol()
    prefill = prefills()[variant]
    msgs = [m for m in msgs if not (m["role"] == "assistant" and m["content"] == "")]
    head = _chat(tok, msgs)
    txt = head + prefill.lstrip()
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = list(enc["input_ids"]), enc["offset_mapping"]
    a = txt.rfind("Trial 1")
    b = txt.find("<|im_end|>", a)
    span = [i for i, (x, y) in enumerate(off) if x >= a and y <= b]
    pre = [i for i, (x, y) in enumerate(off) if x >= len(head)]
    if variant == "default":
        assert len(ids) == POS["T"], len(ids)
        assert pre == POS["prefill"], (pre[0], pre[-1])
    assert pre[0] == POS["prefill"][0] and pre[-1] == len(ids) - 1, (pre[0], pre[-1])
    assert span == POS["span"], (span[0], span[-1])
    assert txt[off[POS["cq"]][0]:off[POS["cq"]][1]] == "," and txt[off[POS["cq"]][0] - 5:off[POS["cq"]][0]] == "If so"
    assert txt[off[-1][0]:off[-1][1]] == ' "'
    return ids


def render_neutral(tok):
    txt = _chat(tok, [{"role": "user", "content": NEUTRAL_TEXT}])
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = list(enc["input_ids"]), enc["offset_mapping"]
    a = txt.find(NEUTRAL_TEXT)
    span = [i for i, (x, y) in enumerate(off) if x >= a and y <= a + len(NEUTRAL_TEXT)]
    assert len(ids) == NEUTRAL["T"] and span == NEUTRAL["span"], (len(ids), span)
    return ids


def render_tell(tok, w):
    return list(tok(_chat(tok, [{"role": "user", "content": f"Tell me about {w}"}]), add_special_tokens=False)["input_ids"])


def render_template(tok, tmpl, w):
    return list(tok(_chat(tok, [{"role": "user", "content": tmpl.format(w=w)}]), add_special_tokens=False)["input_ids"])


def spellings(tok, c):
    """Single-token spellings of c: 'quote' (right after the open quote: c, C) and 'all' (c, C, " c", " C")."""
    def one(s):
        t = tok.encode(s, add_special_tokens=False)
        return t[0] if len(t) == 1 else None
    _, prefill, _, _ = protocol()
    base = tok.encode(prefill, add_special_tokens=False)
    quote = []
    for s in (c, c.capitalize()):
        ctx = tok.encode(prefill + s, add_special_tokens=False)
        assert ctx[:len(base)] == base and len(ctx) == len(base) + 1, f"{s!r} is not one token after the quote"
        quote.append(ctx[-1])
    space = [t for t in (one(" " + c), one(" " + c.capitalize())) if t is not None]
    assert one(" " + c) is not None, c
    return {"quote": sorted(set(quote)), "all": sorted(set(quote) | set(space)), "space": sorted(set(space)),
            "space_lower": one(" " + c)}


# ------------------------------------------------------------------------------------------------------- scoring
def _variants(c):
    return (c, c + "s", c + "es")


_PAT = {}


def names(text, c):
    """Exact spelling-variant match (design §7): c, c+s, c+es as a whole word, case-insensitive."""
    if c not in _PAT:
        _PAT[c] = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(v) for v in _variants(c)), re.I)
    return bool(_PAT[c].search(text))


def named_in_quote(text, c):
    """R2: the text before the closing quote, stripped of punctuation and space, equals a variant (case-insensitive)."""
    head = text.split('"')[0]
    word = re.sub(r"[^\w\s-]", "", head).strip().lower()
    return word in _variants(c)


# ------------------------------------------------------------------------------------------------- seeded randomness
def seed_of(*parts):
    return (SEED * 1000003 + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 62)


def gaussian(d, *parts, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed_of(*parts))
    return torch.randn(d, generator=g).to(device)


def unit(v):
    v = v.float()
    return v / v.norm().clamp_min(1e-30)


# ------------------------------------------------------------------------------------------------------- hooks
class BandInject(E._Hooked):
    """Per-row additive write h[b, pos, :] += delta[l][b] at each layer l (float32 residual; positions fixed, so a longer
    sequence during sampling is written only at the prompt positions). Records fidelity per layer (rho, kappa over rows
    with a nonzero write)."""

    def __init__(self, blocks, delta, positions):
        super().__init__(blocks, list(delta))
        self.delta, self.pos = delta, torch.as_tensor(list(positions), dtype=torch.long)
        self.fid = {}

    def _mk(self, l):
        @torch.autocast("cuda", enabled=False)
        def f(m, i, o):
            h = E._out(o)
            assert h.dtype == torch.float32, "BandInject needs the float32-residual regime"
            d = self.delta[l].to(h.device, torch.float32)
            assert d.shape[0] == h.shape[0], (d.shape, h.shape)
            h = h.clone()
            pos = self.pos.to(h.device)
            before = h[:, pos, :].clone()
            h[:, pos, :] = before + d[:, None, :]
            real = h[:, pos, :] - before
            req = d[:, None, :].expand_as(real)
            live = d.norm(dim=-1) > 0
            if bool(live.any()):
                rn, qn = real[live].flatten(1).norm(dim=1), req[live].flatten(1).norm(dim=1)
                kap = (real[live].flatten(1) * req[live].flatten(1)).sum(1) / (rn * qn).clamp_min(1e-30)
                rho = rn / qn.clamp_min(1e-30)
                self.fid[l] = {"rho_min": float(rho.min()), "rho_max": float(rho.max()), "kappa_min": float(kap.min()),
                               "rows": int(live.sum())}
            return E._pack(o, h)
        return f


class BandClamp(E._Hooked):
    """Per-row privilege clamp: h[b] <- h[b] - Q Q^T (h[b] - ref) at every position of every layer in Q_by_row[b]
    (rows without an entry are untouched). ref = h_clean, or, for a re-entry clamp (design §16.3), h_clean + A with
    A = offsets[b][layer] (the accumulated raw injection) added at `offset_pos`; since h - h_clean = A + M, the re-entry
    clamp removes only M's component (what the model's blocks wrote) in span(Q). Logs, per row and layer, the share of
    the perturbation h - ref removed and the post-clamp residual |Q^T (h - ref)| / |Q^T ref|."""

    def __init__(self, blocks, Q_by_row, h_clean, layers, offsets=None, offset_pos=None):
        super().__init__(blocks, layers)
        self.Q, self.hc = Q_by_row, h_clean
        self.off = offsets or {}
        self.off_pos = None if offset_pos is None else torch.as_tensor(list(offset_pos), dtype=torch.long)
        self.log = {}

    def _mk(self, l):
        @torch.autocast("cuda", enabled=False)
        def f(m, i, o):
            h = E._out(o)
            assert h.dtype == torch.float32
            rows = [b for b, q in self.Q.items() if l in q]
            if not rows:
                return None
            h = h.clone()
            hc = self.hc[l].to(h.device, torch.float32)[: h.shape[1]]
            for b in rows:
                Q = self.Q[b][l].to(h.device, torch.float32)
                ref = hc
                if b in self.off and l in self.off[b]:
                    ref = hc.clone()
                    pos = self.off_pos.to(h.device)
                    ref[pos] = ref[pos] + self.off[b][l].to(h.device, torch.float32)
                pert = h[b] - ref
                coord = pert @ Q
                removed = coord @ Q.T
                h[b] = h[b] - removed
                pn = float(pert.norm())
                resid = float(((h[b] - ref) @ Q).norm()) / max(float((ref @ Q).norm()), 1e-30)
                self.log.setdefault(b, {})[l] = {"removed_share": float(removed.norm()) / pn if pn > 0 else 0.0,
                                                 "residual": resid}
            return E._pack(o, h)
        return f


class RowRecorder(E._Hooked):
    """Record block outputs at `positions` for every row: {l: [B, P, d] float32 on device}."""

    def __init__(self, blocks, layers, positions):
        super().__init__(blocks, layers)
        self.pos = torch.as_tensor(list(positions), dtype=torch.long)
        self.acts = {}

    def _mk(self, l):
        def f(m, i, o):
            h = E._out(o)
            self.acts[l] = h[:, self.pos.to(h.device), :].detach().float().clone()
        return f


# ---------------------------------------------------------------------------------------------- forwards and sampling
def text_model(model):
    return (model.get_base_model() if hasattr(model, "get_base_model") else model).model


@torch.no_grad()
def forward_rows(model, ids, B, W_U, logit_pos, writes=(), record=(), record_pos=(), fp32=True):
    """One batched teacher-forced forward of B identical rows. Returns (log-probs [B, P, V] float32 at logit_pos,
    {layer: [B, len(record_pos), d]})."""
    R.assert_subject_model_intact(model)
    R.assert_subject_model_resident(model)
    blocks = R.blocks(model)
    x = torch.tensor([list(ids)] * B, device=W_U.device)
    rec = RowRecorder(blocks, record, record_pos) if record else None
    regime = E.Fp32Residual(blocks, 0) if fp32 else None
    with E.Both(regime, *writes, rec):
        fh = text_model(model)(input_ids=x, use_cache=False).last_hidden_state[:, list(logit_pos), :].float()
    with torch.autocast("cuda", enabled=False):
        lp = torch.log_softmax(fh @ W_U.T, -1)
    return lp, (rec.acts if rec else {})


def top_p_sample(logits, gen, top_p=0.95, temperature=1.0):
    """One token per row from float32 logits [B, V] (temperature, then nucleus top-p; HF semantics: keep the smallest
    prefix whose mass reaches top_p)."""
    probs = torch.softmax(logits.float() / temperature, -1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = sp.cumsum(-1)
    sp = torch.where(cum - sp > top_p, torch.zeros_like(sp), sp)
    pick = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)
    return si.gather(-1, pick)[:, 0]


@torch.no_grad()
def sample_rows(model, ids, W_U, delta, positions, n_new, seed, fp32=True):
    """Cache-free sampling: each step re-runs the full sequence with BandInject at the prompt positions only.
    delta: {layer: [B, d]} (B rows). Returns [B, n_new] token ids (CPU)."""
    blocks = R.blocks(model)
    B = next(iter(delta.values())).shape[0]
    gen = torch.Generator(device=W_U.device).manual_seed(int(seed))
    x = torch.tensor([list(ids)] * B, device=W_U.device)
    for _ in range(n_new):
        inj = BandInject(blocks, delta, positions)
        regime = E.Fp32Residual(blocks, 0) if fp32 else None
        with E.Both(regime, inj):
            fh = text_model(model)(input_ids=x, use_cache=False).last_hidden_state[:, -1, :].float()
        with torch.autocast("cuda", enabled=False):
            nxt = top_p_sample(fh @ W_U.T, gen)
        x = torch.cat([x, nxt[:, None]], 1)
    return x[:, len(ids):].cpu()


@torch.no_grad()
def capture_last(model, id_lists, layers, bs=48):
    """Stock bf16 forward (the reader-input convention): last-token residual at `layers` for each id list.
    Prompts are batched by equal length (no padding). Returns [N, len(layers), d] float32 CPU."""
    R.assert_subject_model_intact(model)
    R.assert_subject_model_resident(model)
    blocks = R.blocks(model)
    out = torch.zeros(len(id_lists), len(layers), R.D_MODEL)
    by_len = {}
    for i, ids in enumerate(id_lists):
        by_len.setdefault(len(ids), []).append(i)
    dev = next(model.parameters()).device
    for T, idx in sorted(by_len.items()):
        for j in range(0, len(idx), bs):
            chunk = idx[j:j + bs]
            rec = RowRecorder(blocks, layers, [T - 1])
            with rec:
                text_model(model)(input_ids=torch.tensor([id_lists[i] for i in chunk], device=dev), use_cache=False)
            for li, l in enumerate(layers):
                out[chunk, li] = rec.acts[l][:, 0].cpu()
    return out


# ---------------------------------------------------------------------------------------------- model-side metrics
def rank_of(lp_row, toks):
    """1-based rank of the best of `toks` in a [V] score vector."""
    best = lp_row[list(toks)].max()
    return int((lp_row > best).sum()) + 1


def ranks_many(score_row, tok_sets):
    """1-based rank of the best token of each set in tok_sets (list of lists) for one [V] score vector."""
    best = torch.stack([score_row[list(t)].max() for t in tok_sets])
    return ((score_row[None, :] > best[:, None]).sum(1) + 1).tolist()


def intro_metrics(lp, lp_clean, sp):
    """lp, lp_clean: [B, 14, V] / [14, V] log-probs at positions 125..138 of the introspection prompt.
    Returns per row: R1 (reciprocal rank at C_o), R1p (log-sum-exp log-prob at C_o over the quote spellings) and its
    delta vs clean, B3 (max probability of any single-token spelling at the prefill-predicting positions 125..137),
    B4 (mean KL(clean || injected) at 126..137)."""
    co = lp.shape[1] - 1
    q, a = list(sp["quote"]), list(sp["all"])
    out = []
    r1p_clean = float(torch.logsumexp(lp_clean[co, q], 0))
    pc = lp_clean.exp()
    for b in range(lp.shape[0]):
        r1p = float(torch.logsumexp(lp[b, co, q], 0))
        b3 = float(lp[b, :co, a].exp().sum(-1).max())
        kl = (pc[1:co] * (lp_clean[1:co] - lp[b, 1:co])).sum(-1)
        out.append({"R1": 1.0 / rank_of(lp[b, co], q), "R1p": r1p, "dR1p": r1p - r1p_clean, "B3": b3,
                    "B4": float(kl.mean())})
    return out


def neutral_metrics(lp, lp_clean, sp):
    """lp: [B, 8, V] at the 8 positions predicting the clean continuation. B1 = mean KL(clean || injected);
    B2 = max over positions of the change in log P(any single-token spelling)."""
    a = list(sp["all"])
    pc = lp_clean.exp()
    base = torch.logsumexp(lp_clean[:, a], -1)
    out = []
    for b in range(lp.shape[0]):
        kl = (pc * (lp_clean - lp[b])).sum(-1)
        out.append({"B1": float(kl.mean()), "B2": float((torch.logsumexp(lp[b][:, a], -1) - base).max())})
    return out


def forced_candidates(tok, prefix_text, words):
    """Design §16.7 11b. The reader's answer is teacher-forced to start with `prefix_text`; returns (prefix ids,
    {w: [ids]}) with, for each word, its space spellings (" w", " W") that are one token right after the prefix.
    Words with neither are left out."""
    base = tok.encode(prefix_text, add_special_tokens=False)
    out = {}
    for w in words:
        ids = set()
        for sp in (" " + w, " " + w.capitalize()):
            t = tok.encode(prefix_text + sp, add_special_tokens=False)
            if t[:len(base)] == base and len(t) == len(base) + 1:
                ids.add(t[-1])
        if ids:
            out[w] = sorted(ids)
    return base, out


def specificity(dw, c, dev):
    """Design §16.6 10b. dw: {word: ΔR1′} for one row over the word set W (which contains `dev`); c: the planted
    concept. ID: c has the largest ΔR1′ among `dev`; SM = ΔR1′(c) − median over W∖{c}; pct: share of W∖{c} below c;
    generic: the median over W∖{c} (the rise of every other word)."""
    others = np.array([v for w, v in dw.items() if w != c], float)
    gen = float(np.median(others))
    return {"ID": max(dev, key=lambda w: dw[w]) == c, "SM": float(dw[c]) - gen,
            "pct": float((others < dw[c]).mean()), "generic": gen}


# ---------------------------------------------------------------------------------------------- G1 and onset rules
def g1_decide(rows, grid):
    """rows: dicts {concept, cond (1, 2 or 7), s, R2, B3, B1, R1} for the gate concepts.
    Returns the pass decision and the calibration action (design §9 G1)."""
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    res = {}
    for cond in (1, 2):
        per_s = []
        for s in grid:
            r2 = [r["R2"] for r in rows if r["cond"] == cond and r["s"] == s]
            r1 = [r["R1"] for r in rows if r["cond"] == cond and r["s"] == s]
            per_s.append((mean(r2), float(np.median(r1)) if r1 else 0.0, -s, s))
        best = max(per_s)                     # peak by mean R2, ties by median R1, then the lower strength
        s_b = best[3]
        r2 = best[0]
        r2_rand = mean([r["R2"] for r in rows if r["cond"] == 7 and r["s"] == s_b])
        b3 = float(np.median([r["B3"] for r in rows if r["cond"] == cond and r["s"] == s_b]))
        ok = r2 >= 0.25 and (r2 - r2_rand) >= 0.20 and b3 < 0.05
        res[cond] = {"best_s": s_b, "R2": r2, "R2_rand": r2_rand, "B3_median": b3, "pass": bool(ok)}
    action = None
    if res[1]["best_s"] == grid[-1]:
        action = "x4"
    else:
        hi = [s for s in grid if s >= 1 / 8]
        b1_hi = [float(np.median([r["B1"] for r in rows if r["cond"] == 1 and r["s"] == s])) for s in hi]
        if hi and all(b > 5.0 for b in b1_hi) and res[1]["best_s"] == grid[0]:
            action = "/4"
    return {"pass": bool(res[1]["pass"] or res[2]["pass"]), "by_cond": res, "calibration": action}


def onset(rows, grid):
    """rows: {concept, cond (4 or 7), seed, s, dR1p} on the dev concepts. s* = the smallest strength where the median
    over concepts of dR1p(u_perp) exceeds the 95th percentile of dR1p under rand-dir (all seeds x concepts)."""
    trace = []
    s_star = None
    for s in grid:
        null = [r["dR1p"] for r in rows if r["cond"] == 7 and r["s"] == s]
        perp = [r["dR1p"] for r in rows if r["cond"] == 4 and r["s"] == s]
        q95 = float(np.quantile(null, 0.95)) if null else float("nan")
        med = float(np.median(perp)) if perp else float("nan")
        trace.append({"s": s, "null95": q95, "median_perp": med, "above": bool(med > q95)})
        if s_star is None and med > q95:
            s_star = s
    return s_star, trace


def text_strengths(s_star, grid):
    """{s*/4, s*, 4 s*} mapped to grid indices (two grid steps = x4), kept distinct and inside the grid; without an
    onset, the three highest grid strengths."""
    if s_star is None:
        return list(grid[-3:])
    i = grid.index(s_star)
    want = [i - 2, i, i + 2]
    idx = []
    for j in want:
        j = min(max(j, 0), len(grid) - 1)
        while j in idx:
            j = j + 1 if j + 1 < len(grid) and j + 1 not in idx else j - 1
        idx.append(j)
    return [grid[j] for j in sorted(idx)]
