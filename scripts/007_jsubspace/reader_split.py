"""007 reader-side scoring (design §4, §7; amendment A1): the Oracle (L40, L44) and the NLA (L42) full-sequence colour
likelihoods on every piece of the brew captures, plus the no-activation row, the per-piece positive control, the
prompt-only reader floor, the generic mean (wikitext), qualitative samples and the stop gates (G-score1/2, G-repro,
G-inject with the scale check, G-mem). Reader-side code; never imported by label code.

  python reader_split.py stage1a --run-id r1      # brew dev, ':' (primary)
  python reader_split.py stage1b --run-id r1      # brew dev, '</think>' (needs stage1a's u_gen)

Every (reader, layer, piece) block writes one file and is skipped on a rerun. A shared-path gate failure for a reader
(G-score1/2, G-inject, G-repro) is recorded in outputs/007_jsubspace/disabled_readers.json and disables that reader in
every later stage (A1).
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "005_planted_concepts"))
import brew_material as BM                        # noqa: E402
import p005 as P5                                 # noqa: E402  (RowRecorder)
import readers as RD                              # noqa: E402
import registry as R                              # noqa: E402
import dstat                                      # noqa: E402
import p007 as C                                  # noqa: E402

PREFIX0 = {"oracle": "-", "nla": "<explanation>\nThe text is about"}      # 006's primary prefixes
WANT = {"emit_asst": ("assistant", "\n"), "emit_think": ("</think>", "\n\n"), "emit_colon": ("Answer", ":")}
DISABLED = C.OUT / "disabled_readers.json"
WSB_PO = Path("/workspace/sources/repos/workspace-bench/evals/baselines/prompt_only.json")
PO_PREFILL = "The model is thinking about"


class GateFailure(Exception):
    def __init__(self, reader, gate, detail):
        super().__init__(f"{reader}: {gate} failed: {detail}")
        self.reader, self.gate, self.detail = reader, gate, detail


# ------------------------------------------------------------------------------------------------ speed-up (exact)
_ORIG_SEQ_LP = RD.seq_logprob_from_hidden


def seq_logprob_fast(Hn, W32, gather):
    """Same arithmetic as readers.seq_logprob_from_hidden (float32 logits at the gathered positions, log-softmax, sum per
    row), vectorised over rows instead of a Python loop. Checked against the original at run start (G-fast)."""
    rows = [r for r, g in enumerate(gather) for _ in g]
    pos = [p for g in gather for p, _ in g]
    tgt = [t for g in gather for _, t in g]
    dev = Hn.device
    with torch.autocast(dev.type, enabled=False):
        H = Hn[torch.tensor(rows, device=dev), torch.tensor(pos, device=dev)].float()
        out = torch.empty(len(rows), device=dev)
        for j in range(0, len(rows), 1024):                                   # bound the [N, V] logits buffer
            lp = torch.log_softmax(H[j:j + 1024] @ W32.T, -1)
            out[j:j + 1024] = lp[torch.arange(lp.shape[0], device=dev), torch.tensor(tgt[j:j + 1024], device=dev)]
    seg = torch.tensor(rows, device=dev)
    return torch.zeros(len(gather), device=dev).index_add_(0, seg, out)


def spellings(tok, prefix):
    base = tok.encode(prefix, add_special_tokens=False)
    seqs, owner = [], []
    for c in C.COLOURS:
        for form in (" " + c, " " + c.capitalize()):
            full = tok.encode(prefix + form, add_special_tokens=False)
            assert full[:len(base)] == base, (prefix[-40:], form)
            seqs.append(full[len(base):])
            owner.append(c)
    return base, seqs, owner


def _rows_for(P, cand_seqs, pad):
    """Rows for one prefix P: row 0 = P alone (its last position scores every single-token spelling), then one row per
    multi-token spelling (P + spelling, full-sequence scored as in readers.cont_batch). Right-padded to a common T."""
    single = [i for i, s in enumerate(cand_seqs) if len(s) == 1]
    multi = [i for i, s in enumerate(cand_seqs) if len(s) > 1]
    conts = [()] + [tuple(cand_seqs[i]) for i in multi]
    T = len(P) + max(len(c) for c in conts)
    rows, mask, gather = [], [], []
    for c in conts:
        r = list(P) + list(c)
        rows.append(r + [pad] * (T - len(r)))
        mask.append([1] * len(r) + [0] * (T - len(r)))
        gather.append([(len(P) - 1 + k, int(c[k])) for k in range(len(c))])
    return rows, mask, gather[1:], single, multi, [int(cand_seqs[i][0]) for i in single]


def _assemble(Hn, W32, row0, Lps, gms, single, multi, stok, S):
    """Hn [rows, T, d]. row0[a] = index of activation a's prefix row; its multi rows follow it. Lps[a] = its prefix
    length; gms[a] = gathers of its multi rows. Returns [n_act, S] float32 full-sequence log-probs."""
    dev = Hn.device
    n = len(row0)
    out = torch.empty(n, S, device=dev)
    with torch.autocast(dev.type, enabled=False):
        pre = Hn[torch.tensor(row0, device=dev), torch.tensor([L - 1 for L in Lps], device=dev)].float()
        out[:, single] = torch.log_softmax(pre @ W32.T, -1)[:, stok]
    if multi:
        ridx = [row0[a] + 1 + j for a in range(n) for j in range(len(multi))]
        g = [gms[a][j] for a in range(n) for j in range(len(multi))]
        out[:, multi] = seq_logprob_fast(Hn[torch.tensor(ridx, device=dev)], W32, g).view(n, len(multi))
    return out


def _own_logit_check(lg, row0, Lps, gms, single, multi, stok, res):
    """The model's own logits (lm_head path) for the same quantities; max |difference| (G-score2)."""
    n = len(row0)
    own = torch.empty(n, len(single) + len(multi), device=lg.device)
    for a in range(n):
        lp0 = torch.log_softmax(lg[row0[a], Lps[a] - 1].float(), -1)
        own[a, single] = lp0[stok]
        for j, gi in enumerate(gms[a]):
            r = row0[a] + 1 + j
            own[a, multi[j]] = sum(torch.log_softmax(lg[r, p_].float(), -1)[t_] for p_, t_ in gi)
    return float((own.cpu() - res.cpu()).abs().max())


def oracle_fast(ol, hs, layer, prefix_ids, cand_seqs, W32, acts_per_batch=16, check=False):
    """readers.OracleLens.score_seq with 2 rows per activation instead of 24 (same injection, same arithmetic)."""
    ids, slot, _ = ol.carrier(layer)
    pad = ol.tok.pad_token_id if ol.tok.pad_token_id is not None else ol.tok.eos_token_id
    P = list(ids) + list(prefix_ids)
    rows, mask, gm, single, multi, stok = _rows_for(P, cand_seqs, pad)
    R_ = len(rows)
    dev = ol.model.device
    emb = ol.model.get_input_embeddings()(torch.tensor(rows, device=dev))
    am = torch.tensor(mask, device=dev)
    inner = ol.model.get_base_model().model
    out = []
    for j in range(0, len(hs), acts_per_batch):
        ch = hs[j:j + acts_per_batch]
        e = emb.repeat(len(ch), 1, 1).clone()
        for a, h in enumerate(ch):
            if h is None:                                                  # no-activation row
                continue
            v = h.to(dev).float()
            e[a * R_:(a + 1) * R_, slot, :] = (ol.c["alpha"] * v / v.norm().clamp_min(1e-9)).to(e.dtype)
        m = am.repeat(len(ch), 1)
        Hn = inner(inputs_embeds=e, attention_mask=m, use_cache=False).last_hidden_state
        row0 = [a * R_ for a in range(len(ch))]
        res = _assemble(Hn, W32, row0, [len(P)] * len(ch), [gm] * len(ch), single, multi, stok, len(cand_seqs))
        if check and j == 0:
            k = min(2, len(ch))
            lg = ol.model(inputs_embeds=e[:k * R_], attention_mask=m[:k * R_], use_cache=False).logits
            ol.last_check = _own_logit_check(lg, row0[:k], [len(P)] * k, [gm] * k, single, multi, stok, res[:k])
            del lg
        out.append(res.cpu())
        del Hn, e
    return torch.cat(out)


def nla_fast(nla, hs, prefix_ids, cand_seqs, acts_per_batch=16, check=False):
    """readers.NLArl.score_seq with 2 rows per activation instead of 24 (same Karvonen add at block 1)."""
    if not hasattr(nla, "_W32"):
        nla._W32 = nla.model.get_output_embeddings().weight.detach().float()
    pad = nla.tok.pad_token_id if nla.tok.pad_token_id is not None else nla.tok.eos_token_id
    P = list(nla.ids) + list(prefix_ids)
    rows, mask, gm, single, multi, stok = _rows_for(P, cand_seqs, pad)
    R_ = len(rows)
    dev = nla.model.device
    x0, m0 = torch.tensor(rows, device=dev), torch.tensor(mask, device=dev)
    inner = nla.model.get_base_model().model
    out = []
    for j in range(0, len(hs), acts_per_batch):
        ch = hs[j:j + acts_per_batch]
        V = torch.stack([h.float() if h is not None else torch.zeros(R.D_MODEL) for h in ch]).repeat_interleave(R_, 0)
        x, m = x0.repeat(len(ch), 1), m0.repeat(len(ch), 1)
        row0 = [a * R_ for a in range(len(ch))]
        try:
            nla._v["v"] = V
            Hn = inner(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state
            res = _assemble(Hn, nla._W32, row0, [len(P)] * len(ch), [gm] * len(ch), single, multi, stok, len(cand_seqs))
            if check and j == 0:
                k = min(2, len(ch))
                nla._v["v"] = V[:k * R_]
                lg = nla.model(input_ids=x[:k * R_], attention_mask=m[:k * R_], use_cache=False).logits
                nla.last_check = _own_logit_check(lg, row0[:k], [len(P)] * k, [gm] * k, single, multi, stok, res[:k])
                del lg
        finally:
            nla._v["v"] = None
        out.append(res.cpu())
        del Hn
    return torch.cat(out)


def floor_ok(a, b, mx=0.25, md=0.05):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return float(np.max(d)), float(np.median(d)), bool(np.max(d) <= mx and np.median(d) <= md)


# ------------------------------------------------------------------------------------------------ the stage
class Stage:
    def __init__(self, stage, run_id, nla_posthoc=False):
        self.posthoc = nla_posthoc                                         # A2: NLA only, 006's exact 24-row scorer
        self.stage = stage
        self.cname = {"stage1a": "colon", "stage1b": "think", "brew_test": "colon"}[stage]
        self.cell = C.CELLS[self.cname]
        self.dir = C.OUT / stage / run_id
        (self.dir / "ll").mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.log = C.Logger(self.dir / "log.txt")
        self.test = stage == "brew_test"
        self.order = C.TEST_PIECES if self.test else C.PIECE_ORDER
        src = (C.OUT / "brew_test" / "pieces") if self.test else C.STAGE0
        self.roles = json.loads(((C.OUT / "brew_test") if self.test else C.STAGE0).joinpath("roles.json").read_text())
        self.n = len(self.roles["renders"])
        self.pieces = {L: dict(np.load(src / f"pieces_{self.cname}_L{L}.npz")) for L in C.LAYERS}
        self.pc = {L: dict(np.load(src / f"pcpieces_{self.cname}_L{L}.npz")) for L in C.LAYERS}
        self.bases = torch.load(C.STAGE0 / "bases_f32.pt")
        self.disabled = json.loads(DISABLED.read_text()) if DISABLED.exists() else {}

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1, default=float))

    def gpu_time(self, key, t):
        self.state.setdefault("seconds", {})[key] = self.state.get("seconds", {}).get(key, 0) + (time.time() - t)
        self.save()

    def disable(self, err):
        self.disabled[err.reader] = {"gate": err.gate, "detail": err.detail, "stage": self.stage,
                                     "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        DISABLED.write_text(json.dumps(self.disabled, indent=1))
        self.state.setdefault("gate_failures", []).append(self.disabled[err.reader])
        self.save()
        self.log("STOP GATE:", err)

    # ---------------------------------------------------------------- generic mean (wikitext, stage 1a Phase A)
    def generic_mean(self, model, tok):
        f = C.OUT / "stage1a" / "u_gen.pt"
        if f.exists():
            return torch.load(f)
        assert self.stage == "stage1a", "stage1a must have produced u_gen.pt"
        import pandas as pd
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
        texts = pd.read_parquet(p)["text"].tolist()
        sel, ids = [], []
        for i, t in enumerate(texts):
            s = t.strip()
            if not s or s.startswith("="):
                continue
            x = tok.encode(s, add_special_tokens=False)
            if len(x) >= 128:
                sel.append(i)
                ids.append(x[:128])
            if len(ids) == 200:
                break
        assert len(ids) == 200
        blocks = R.blocks(model)
        sums = {L: torch.zeros(C.D, dtype=torch.float64) for L in C.LAYERS}
        cnt = 0
        dev = next(model.parameters()).device
        with torch.no_grad():
            for j in range(0, 200, 25):
                rec = P5.RowRecorder(blocks, list(C.LAYERS), list(range(16, 128)))
                with rec:
                    P5.text_model(model)(input_ids=torch.tensor(ids[j:j + 25], device=dev), use_cache=False)
                for L in C.LAYERS:
                    sums[L] += rec.acts[L].double().sum((0, 1)).cpu()
                cnt += rec.acts[C.LAYERS[0]].shape[0] * rec.acts[C.LAYERS[0]].shape[1]
        u = {L: (sums[L] / cnt) / (sums[L] / cnt).norm() for L in C.LAYERS}
        torch.save({L: v.float() for L, v in u.items()}, f)
        cosb = {L: float(u[L].float() @ torch.load(C.STAGE0 / f"u_brew_{self.cname}_L{L}.pt")) for L in C.LAYERS}
        C.dump(C.OUT / "stage1a" / "u_gen_meta.json", {"source": "Salesforce/wikitext wikitext-103-raw-v1 test parquet",
                                                         "row_indices": sel, "tokens_per_seq": 128, "positions": "16..127",
                                                         "n_vectors": cnt, "cos_u_gen_u_brew_colon": cosb})
        self.log("u_gen built; cos(u_gen, u_brew ':')", cosb)
        return {L: v.float() for L, v in u.items()}

    def add_generic(self, u_gen):
        for L in C.LAYERS:
            u = u_gen[L].double()
            H = torch.from_numpy(self.pieces[L]["whole"]).double()
            self.pieces[L]["GMR"] = (H - (H @ u).unsqueeze(1) * u.unsqueeze(0)).float().numpy()
            h = torch.from_numpy(self.pc[L]["whole"]).double()
            g = h - (h @ u).unsqueeze(1) * u.unsqueeze(0)
            QJ = self.bases[f"J_L{L}"].double()
            self.pc[L]["GMR"] = g.float().numpy()                         # the brew generic operator on the colour acts
            self.pc[L]["J_gc@brew"] = ((g @ QJ) @ QJ.T).float().numpy()
            import pieces as PC
            for r in range(C.N_RAND_C):
                QR = PC.rand_basis_f32(C.rand_seed("r1024", r), C.K)[0].double()
                self.pc[L][f"R{r}_gc@brew"] = ((g @ QR) @ QR.T).float().numpy()
            assert float((torch.from_numpy(self.pieces[L]["GMR"]).double() @ u).abs().max()) <= \
                1e-6 * float(H.norm(dim=1).max()), "G-recon (generic) failed"

    # ---------------------------------------------------------------- prompt-only reader floor (stock model)
    def prompt_only(self, model, tok):
        f = self.dir / "ll" / f"promptonly_{self.cname}.npz"
        if f.exists():
            return
        t0 = time.time()
        tmpl = json.loads(WSB_PO.read_text())["_source"]["prompt_template"]
        keys = [r["key"] for r in self.roles["renders"]]
        by_key = {}
        for e in BM.brew_pairs(tok):
            for rr in (e["r_m"], e["r_p"]):
                by_key[BM.ids_key(rr.ids)] = rr
        inner = P5.text_model(model)
        W32 = model.get_output_embeddings().weight.detach().float()
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        LL = np.zeros((len(keys), 24), np.float32)
        owner = None
        with torch.no_grad():
            items = []
            for i, k in enumerate(keys):
                rr = by_key[k]
                by = {g: p for p, g in rr.regions.items() if g.startswith("emit_")}
                p = by[self.cell]
                assert rr.decoded[p - 1] == WANT[self.cell][0] and rr.decoded[p] == WANT[self.cell][1], (rr.item, self.cell)
                content = tmpl.format(token=tok.decode([rr.ids[p]]), prefix=tok.decode(rr.ids[:p + 1]))
                text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) + PO_PREFILL
                base, seqs, owner = spellings(tok, text)
                items.append((i, base, seqs))
            for j in range(0, len(items), 16):
                ch = items[j:j + 16]
                per = [_rows_for(b, s_, pad) for _, b, s_ in ch]
                T = max(len(pr[0][0]) for pr in per)
                rows, mask, row0, Lps, gms = [], [], [], [], []
                for (_, b, s_), (rw, mk, gm, single, multi, stok) in zip(ch, per):
                    row0.append(len(rows))
                    Lps.append(len(b))
                    gms.append(gm)
                    for r_, m_ in zip(rw, mk):
                        rows.append(r_ + [pad] * (T - len(r_)))
                        mask.append(m_ + [0] * (T - len(m_)))
                x = torch.tensor(rows, device=W32.device)
                m = torch.tensor(mask, device=W32.device)
                Hn = inner(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state
                res = _assemble(Hn, W32, row0, Lps, gms, single, multi, stok, 24).cpu().numpy()
                for (ii, _, _), rowv in zip(ch, res):
                    LL[ii] = rowv
                del Hn
        np.savez(f, LL=LL, owner=np.array(owner), template=tmpl, prefill=PO_PREFILL)
        ref = C.OUT / "stage1a" / "r1_aborted_before_reader_outcomes" / "ll" / f"promptonly_{self.cname}.npz"
        if ref.exists() and self.stage == "stage1a":                       # G-fastpath for prompt-only: vs r1's 24-row path
            mx, md, ok = floor_ok(LL, np.load(ref)["LL"])
            self.state["G_fastpath_prompt_only"] = {"max": mx, "median": md, "pass": ok}
            self.log(f"G-fastpath prompt-only vs the 24-row path: max {mx:.4f} median {md:.4f}")
        LLc = dstat.colour_ll(LL, owner)
        self.state.setdefault("prompt_only", {})[self.cname] = {r: dstat.D_mean(LLc, self.roles["pairs"], r)
                                                                 for r in ("z", "start", "answer", "off")}
        self.gpu_time("prompt_only", t0)
        self.log("prompt-only reader floor done", self.state["prompt_only"][self.cname])

    # ---------------------------------------------------------------- one reader
    def reader(self, name, L, score, score_next, tok, read_batch, noact_direct, inject_check, noact_written):
        """score(hs, base, seqs, check) -> [n, S]; returns nothing; writes ll files. Raises GateFailure."""
        tag = f"{name}_L{L}"
        base, seqs, owner = spellings(tok, PREFIX0[name])
        P = self.pieces[L]
        acts = lambda nm, idx=None: [torch.from_numpy(P[nm][i]) for i in (range(P[nm].shape[0]) if idx is None else idx)]
        st = self.state.setdefault("readers", {}).setdefault(tag, {})
        if self.posthoc and "acts_per_batch" not in st:                    # A2: 006's settings, no probe
            st["acts_per_batch"] = 2
            st["scorer"] = "readers.NLArl.score_seq, 24 rows per activation, acts_per_batch 2, original seq_logprob (006)"
        # throughput probe (engineering only; before any outcome; execution note §20)
        if "acts_per_batch" not in st:
            t0 = time.time()
            hs = acts("whole", range(32))
            times = {}
            for apb in (8, 16, 32, 64):
                try:
                    score(hs[:4], base, seqs, False, min(apb, 4))         # warm-up
                    torch.cuda.synchronize()
                    t = time.time()
                    score(hs, base, seqs, False, apb)
                    torch.cuda.synchronize()
                    times[apb] = time.time() - t
                except torch.cuda.OutOfMemoryError:
                    times[apb] = float("inf")
                    gc.collect()
                    torch.cuda.empty_cache()
            st["acts_per_batch"] = min(times, key=times.get)
            st["probe_seconds"] = times
            self.gpu_time(f"{tag}_probe", t0)
            self.log(f"{tag} throughput probe {times} -> acts_per_batch {st['acts_per_batch']}")
        apb = st["acts_per_batch"]
        # G-score1: score_seq vs score_next on single-token spellings (4 whole acts)
        if "G_score1" not in st:
            t0 = time.time()
            single = [i for i, s in enumerate(seqs) if len(s) == 1]
            a = score(acts("whole", range(4)), base, seqs, False, apb)[:, single]
            b = score_next(acts("whole", range(4)), base, [seqs[i][0] for i in single])
            mx, md, ok = floor_ok(a.numpy(), b.cpu().numpy())
            # G-fastpath: the 2-row scorer vs the original 24-row readers.score_seq, every spelling (bf16 floor)
            if self.posthoc:
                fok = True
                st["G_fastpath"] = {"applicable": False, "note": "A2: the scorer is the original 24-row path"}
            else:
                fa = score(acts("whole", range(4)), base, seqs, False, apb).numpy()
                fo = self.score_orig(acts("whole", range(4)), base, seqs).numpy()
                fmx, fmd, fok = floor_ok(fa, fo)
                st["G_fastpath"] = {"max": fmx, "median": fmd, "pass": fok}
                self.log(f"G-fastpath {tag}: max {fmx:.4f} median {fmd:.4f}")
            ok = ok and fok
            st["G_score1"] = {"max": mx, "median": md, "pass": ok, "fastpath_pass": fok}
            self.gpu_time(f"{tag}_gates", t0)
            self.log(f"G-score1 {tag}: max {mx:.4f} median {md:.4f}")
            if not ok:
                raise GateFailure(name, "G-score1", st["G_score1"])
        elif not st["G_score1"]["pass"]:
            raise GateFailure(name, "G-score1", st["G_score1"])
        # whole + G-score2 + G-repro (vs 006)
        fw = self.dir / "ll" / f"{tag}_{self.cname}_whole.npy"
        if not fw.exists():
            t0 = time.time()
            score(acts("whole", range(2)), base, seqs, True, 2)              # G-score2: own-logit check, 2 acts
            chk = self.last_check()
            LLw = score(acts("whole"), base, seqs, False, apb).numpy()
            st["G_score2_own_logits_maxabs"] = chk
            if self.test:                                                  # new captures: no 006 reference
                st["G_repro"] = {"applicable": False, "pass": bool(chk <= 1.0)}
                mx = md = dz_ours = dz_006 = float("nan")
            else:
                ref = np.load(C.LL006 / f"ll_{name}_L{L}_p0.npz")
                mx, md, ok = floor_ok(LLw, ref[self.cell][:231])
                dz_ours = dstat.D_mean(dstat.colour_ll(LLw, owner), self.roles["pairs"], "z")[0]
                dz_006 = dstat.D_mean(dstat.colour_ll(ref[self.cell][:231], ref["owner"]), self.roles["pairs"], "z")[0]
                st["G_repro"] = {"max": mx, "median": md, "Dz_ours": dz_ours, "Dz_006": dz_006,
                                 "pass": bool(ok and abs(dz_ours - dz_006) <= 0.1 and chk <= 1.0)}
            self.gpu_time(f"{tag}_pieces", t0)
            self.log(f"G-score2 {tag}: {chk:.3f}; G-repro: max {mx:.4f} median {md:.4f} Dz {dz_ours:.3f} vs 006 {dz_006:.3f}")
            if not st["G_repro"]["pass"]:
                self.save()
                raise GateFailure(name, "G-repro/G-score2", st["G_repro"])
            np.save(fw, LLw)
        elif not st.get("G_repro", {}).get("pass", False):
            raise GateFailure(name, "G-repro", st.get("G_repro"))
        # G-inject (+ scale check) and the no-activation row
        if "G_inject" not in st:
            t0 = time.time()
            inj = inject_check([torch.from_numpy(P[nm]) for nm in self.order if nm in P])
            rng = np.random.default_rng(C.SEED)
            idx = sorted(rng.choice(self.n, 8, replace=False).tolist())
            sc = {}
            for nm in ("whole", "J", "MR"):
                h1 = acts(nm, idx)
                one = score(h1, base, seqs, False, apb).numpy()
                for s in (0.5, 2.0):
                    sc[f"{nm}x{s}"] = floor_ok(score([h * s for h in h1], base, seqs, False, apb).numpy(), one)
            none = score([None], base, seqs, False, apb).numpy()
            wrote_nothing = noact_written()
            direct = noact_direct(base, seqs)
            na = floor_ok(none, direct)
            na = (na[0], na[1], bool(na[2] and wrote_nothing))
            ok = inj["pass"] and all(v[2] for v in sc.values()) and na[2]
            st["G_inject"] = {"inject": inj, "scale": {k: {"max": v[0], "median": v[1], "pass": v[2]} for k, v in sc.items()},
                              "noact_vs_direct": {"max": na[0], "median": na[1], "pass": na[2]}, "pass": bool(ok)}
            np.save(self.dir / "ll" / f"{tag}_{self.cname}_none.npy", none)
            self.gpu_time(f"{tag}_gates", t0)
            self.log(f"G-inject {tag}: {st['G_inject']['pass']} (inject {inj['pass']}, scale "
                     f"{max(v[0] for v in sc.values()):.4f} max, no-act vs direct {na[0]:.4f})")
            if not ok:
                self.save()
                raise GateFailure(name, "G-inject", st["G_inject"])
        elif not st["G_inject"]["pass"]:
            raise GateFailure(name, "G-inject", st["G_inject"])
        self.save()
        # positive control (all piece types, colour-mean and frozen brew operators)
        fpc = self.dir / "ll" / f"{tag}_{self.cname}_positive.npz"
        if not fpc.exists():
            t0 = time.time()
            Q = self.pc[L]
            names = [nm for nm in Q if (self.stage in ("stage1a", "brew_test") or "PCA" in nm)]
            outp = {nm: score([torch.from_numpy(Q[nm][i]) for i in range(12)], base, seqs, False, apb).numpy() for nm in names}
            np.savez(fpc, **outp, owner=np.array(owner))
            self.gpu_time(f"{tag}_positive", t0)
            self.log(f"{tag} positive control: {len(names)} piece types scored")
        # pieces in the pre-registered order
        for nm in self.order:
            f = self.dir / "ll" / f"{tag}_{self.cname}_{nm}.npy"
            if f.exists():
                continue
            t0 = time.time()
            np.save(f, score(acts(nm), base, seqs, False, apb).numpy())
            self.gpu_time(f"{tag}_pieces", t0)
            self.log(f"{tag} {nm}: {P[nm].shape[0]} reads in {time.time() - t0:.0f}s")
        np.save(self.dir / "ll" / "owner.npy", np.array(owner))
        # qualitative samples (stage 1a only)
        fq = self.dir / f"qualitative_{tag}.jsonl"
        if self.stage == "stage1a" and not fq.exists():
            t0 = time.time()
            rng = np.random.default_rng(C.SEED)
            tabs = sorted(rng.choice(48, 8, replace=False).tolist())
            rows = []
            for tb in tabs:
                prs = [p for p in self.roles["pairs"] if p["table"] == tb]
                pp = prs[int(rng.integers(len(prs)))]
                rows += [(tb, pp["m"], "member"), (tb, pp["p"], "partner")]
            with open(fq, "w") as fo:
                for nm in C.QUAL_PIECES:
                    hs = [torch.from_numpy(P[nm][i]) for _, i, _ in rows]
                    outs = read_batch(hs, 2, C.SEED)
                    for (tb, i, role), smp in zip(rows, outs):
                        fo.write(json.dumps({"reader": name, "layer": L, "cell": self.cell, "table": tb,
                                             "item": self.roles["renders"][i]["item"], "render": i, "role": role,
                                             "path": [self.roles["renders"][i][k] for k in ("start", "z", "answer")],
                                             "piece": nm, "samples": smp}, ensure_ascii=False) + "\n")
            self.gpu_time(f"{tag}_qualitative", t0)
            self.log(f"{tag} qualitative samples written")

    @torch.no_grad()
    def g_fast(self, model, tok, W32, pad):
        """G-fast: the vectorised log-prob equals readers.seq_logprob_from_hidden on a real batch."""
        base, seqs, _ = spellings(tok, PO_PREFILL)
        rows, mask, gather = RD.cont_batch(tok.encode("Colours: red, blue." + PO_PREFILL, add_special_tokens=False), seqs, pad)
        Hn = P5.text_model(model)(input_ids=torch.tensor(rows, device=W32.device),
                                  attention_mask=torch.tensor(mask, device=W32.device), use_cache=False).last_hidden_state
        d = float((seq_logprob_fast(Hn, W32, gather) - _ORIG_SEQ_LP(Hn, W32, gather)).abs().max())
        del Hn
        return d

    # ---------------------------------------------------------------- run
    def run_nla_posthoc(self):
        """A2: the NLA L42 on 006's exact scorer. No subject forward (pieces, u_gen and prompt-only already exist)."""
        t_start = time.time()
        assert RD.seq_logprob_from_hidden is _ORIG_SEQ_LP, "A2 requires the original log-prob function"
        if not self.test:
            self.add_generic(torch.load(C.OUT / "stage1a" / "u_gen.pt"))
        t0 = time.time()
        try:
            nla = RD.NLArl()
        except RuntimeError as e:
            self.state.setdefault("gate_failures", []).append({"gate": "G-mem", "detail": str(e)})
            self.save()
            self.log("STOP GATE (A2): G-mem", e)
            return
        self.gpu_time("nla_load", t0)
        self.last_check = lambda: getattr(nla, "last_check", float("nan"))
        writes = []
        try:
            def score(hs, b, s, check, apb):
                out = nla.score_seq(hs, b, s, acts_per_batch=2, check=check)
                if nla.last_write is not None:
                    writes.append(dict(nla.last_write))
                return out

            def score_next(hs, b, ci):
                return nla.score_next(hs, b, ci)[0]

            def noact_direct(b, s):
                pad_ = nla.tok.pad_token_id if nla.tok.pad_token_id is not None else nla.tok.eos_token_id
                rows, mask, gather = RD.cont_batch(list(nla.ids) + list(b), s, pad_)
                nla._v["v"] = None
                with torch.no_grad():
                    Hn = nla.model.get_base_model().model(input_ids=torch.tensor(rows, device=nla.model.device),
                                                          attention_mask=torch.tensor(mask, device=nla.model.device),
                                                          use_cache=False).last_hidden_state
                if not hasattr(nla, "_W32"):
                    nla._W32 = nla.model.get_output_embeddings().weight.detach().float()
                return _ORIG_SEQ_LP(Hn, nla._W32, gather).cpu().numpy()[None]

            def inject_check(arrs):
                zero = sum(int((A.float().norm(dim=1) <= 1e-6).sum()) for A in arrs)
                live = [w for w in writes if w.get("rows", 0) > 0]
                ok = bool(live) and zero == 0 and min(w["rel_min"] for w in live) >= 0.99 and \
                    max(w["rel_max"] for w in live) <= 1.01 and min(w["cos_min"] for w in live) >= 0.99
                return {"writes_checked": len(live), "zero_rows": zero, "pass": ok}

            def read_batch(hs, k, seed):
                out = []
                for j in range(0, len(hs), 32):
                    out += nla.read_batch(hs[j:j + 32], k=k, seed=seed)
                return out
            self.reader("nla", 42, score, score_next, nla.tok, read_batch, noact_direct, inject_check,
                        lambda: nla.last_write is not None and nla.last_write.get("rows", -1) == 0)
        except GateFailure as e:
            self.state.setdefault("gate_failures", []).append({"reader": e.reader, "gate": e.gate, "detail": e.detail})
            self.save()
            self.log("STOP GATE (A2):", e)
        finally:
            nla.close()
        self.state["wall_minutes"] = (time.time() - t_start) / 60
        self.save()
        self.log(f"{self.stage} NLA post hoc done ({self.state['wall_minutes']:.1f} min)")

    def run(self):
        if self.posthoc:
            return self.run_nla_posthoc()
        t_start = time.time()
        RD.seq_logprob_from_hidden = seq_logprob_fast
        tok = R.make_tokenizer()
        model = R.load_model()
        W32 = model.get_output_embeddings().weight.detach().float()
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        dfast = self.g_fast(model, tok, W32, pad)
        self.state["G_fast_maxabs"] = dfast
        self.log(f"G-fast (vectorised vs original log-prob) max |diff| {dfast:.2e}")
        assert dfast <= 1e-3, "G-fast failed"
        if not self.test:
            u_gen = self.generic_mean(model, tok)
            self.add_generic(u_gen)
        self.prompt_only(model, tok)
        # ---------------- Phase A: the Oracle
        if "oracle" in self.disabled:
            self.log("oracle disabled by an earlier shared-path gate failure:", self.disabled["oracle"])
        else:
            ol = RD.OracleLens(model, tok)
            self.last_check = lambda: getattr(ol, "last_check", float("nan"))
            try:
                for L in (40, 44):
                    def score(hs, b, s, check, apb, L=L):
                        return oracle_fast(ol, hs, L, b, s, W32, acts_per_batch=apb, check=check)

                    def score_orig(hs, b, s, L=L):
                        return ol.score_seq(hs, L, b, s, W32, acts_per_batch=2, check=False)

                    def score_next(hs, b, ci, L=L):
                        return ol.score_next(hs, L, b, ci, W32)[0]

                    def noact_direct(b, s, L=L):
                        ids, _, _ = ol.carrier(L)
                        rows, mask, gather = RD.cont_batch(list(ids) + list(b), s, pad)
                        with torch.no_grad():
                            Hn = ol.model.get_base_model().model(input_ids=torch.tensor(rows, device=W32.device),
                                                                 attention_mask=torch.tensor(mask, device=W32.device),
                                                                 use_cache=False).last_hidden_state
                        return seq_logprob_fast(Hn, W32, gather).cpu().numpy()[None]

                    def inject_check(arrs):
                        alpha = ol.c["alpha"]
                        worst, zero = 0.0, 0
                        for A in arrs:
                            nrm = A.float().norm(dim=1)
                            zero += int((nrm <= 1e-6).sum())
                            v = (alpha * A.float() / nrm.clamp_min(1e-9).unsqueeze(1)).to(torch.bfloat16).float().norm(dim=1)
                            worst = max(worst, float(((v - alpha).abs() / alpha).max()))
                        return {"max_rel_norm_err": worst, "zero_rows": zero, "pass": worst <= 1e-2 and zero == 0}

                    def read_batch(hs, k, seed, L=L):
                        out = []
                        for j in range(0, len(hs), 32):
                            out += ol.read_batch(hs[j:j + 32], L, k=k, seed=seed)
                        return out
                    self.score_orig = score_orig
                    self.reader("oracle", L, score, score_next, tok, read_batch, noact_direct, inject_check,
                                lambda: True)                              # Oracle: the None row skips the slot write
            except GateFailure as e:
                self.disable(e)
            finally:
                base_model = ol.unload()
                assert base_model is model
                del ol, base_model
                gc.collect()
        # ---------------- teardown, Phase B: the NLA
        self.log("phase B: free the subject; load nla-rl")
        del model, W32
        gc.collect()
        torch.cuda.empty_cache()
        if "nla" in self.disabled:
            self.log("nla disabled by an earlier shared-path gate failure:", self.disabled["nla"])
        else:
            t0 = time.time()
            try:
                nla = RD.NLArl()
            except RuntimeError as e:                                     # G-mem
                self.disable(GateFailure("nla", "G-mem", str(e)))
                nla = None
            self.gpu_time("nla_load", t0)
            if nla is not None:
                self.last_check = lambda: getattr(nla, "last_check", float("nan"))
                try:
                    L = 42
                    writes = []

                    def score(hs, b, s, check, apb):
                        out = nla_fast(nla, hs, b, s, acts_per_batch=apb, check=check)
                        if nla.last_write is not None:
                            writes.append(dict(nla.last_write))
                        return out

                    def score_orig(hs, b, s):
                        return nla.score_seq(hs, b, s, acts_per_batch=2, check=False)

                    def score_next(hs, b, ci):
                        return nla.score_next(hs, b, ci)[0]

                    def noact_direct(b, s):
                        pad_ = nla.tok.pad_token_id if nla.tok.pad_token_id is not None else nla.tok.eos_token_id
                        rows, mask, gather = RD.cont_batch(list(nla.ids) + list(b), s, pad_)
                        nla._v["v"] = None                                    # hook disabled: no write at all
                        with torch.no_grad():
                            Hn = nla.model.get_base_model().model(input_ids=torch.tensor(rows, device=nla.model.device),
                                                                  attention_mask=torch.tensor(mask, device=nla.model.device),
                                                                  use_cache=False).last_hidden_state
                        if not hasattr(nla, "_W32"):
                            nla._W32 = nla.model.get_output_embeddings().weight.detach().float()
                        return seq_logprob_fast(Hn, nla._W32, gather).cpu().numpy()[None]

                    def inject_check(arrs):
                        zero = sum(int((A.float().norm(dim=1) <= 1e-6).sum()) for A in arrs)
                        live = [w for w in writes if w.get("rows", 0) > 0]
                        rel_lo = min((w["rel_min"] for w in live), default=float("nan"))
                        rel_hi = max((w["rel_max"] for w in live), default=float("nan"))
                        cos_lo = min((w["cos_min"] for w in live), default=float("nan"))
                        ok = bool(live) and zero == 0 and rel_lo >= 0.99 and rel_hi <= 1.01 and cos_lo >= 0.99
                        return {"writes_checked": len(live), "rel_min": rel_lo, "rel_max": rel_hi, "cos_min": cos_lo,
                                "zero_rows": zero, "pass": ok}

                    def read_batch(hs, k, seed):
                        out = []
                        for j in range(0, len(hs), 32):
                            out += nla.read_batch(hs[j:j + 32], k=k, seed=seed)
                        return out
                    self.score_orig = score_orig
                    self.reader("nla", L, score, score_next, nla.tok, read_batch, noact_direct, inject_check,
                                lambda: nla.last_write is not None and nla.last_write.get("rows", -1) == 0)
                except GateFailure as e:
                    self.disable(e)
                finally:
                    nla.close()
        self.state["wall_minutes"] = (time.time() - t_start) / 60
        self.save()
        self.log(f"{self.stage} done ({self.state['wall_minutes']:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1a", "stage1b", "brew_test"])
    ap.add_argument("--run-id", default="r1")
    ap.add_argument("--nla-posthoc", action="store_true", help="A2: the NLA only, on 006's exact 24-row scorer")
    a = ap.parse_args()
    d = C.OUT / a.stage / a.run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"stage": a.stage, "run_id": a.run_id, "manifest": R.manifest_block(),
                                                 "nla_posthoc": a.nla_posthoc,
                                                 "design": "docs/design_specs/007_jsubspace.md (v2 + A1 + A2)"}, indent=1))
    Stage(a.stage, a.run_id, nla_posthoc=a.nla_posthoc).run()


if __name__ == "__main__":
    main()
