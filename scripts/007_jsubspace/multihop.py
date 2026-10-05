"""007 Stage 3 — multihop Part A (design §8 + A1; secondary; expected to say little about J). Reader-side.

  python multihop.py --run-id m1

One process: CPU set-up (split, relation-level pairs, candidate sets) -> Phase A (subject resident: captures at the
bank's read position, positive-control activations, G-cw against Cathryn's L42 residuals, pieces with PCA-k frozen
before any read, prompt-only reader floor, then the Oracle L40/L44) -> teardown -> Phase B (the NLA L42). Scores are
saved per (reader, layer, piece) with the per-group candidate layout; D and the restatement flags are computed by
analysis.py (CPU). Shared-path gate failures disable that reader (A1) and are recorded.
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "005_planted_concepts"))
import p005 as P5                                 # noqa: E402
import readers as RD                              # noqa: E402
import registry as R                              # noqa: E402
import wsb                                        # noqa: E402
import p007 as C                                  # noqa: E402
import reader_split as RS                         # noqa: E402
from pieces import f32_orth, rand_basis_f32       # noqa: E402

BANK = Path("/workspace/sources/repos/workspace-bench/evals/multihop/items.json")
CW_RESID = Path("/workspace/sources/repos/nla-vs-lenses/runs/abl50/resid_L42.pt")
RESTATE_PIECES = ("whole", "J", "rest_J", "R0", "PCA", "MR")
PIECES21 = (["whole", "J", "rest_J", "PCA", "rest_PCA", "LL", "rest_LL", "MR"] + [f"R{r}" for r in range(10)] +
            [f"rest_R{r}" for r in range(3)])


def setup():
    items = json.loads(BANK.read_text())["items"]
    for i, it in enumerate(items):
        it["idx"] = i
        it["split"] = "dev" if i < 50 else "test"
        it["key"] = it["name"].split("-")[-1]
        it["bridge"] = it["intermediates"][0]
    groups = {}
    for it in items:
        groups.setdefault((it["split"], it["key"]), []).append(it["idx"])
    pairs = []
    for (sp, key), idx in sorted(groups.items()):
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if items[i]["bridge"] != items[j]["bridge"] and items[i]["target"] != items[j]["target"]:
                    pairs.append({"i": i, "j": j, "split": sp, "key": key})
    paired = sorted({p["i"] for p in pairs} | {p["j"] for p in pairs})
    gstr = {}
    for p in pairs:
        g = (p["split"], p["key"])
        for x in (p["i"], p["j"]):
            gstr.setdefault(g, set()).update([items[x]["bridge"], items[x]["target"]])
    gstr = {f"{sp}|{k}": sorted(v) for (sp, k), v in gstr.items()}
    return items, pairs, paired, gstr


def string_spellings(tok, prefix, strings):
    """Spellings {" s", " S(wapcase first)"} of every candidate string after `prefix`; returns (base, seqs, owner)."""
    base = tok.encode(prefix, add_special_tokens=False)
    seqs, owner = [], []
    for s in strings:
        forms = list(dict.fromkeys([" " + s, " " + s[0].swapcase() + s[1:]]))
        for form in forms:
            full = tok.encode(prefix + form, add_special_tokens=False)
            assert full[:len(base)] == base, (prefix[-30:], form)
            seqs.append(full[len(base):])
            owner.append(s)
    return base, seqs, owner


def cue_words(prompt, exclude):
    body = prompt.split("Fact:", 1)[-1]
    ws = re.findall(r"\b[A-Z][\w'’.-]*\b|\b\d+\b", body)
    ex = {"Fact", "The", "In"} | {w.lower() for w in exclude}
    return sorted({w for w in ws if w not in ("Fact", "The", "In") and w.lower() not in ex})


class Stage3:
    def __init__(self, run_id):
        self.dir = C.OUT / "stage3" / run_id
        (self.dir / "ll").mkdir(parents=True, exist_ok=True)
        self.log = C.Logger(self.dir / "log.txt")
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.items, self.pairs, self.paired, self.gstr = setup()
        self.disabled = json.loads(RS.DISABLED.read_text()) if RS.DISABLED.exists() else {}
        C.dump(self.dir / "design_material.json", {
            "split": "dev = items 1-50 (touched), test = items 51-100", "key": "item name's final field",
            "pairs": self.pairs, "paired_items": self.paired, "group_strings": self.gstr,
            "n_pairs": {sp: sum(p["split"] == sp for p in self.pairs) for sp in ("dev", "test")},
            "cue_words": {it["idx"]: cue_words(it["prompt"], [it["bridge"], it["target"]]) for it in self.items}})

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1, default=float))

    # ------------------------------------------------ Phase A helpers
    def capture(self, model, tok):
        f = self.dir / "captures.npz"
        if f.exists():
            z = np.load(f)
            return z["X"], z["pos"]
        blocks = R.blocks(model)
        dev = next(model.parameters()).device
        X = np.zeros((len(self.items), len(C.LAYERS), C.D), np.float32)
        pos = np.zeros(len(self.items), int)
        dec = []
        with torch.no_grad():
            for it in self.items:
                ids = wsb.render_text(tok, "multihop", it["prompt"])
                off = it["readout"]["offsets"][0]
                p = len(ids) + off
                pos[it["idx"]] = p
                rec = P5.RowRecorder(blocks, list(C.LAYERS), [p])
                with rec:
                    P5.text_model(model)(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                for li, L in enumerate(C.LAYERS):
                    X[it["idx"], li] = rec.acts[L][0, 0].cpu().numpy()
                dec.append(tok.decode([ids[p]]))
        # G-cap: the read token is the item's last word (or the opening quote of `is "`)
        bad = []
        for it, t in zip(self.items, dec):
            last = it["prompt"].rstrip().split()[-1]
            if not (t.strip() and (t.strip() in last or last.endswith(t.strip()) or t.strip() == '"')):
                bad.append((it["name"], t, last))
        self.state["G_cap"] = {"n_items": len(self.items), "mismatches": bad, "pass": not bad}
        np.savez(f, X=X, pos=pos)
        self.log("captures", X.shape, "G-cap", "pass" if not bad else f"FAIL {bad[:3]}")
        return X, pos

    def g_cw(self, X):
        cw = torch.load(CW_RESID, weights_only=False)
        by = {it["name"]: it for it in self.items}
        rows = []
        for c in cw:
            it = by.get(c["id"])
            if it is None or c["layer"] != 42:
                continue
            same = int(c["pos"]) == int(it["readout"]["offsets"][0])
            if same:
                h = X[it["idx"], C.LAYERS.index(42)]
                rows.append(float(np.dot(h, c["h"].float().numpy()) / (np.linalg.norm(h) * float(c["h"].norm()))))
        self.state["G_cw"] = {"n_matched_positions": len(rows), "n_cw": len(cw), "cos_min": min(rows) if rows else None,
                              "cos_median": float(np.median(rows)) if rows else None,
                              "pass": bool(rows) and min(rows) >= 0.999}
        self.log("G-cw", self.state["G_cw"])

    def pieces(self, X):
        """The 21 v1 pieces (design §8.3) for every item at L40/42/44; PCA-k Δ-matched on dev and frozen here."""
        B = torch.load(C.STAGE0 / "bases_f32.pt")
        Rb = {r: rand_basis_f32(C.rand_seed("r1024", r), C.K)[0].double() for r in range(10)}
        out, meta = {}, {}
        items = self.items
        key_of = np.array([it["key"] for it in items])
        sp_of = np.array([it["split"] for it in items])
        dev_idx = [it["idx"] for it in items if it["split"] == "dev"]
        keys_dev = sorted({key_of[i] for i in dev_idx})
        rng = np.random.default_rng(C.SEED)
        perm = rng.permutation(len(keys_dev))
        kfold = {keys_dev[k]: int(f) for f in range(4) for k in perm[f::4]}
        Pj = lambda A, Q: (A @ Q) @ Q.T
        for li, L in enumerate(C.LAYERS):
            H = torch.from_numpy(X[:, li].astype(np.float64))
            QJ, QL = B[f"J_L{L}"].double(), B["LL"].double()

            def wpca(rows):                                                # within-key centred PCA (004 style)
                rows = list(rows)
                Hs = H[rows]
                ks = key_of[rows]
                Cm = torch.zeros_like(Hs)
                for k in np.unique(ks):
                    m = torch.from_numpy(ks == k)
                    if int(m.sum()) >= 2:
                        Cm[m] = (Hs[m] - Hs[m].mean(0, keepdim=True)) / float(np.sqrt(int(m.sum())))
                _, S, Vh = torch.linalg.svd(Cm, full_matrices=False)
                r_ = int((S > S[0] * 1e-8).sum())
                return Vh[:r_].T
            Vall = wpca(dev_idx)
            Vf = {f: wpca([i for i in dev_idx if kfold[key_of[i]] != f]) for f in range(4)}
            # k: delta-matched to J's median dev pair delta-share (outcome-blind)
            dpairs = [p for p in self.pairs if p["split"] == "dev"]
            Dl = torch.stack([H[p["i"]] - H[p["j"]] for p in dpairs])
            jds = float(np.median(((Pj(Dl, QJ) ** 2).sum(1) / (Dl ** 2).sum(1)).numpy()))
            kmax = min(V.shape[1] for V in list(Vf.values()) + [Vall])
            best = None
            for k in range(1, kmax + 1):
                ds = []
                for p in dpairs:
                    f = kfold[key_of[p["i"]]]
                    Qk = Vf[f][:, :k]
                    d = H[p["i"]] - H[p["j"]]
                    ds.append(float(((Qk.T @ d) ** 2).sum() / (d ** 2).sum()))
                gap = abs(float(np.median(ds)) - jds)
                if best is None or gap < best[1]:
                    best = (k, gap, float(np.median(ds)))
            k = best[0]
            meta[L] = {"J_dev_median_delta_share": jds, "PCA_k": k, "PCA_dev_median_delta_share": best[2], "kmax": kmax}
            Hp = torch.empty_like(H)
            for it in items:
                i = it["idx"]
                V = Vf[kfold[key_of[i]]] if it["split"] == "dev" else Vall
                Qk = f32_orth(V[:, :k].float())[0].double()
                Hp[i] = Qk @ (Qk.T @ H[i])
            # mean-removed: leave-one-key-out mean within the split
            MR = torch.empty_like(H)
            for it in items:
                i = it["idx"]
                m = torch.from_numpy((sp_of == it["split"]) & (key_of != it["key"]))
                u = H[m].mean(0)
                u = u / u.norm()
                MR[i] = H[i] - (H[i] @ u) * u
            pcs = {"whole": H, "J": Pj(H, QJ), "rest_J": H - Pj(H, QJ), "PCA": Hp, "rest_PCA": H - Hp,
                   "LL": Pj(H, QL), "rest_LL": H - Pj(H, QL), "MR": MR}
            for r in range(10):
                pcs[f"R{r}"] = Pj(H, Rb[r])
                if r < 3:
                    pcs[f"rest_R{r}"] = H - pcs[f"R{r}"]
            out[L] = {k_: v.float().numpy() for k_, v in pcs.items()}
            np.savez(self.dir / f"pieces_L{L}.npz", **out[L])
        self.state["pieces_meta"] = meta
        self.log("pieces", meta)
        return out

    def pc_acts(self, model, tok):
        f = self.dir / "positive_acts.pt"
        if f.exists():
            return torch.load(f)
        bridges = sorted({self.items[i]["bridge"] for i in self.paired})
        ids = [P5.render_tell(tok, b) for b in bridges]
        acts = P5.capture_last(model, ids, list(C.LAYERS))                 # [n, 3, d] stock, last token
        out = {"bridges": bridges, "acts": acts}
        torch.save(out, f)
        return out

    def pc_pieces(self, pc, L):
        """10 piece types of the bridge activations (design §8.5): fixed projectors as for the items; PCA with the
        all-dev basis; mean-removed with the leave-one-out mean of the other bridge activations."""
        B = torch.load(C.STAGE0 / "bases_f32.pt")
        h = pc["acts"][:, C.LAYERS.index(L)].double()
        n = h.shape[0]
        mu = (h.sum(0, keepdim=True) - h) / (n - 1)
        U = mu / mu.norm(dim=1, keepdim=True)
        Pj = lambda A, Q: (A @ Q) @ Q.T
        QJ, QL = B[f"J_L{L}"].double(), B["LL"].double()
        R0 = rand_basis_f32(C.rand_seed("r1024", 0), C.K)[0].double()
        P = np.load(self.dir / f"pieces_L{L}.npz")
        k = self.state["pieces_meta"][str(L)]["PCA_k"] if str(L) in self.state["pieces_meta"] else self.state["pieces_meta"][L]["PCA_k"]
        return {"whole": h, "J": Pj(h, QJ), "rest_J": h - Pj(h, QJ), "LL": Pj(h, QL), "rest_LL": h - Pj(h, QL),
                "MR": h - (h * U).sum(1, keepdim=True) * U, "R0": Pj(h, R0), "rest_R0": h - Pj(h, R0),
                "_k": k}

    def prompt_only(self, model, tok):
        f = self.dir / "ll" / "promptonly.npz"
        if f.exists():
            return
        tmpl = json.loads(RS.WSB_PO.read_text())["_source"]["prompt_template"]
        inner = P5.text_model(model)
        W32 = model.get_output_embeddings().weight.detach().float()
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        out = {}
        with torch.no_grad():
            for g, strings in self.gstr.items():
                sp, key = g.split("|")
                idx = [i for i in self.paired if self.items[i]["split"] == sp and self.items[i]["key"] == key]
                res = []
                for i in idx:
                    it = self.items[i]
                    ids = wsb.render_text(tok, "multihop", it["prompt"])
                    p = len(ids) + it["readout"]["offsets"][0]
                    content = tmpl.format(token=tok.decode([ids[p]]), prefix=tok.decode(ids[:p + 1]))
                    text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False) + RS.PO_PREFILL
                    base, seqs, owner = string_spellings(tok, text, strings)
                    rows, mask, gm, single, multi, stok = RS._rows_for(base, seqs, pad)
                    Hn = inner(input_ids=torch.tensor(rows, device=W32.device),
                               attention_mask=torch.tensor(mask, device=W32.device), use_cache=False).last_hidden_state
                    res.append(RS._assemble(Hn, W32, [0], [len(base)], [gm], single, multi, stok, len(seqs)).cpu().numpy()[0])
                    del Hn
                out[g] = {"items": idx, "LL": np.stack(res), "owner": owner}
        np.savez(f, **{f"{g}|LL": v["LL"] for g, v in out.items()}, **{f"{g}|items": np.array(v["items"]) for g, v in out.items()},
                 **{f"{g}|owner": np.array(v["owner"]) for g, v in out.items()})
        self.log("prompt-only done")

    # ------------------------------------------------ one reader-layer
    def reader(self, name, L, P, score, score_orig, score_next, tok, prefix, read_batch, inject_check, pc):
        tag = f"{name}_L{L}"
        st = self.state.setdefault("readers", {}).setdefault(tag, {})
        acts = lambda nm, idx: [torch.from_numpy(P[nm][i]) for i in idx]
        # gates on 4 paired items with the colour spellings (scorer path) and their group strings (fast path)
        base_c, seqs_c, _ = RS.spellings(tok, prefix)
        four = self.paired[:4]
        if "G_score1" not in st:
            single = [i for i, s in enumerate(seqs_c) if len(s) == 1]
            a = score(acts("whole", four), base_c, seqs_c, False, 16)[:, single]
            b = score_next(acts("whole", four), base_c, [seqs_c[i][0] for i in single])
            mx, md, ok = RS.floor_ok(a.numpy(), b.cpu().numpy())
            fa = score(acts("whole", four), base_c, seqs_c, False, 16).numpy()
            fo = score_orig(acts("whole", four), base_c, seqs_c).numpy()
            fmx, fmd, fok = RS.floor_ok(fa, fo)
            score(acts("whole", four[:2]), base_c, seqs_c, True, 2)
            chk = self.last_check()
            inj = inject_check([torch.from_numpy(P[nm]) for nm in PIECES21])
            sc = {}
            one = score(acts("whole", four), base_c, seqs_c, False, 16).numpy()
            for s_ in (0.5, 2.0):
                sc[s_] = RS.floor_ok(score([h * s_ for h in acts("whole", four)], base_c, seqs_c, False, 16).numpy(), one)
            ok_all = ok and fok and chk <= 1.0 and inj["pass"] and all(v[2] for v in sc.values())
            st["G_score1"] = {"max": mx, "median": md, "fastpath": {"max": fmx, "median": fmd, "pass": fok},
                              "G_score2": chk, "inject": inj, "scale": {str(k): v for k, v in sc.items()}, "pass": bool(ok_all)}
            self.save()
            self.log(f"{tag} gates: {st['G_score1']['pass']} (score1 {mx:.3f}, fastpath {fmx:.3f}, score2 {chk:.3f})")
            if not ok_all:
                raise RS.GateFailure(name, "G-score1/G-fastpath/G-inject (multihop)", st["G_score1"])
        elif not st["G_score1"]["pass"]:
            raise RS.GateFailure(name, "G-score1 (multihop)", st["G_score1"])
        # per group: every piece + no-activation, the group's candidate strings
        for g, strings in self.gstr.items():
            sp, key = g.split("|")
            idx = [i for i in self.paired if self.items[i]["split"] == sp and self.items[i]["key"] == key]
            base, seqs, owner = string_spellings(tok, prefix, strings)
            f = self.dir / "ll" / f"{tag}_{g.replace('|', '__')}.npz"
            if f.exists():
                continue
            t0 = time.time()
            arrs = {nm: score(acts(nm, idx), base, seqs, False, 16).numpy() for nm in PIECES21}
            arrs["none"] = score([None], base, seqs, False, 16).numpy()
            np.savez(f, **arrs, items=np.array(idx), owner=np.array(owner))
            self.log(f"{tag} group {g}: {len(idx)} items x {len(PIECES21)} pieces x {len(seqs)} spellings in {time.time() - t0:.0f}s")
        # positive control: bridge activations, scored against the group's bridges (within split/key) -> all bridges
        fpc = self.dir / "ll" / f"{tag}_positive.npz"
        if not fpc.exists():
            pp = self.pc_pieces(pc, L)
            k_ = pp.pop("_k")
            bridges = pc["bridges"]
            base, seqs, owner = string_spellings(tok, prefix, bridges)
            outp = {nm: score([v[i].float() for i in range(v.shape[0])], base, seqs, False, 16).numpy() for nm, v in pp.items()}
            np.savez(fpc, **outp, bridges=np.array(bridges), owner=np.array(owner))
            self.log(f"{tag} positive control scored ({len(bridges)} bridges)")
        # restatement samples (k = 2)
        fq = self.dir / f"restatement_{tag}.jsonl"
        if not fq.exists():
            with open(fq, "w") as fo:
                for nm in RESTATE_PIECES:
                    outs = read_batch(acts(nm, self.paired), 2, C.SEED)
                    for i, smp in zip(self.paired, outs):
                        fo.write(json.dumps({"reader": name, "layer": L, "item": self.items[i]["name"], "idx": i,
                                             "split": self.items[i]["split"], "piece": nm, "samples": smp},
                                            ensure_ascii=False) + "\n")
            self.log(f"{tag} restatement samples written")

    def run(self):
        t_start = time.time()
        RD.seq_logprob_from_hidden = RS.seq_logprob_fast
        tok = R.make_tokenizer()
        model = R.load_model()
        X, pos = self.capture(model, tok)
        if not self.state["G_cap"]["pass"]:
            self.log("STOP GATE: G-cap failed; Stage 3 skipped")
            self.save()
            return
        self.g_cw(X)
        P = self.pieces(X)
        pc = self.pc_acts(model, tok)
        self.prompt_only(model, tok)
        self.save()
        W32 = model.get_output_embeddings().weight.detach().float()
        if "oracle" in self.disabled:
            self.log("oracle disabled (A1):", self.disabled["oracle"])
        else:
            ol = RD.OracleLens(model, tok)
            self.last_check = lambda: getattr(ol, "last_check", float("nan"))
            try:
                for L in (40, 44):
                    def score(hs, b, s, check, apb, L=L):
                        return RS.oracle_fast(ol, hs, L, b, s, W32, acts_per_batch=apb, check=check)

                    def score_orig(hs, b, s, L=L):
                        return ol.score_seq(hs, L, b, s, W32, acts_per_batch=2, check=False)

                    def score_next(hs, b, ci, L=L):
                        return ol.score_next(hs, L, b, ci, W32)[0]

                    def inject_check(arrs):
                        worst, zero = 0.0, 0
                        for A in arrs:
                            nrm = A.float().norm(dim=1)
                            zero += int((nrm <= 1e-6).sum())
                            v = (ol.c["alpha"] * A.float() / nrm.clamp_min(1e-9).unsqueeze(1)).to(torch.bfloat16).float().norm(dim=1)
                            worst = max(worst, float(((v - ol.c["alpha"]).abs() / ol.c["alpha"]).max()))
                        return {"max_rel_norm_err": worst, "zero_rows": zero, "pass": worst <= 1e-2 and zero == 0}

                    def read_batch(hs, k, seed, L=L):
                        out = []
                        for j in range(0, len(hs), 32):
                            out += ol.read_batch(hs[j:j + 32], L, k=k, seed=seed)
                        return out
                    self.reader("oracle", L, P[L], score, score_orig, score_next, tok, RS.PREFIX0["oracle"], read_batch,
                                inject_check, pc)
            except RS.GateFailure as e:
                self.disable(e)
            finally:
                base_model = ol.unload()
                del ol, base_model
                gc.collect()
        del model, W32
        gc.collect()
        torch.cuda.empty_cache()
        if "nla" in self.disabled:
            self.log("nla disabled (A1):", self.disabled["nla"])
        else:
            try:
                nla = RD.NLArl()
            except RuntimeError as e:
                self.disable(RS.GateFailure("nla", "G-mem", str(e)))
                nla = None
            if nla is not None:
                self.last_check = lambda: getattr(nla, "last_check", float("nan"))
                writes = []
                try:
                    def score(hs, b, s, check, apb):
                        out = RS.nla_fast(nla, hs, b, s, acts_per_batch=apb, check=check)
                        if nla.last_write is not None:
                            writes.append(dict(nla.last_write))
                        return out

                    def score_orig(hs, b, s):
                        return nla.score_seq(hs, b, s, acts_per_batch=2, check=False)

                    def score_next(hs, b, ci):
                        return nla.score_next(hs, b, ci)[0]

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
                    self.reader("nla", 42, P[42], score, score_orig, score_next, nla.tok, RS.PREFIX0["nla"], read_batch,
                                inject_check, pc)
                except RS.GateFailure as e:
                    self.disable(e)
                finally:
                    nla.close()
        self.state["wall_minutes"] = (time.time() - t_start) / 60
        self.save()
        self.log(f"stage 3 done ({self.state['wall_minutes']:.1f} min)")

    def disable(self, err):
        self.disabled[err.reader] = {"gate": err.gate, "detail": err.detail, "stage": "stage3",
                                     "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        RS.DISABLED.write_text(json.dumps(self.disabled, indent=1, default=float))
        self.state.setdefault("gate_failures", []).append(self.disabled[err.reader])
        self.save()
        self.log("STOP GATE:", err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="m1")
    a = ap.parse_args()
    d = C.OUT / "stage3" / a.run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"run_id": a.run_id, "manifest": R.manifest_block(),
                                                 "design": "docs/design_specs/007_jsubspace.md (v2 + A1)"}, indent=1))
    Stage3(a.run_id).run()


if __name__ == "__main__":
    main()
