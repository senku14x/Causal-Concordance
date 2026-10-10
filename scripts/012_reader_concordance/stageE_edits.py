"""012 Stage E step 4 (design `docs/design_specs/012E_ar_concept_swaps.md` §2, §7-§9, §15): multihop edits along the
Oracle reconstructor's (AR) concept differences, the J-row and class-mean references, and their controls. EXPLORATORY.

Regime (design §2, PI fix 1): Jan's phrase-lens `ModelRunner` in bf16 with `BandEditor` (every prompt position, once
per band layer, prefill only), float32 only for the final logits (as stageB_capture.py). Band L36-47. AR vectors at
the nearest tag at or below each layer (L36 for 36-39, L40 for 40-43, L44 for 44-47).

Modes
  parity   G-parity: clean, the J-row swap and its random control at L36-47 on every Stage B-evaluated item, against
           Stage A's j_swap_L36-47 rows; G0 (alpha 0) and the identity-clamp floors (Jan's SwapEdit; the additive clamp)
  cmlever  the class-mean full swap and its "true source / random target" control on every cohort item whose src and
           tgt are both covered (fix 2's lever rule for the size reference)
  smoke    every §7 arm on a few items (--items or seeded draw)

Arms (§7): additive clamps h <- h + u (c_clean + k - c), c = u.h, set per layer (no compounding), k matched per
(layer, position) to the class-mean swap's clean-state edit size (J-row size where the item has no class means, and as
a sensitivity). Edit sizes are logged at every edited position and layer, with the coordinate error after the bf16
write.

  CUDA_VISIBLE_DEVICES=k PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src \
      python stageE_edits.py {parity|cmlever|smoke} [--shard i --nshard n] [--items a,b] [--n-smoke 5] [--tag t]
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from phrase_lens.benchmarks import load_benchmark
from phrase_lens.interventions import SwapEdit, answer_generation_budget, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.metrics import causal_outcome
from phrase_lens.modeling import ModelRunner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import stageE_common as E                                                   # noqa: E402

MODEL, REV = "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
BENCH, ARM_ID = "causal_st_multihop", "j"
SA = ROOT / "outputs/012_reader_concordance/stageA"
BAND = list(range(36, 48))
TAG = {L: (36 if L < 40 else 40 if L < 44 else 44) for L in BAND}
D = 5120
SEED = 20261010
PHRASES = ("starts", "mentions")
RESCUED = ("Brazil", "Canada", "China", "Denmark", "Fuji", "Miami", "Nile", "Olympus", "Sydney")   # §15, POST HOC


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def seed_of(*parts):
    return SEED + zlib.crc32("/".join(map(str, parts)).encode()) % 1_000_000


def lse(xs):
    return float(torch.logsumexp(torch.tensor(xs, dtype=torch.float64), 0))


# ------------------------------------------------------------------------------------------------ edit objects
class ClampEdit:
    """General per-layer clamp: c = R (h - mu); h <- h + A (t - c); t fixed per layer and position from the clean run.
    A [D, k], R [k, D], mu [D], target [k, P]."""
    def __init__(self, band, A, R, mu, target, kind):
        self.band, self.A, self.R, self.mu, self.target, self.kind = list(band), A, R, mu, target, kind
        self.targets = target                                               # Jan-compatible attribute name

    def __call__(self, layer, residual):
        A, R, mu, t = (x[layer].to(residual) for x in (self.A, self.R, self.mu, self.target))
        c = R @ (residual - mu).T                                           # [k, P]
        return residual + (A @ (t - c)).T

    @property
    def flags(self):
        return {"edit": self.kind, "n_band_layers": len(self.band)}


def additive(u, k, clean, kind):
    """u {L: [D] unit}, k {L: [P] sizes}, clean {L: [P, D]}: clamp the coordinate on u to clean + k."""
    A = {L: u[L][:, None] for L in u}
    R = {L: u[L][None, :] for L in u}
    mu = {L: torch.zeros(D) for L in u}
    tg = {L: (clean[L].float() @ u[L])[None, :] + k[L][None, :] for L in u}
    return ClampEdit(list(u), A, R, mu, tg, kind)


def swap_sizes(s, t, clean):
    """Clean-state edit size of Jan's swap per layer and position: || V (swap(c) - c) ||, V = [s, t]."""
    out = {}
    for i, L in enumerate(BAND):
        V = torch.stack([s[i], t[i]], 1).double()
        c = torch.linalg.pinv(V) @ clean[L].double().T
        out[L] = (V @ (c[[1, 0]] - c)).norm(dim=0).float()
    return out


class Logged:
    """Records what an edit did at every position of every band layer (first call per layer = the prefill)."""
    def __init__(self, edit, coord_fn=None):
        self.edit, self.band, self.log, self.coord_fn = edit, edit.band, {}, coord_fn

    def __call__(self, layer, residual):
        out = self.edit(layer, residual)
        if layer not in self.log:
            dh = (out - residual).norm(dim=-1)
            h = residual.norm(dim=-1).clamp_min(1e-12)
            rec = {"rel_dh": (dh / h).tolist()}
            if self.coord_fn is not None:
                rec.update(self.coord_fn(layer, out))
            self.log[layer] = rec
        return out

    @property
    def flags(self):
        return getattr(self.edit, "flags", {})


def coord_err_clamp(edit):
    def f(layer, out):
        A, R, mu, t = (x[layer].to(out) for x in (edit.A, edit.R, edit.mu, edit.target))
        e32 = (R @ (out - mu).T - t).abs().max()
        ebf = (R @ (out.to(torch.bfloat16).float() - mu).T - t).abs().max()
        return {"coord_err_fp32": float(e32), "coord_err_bf16": float(ebf)}
    return f


def coord_err_swap(edit):
    def f(layer, out):
        if layer not in edit.targets:
            return {}
        i = edit.band.index(layer)
        pinv = torch.linalg.pinv(torch.stack([edit.s[i], edit.t[i]], 1).to(out))
        t = edit.targets[layer].to(out)
        return {"coord_err_fp32": float((pinv @ out.T - t).abs().max()),
                "coord_err_bf16": float((pinv @ out.to(torch.bfloat16).float().T - t).abs().max())}
    return f


# ------------------------------------------------------------------------------------------------ material
def load_material(with_cm):
    arz = torch.load(E.OUT / "ar" / "ar_outputs.pt", weights_only=False)["outputs"]["multihop"]
    Q = {L: torch.load(E.OUT / "bases" / f"J_raw_L{L}.pt", weights_only=False)["Q"][:, :1024].float() for L in BAND}
    cm = None
    if with_cm:
        caps, cov = {}, {}
        G = E.OUT / "gate"
        for pat, src in (("gate_shard*.json", "v1"), ("gate_v2*_shard*.json", "v2"), ("gate_easy_shard*.json", "easy")):
            for f in sorted(G.glob(pat)):
                for c, v in json.loads(f.read_text())["concepts"].items():
                    if src == "easy" and c not in RESCUED:                 # style controls never replace material
                        continue
                    cov[c] = {"dropped": v["dropped"], "source": src}
        for pat in ("caps_shard*.pt", "caps_v2*_shard*.pt", "caps_easy_shard*.pt"):
            for f in sorted(G.glob(pat)):
                for c, v in torch.load(f, weights_only=False)["caps"].items():
                    if "easy" in f.name and c not in RESCUED:
                        continue
                    caps[c] = v
        kept = sorted(c for c, v in cov.items() if not v["dropped"] and caps.get(c, {}).get("final") is not None)
        means = {c: caps[c]["final"].float().mean(0) for c in kept}                      # [12 layers, D]
        mu = torch.stack(list(means.values())).mean(0)
        cm = {"dirs": {c: unit(m - mu) for c, m in means.items()},                      # [12, D] per concept
              "covered": kept, "source": {c: cov[c]["source"] for c in kept}}
    return arz, Q, cm


def ar_mean(arz, c, sets, tag):
    xs = [arz[c][s][tag].float() for s in sets if s in arz.get(c, {}) and tag in arz[c][s]]
    return torch.cat(xs).mean(0) if xs else None


def ar_diff(arz, x, y, sets):
    out = {}
    for L in BAND:
        a, b = ar_mean(arz, x, sets, TAG[L]), ar_mean(arz, y, sets, TAG[L])
        if a is None or b is None:
            return None
        out[L] = b - a
    return out


def jremove(d, Q):
    return {L: unit(d[L] - Q[L] @ (Q[L].T @ d[L])) for L in d}


def rremove(d, name):
    out = {}
    for L in d:
        g = torch.Generator().manual_seed(seed_of(name, "rrem", L))
        P, _ = torch.linalg.qr(torch.randn(D, 1024, generator=g))
        out[L] = unit(d[L] - P @ (P.T @ d[L]))
    return out


def rand_dirs(name, tag, Q=None):
    out = {}
    for L in BAND:
        g = torch.Generator().manual_seed(seed_of(name, tag, L))
        v = torch.randn(D, generator=g)
        if Q is not None:
            v = v - Q[L] @ (Q[L].T @ v)
        out[L] = unit(v)
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("parity", "cmlever", "smoke"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--items", default="")
    ap.add_argument("--n-smoke", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out_dir = E.OUT / "edits"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{a.mode}{'_' + a.tag if a.tag else ''}_shard{a.shard}"
    t0 = time.time()

    cohort = {r["name"]: r for r in E.cohort()}
    stageB = {json.loads(l)["name"]: json.loads(l) for l in open(E.STAGEB_ITEMS)}
    bench = {it["name"]: it for it in load_benchmark(BENCH)["items"]}
    arz, Q, cm = load_material(with_cm=a.mode in ("cmlever", "smoke"))
    if a.mode == "parity":
        names = [n for n, r in stageB.items() if r.get("status") == "evaluated"]
    elif a.mode == "cmlever":
        names = [n for n in cohort if cohort[n]["src"] in cm["dirs"] and cohort[n]["tgt"] in cm["dirs"]]
    else:
        pool = sorted(n for n in cohort if cohort[n]["src"] in cm["dirs"] and cohort[n]["tgt"] in cm["dirs"])
        names = a.items.split(",") if a.items else list(np.random.default_rng(11).choice(pool, a.n_smoke, replace=False))
    names = sorted(names)[a.shard::a.nshard]
    stageA = {r["name"]: r for r in map(json.loads, open(SA / "runs/j_swap_L36-47.jsonl"))}

    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks = runner.tokenizer, runner.blocks
    bank = DirectionBank.load(SA / "bank_j")
    W32 = runner.model.get_output_embeddings().weight.detach().float()
    norm = runner.final_norm
    WH = None
    if a.mode == "smoke":
        sys.path.insert(0, str(ROOT / "scripts" / "011_olens_swaps"))
        import ar_vectors as AV
        WH = {t: AV.whitener(t) for t in (36, 40, 44)}
    print(f"loaded in {time.time() - t0:.0f}s; mode {a.mode}; {len(names)} items", flush=True)

    rows = []
    for k_, nm in enumerate(names):
        t1 = time.time()
        it = bench[nm]
        prompt = it["prompt"].rstrip()
        ids = runner.encode(prompt)
        n, pos = len(ids), len(ids) - 1
        alts_a, alts_b = it.get("y_a_alts", []), it.get("y_b_alts", [])
        sa = [answer_generation_budget(tok, it["prompt"], [it["y_a"], *alts_a]),
              answer_generation_budget(tok, it["prompt"], [it["y_b"], *alts_b])]
        budget = max(sa)
        spell_a, spell_b = [it["y_a"], *alts_a], [it["y_b"], *alts_b]
        dec = lambda x: tok.decode(x, skip_special_tokens=True)            # noqa: E731

        def one(s, edit):
            grab = {}
            hk = norm.register_forward_pre_hook(lambda _m, args: grab.__setitem__("x", args[0]))
            try:
                r = runner.continuation_scores(prompt, s, edit=edit)
            finally:
                hk.remove()
            m0, tg = r["prompt_tokens"], r["token_ids"]
            with torch.no_grad():
                y = norm(grab["x"][0, m0 - 1:m0 - 1 + len(tg)].float())
                lp32 = torch.log_softmax(y @ W32.T, -1)[torch.arange(len(tg)), torch.tensor(tg)].double().sum()
            return float(lp32), r["logprob_sum"]

        def scores(edit):
            both = {s: one(s, edit) for s in dict.fromkeys(spell_a + spell_b)}
            lp, lpj = {s: v[0] for s, v in both.items()}, {s: v[1] for s, v in both.items()}
            mk = lambda d_: lse([d_[s] for s in spell_b]) - lse([d_[s] for s in spell_a])   # noqa: E731
            return {"lp": lp, "m": mk(lp), "m_bf16_jan": mk(lpj)}

        arms = {}

        def run(armname, edit, coord_fn=None, collision=False, **meta):
            lg = Logged(edit, coord_fn) if edit is not None else None
            gids = runner.generate(ids, edit=lg, max_new_tokens=budget, return_token_ids=True)
            tp = dec(gids[:sa[1]])
            rec = {"text": dec(gids), "target_prefix": tp, **scores(edit), **meta}
            if armname != "clean":
                rec["success"] = causal_outcome(it, tp, condition="swap", collision=collision)
                rec["dm"] = rec["m"] - arms["clean"]["m"]
                rec["greedy_same_as_clean"] = gids == arms["clean"]["ids"]
                rec["edit_log"] = {str(L): v for L, v in lg.log.items()}
                rec["flags"] = getattr(edit, "flags", {})
            else:
                rec["ids"] = gids
            arms[armname] = rec

        run("clean", None)
        arms["clean"]["clean_correct"] = causal_outcome(it, dec(arms["clean"]["ids"][:sa[0]]), condition="clean")
        cap = runner.capture_ids(ids, positions=list(range(n)), layers=BAND)
        clean = {L: cap[i] for i, L in enumerate(BAND)}
        s_dirs, t_dirs = bank.concept(it["src"], BAND), bank.concept(it["tgt"], BAND)
        rs = [BENCH, nm, ARM_ID, "rand_swap"]
        Jsw = SwapEdit(s_dirs, t_dirs, clean, BAND, 1.0)
        sd = []
        for j, L in enumerate(BAND):
            V = torch.stack([s_dirs[j], t_dirs[j]], 1)
            c = torch.linalg.pinv(V) @ clean[L].float().T
            sd.append(float(c[0, pos] - c[1, pos]))
        j_dir = "toward_tgt" if np.mean(sd) > 0 else "toward_src"
        row = {"name": nm, "src": it["src"], "tgt": it["tgt"], "y_a": it["y_a"], "y_b": it["y_b"], "n_prompt": n,
               "read_pos": pos, "in_cohort": nm in cohort, "j_edit_direction": j_dir}

        if a.mode in ("parity", "smoke"):
            run("J_L36-47", Jsw, coord_err_swap(Jsw), collision=Jsw.flags["proxy_collision_all_layers"])
            Jr = SwapEdit(s_dirs, random_directions(s_dirs.shape[-1], rs, BAND), clean, BAND, 1.0)
            run("rand_L36-47", Jr, coord_err_swap(Jr))
        if a.mode == "parity":
            z = SwapEdit(s_dirs, t_dirs, clean, BAND, 0.0)
            ident = SwapEdit(s_dirs, t_dirs, clean, BAND, 1.0)
            ident.targets = {L: t[[1, 0]] for L, t in ident.targets.items()}
            u0 = ar_diff(arz, it["src"], it["tgt"], PHRASES)
            u0 = {L: unit(v) for L, v in u0.items()} if u0 else rand_dirs(nm, "ident_add")
            ia = additive(u0, {L: torch.zeros(n) for L in BAND}, clean, "identity_additive")
            g0 = {}
            for gname, e in (("alpha0", z), ("identity_swap", ident), ("identity_additive", ia)):
                sc = scores(e)
                gi = runner.generate(ids, edit=e, max_new_tokens=budget, return_token_ids=True)
                g0[gname] = {"dm": sc["m"] - arms["clean"]["m"], "greedy_same": gi == arms["clean"]["ids"],
                             "max_abs_dlp": max(abs(sc["lp"][s] - arms["clean"]["lp"][s]) for s in sc["lp"])}
            row["G0"] = g0
            r = stageA.get(nm)
            if r is not None:
                res = r["result"]
                row["parity"] = {"stageA_success": r["success"], "ours_success": arms["J_L36-47"]["success"],
                                 "stageA_random_success": r["random_success"],
                                 "ours_random_success": arms["rand_L36-47"]["success"],
                                 "clean_same": res["clean"] == arms["clean"]["text"],
                                 "J_same": res["intervention"] == arms["J_L36-47"]["text"],
                                 "rand_same": res["random"] == arms["rand_L36-47"]["text"],
                                 "stageA_clean_correct": r["clean_correct"]}

        if a.mode in ("cmlever", "smoke") and it["src"] in cm["dirs"] and it["tgt"] in cm["dirs"]:
            cs, ct = cm["dirs"][it["src"]], cm["dirs"][it["tgt"]]
            CMsw = SwapEdit(cs, ct, clean, BAND, 1.0)
            cdir = []
            for j, L in enumerate(BAND):
                V = torch.stack([cs[j], ct[j]], 1)
                c = torch.linalg.pinv(V) @ clean[L].float().T
                cdir.append(float(c[0, pos] - c[1, pos]))
            row["cm_edit_direction"] = "toward_tgt" if np.mean(cdir) > 0 else "toward_src"
            row["cm_source"] = {"src": cm["source"][it["src"]], "tgt": cm["source"][it["tgt"]]}
            run("CM_swap", CMsw, coord_err_swap(CMsw), collision=CMsw.flags["proxy_collision_all_layers"])
            CMr = SwapEdit(cs, random_directions(D, [BENCH, nm, "cm", "rand_swap"], BAND), clean, BAND, 1.0)
            run("CM_rand", CMr, coord_err_swap(CMr))
            if a.mode == "smoke":
                jr = lambda d_: torch.stack([unit(d_[j] - Q[L] @ (Q[L].T @ d_[j])) for j, L in enumerate(BAND)])  # noqa: E731
                CMj = SwapEdit(jr(cs), jr(ct), clean, BAND, 1.0)
                run("CM_swap_Jrem", CMj, coord_err_swap(CMj))

        if a.mode == "smoke":
            cm_ok = "CM_swap" in arms
            k_cm = swap_sizes(cm["dirs"][it["src"]], cm["dirs"][it["tgt"]], clean) if cm_ok else None
            k_j = swap_sizes(s_dirs, t_dirs, clean)
            K = k_cm if cm_ok else k_j
            row["size_reference"] = "class-mean swap" if cm_ok else "J-row swap (fallback)"
            row["ref_size_rel_read"] = {str(L): float(K[L][pos] / clean[L][pos].norm()) for L in BAND}
            row["ref_size_rel_median_all_pos"] = {str(L): float((K[L] / clean[L].norm(dim=-1)).median()) for L in BAND}

            def add(armname, d, KK=None, **meta):
                if d is None:
                    arms[armname] = {"skipped": "no vector"}
                    return
                u = {L: unit(v) for L, v in d.items()}
                e = additive(u, KK or K, clean, armname)
                run(armname, e, coord_err_clamp(e), **meta)

            full = ar_diff(arz, it["src"], it["tgt"], PHRASES)
            add("AR_full", full)
            add("AR_Jrem", jremove(full, Q) if full else None)
            add("AR_rrem", rremove(full, nm) if full else None)
            add("AR_Jrem_Jsize", jremove(full, Q) if full else None, KK=k_j)
            for s_ in ("starts", "mentions", "template", "word"):
                d_ = ar_diff(arz, it["src"], it["tgt"], (s_,))
                add(f"AR_{s_}", d_)
                add(f"AR_{s_}_Jrem", jremove(d_, Q) if d_ else None)
            add("AR_answer", ar_diff(arz, it["y_a"], it["y_b"], PHRASES + ("template", "word")))
            others = [r for r in cohort.values() if E.rel(r["name"]) != E.rel(nm)
                      and not ({r["src"], r["tgt"]} & {it["src"], it["tgt"]})]
            mm = others[np.random.default_rng(seed_of(nm, "mm") % 2**32).integers(len(others))]
            row["mismatched_pair"] = [mm["src"], mm["tgt"]]
            mmd = ar_diff(arz, mm["src"], mm["tgt"], PHRASES)
            add("AR_mismatched", mmd)
            add("AR_mismatched_Jrem", jremove(mmd, Q) if mmd else None)
            for i in range(2):
                add(f"rand_full_{i}", rand_dirs(nm, f"rf{i}"))
                add(f"rand_Jcomp_{i}", rand_dirs(nm, f"rjc{i}", Q))
            # the TA's whitened coordinate swap (011 C_white), nearest tag
            V_all = {}
            for t in (36, 40, 44):
                w = WH[t]
                mh = sorted({r["src"] for r in cohort.values()} | {r["tgt"] for r in cohort.values()})
                Wm = {c: (w["W"].float() @ (ar_mean(arz, c, PHRASES, t) - w["mu"].float())) for c in mh}
                cen = torch.stack(list(Wm.values())).mean(0)
                V_all[t] = {c: v - cen for c, v in Wm.items()}
            A_, R_, mu_, tg_ = {}, {}, {}, {}
            for L in BAND:
                t, w = TAG[L], WH[TAG[L]]
                Vt = torch.stack([V_all[t][it["src"]], V_all[t][it["tgt"]]], 1)
                R_[L] = torch.linalg.pinv(Vt) @ w["W"].float()
                A_[L] = w["Wi"].float() @ Vt
                mu_[L] = w["mu"].float()
                c = R_[L] @ (clean[L].float() - mu_[L]).T
                tg_[L] = c[[1, 0]]
            Cw = ClampEdit(BAND, A_, R_, mu_, tg_, "C_white")
            run("C_white", Cw, coord_err_clamp(Cw))

        row["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in arms.items()}
        row["sec"] = round(time.time() - t1, 1)
        rows.append(row)
        brief = " ".join(f"{k}:{v.get('dm', 0):+.2f}{'*' if v.get('success') else ''}" for k, v in arms.items()
                         if k != "clean" and "dm" in v)
        print(f"[{k_ + 1}/{len(names)}] {nm} {row['sec']}s {brief}", flush=True)

    with open(out_dir / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {out_dir / name}.jsonl", flush=True)


if __name__ == "__main__":
    main()
