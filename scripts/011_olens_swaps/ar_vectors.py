"""011: colour vectors from the Oracle Lens's own reconstructor (AR), for Oracle-space swaps (Camila's suggestion via the PI,
2026-10-06). READER-SIDE code (the Oracle Lens's AR); never imported by label code. One dedicated process: the AR's LoRA
is attached to this process's copy of the subject model, which is then marked as modified.

  python ar_vectors.py --run-id v1

AR (agu18dec/olens_and_ar @ c95215d, ar_ptag_pooled; README): Qwen3.6-27B text model + LoRA r16/a32 on every linear,
one shared Linear(5120, 5120) head; input = "[Layer NN]" tag (6 tokens, from heads.pt) + span; read the final hidden
state at the span's last real token; trained to predict the activation preceding the span, MSE in whitened space
(pooled whitener: agu18dec/local-workspace ckpts/ar/ptag/whiteners/whitening_L{L}.safetensors, cov + mu, lmsys-chat-1m).

Undocumented conventions are settled empirically FIRST (§validation): which layer index the tag means (block L or
block L-1 output), whether the head outputs ZCA-whitened / PCA-whitened / raw vectors, and whether the read is post- or
pre-final-norm. Acceptance: the best convention reaches FVE >= 0.10 at L44 on held-out assistant text (published val
FVE 0.220) and beats every other convention by >= 0.05; otherwise no vectors are written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import registry as R                                             # noqa: E402

PROJECT = HERE.parents[1]
OUT = PROJECT / "outputs" / "011_olens_swaps"
RES = PROJECT / "results" / "011_olens_swaps"
AR_REPO, AR_REV = "agu18dec/olens_and_ar", "c95215d5d2f6250f20c305ae55c6e4d7569f92dc"
WH_REPO = "agu18dec/local-workspace"
LAYERS = (40, 44, 48)
COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
SEED = 20261006
DIAG_ONLY = [False]
VAL_PROMPTS = [
    "Explain how a refrigerator keeps food cold.", "Write a short paragraph about the history of the bicycle.",
    "What are three tips for learning a new language?", "Describe a sunset over the ocean.",
    "How does compound interest work? Give an example.", "Summarize the plot of Romeo and Juliet in a few sentences.",
    "Why is the sky blue?", "Give me a simple recipe for pancakes.", "What is the difference between weather and climate?",
    "Write a polite email asking a colleague for a meeting next week.", "Explain photosynthesis to a ten-year-old.",
    "List some common causes of headaches.",
]


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, *a):
        msg = f"[{time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - self.t0:6.0f}s] " + " ".join(map(str, a))
        print(msg, flush=True)
        with open(self.path, "a") as f:
            f.write(msg + "\n")


def whitener(L):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    p = hf_hub_download(WH_REPO, f"ckpts/ar/ptag/whiteners/whitening_L{L}.safetensors", repo_type="dataset")
    with safe_open(p, "pt") as f:
        cov, mu, md = f.get_tensor("cov").double(), f.get_tensor("mu").double(), f.metadata()
    ev, U = torch.linalg.eigh(cov)
    assert float(ev[0]) > 0, f"L{L}: whitener covariance not positive definite (min eig {float(ev[0])})"
    return {"mu": mu, "ev": ev, "U": U, "W": (U * ev.rsqrt()) @ U.T, "Wi": (U * ev.sqrt()) @ U.T,
            "P": (U * ev.rsqrt()).T, "meta": md, "eig_min": float(ev[0]), "eig_max": float(ev[-1])}


@torch.no_grad()
def gen_texts(model, tok, log):
    """Stock assistant responses (greedy, 160 tokens) for the validation spans."""
    out = []
    for q in VAL_PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        x = torch.tensor([ids], device=model.device)
        g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=160,
                           pad_token_id=tok.eos_token_id)
        out.append((ids, [int(t) for t in g[0, len(ids):]]))
    log(f"generated {len(out)} validation responses")
    return out


@torch.no_grad()
def stock_targets(model, texts, log):
    """Validation spans and the stock residual preceding each span, at block outputs L-1 and L (both conventions)."""
    rng = np.random.default_rng(SEED)
    blocks = R.blocks(model)
    want = sorted({l for L in LAYERS for l in (L - 1, L)})
    spans, targets = [], {l: [] for l in want}
    for prompt, resp in texts:
        full = prompt + resp
        store = {}
        hs = [blocks[l].register_forward_hook(lambda m, i, o, l=l: store.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
              for l in want]
        try:
            model(torch.tensor([full], device=model.device), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        start = len(prompt) + 1
        for _ in range(64):
            n = int(rng.choice([1, 2, 4, 8, 16]))
            p = int(rng.integers(start, len(full) - n))
            spans.append(full[p:p + n])
            for l in want:
                targets[l].append(store[l][p - 1])
    log(f"{len(spans)} validation spans")
    return spans, {l: torch.stack(v).double() for l, v in targets.items()}


def attach_ar(model, log):
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    cfg = json.load(open(hf_hub_download(AR_REPO, "ar_ptag_pooled/lora/adapter_config.json", revision=AR_REV)))
    sd = load_file(hf_hub_download(AR_REPO, "ar_ptag_pooled/lora/adapter_model.safetensors", revision=AR_REV))
    sd = {k.replace("._orig_mod", ""): v for k, v in sd.items()}
    text = model.model
    lc = LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], target_modules=cfg["target_modules"], lora_dropout=0.0,
                    bias="none", use_rslora=bool(cfg.get("use_rslora")))
    pt = get_peft_model(text, lc)
    res = set_peft_model_state_dict(pt, sd)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    n_lora = sum(1 for n, _ in pt.named_parameters() if "lora_" in n)
    assert not unexpected, f"unexpected LoRA keys: {unexpected[:5]}"
    assert n_lora == len(sd), f"LoRA parameter count {n_lora} != checkpoint {len(sd)}"
    heads = torch.load(hf_hub_download(AR_REPO, "ar_ptag_pooled/heads.pt", revision=AR_REV), map_location="cpu",
                       weights_only=False)
    head = torch.nn.Linear(5120, 5120).to(model.device, torch.float32)
    head.load_state_dict({k.split(".", 1)[1]: v for k, v in heads["head"].items()})        # keys are "linear.*"
    setattr(model, R.MODIFIED_ATTR, "011 AR LoRA attached (Oracle Lens reconstructor); not the subject")
    log(f"AR attached: {n_lora} LoRA tensors; tags for layers {heads['layers']}")
    return pt, head, heads


@torch.no_grad()
def ar_forward(pt, head, heads, spans, L, bs=32):
    """Returns (post-norm read, pre-norm read) head outputs [n, 5120] float64."""
    tags = heads["tag_ids"][heads["layers"].index(L)].tolist()
    dev = head.weight.device
    outs_post, outs_pre = [], []
    norm = pt.base_model.model.norm
    for j in range(0, len(spans), bs):
        ch = [tags + list(s) for s in spans[j:j + bs]]
        T = max(len(c) for c in ch)
        x = torch.tensor([c + [0] * (T - len(c)) for c in ch], device=dev)
        m = torch.tensor([[1] * len(c) + [0] * (T - len(c)) for c in ch], device=dev)
        keep = {}
        hk = norm.register_forward_hook(lambda mod, i, o: keep.__setitem__("pre", i[0]))
        try:
            hs = pt(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state
        finally:
            hk.remove()
        last = m.sum(1) - 1
        r = torch.arange(len(ch), device=dev)
        post = hs[r, last].float()
        pre = keep["pre"][r, last].float()
        with torch.autocast("cuda", enabled=False):
            outs_post.append(head(post).double().cpu())
            outs_pre.append(head(pre).double().cpu())
    return torch.cat(outs_post), torch.cat(outs_pre)


def fve(yhat, y):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def diag(yhat, y):
    """Scale-free diagnostics: mean cosine, FVE after a per-sample least-squares scale, retrieval@1 (is the true target
    the most cosine-similar among all targets?)."""
    yn = torch.nn.functional.normalize(y, dim=1)
    hn = torch.nn.functional.normalize(yhat, dim=1)
    cos = (yn * hn).sum(1)
    a = (yhat * y).sum(1) / (yhat * yhat).sum(1).clamp_min(1e-30)
    fs = float(1 - ((y - a[:, None] * yhat) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
    ret = float((torch.argmax(hn @ yn.T, dim=1) == torch.arange(len(y))).double().mean())
    return {"cos_mean": float(cos.mean()), "fve_scaled": fs, "ret1": ret, "fve_raw": fve(yhat, y),
            "norm_ratio_median": float((yhat.norm(dim=1) / y.norm(dim=1)).median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v1")
    ap.add_argument("--diag-only", action="store_true")
    a = ap.parse_args()
    DIAG_ONLY[0] = a.diag_only
    out = OUT / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "log.txt")
    tok = R.make_tokenizer()
    WH = {L: whitener(L) for L in LAYERS}
    log("whiteners", {L: {"eig_min": w["eig_min"], "eig_max": w["eig_max"], **w["meta"]} for L, w in WH.items()})
    model = R.load_model()
    texts = gen_texts(model, tok, log)
    spans, tg = stock_targets(model, texts, log)
    pt, head, heads = attach_ar(model, log)
    val = {}
    DIAG = {}
    for L in LAYERS:
        post, pre = ar_forward(pt, head, heads, spans, L)
        w = WH[L]
        for lname, l in (("block_L", L), ("block_L-1", L - 1)):
            h = tg[l]
            ys = {"zca": (h - w["mu"]) @ w["W"].T, "pca": (h - w["mu"]) @ w["P"].T, "raw": h}
            for read, yhat in (("post_norm", post), ("pre_norm", pre)):
                for space, y in ys.items():
                    val[f"L{L}|{lname}|{read}|{space}"] = fve(yhat, y)
                    DIAG[f"L{L}|{lname}|{read}|{space}"] = diag(yhat, y)
    (RES / "ar_diagnostics.json").parent.mkdir(parents=True, exist_ok=True)
    (RES / "ar_diagnostics.json").write_text(json.dumps(DIAG, indent=1))
    log("diagnostics (top by ret@1)", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in
                                                  sorted(DIAG.items(), key=lambda kv: -kv[1]["ret1"])[:6]}))
    if DIAG_ONLY[0]:
        log("diagnostics only; stopping before acceptance")
        return
    best = max(val, key=val.get)
    b_layer, b_read, b_space = best.split("|")[1:]
    at44 = {k: v for k, v in val.items() if k.startswith("L44|")}
    second = sorted(at44.values())[-2]
    conv = f"{b_layer}|{b_read}|{b_space}"
    # the read/space convention must be clear (>= 0.05 above every other read/space at the best layer index); the layer
    # index (block L vs L-1, adjacent and therefore similar) is taken as the argmax and its margin is reported
    rivals = [v for k, v in at44.items() if k.split("|")[1] == b_layer and k != f"L44|{conv}"]
    ok = val[f"L44|{conv}"] >= 0.10 and val[f"L44|{conv}"] - max(rivals) >= 0.05
    other_layer = "block_L-1" if b_layer == "block_L" else "block_L"
    rep = {"run_id": a.run_id, "n_spans": len(spans), "fve": val, "best": best, "convention": conv,
           "fve_L44_best": val[f"L44|{conv}"], "fve_L44_second": second, "accepted": bool(ok),
           "layer_index_margin_L44": val[f"L44|{conv}"] - val[f"L44|{other_layer}|{b_read}|{b_space}"],
           "published_val_fve": {"L40": 0.224, "L44": 0.2202, "L48": 0.2277}, "manifest": R.manifest_block()}
    log("validation", json.dumps({k: round(v, 3) for k, v in sorted(val.items(), key=lambda kv: -kv[1])[:8]}), "accepted", ok)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "ar_validation.json").write_text(json.dumps(rep, indent=1))
    if not ok:
        log("STOP: no convention meets the acceptance rule; no vectors written")
        return
    vecs = {}
    for L in LAYERS:
        for form, fmt in (("space", " {}"), ("nospace", "{}")):
            sp = [tok.encode(fmt.format(c), add_special_tokens=False) for c in COLOURS]
            post, pre = ar_forward(pt, head, heads, sp, L)
            yhat = post if b_read == "post_norm" else pre
            w = WH[L]
            if b_space == "zca":
                white, raw = yhat, yhat @ w["Wi"].T + w["mu"]
            elif b_space == "pca":
                white, raw = yhat @ w["U"].T, (yhat * w["ev"].sqrt()) @ w["U"].T + w["mu"]     # zca = U pca
            else:
                white, raw = (yhat - w["mu"]) @ w["W"].T, yhat
            vecs[(L, form)] = {"zca_white": white.float(), "raw": raw.float(), "n_tokens": [len(s) for s in sp]}
        Z = vecs[(L, "space")]["zca_white"].double()
        Zc = Z - Z.mean(0)
        C = torch.nn.functional.normalize(Zc, dim=1) @ torch.nn.functional.normalize(Zc, dim=1).T
        rep.setdefault("colour_geometry", {})[f"L{L}"] = {
            "norm_median": float(Z.norm(dim=1).median()), "centred_norm_median": float(Zc.norm(dim=1).median()),
            "centred_cos_offdiag_mean": float((C.sum() - C.diag().sum()) / (12 * 11)),
            "space_vs_nospace_cos": float(torch.nn.functional.cosine_similarity(
                vecs[(L, "space")]["zca_white"].double(), vecs[(L, "nospace")]["zca_white"].double()).mean())}
    torch.save({"colours": COLOURS, "layers": LAYERS, "convention": conv, "vecs": vecs,
                "note": "zca_white = W (h - mu) with the AR's pooled whitener; raw = mu + W^-1 zca_white"}, out / "ar_colour_vectors.pt")
    (RES / "ar_validation.json").write_text(json.dumps(rep, indent=1))
    log("colour vectors written", json.dumps(rep.get("colour_geometry"), default=float))


if __name__ == "__main__":
    main()
