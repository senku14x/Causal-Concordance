"""Readers: the instruments under test. Label code must never import this module (tests/test_import_boundary.py).

Every reader maps h, the float32 output of block `layer` at one position (stock bf16 forward with any adapter disabled,
WSB's Backend.capture convention), to a JSON-able readout.

  logit_lens     top-k of W_U . final_norm(h)                                   (WSB LogitLens)
  J standard     top-k of W_U . final_norm(J_L h)                               (WSB RLens / RHUT readout)
  J cosine       top-k of (W_U J_L h) / ||J_L^T W_U[t]||                        (WSB JLens: the J_NP arm of record)
  oracle_lens    WSB OLens on the public adapter agu18dec/olens_and_ar:olens_s3d_rl600; marker embedding row <-
                 alpha . unit(h), layer named in the prompt, adapter disabled while capturing
  nla AV         ceselder/qwen3.6-27b-nla-L42: base + av_sft_lora (merged) + av_rl_lora_step400; Karvonen add
                 h' = h + ||h|| v/||v|| at the output of block 1 at the marker (bf16 arithmetic, as EasyNLA and WSB)
  nla AR         rl_critic_step400 (43 blocks, no final norm) + float32 value head on normalize(h_last, sqrt(d));
                 FVE = 1 - mse(n(pred), n(gold)) / mean((n(v) - mean n(v))^2) over a reference set
  nla-rl AV      ceselder/qwen3.6-27b-nla-rl (WSB's NLA; the PI's default from 2026-10-04): av_base loaded as a separate
                 bf16 model + PEFT adapter av_rl_adapters/iter_000400 attached, not merged; the same Karvonen add at block
                 1; the subject model is never touched (class NLArl). No AR wrapper yet.
"""
import contextlib
import glob
import math
import os
import re

import torch

import registry as R

COLOURS = ["black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white"]
SAMPLING_WSB = dict(temperature=1.0, top_p=0.95, top_k=64, max_new_tokens=256)


def snapshot(repo, rev):
    return os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "snapshots", rev)


def colours_named(text):
    """Brew colour words in a readout (whole words, case-insensitive), in order of first mention."""
    seen = []
    for w in re.findall(r"[A-Za-z]+", text.lower()):
        if w in COLOURS and w not in seen:
            seen.append(w)
    return seen


# ------------------------------------------------------------------------------------------- lens files
def lens_path(name):
    e = R.REGISTRY["lenses"][name]
    repo_dir = "models--" + e["hf_repo"].replace("/", "--")
    p = os.path.join(os.environ["HF_HOME"], "hub", repo_dir, "snapshots", e["hf_revision"], e["file"])
    if not os.path.exists(p):
        hits = sorted(glob.glob(os.path.join(os.environ["HF_HOME"], "hub", repo_dir, "snapshots", "*", e["file"])))
        assert hits, f"lens file for {name} not found under {repo_dir}"
        p = hits[-1]
    return p


def load_lenses(names=("J_NP",), layers=None, device="cuda", verify_sha=True):
    """Load Jacobian lenses by registry name with jlens (pinned commit). Only `layers` are kept on device
    (float32); all others are dropped to save memory when `layers` is given."""
    from jlens.lens import JacobianLens
    out = {}
    for nm in names:
        p = lens_path(nm)
        if verify_sha:
            got = R.sha256(p)
            assert got == R.REGISTRY["lenses"][nm]["sha256"], f"{nm} sha256 mismatch: {got}"
        L = JacobianLens.load(p)
        assert list(L.source_layers) == list(range(R.N_SRC)), f"{nm}: unexpected layer map"
        keep = L.jacobians if layers is None else {l: L.jacobians[l] for l in layers}
        L.jacobians = {l: J.float().to(device) for l, J in keep.items()}
        out[nm] = L
    return out


# --------------------------------------------------------------------------------------------- capture
@torch.no_grad()
def capture_stock(model, ids, layers, positions):
    """{layer: [P, d] float32 cpu}: one stock bf16 forward (adapter disabled if one is mounted). Refuses a model whose
    weights a reader merged in place (disabling an adapter cannot undo a merge)."""
    import engine as E
    R.assert_subject_model_intact(model)
    R.assert_subject_model_resident(model)
    blocks = R.blocks(model)
    rec = E.Recorder(blocks, layers)
    off = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    with off, rec:
        model(torch.tensor([list(ids)], device=model.device), logits_to_keep=1)
    return {l: rec.acts[l][list(positions)].cpu() for l in layers}


# ---------------------------------------------------------------------------------------- vector lenses
def _topk(tok, scores, k):
    v, i = torch.topk(scores, k)
    return [(tok.decode([int(t)]), round(float(s), 4)) for s, t in zip(v, i)]


class VectorLenses:
    """Logit lens and the three Jacobian lenses (standard and cosine readouts) at a fixed set of layers."""

    def __init__(self, model, tok, layers, lens_names=("J_NP", "J_CB", "R_CB")):
        self.model, self.tok, self.layers = model, tok, list(layers)
        self.norm = R.final_norm(model)
        self.W = model.get_output_embeddings().weight.detach().float()
        self.lenses = load_lenses(lens_names, layers=self.layers, device=self.W.device)
        self._den = {}

    def _n(self, x):
        return self.norm(x.to(self.W.device).float()).float()

    @torch.no_grad()
    def logit_lens(self, h, k=10):
        return _topk(self.tok, self._n(h) @ self.W.T, k)

    @torch.no_grad()
    def j_standard(self, name, h, layer, k=10):
        t = self.lenses[name].transport(h.to(self.W.device).float(), layer)
        return _topk(self.tok, self._n(t) @ self.W.T, k)

    @torch.no_grad()
    def j_cosine(self, name, h, layer, k=10):
        J = self.lenses[name].jacobians[layer]
        key = (name, layer)
        if key not in self._den:                        # ||J^T W_U[t]|| per token, in row chunks (5 GB otherwise)
            self._den[key] = torch.cat([(self.W[i:i + 32768] @ J).norm(dim=1) for i in range(0, self.W.shape[0], 32768)])
        s = ((h.to(J.device).float() @ J.T) @ self.W.T) / self._den[key].clamp_min(1e-9)
        return _topk(self.tok, s, k)


# ---------------------------------------------------------------------------------------------- oracle
class OracleLens:
    """WSB produce/methods.py OLens, adapter from the public model repo."""

    def __init__(self, model, tok):
        from peft import PeftModel
        R.assert_subject_model_resident(model)           # a parked (SecReplay-built) model cannot generate
        c = R.REGISTRY["readers"]["oracle_lens"]
        self.c = c["contract"]
        self.dir = os.path.join(snapshot(c["hf_repo"], c["hf_revision"]), c["subfolder"])
        self.tok = tok
        self.model = PeftModel.from_pretrained(model, self.dir)
        self.model.eval()
        self._slots = {}

    def carrier(self, layer):
        if layer in self._slots:
            return self._slots[layer]
        for code in range(0x3200, 0x3400):
            ch = chr(code)
            ids = self.tok(ch, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                continue
            out = self.tok.apply_chat_template([{"role": "user", "content": self.c["prompt"].format(layer=layer, char=ch)}],
                                               tokenize=True, add_generation_prompt=True, enable_thinking=False)
            rendered = list(out["input_ids"] if hasattr(out, "keys") else out)
            slots = [i for i, t in enumerate(rendered) if t == ids[0]]
            if len(slots) == 1:
                self._slots[layer] = (rendered, slots[0], ch)
                return self._slots[layer]
        raise ValueError("no enclosed ideograph survives as one token in the carrier prompt")

    @torch.no_grad()
    def read(self, h, layer, k=4, seed=0, inject=True, sampling=SAMPLING_WSB):
        ids, slot, _ = self.carrier(layer)
        dev = self.model.device
        e = self.model.get_input_embeddings()(torch.tensor([ids], device=dev)).clone()
        if inject:
            v = h.to(dev).float()
            e[0, slot, :] = (self.c["alpha"] * v / v.norm().clamp_min(1e-9)).to(e.dtype)
        torch.manual_seed(seed)
        out = self.model.generate(inputs_embeds=e, attention_mask=torch.ones(1, len(ids), device=dev, dtype=torch.long),
                                  do_sample=True, temperature=sampling["temperature"], top_p=sampling["top_p"],
                                  top_k=sampling["top_k"], max_new_tokens=sampling["max_new_tokens"],
                                  num_return_sequences=k, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]

    @torch.no_grad()
    def read_batch(self, hs, layer, k=4, seed=0, sampling=SAMPLING_WSB):
        """Several activations in one generate call (same carrier, one row per activation; None = no activation).
        Returns a list (one per activation) of k samples."""
        ids, slot, _ = self.carrier(layer)
        dev = self.model.device
        base = self.model.get_input_embeddings()(torch.tensor([ids], device=dev))[0]
        e = base.unsqueeze(0).repeat(len(hs), 1, 1).clone()
        for b, h in enumerate(hs):
            if h is not None:
                v = h.to(dev).float()
                e[b, slot, :] = (self.c["alpha"] * v / v.norm().clamp_min(1e-9)).to(e.dtype)
        torch.manual_seed(seed)
        out = self.model.generate(inputs_embeds=e, attention_mask=torch.ones(len(hs), len(ids), device=dev, dtype=torch.long),
                                  do_sample=True, temperature=sampling["temperature"], top_p=sampling["top_p"],
                                  top_k=sampling["top_k"], max_new_tokens=sampling["max_new_tokens"],
                                  num_return_sequences=k, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        texts = [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]
        return [texts[b * k:(b + 1) * k] for b in range(len(hs))]

    def unload(self):
        """Remove the adapter (no merge) and return the base model."""
        return self.model.unload()


# ------------------------------------------------------------------------------------------------- NLA
def nla_paths():
    n = R.REGISTRY["readers"]["nla_L42"]
    root = snapshot(n["hf_repo"], n["hf_revision"])
    return root, {k: os.path.join(root, n[k]) for k in ("av_sft_lora", "av_rl_lora", "ar_critic")}, n


def nla_meta():
    import yaml
    root, _, n = nla_paths()
    with open(os.path.join(root, n["meta"])) as f:
        return yaml.safe_load(f)


def extract_explanation(text):
    """EasyNLA nla/schema.py: payload between <explanation> tags, stripped; None on miss."""
    m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return m.group(1).strip() if m else None


class NLAVerbalizer:
    """AV of nla-L42. MODIFIES `model` IN PLACE (the SFT LoRA is merged into the base weights) and marks it, so any
    later capture or labelling call on that object raises. Capture every subject activation first, or use a fresh model."""

    def __init__(self, model, tok_av):
        from peft import PeftModel
        R.assert_subject_model_resident(model)           # a parked (SecReplay-built) model cannot be merged into
        _, p, _ = nla_paths()
        meta = nla_meta()
        self.tok = tok_av
        merged = PeftModel.from_pretrained(model, p["av_sft_lora"]).merge_and_unload()
        for o in (model, merged):                       # the same object; mark it so no capture/label code reuses it
            setattr(o, R.MODIFIED_ATTR, "nla-L42 av_sft_lora merged in place by readers.NLAVerbalizer")
        self.model = PeftModel.from_pretrained(merged, p["av_rl_lora"])
        self.model.eval()
        tm = meta["tokens"]
        out = tok_av.apply_chat_template(
            [{"role": "user", "content": meta["prompt_templates"]["actor"].format(injection_char=tm["injection_char"])}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        self.ids = list(out["input_ids"] if hasattr(out, "keys") else out)
        (self.marker,) = [i for i, t in enumerate(self.ids) if t == int(tm["injection_token_id"])]
        assert self.ids[self.marker - 1] == tm["injection_left_neighbor_id"] and \
            self.ids[self.marker + 1] == tm["injection_right_neighbor_id"], "marker neighbours differ from the contract"
        self._v = {}
        (block1,) = [m for n, m in self.model.named_modules() if n.endswith("model.layers.1")]
        self._hook = block1.register_forward_hook(self._inject)
        self.injections = 0

    def _inject(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = self._v.get("v")
        mp = self.marker
        if v is None or h.shape[1] <= mp:
            return out
        if v.dim() == 1:                                   # one vector for every row (contract: bf16 arithmetic)
            hp = h[:, mp, :].clone()
            vu = v.to(h.device, h.dtype) / (v.to(h.device, h.dtype).norm() + 1e-9)
            h[:, mp, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
        else:                                              # one vector per row; all-zero rows = no activation
            V = v.to(h.device, h.dtype)
            assert V.shape[0] == h.shape[0], (V.shape, h.shape)
            for b in range(h.shape[0]):
                if float(V[b].float().norm()) == 0.0:
                    continue
                hp = h[b, mp, :].clone()
                vu = V[b] / (V[b].norm() + 1e-9)
                h[b, mp, :] = hp + hp.norm() * vu
        self.injections += 1
        return out

    @torch.no_grad()
    def read(self, h, k=4, seed=0, sampling=SAMPLING_WSB):
        """h=None runs the no-activation arm (nothing injected)."""
        self._v["v"] = None if h is None else h.float()
        try:
            torch.manual_seed(seed)
            x = torch.tensor([self.ids], device=self.model.device)
            out = self.model.generate(x, attention_mask=torch.ones_like(x), do_sample=True,
                                      temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
                                      max_new_tokens=sampling["max_new_tokens"], num_return_sequences=k,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        finally:
            self._v["v"] = None
        return self.tok.batch_decode(out[:, len(self.ids):], skip_special_tokens=True)

    @torch.no_grad()
    def read_batch(self, hs, k=4, seed=0, sampling=SAMPLING_WSB):
        """Several activations in one generate call (None = no activation). Returns one list of k raw samples per
        activation (rows are expanded k times by num_return_sequences, so the per-row vectors are too)."""
        d = next(h for h in hs if h is not None).shape[-1] if any(h is not None for h in hs) else 5120
        V = torch.stack([h.float() if h is not None else torch.zeros(d) for h in hs]).repeat_interleave(k, 0)
        self._v["v"] = V
        try:
            torch.manual_seed(seed)
            x = torch.tensor([self.ids] * len(hs), device=self.model.device)
            out = self.model.generate(x, attention_mask=torch.ones_like(x), do_sample=True,
                                      temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
                                      max_new_tokens=sampling["max_new_tokens"], num_return_sequences=k,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        finally:
            self._v["v"] = None
        texts = self.tok.batch_decode(out[:, len(self.ids):], skip_special_tokens=True)
        return [texts[b * k:(b + 1) * k] for b in range(len(hs))]

    def close(self):
        self._hook.remove()


class NLAReconstructor:
    """AR of nla-L42 (EasyNLA NLACriticModel + critic_predict)."""

    def __init__(self, device="cuda"):
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM
        _, p, _ = nla_paths()
        self.meta = nla_meta()
        m = AutoModelForCausalLM.from_pretrained(p["ar_critic"], dtype=torch.bfloat16, device_map=device)
        m.lm_head = torch.nn.Identity()
        m.model.norm = torch.nn.Identity()
        self.model = m.eval()
        self.n_layers = len(m.model.layers)
        d = m.config.get_text_config().hidden_size
        self.head = torch.nn.Linear(d, d, bias=False).to(device=device, dtype=torch.float32)
        self.head.load_state_dict(load_file(os.path.join(p["ar_critic"], "value_head.safetensors")))
        self.scale = math.sqrt(d)

    def norm(self, v):
        v = v.float()
        return v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-12) / self.scale)

    @torch.no_grad()
    def predict(self, tok, explanation):
        text = self.meta["prompt_templates"]["critic"].format(explanation=explanation)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids], device=self.model.device)
        hs = self.model.model(input_ids=x, use_cache=False).last_hidden_state[0, -1].float()
        with torch.autocast("cuda", enabled=False):
            return self.head(self.norm(hs))

    def fve(self, pred, gold, reference):
        """Paper FVE against the reference set's variance of normalised activations."""
        ref = self.norm(reference.to(pred.device))
        denom = float(((ref - ref.mean(0, keepdim=True)) ** 2).mean())
        mse = float(((self.norm(pred) - self.norm(gold.to(pred.device))) ** 2).mean())
        return 1.0 - mse / denom, mse, denom


def nla_reference_activations():
    import pandas as pd
    r = R.REGISTRY["readers"]["nla_reference_activations"]
    p = os.path.join(snapshot(r["hf_repo"], r["hf_revision"]), r["file"])
    df = pd.read_parquet(p)
    col = "activation_vector" if "activation_vector" in df.columns else df.columns[-1]
    return torch.tensor([list(v) for v in df[col]], dtype=torch.float32), df


# ------------------------------------------------------------------------------------- nla-rl (separate AV model)
def nla_rl_paths():
    n = R.REGISTRY["readers"]["nla_rl"]
    return snapshot(n["hf_repo"], n["hf_revision"]), n


def verify_nla_rl(root=None, entry=None, workers=4):
    """sha256 of every pinned nla-rl file against the registry. Returns {relative path: sha256}; raises on a missing
    file or a mismatch."""
    from concurrent.futures import ThreadPoolExecutor
    if root is None:
        root, entry = nla_rl_paths()
    rels = sorted(entry["sha256"])
    missing = [r for r in rels if not os.path.exists(os.path.join(root, r))]
    if missing:
        raise FileNotFoundError(f"nla-rl files not downloaded: {missing} (python scripts/common/artifacts.py download)")
    with ThreadPoolExecutor(workers) as ex:
        got = dict(zip(rels, ex.map(lambda r: R.sha256(os.path.join(root, r)), rels)))
    bad = {r: g for r, g in got.items() if g != entry["sha256"][r]}
    if bad:
        raise RuntimeError(f"nla-rl sha256 mismatch: {bad}")
    return got


def nla_rl_prompt_ids(tok, meta):
    """WSB's NLA prompt: the meta's AV template around the marker char, chat template, generation prompt, no thinking."""
    tm = meta["tokens"]
    tmpl = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
    out = tok.apply_chat_template([{"role": "user", "content": tmpl.format(injection_char=tm["injection_char"])}],
                                  tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = list(out["input_ids"] if hasattr(out, "keys") else out)
    (marker,) = [i for i, t in enumerate(ids) if t == int(tm["injection_token_id"])]
    assert ids[marker - 1] == tm["injection_left_neighbor_id"] and \
        ids[marker + 1] == tm["injection_right_neighbor_id"], "marker neighbours differ from the contract"
    return ids, marker


def karvonen_add_(h, v, marker):
    """WSB's NLA hook arithmetic, in place on h [B, T, d]: h[:, m] <- h[:, m] + ||h[:, m]|| * v.to(h.dtype) / (||v|| + 1e-9),
    with ||v|| taken on the float32 v. v is [d] (every row) or [B, d] (one per row; an all-zero row is left alone, the
    no-activation arm). No-op when T <= marker (decode steps). Returns whether anything was written."""
    if v is None or h.shape[1] <= marker:
        return False
    v = v.to(h.device)
    if v.dim() == 1:
        hp = h[:, marker, :]
        vu = v.to(h.dtype) / (v.norm() + 1e-9)
        h[:, marker, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
        return True
    assert v.shape[0] == h.shape[0], (v.shape, h.shape)
    for b in range(h.shape[0]):
        if float(v[b].norm()) == 0.0:
            continue
        hp = h[b, marker, :]
        vu = v[b].to(h.dtype) / (v[b].norm() + 1e-9)
        h[b, marker, :] = hp + hp.norm() * vu
    return True


def wsb_explanation(text):
    """WSB's NLA post-processing: text before </explanation>, a leading <explanation> removed, stripped."""
    return text.split("</explanation>")[0].removeprefix("<explanation>").strip()


def check_gpu_room(device, need_bytes, margin=4 << 30):
    """Refuse to load when the device lacks room (e.g. the 50 GB subject is still resident)."""
    if not str(device).startswith("cuda"):
        return
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    if free < need_bytes + margin:
        raise RuntimeError(f"{free / 1e9:.1f} GB free on {device}; nla-rl needs {need_bytes / 1e9:.1f} GB + "
                           f"{margin / 1e9:.0f} GB: capture subject activations first, then free the subject "
                           f"(del model; gc.collect(); torch.cuda.empty_cache())")


class NLArl:
    """AV of ceselder/qwen3.6-27b-nla-rl, WSB's NLA (produce/methods.py): av_base as a SEPARATE bf16 model with the pinned
    PEFT adapter (iter_000400) attached, never merged. The subject model is never touched; this reader's own model is
    marked so no capture or labelling entry point accepts it. av_base (53.8 GB) and the subject (50.1 GB) do not fit on
    one 96 GB GPU together: capture subject activations first and free the subject. Trained on layer-42 residuals."""

    def __init__(self, device="cuda", verify=True):
        import yaml
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        root, n = nla_rl_paths()
        self.hashes = verify_nla_rl() if verify else None
        base, adir = os.path.join(root, n["av_base"]), os.path.join(root, n["adapter"])
        with open(os.path.join(root, n["meta"])) as f:
            self.meta = yaml.safe_load(f)
        self.tok = AutoTokenizer.from_pretrained(base)
        self.ids, self.marker = nla_rl_prompt_ids(self.tok, self.meta)
        need = sum(os.path.getsize(os.path.join(root, r)) for r in n["sha256"] if r.endswith(".safetensors"))
        check_gpu_room(device, need)
        m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map=device)
        self.model = PeftModel.from_pretrained(m, adir, torch_dtype=torch.bfloat16)   # WSB's call, verbatim
        self.model.eval()
        self.model.requires_grad_(False)
        for o in (self.model, self.model.get_base_model()):
            setattr(o, R.MODIFIED_ATTR, "nla-rl reader model (av_base + adapter), not the subject")
        self.scaling = sorted({float(s) for mod in self.model.modules()          # LoRA layers keep a dict;
                               if isinstance(getattr(mod, "scaling", None), dict)  # attention keeps a float
                               for s in mod.scaling.values()})
        (block1,) = [mod for name, mod in self.model.named_modules() if name.endswith("model.layers.1")]
        self._v = {}
        self._hook = block1.register_forward_hook(self._inject)
        self.injections = 0
        self.last_write = None

    def _inject(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = self._v.get("v")
        if v is None or h.shape[1] <= self.marker:
            return out
        before = h[:, self.marker, :].float().clone()
        if karvonen_add_(h, v, self.marker):
            d = h[:, self.marker, :].float() - before
            vv = v.to(h.device).float().expand_as(d) if v.dim() == 1 else v.to(h.device).float()
            live = vv.norm(dim=-1) > 0
            rel = (d.norm(dim=-1) / before.norm(dim=-1))[live]
            cos = torch.nn.functional.cosine_similarity(d, vv, dim=-1)[live]
            self.last_write = {"rows": int(live.sum()), "rel_min": float(rel.min()), "rel_max": float(rel.max()),
                               "cos_min": float(cos.min())} if bool(live.any()) else {"rows": 0}
            self.injections += 1
        return out

    @torch.no_grad()
    def read(self, h, k=1, seed=0, sampling=SAMPLING_WSB):
        """k raw samples for one activation h (float32 [d]); h=None runs the no-activation arm."""
        return self.read_batch([h], k=k, seed=seed, sampling=sampling)[0]

    @torch.no_grad()
    def read_batch(self, hs, k=1, seed=0, sampling=SAMPLING_WSB):
        """Several activations in one generate call (None = no activation). Returns one list of k raw samples per
        activation (rows are expanded k times by num_return_sequences, so the per-row vectors are too)."""
        if all(h is None for h in hs):
            V = None
        else:
            d = next(h for h in hs if h is not None).shape[-1]
            V = torch.stack([h.float() if h is not None else torch.zeros(d) for h in hs]).repeat_interleave(k, 0)
            if len(hs) == 1:
                V = V[0]                                       # one vector for every row: WSB's exact path
        self._v["v"] = V
        try:
            torch.manual_seed(seed)
            x = torch.tensor([self.ids] * len(hs), device=self.model.device)
            out = self.model.generate(x, attention_mask=torch.ones_like(x), do_sample=sampling["temperature"] > 0,
                                      temperature=max(sampling["temperature"], 1e-5), top_p=sampling["top_p"],
                                      top_k=sampling["top_k"], max_new_tokens=sampling["max_new_tokens"],
                                      num_return_sequences=k, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        finally:
            self._v["v"] = None
        texts = self.tok.batch_decode(out[:, len(self.ids):], skip_special_tokens=True)
        return [texts[b * k:(b + 1) * k] for b in range(len(hs))]

    def close(self):
        """Remove the hook and free the GPU."""
        import gc
        self._hook.remove()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
