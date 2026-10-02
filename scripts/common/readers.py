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
    """{layer: [P, d] float32 cpu}: one stock bf16 forward (adapter disabled if one is mounted)."""
    import engine as E
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
    """AV of nla-L42. MODIFIES `model` IN PLACE (the SFT LoRA is merged into the base weights): capture first."""

    def __init__(self, model, tok_av):
        from peft import PeftModel
        _, p, _ = nla_paths()
        meta = nla_meta()
        self.tok = tok_av
        merged = PeftModel.from_pretrained(model, p["av_sft_lora"]).merge_and_unload()
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
        hp = h[:, mp, :].clone()
        vu = v.to(h.device, h.dtype) / (v.to(h.device, h.dtype).norm() + 1e-9)
        h[:, mp, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
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
