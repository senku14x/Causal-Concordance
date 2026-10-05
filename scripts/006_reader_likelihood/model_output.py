"""Design 006b (docs/design_specs/006b_model_output_baseline.md): the subject model's own final-layer log-probs for the
12 colours at the three brew cells (assistant newline, </think> newline, ':'), full-sequence scoring of every spelling
token, stock forward, float32 logits from the final norm. Reader-side scoring helpers (readers.cont_batch).

  python model_output.py --run-id m1
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
import brew_material as BM                        # noqa: E402
import readers as RD                              # noqa: E402
import registry as R                              # noqa: E402

COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
CELLS = ("emit_asst", "emit_think", "emit_colon")
WANT = {"emit_asst": ("assistant", "\n"), "emit_think": ("</think>", "\n\n"), "emit_colon": ("Answer", ":")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="m1")
    a = ap.parse_args()
    out = R.PROJECT / "outputs" / "006_reader_likelihood" / f"model_{a.run_id}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = json.loads((R.PROJECT / "outputs/001_leverage_profile/run/captures_brew_meta.json").read_text())
    keys = [r["key"] for r in meta["renders"] if r["role"] in ("member", "partner")]
    tok = R.make_tokenizer()
    by_key = {}
    for e in BM.brew_pairs(tok):
        for rr in (e["r_m"], e["r_p"]):
            by_key[BM.ids_key(rr.ids)] = rr
    missing = [k for k in keys if k not in by_key]
    assert not missing, f"{len(missing)} capture keys not rebuilt"
    renders = [by_key[k] for k in keys]
    pos = {}
    for rr in renders:
        by = {g: p for p, g in rr.regions.items() if g.startswith("emit_")}
        for c in CELLS:
            assert rr.decoded[by[c] - 1] == WANT[c][0] and rr.decoded[by[c]] == WANT[c][1], (rr.item, c)
        pos[BM.ids_key(rr.ids)] = by
    P = {c: sorted({pos[k][c] for k in keys}) for c in CELLS}
    assert all(len(v) == 1 for v in P.values()), P                     # one position per cell across renders
    seqs, owner = [], []
    for c in COLOURS:
        for form in (" " + c, " " + c.capitalize()):
            seqs.append(tok.encode(form, add_special_tokens=False))
            owner.append(c)
    model = R.load_model()
    inner = (model.get_base_model() if hasattr(model, "get_base_model") else model).model
    W32 = model.get_output_embeddings().weight.detach().float()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    LL = np.zeros((len(renders), len(CELLS), len(seqs)), np.float32)
    state = {"positions": {c: P[c][0] for c in CELLS}, "n_renders": len(renders), "spellings": [tok.decode(s) for s in seqs]}
    with torch.no_grad():
        # gate G-score-m: cont_batch path vs the full model's own logits at ':' (single-token spellings, 4 renders)
        single = [i for i, s in enumerate(seqs) if len(s) == 1]
        diffs = []
        for rr in renders[:4]:
            p = P["emit_colon"][0]
            rows, mask, gather = RD.cont_batch(rr.ids[:p + 1], [seqs[i] for i in single], pad)
            x, m = torch.tensor(rows, device=W32.device), torch.tensor(mask, device=W32.device)
            lp = RD.seq_logprob_from_hidden(inner(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state, W32, gather)
            own = torch.log_softmax(model(input_ids=torch.tensor([rr.ids[:p + 1]], device=W32.device), use_cache=False).logits[0, -1].float(), -1)
            diffs.append((lp.cpu() - own[[seqs[i][0] for i in single]].cpu()).abs())
        d = torch.cat(diffs)
        state["G_score_m"] = {"max": float(d.max()), "median": float(d.median())}
        print("G-score-m", state["G_score_m"], flush=True)
        assert float(d.max()) <= 0.25 and float(d.median()) <= 0.05, state["G_score_m"]
        for ci_, c in enumerate(CELLS):
            p = P[c][0]
            for j in range(0, len(renders), 4):
                ch = renders[j:j + 4]
                rows, mask, gather = [], [], []
                for rr in ch:
                    r_, m_, g_ = RD.cont_batch(rr.ids[:p + 1], seqs, pad)
                    rows += r_
                    mask += m_
                    gather += g_
                x, m = torch.tensor(rows, device=W32.device), torch.tensor(mask, device=W32.device)
                lp = RD.seq_logprob_from_hidden(inner(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state, W32, gather)
                LL[j:j + len(ch), ci_] = lp.view(len(ch), len(seqs)).cpu().numpy()
            print(f"{c} done ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out / "ll_model.npz", LL=LL, owner=np.array(owner), keys=np.array(keys), cells=np.array(CELLS))
    state["wall_minutes"] = (time.time() - t0) / 60
    (out / "state.json").write_text(json.dumps(state, indent=1))
    print("006b done", f"{state['wall_minutes']:.1f} min")


if __name__ == "__main__":
    main()
