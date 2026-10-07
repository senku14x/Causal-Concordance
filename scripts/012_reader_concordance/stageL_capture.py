"""012 Stage L model side (design §22; EXPLORATORY; label code, no reader imports): Stage J's hold states captured at
more layers. One prompt per hold (clue x carrier; hold states are question-independent, Stage J), Stage J's edits
exactly (J_NP SwapEdit band L24-35 masked to positions < H1; random-target seed root as Stage I/J), states clean /
J_clue / rand_clue, block outputs at H1 / H2 for L40/42/44/48/52/56/60/63 in the prefill (same BandEditor + Cap path,
one generated token). Gate: L40/42/44 reproduce stageJ/j2_hold_captures.pt.

  PYTHONPATH=/workspace/sources/ext/phrase-lens-src/src python stageL_capture.py
"""
import json
import time

import torch
from phrase_lens.interventions import BandEditor, SwapEdit, random_directions
from phrase_lens.lenses import DirectionBank
from phrase_lens.modeling import ModelRunner

from stageB_capture import MODEL, REV, ROOT, SA
from stageF_capture import Masked
from stageI_capture import BAND, Cap, D as DI

LAYERS = [40, 42, 44, 48, 52, 56, 60, 63]
ARMS = ["clean", "J_clue", "rand_clue"]
DJ = ROOT / "outputs/012_reader_concordance/stageJ"
OUT = ROOT / "outputs/012_reader_concordance/stageL"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((DI / "prompts.json").read_text())
    capJ = torch.load(DJ / "j2_hold_captures.pt")
    runner = ModelRunner.from_pretrained(MODEL, device="cuda", dtype="bfloat16", revision=REV, local_files_only=True)
    tok, blocks = runner.tokenizer, runner.blocks
    bank = DirectionBank.load(SA / "bank_j")
    enc = lambda s: list(tok(s, add_special_tokens=True).input_ids)
    first = {}
    for it in spec["items"]:
        first.setdefault(f"{it['clue_id']}|{it['carrier']}", it)
    holds = [hk for hk in capJ["holds"]]
    X, parity = [], {}
    for hk in holds:
        it = first[hk]
        ids = enc(it["prompt"].rstrip())
        n = len(ids)
        H1, H2 = len(enc(it["prefix_H1"])) - 1, len(enc(it["prefix_H2"])) - 1
        c = runner.capture_ids(ids, positions=list(range(n)), layers=BAND)
        clean_acts = {L: c[i] for i, L in enumerate(BAND)}
        s_dirs, t_dirs = bank.concept(it["src"], BAND), bank.concept(it["tgt"], BAND)
        r_dirs = random_directions(s_dirs.shape[-1], ["012I_delayed", it["clue_id"], "j", "rand_swap"], BAND)
        J, Rr = SwapEdit(s_dirs, t_dirs, clean_acts, BAND, 1.0), SwapEdit(s_dirs, r_dirs, clean_acts, BAND, 1.0)
        make = {"clean": lambda: None, "J_clue": lambda: Masked(J, range(H1), n), "rand_clue": lambda: Masked(Rr, range(H1), n)}
        per_arm = []
        for arm in ARMS:
            e = make[arm]()
            ctx = BandEditor(blocks, e.band, e, n) if e is not None else torch.no_grad()
            with ctx, Cap(blocks, LAYERS, n, [H1, H2]) as cap:
                runner.generate(ids, edit=None, max_new_tokens=1, return_token_ids=True)
            per_arm.append(torch.stack([cap.acts[L] for L in LAYERS], 1))      # [pos(2), layer, d]
        Xh = torch.stack(per_arm)                                                  # [arm, pos, layer, d]
        X.append(Xh)
        i = capJ["holds"].index(hk)
        diffs = []
        for a_i, arm in enumerate(ARMS):
            for L in (40, 42, 44):
                ref = capJ["X"][i, capJ["arms"].index(arm), :, capJ["layers"].index(L)]
                got = Xh[a_i, :, LAYERS.index(L)]
                diffs.append(float((got - ref).abs().max() / ref.norm(dim=-1).max()))
        parity[hk] = max(diffs)
        print(f"[L] {hk} parity rel {parity[hk]:.2e} ({time.time() - t0:.0f}s)", flush=True)
    X = torch.stack(X)                                                             # [hold, arm, pos, layer, d]
    torch.save({"holds": holds, "arms": ARMS, "positions": ["H1", "H2"], "layers": LAYERS, "X": X, "meta": capJ["meta"]},
               OUT / "l_hold_captures.pt")
    g = {"parity_rel_max": max(parity.values()), "parity_bitwise_n": sum(v == 0.0 for v in parity.values()), "n_holds": len(holds),
         "pass": max(parity.values()) <= 1e-3, "wall_s": round(time.time() - t0, 1), "per_hold": parity}
    (ROOT / "results/012_reader_concordance/analysis/stageL_gates.json").write_text(json.dumps(g, indent=1))
    print(json.dumps({k: v for k, v in g.items() if k != "per_hold"}), flush=True)


if __name__ == "__main__":
    main()
