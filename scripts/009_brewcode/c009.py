"""009 (brew-code bank) shared constants and small helpers (design docs/design_specs/008_009_trackb_brewcode.md, Part 2).
Reader-free: no lens, AR or SAE code may be imported by any 009 module (label/reader import boundary)."""
import json
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "outputs" / "009_brewcode"
RES = PROJECT / "results" / "009_brewcode"
BANK_DIR = OUT / "bank"

BANK_SEED = 20261006          # dev stream; the test stream uses BANK_SEED + 1 (separate, so dev never depends on test)
BOOT_SEED = 20261004
N_DEV, N_TEST = 300, 150
N_FOLDS = 5
COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
N_RULES = 10
CODE_RANGE = (10, 99)          # inclusive
CAP_LAYERS = (40, 42, 44, 48, 52, 56, 60)
EMIT_CELLS = ("emit_asst", "emit_think", "emit_colon")
TASKS = ("final", "start")     # final-code (needs z) / start-code (does not)
DELTA = 0.5                    # nats, the output-facing equivalence margin (design §2.5.1)
GATE_MIN_SHARE, GATE_MIN_PAIRS = 0.25, 200    # design §2.4

INSTRUCTION = "Here is a colour-mixing problem. Answer with a single two-digit code and nothing else."
PROBLEM = "Problem: A potion changes colour each time it is stirred, according to these rules:"
STIR = {2: "You stir the potion twice.", 1: "You stir the potion once."}
START_FMT = "The potion starts out {}."
QUESTION = {"final": "What is the code of the potion's colour at the end?",
            "start": "What is the code of the potion's colour at the start?"}
QUESTION_TOKENS = 13           # both questions; they differ only at token 12 (" end" / " start")


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, *a):
        msg = f"[{time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - self.t0:7.0f}s] " + " ".join(map(str, a))
        print(msg, flush=True)
        with open(self.path, "a") as f:
            f.write(msg + "\n")


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
