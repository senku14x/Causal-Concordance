"""008 (Track B) shared constants and helpers (design docs/design_specs/008_009_trackb_brewcode.md, Part 1).
Reader-side geometry (J-lens atoms) is used for interventions only; no labels are computed here."""
import json
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "outputs" / "008_trackb"
RES = PROJECT / "results" / "008_trackb"
CAPS001_META = PROJECT / "outputs" / "001_leverage_profile" / "run" / "captures_brew_meta.json"
COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
BAND = tuple(range(40, 49))                 # L40-48, every layer (block outputs; all inside SEC's fp32 region)
SIGMA_LAYERS = BAND
EMIT = ("emit_asst", "emit_think", "emit_colon")
SITES = {"emit3": EMIT, "colon": ("emit_colon",)}
K_PURSUIT = 25
BOOT_SEED = 20261004
SEED = 20261006
FLOOR = 2e-4
POWER_MIN_S = 0.15


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
