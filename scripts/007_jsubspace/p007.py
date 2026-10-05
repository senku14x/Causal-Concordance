"""007 shared constants and small helpers (design docs/design_specs/007_jsubspace.md). Reader-side; never imported by
label code."""
import json
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "outputs" / "007_jsubspace"
RES = PROJECT / "results" / "007_jsubspace"
STAGE0 = OUT / "stage0"
CAPS = PROJECT / "outputs" / "001_leverage_profile" / "run" / "captures_brew.npz"
CAPS_META = PROJECT / "outputs" / "001_leverage_profile" / "run" / "captures_brew_meta.json"
POS_ACTS = PROJECT / "outputs" / "006_reader_likelihood" / "run1" / "positive_acts.pt"     # [12, (40,42,44,56,60), d]
LL006 = PROJECT / "outputs" / "006_reader_likelihood" / "run1"

COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
SEED = 20261005
BOOT_SEED = 20261004
K = 1024
D = 5120
LAYERS = (40, 42, 44)
CAP_LAYERS = (40, 42, 44, 56, 60)
CELLS = {"colon": "emit_colon", "think": "emit_think"}
CELL_INDEX = {"emit_asst": 0, "emit_think": 1, "emit_colon": 2}
PCA_K = {("colon", 40): 4, ("colon", 42): 4, ("colon", 44): 4, ("think", 40): 4, ("think", 42): 3, ("think", 44): 4}
READERS = (("oracle", 40), ("oracle", 44), ("nla", 42))                 # (reader, layer); reader layer = capture layer
DELTA = 0.5                                                               # nats, the pre-registered margin

# Stage 1a/1b pieces (26). Scoring order (design §4.5): whole first (G-repro), then the decisive pieces.
N_RAND, N_REST_RAND, N_RAND_C = 10, 3, 3
PIECE_ORDER = (["whole", "J", "rest_J", "PCA", "J_c"] + [f"R{r}_c" for r in range(N_RAND_C)] +
               ["MR", "rest_PCA", "GMR", "LL", "rest_LL"] + [f"R{r}" for r in range(N_RAND)] +
               [f"rest_R{r}" for r in range(N_REST_RAND)])
assert len(PIECE_ORDER) == 26 and len(set(PIECE_ORDER)) == 26
# brew test pieces (15; design §7.3 + open point O2)
TEST_PIECES = (["whole", "J", "rest_J", "PCA", "MR", "J_c"] + [f"R{r}_c" for r in range(3)] +
               [f"R{r}" for r in range(3)] + [f"rest_R{r}" for r in range(3)])
assert len(TEST_PIECES) == 15
COMPLEMENTS = [("J", "rest_J"), ("PCA", "rest_PCA"), ("LL", "rest_LL")] + [(f"R{r}", f"rest_R{r}") for r in range(N_REST_RAND)]
QUAL_PIECES = ("whole", "J", "rest_J", "PCA", "MR", "GMR", "R0")


def rand_seed(kind, r):
    """Random-subspace seeds. Stage 1 draws use SEED + r (report 22's bases). Stage 2's extra families (fixed at Stage 0,
    before any outcome): dims 512 / 2,560 use SEED + 100 + r (design §5.2); the delta-matched family uses SEED + 200 + r
    (the design left it unspecified)."""
    return {"r1024": SEED + r, "r512": SEED + 100 + r, "r2560": SEED + 100 + r, "rdelta": SEED + 200 + r}[kind]


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
