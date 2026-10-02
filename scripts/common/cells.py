"""The cell format: one token position at one layer of one item. Every label and every readout refers to a Cell.

Fields (fixed by AMENDMENT_01 §3): item, pair_id, position, layer, family, half, split.
`position` is an absolute index into the item's own render (pairs are token-aligned, so it indexes the
partner's render too). `layer` is a block-output index (0..63). `half` is the cross-fitting half (A/B),
`split` the dev/test split.
"""
import json
from dataclasses import asdict, dataclass, fields

FAMILIES = ("brew", "chain", "relational", "multihop", "role_bound")
HALVES = ("A", "B")
SPLITS = ("dev", "test")
N_BLOCKS = 64


@dataclass(frozen=True, order=True)
class Cell:
    item: str
    pair_id: str
    position: int
    layer: int
    family: str
    half: str
    split: str

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        if self.half not in HALVES:
            raise ValueError(f"half must be one of {HALVES}, got {self.half!r}")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        if not (0 <= int(self.layer) < N_BLOCKS):
            raise ValueError(f"layer {self.layer} outside 0..{N_BLOCKS - 1}")
        if int(self.position) < 0:
            raise ValueError("position must be a non-negative absolute index")
        if not self.item or not self.pair_id:
            raise ValueError("item and pair_id must be non-empty")

    @property
    def key(self):
        return f"{self.family}|{self.pair_id}|{self.item}|p{self.position}|L{self.layer}"

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        extra = set(d) - names
        if extra:
            raise ValueError(f"unexpected keys {sorted(extra)}")
        return cls(**{k: (int(v) if k in ("position", "layer") else v) for k, v in d.items()})

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))


def write_jsonl(cells, path):
    with open(path, "w") as f:
        for c in cells:
            f.write(c.to_json() + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [Cell.from_json(line) for line in f if line.strip()]
