"""ICLR-style figure helper: the iclrplot module vendored from RHUT (`scripts/common/iclr/`, the user's iclr-plots
skill) plus this project's fixed colour per reader, so a reader keeps its colour across every figure.

Figures are drawn from committed results/*/analysis JSON only (never from raw arrays), as in RHUT.
"""
import os
import sys

_ICLR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iclr", "scripts")
if _ICLR not in sys.path:
    sys.path.insert(0, _ICLR)
import matplotlib

matplotlib.use("Agg")
import iclrplot as ip  # noqa: E402

READERS = ["logit_lens", "J_NP", "J_CB", "R_CB", "oracle_lens", "nla", "sae", "prompt_only"]
READER_COLOURS = {r: c for r, c in zip(READERS[:-1], ip.CATEGORICAL_EXTENDED)}
READER_COLOURS["prompt_only"] = ip.BASELINE


def setup():
    ip.setup()
    return ip


def reader_colour(name):
    return READER_COLOURS.get(name, ip.NEUTRAL)


def save(fig, path):
    """Write `path`.pdf and `path`.png (directories created) and return the paths."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return ip.finish(fig, path)
