import os

import numpy as np

import figures as F


def test_figure_roundtrip(tmp_path):
    ip = F.setup()
    fig, ax = ip.figure(width="half")
    for i, r in enumerate(["J_NP", "nla", "prompt_only"]):
        ax.plot(np.arange(3), np.arange(3) + i, color=F.reader_colour(r), label=r)
    paths = F.save(fig, str(tmp_path / "sub" / "toy"))
    assert [os.path.splitext(p)[1] for p in paths] == [".pdf", ".png"]
    assert all(os.path.getsize(p) > 0 for p in paths)


def test_reader_colours_distinct():
    cols = [F.reader_colour(r) for r in F.READERS]
    assert len(set(cols)) == len(cols)
