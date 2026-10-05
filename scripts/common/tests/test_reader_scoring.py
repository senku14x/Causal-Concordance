"""CPU tests for design 006's continuation scoring (readers.cont_batch, readers.seq_logprob_from_hidden)."""
import torch

import readers as RD


def test_cont_batch_rows_and_positions():
    rows, mask, gather = RD.cont_batch([5, 6, 7], [[1], [2, 3]], pad_id=0)
    assert rows == [[5, 6, 7, 1, 0], [5, 6, 7, 2, 3]]
    assert mask == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    assert gather == [[(2, 1)], [(2, 2), (3, 3)]]


def test_seq_logprob_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    Hn = torch.randn(2, 5, 8, generator=g)
    W = torch.randn(11, 8, generator=g)
    gather = [[(2, 1)], [(2, 4), (3, 7)]]
    got = RD.seq_logprob_from_hidden(Hn, W, gather)
    lp = torch.log_softmax(Hn @ W.T, -1)
    want = torch.stack([lp[0, 2, 1], lp[1, 2, 4] + lp[1, 3, 7]])
    assert torch.allclose(got, want, atol=1e-6)


def test_padded_batch_equals_unpadded_under_causal_model():
    """A toy causal model (hidden state at t = running mean of embeddings up to t): scoring candidates in one
    right-padded batch must equal scoring each candidate alone, unpadded."""
    g = torch.Generator().manual_seed(1)
    E = torch.randn(20, 6, generator=g)
    W = torch.randn(20, 6, generator=g)

    def hidden(rows):
        x = E[torch.tensor(rows)]                                  # [R, T, d]
        return torch.cumsum(x, 1) / torch.arange(1, x.shape[1] + 1)[None, :, None]
    prefix, cands = [3, 4, 5, 6], [[7], [8, 9], [10, 11, 12]]
    rows, mask, gather = RD.cont_batch(prefix, cands, pad_id=0)
    batched = RD.seq_logprob_from_hidden(hidden(rows), W, gather)
    alone = []
    for c in cands:
        r1, _, g1 = RD.cont_batch(prefix, [c], pad_id=0)
        alone.append(RD.seq_logprob_from_hidden(hidden(r1), W, g1)[0])
    assert torch.allclose(batched, torch.stack(alone), atol=1e-6)
    # and against a direct per-token computation
    h = hidden([prefix + cands[2]])[0]
    lp = torch.log_softmax(h @ W.T, -1)
    direct = lp[3, 10] + lp[4, 11] + lp[5, 12]
    assert torch.allclose(batched[2], direct, atol=1e-6)
