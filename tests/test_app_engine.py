"""CHAT_UI_PLAN.md P2-B/P2-D: engine stages and the on_step callback (tiny model, CPU)."""
import pytest
import torch

from app.tiny import TINY_VOCAB, tiny_decoder

EMPTY = torch.zeros((1, 0), dtype=torch.long)   # report generation seeds with no BOS


def _prefix(dim=64, k=4):
    return torch.randn(1, k, dim, generator=torch.Generator().manual_seed(1))


def test_tiny_vocab_matches_the_m6d_config():
    assert len(TINY_VOCAB) == 97 and len(set(TINY_VOCAB)) == 97


def test_on_step_leaves_cached_beam_output_unchanged():
    model = tiny_decoder()
    plain = model.beam_search_cached(EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=8)
    seen = []
    hooked = model.beam_search_cached(EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=8,
                                      on_step=lambda step, ids: seen.append((step, ids)))
    assert torch.equal(plain, hooked)
    assert [s for s, _ in seen] == list(range(8))
    assert [len(ids) for _, ids in seen] == list(range(1, 9))
    assert seen[-1][1] == hooked[0].tolist()          # the last snapshot is the answer


def test_on_step_leaves_uncached_beam_output_unchanged():
    from scripts.evaluate_report_generation import beam_search_decode
    model = tiny_decoder()
    plain = beam_search_decode(model, EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=6)
    seen = []
    hooked = beam_search_decode(model, EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=6,
                                on_step=lambda step, ids: seen.append(ids))
    assert torch.equal(plain, hooked) and seen[-1] == hooked[0].tolist()


def test_exception_from_on_step_stops_decoding():   # the chat app's cancel mechanism
    class Stop(Exception):
        pass
    calls = []

    def cb(step, ids):
        calls.append(step)
        if step == 2:
            raise Stop()

    with pytest.raises(Stop):
        tiny_decoder().beam_search_cached(EMPTY, prefix_embeds=_prefix(), beam_size=3,
                                          max_new_tokens=8, on_step=cb)
    assert calls == [0, 1, 2]


def test_cached_beam_of_one_equals_the_published_greedy():   # D11
    from scripts.evaluate_report_generation import greedy_decode
    model = tiny_decoder()
    greedy = greedy_decode(model, EMPTY, prefix_embeds=_prefix(), max_new_tokens=8)
    beam1 = model.beam_search_cached(EMPTY, prefix_embeds=_prefix(), beam_size=1, max_new_tokens=8)
    assert torch.equal(greedy, beam1)
