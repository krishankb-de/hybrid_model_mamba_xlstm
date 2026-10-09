"""CHAT_UI_PLAN.md P9-G2: the EOS-stop beam searches (tiny model and scripted logits, CPU, synthetic only).

`beam_search_decode_eos` (scripts/evaluate_report_generation.py) and `HybridLanguageModel.beam_search_cached_eos` sit beside the published
decoders, which stay as they are. The rulings they implement, at every decoding step:
  1. rank all beam x vocab candidates exactly as the published code does and take the top 2 * beam;
  2. walk them in rank order: an EOS candidate within the first `beam` goes to the finished pool (its score is normalised by the length
     with the EOS counted); the others fill the `beam` live slots; an EOS further down is dropped;
  3. stop when `beam` hypotheses have finished, or at the budget;
  4. return the best of finished + live by normalised score (a tie goes to the finished one, then to the live beams in rank order),
     without its trailing EOS, and whether that was a finished one.

The hard gate is the first block: with no EOS in reach each twin returns the published decoder's tokens. Everything after it needs an
EOS that does occur, which no random-init model produces on demand, so the logits are scripted (tests/app_helpers.py) and a slow
oracle written from the rulings (every candidate, a list sort, no cache) is the expected answer.
"""
import argparse
import math
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from app.tiny import TINY_PREFIX_K, TinyTokenizer, TinyTower, tiny_decoder, tiny_prefix_mapper
from tests.app_helpers import TINY_VOCAB_SIZE, markov_script, noise_script, png_bytes, scripted_decoder

EOS = TINY_VOCAB_SIZE - 1                   # the last id of the tiny vocab; the default, 50256, is out of it
EMPTY = torch.zeros((1, 0), dtype=torch.long)
KINDS = ["uncached", "cached"]


def _prefix(dim=64, k=TINY_PREFIX_K):
    return torch.randn(1, k, dim, generator=torch.Generator().manual_seed(1))


def _published(kind, model, ids, prefix, beam, budget, on_step=None, **kw):
    from scripts.evaluate_report_generation import beam_search_decode
    if kind == "cached":
        return model.beam_search_cached(ids, prefix_embeds=prefix, beam_size=beam, max_new_tokens=budget, on_step=on_step, **kw)
    return beam_search_decode(model, ids, prefix_embeds=prefix, beam_size=beam, max_new_tokens=budget, on_step=on_step, **kw)


def _eos(kind, model, ids, prefix, beam, budget, on_step=None, eos=50256, **kw):
    from scripts.evaluate_report_generation import beam_search_decode_eos
    if kind == "cached":
        return model.beam_search_cached_eos(ids, prefix_embeds=prefix, beam_size=beam, max_new_tokens=budget,
                                            eos_token_id=eos, on_step=on_step, **kw)
    return beam_search_decode_eos(model, ids, prefix_embeds=prefix, beam_size=beam, max_new_tokens=budget,
                                  eos_token_id=eos, on_step=on_step, **kw)


def _forbid(model, token):
    """Within the block the model gives `token` the logit -inf: it is a real id of the vocab, and no beam can ever pick it."""
    forward, prefill, step_logits = model.forward, model.prefill, model.step_logits

    def masked_forward(*a, **k):
        out = forward(*a, **k)
        out.logits[..., token] = float("-inf")
        return out

    def masked_prefill(*a, **k):
        logits = prefill(*a, **k)
        logits[..., token] = float("-inf")
        return logits

    def masked_step(*a, **k):
        logits = step_logits(*a, **k)
        logits[..., token] = float("-inf")
        return logits

    stack = ExitStack()
    for name, fn in (("forward", masked_forward), ("prefill", masked_prefill), ("step_logits", masked_step)):
        stack.enter_context(mock.patch.object(model, name, fn))
    return stack


def _oracle(script, eos, beam, budget, length_penalty=1.0):
    """The rulings written out the slow way over a script: every candidate, a stable list sort, no cache. -> (ids, ended_by_eos)."""
    vocab = len(script(0, None))
    live = [((), 0.0)]
    finished = []
    for _ in range(budget):
        length = len(live[0][0]) + 1
        candidates = []
        for tokens, score in live:
            log_probs = torch.log_softmax(script(len(tokens), tokens[-1] if tokens else None).float(), dim=-1)
            candidates += [(tokens + (t,), score + float(log_probs[t])) for t in range(vocab)]
        candidates.sort(key=lambda c: c[1] / (length ** length_penalty), reverse=True)
        kept = []
        for rank, (tokens, score) in enumerate(candidates[:2 * beam]):
            if tokens[-1] == eos:
                if rank < beam:
                    finished.append((score / (length ** length_penalty), tokens[:-1]))
                continue
            kept.append((tokens, score))
            if len(kept) == beam:
                break
        live = kept
        if len(finished) >= beam:
            break
    pool = finished + [(score / (len(tokens) ** length_penalty), tokens) for tokens, score in live]
    best = max(range(len(pool)), key=lambda i: pool[i][0])    # the first maximum: finished in order, then live in rank order
    return list(pool[best][1]), best < len(finished)


# ---------------------------------------------------------------------------------------------------------
# The hard gate: with no EOS in reach, each twin is the published decoder, token for token
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("beam", [1, 3])
def test_with_no_eos_in_reach_each_twin_is_the_published_decoder_token_for_token(kind, beam):
    model = tiny_decoder()
    want = _published(kind, model, EMPTY, _prefix(), beam, 24)
    got, ended = _eos(kind, model, EMPTY, _prefix(), beam, 24)       # the default id, 50256, is no token of the tiny vocab
    assert torch.equal(got, want) and got.dtype == want.dtype and got.shape == (1, 24)
    assert ended is False                                             # a plain bool, not a tensor


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("beam", [1, 3])
def test_an_eos_id_of_the_vocab_that_the_model_cannot_pick_changes_nothing(kind, beam):
    model = tiny_decoder()
    liked = _published(kind, model, EMPTY, _prefix(), beam, 12)[0, 3].item()   # a token this model does use at step 3
    with _forbid(model, liked):
        want = _published(kind, model, EMPTY, _prefix(), beam, 24)
        got, ended = _eos(kind, model, EMPTY, _prefix(), beam, 24, eos=liked)
    assert liked in _published(kind, model, EMPTY, _prefix(), beam, 24)[0].tolist()   # unforbidden, the published run does use it
    assert torch.equal(got, want) and ended is False


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("beam", [1, 3])
@pytest.mark.parametrize("with_prefix", [False, True])
@pytest.mark.parametrize("length_penalty", [1.0, 0.0, 2.0])
def test_parity_holds_for_a_prompt_with_and_without_a_prefix_and_for_any_length_penalty(kind, beam, with_prefix, length_penalty):
    from tests.test_mamba3_numerics import _cached_lm    # the model the cached == uncached tests of M6-D use
    with torch.random.fork_rng(devices=[]):
        model = _cached_lm()
        ids = torch.randint(0, 97, (1, 6))
        prefix = torch.randn(1, 3, 64) if with_prefix else None
    want = _published(kind, model, ids, prefix, beam, 10, length_penalty=length_penalty)
    got, ended = _eos(kind, model, ids, prefix, beam, 10, length_penalty=length_penalty)
    assert torch.equal(got, want) and got.shape == (1, 16) and ended is False


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("beam", [1, 3])
def test_on_step_shows_what_the_published_decoder_shows_while_no_beam_has_ended(kind, beam):
    model = tiny_decoder()
    published, twin = [], []
    _published(kind, model, EMPTY, _prefix(), beam, 16, on_step=lambda step, ids: published.append((step, ids)))
    _eos(kind, model, EMPTY, _prefix(), beam, 16, on_step=lambda step, ids: twin.append((step, ids)))
    assert twin == published and [s for s, _ in twin] == list(range(16))    # once per step, the same ids


@pytest.mark.parametrize("kind", KINDS)
def test_each_twin_decodes_one_sample_at_a_time(kind):
    with pytest.raises(ValueError, match="one sample at a time"):
        _eos(kind, tiny_decoder(), torch.zeros((2, 0), dtype=torch.long), _prefix(), 3, 4)


def test_the_cached_twin_refuses_a_stack_without_a_step_path():
    from tests.test_mamba3_numerics import _cached_lm
    with torch.random.fork_rng(devices=[]):
        model = _cached_lm(layer_pattern=["mamba3", "slstm"], slstm_hidden_dim=64)
    assert not model.supports_cached_decode()
    with pytest.raises(NotImplementedError, match="step"):
        model.beam_search_cached_eos(EMPTY.clone(), prefix_embeds=_prefix(), beam_size=3, max_new_tokens=4)


# ---------------------------------------------------------------------------------------------------------
# A model that ends the report: the search stops there
# ---------------------------------------------------------------------------------------------------------

N = 7    # tokens a scripted report has before it ends


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("beam", [1, 2, 3])
def test_a_model_that_ends_the_report_at_step_n_stops_there_and_returns_it_without_its_eos(kind, beam):
    script = noise_script(5, EOS, lambda n, last: 40.0 if n == N else -1e4)   # after N tokens every beam ends, before that none can
    model, seen, published_seen = tiny_decoder(), [], []
    with scripted_decoder(model, script):
        got, ended = _eos(kind, model, EMPTY, _prefix(), beam, 30, on_step=lambda step, ids: seen.append((step, ids)), eos=EOS)
        want = _published(kind, model, EMPTY, _prefix(), beam, N,           # the published search, cut where the report ends
                          on_step=lambda step, ids: published_seen.append((step, ids)))
    assert ended is True and type(ended) is bool
    assert torch.equal(got, want) and got.shape == (1, N) and EOS not in got[0].tolist()
    assert [step for step, _ in seen] == list(range(N + 1))                 # it stopped at step N: one call per step, that one's too
    assert all(EOS not in ids for _, ids in seen)                           # a live beam never holds an EOS
    assert seen[:N] == published_seen                                       # before step N, what the published decoder shows
    assert len(seen[N][1]) == N + 1                                         # at it, the best live beam: a token past the end


@pytest.mark.parametrize("kind", KINDS)
def test_a_prompt_stays_in_the_returned_ids_and_the_eos_does_not(kind):
    prompt = torch.tensor([[5, 6, 7]])
    script = noise_script(8, EOS, lambda n, last: 40.0 if n == 4 else -1e4)
    model = tiny_decoder()
    with scripted_decoder(model, script, prefix_len=TINY_PREFIX_K + 3):
        got, ended = _eos(kind, model, prompt, _prefix(), 3, 30, eos=EOS)
        want = _published(kind, model, prompt, _prefix(), 3, 4)
    assert ended is True and torch.equal(got, want) and got.shape == (1, 3 + 4)
    assert got[0, :3].tolist() == [5, 6, 7]


@pytest.mark.parametrize("kind", KINDS)
def test_a_report_that_is_ended_by_its_very_first_token_is_empty(kind):
    script = noise_script(2, EOS, lambda n, last: 40.0 if n == 0 else -1e4)
    model = tiny_decoder()
    with scripted_decoder(model, script):
        got, ended = _eos(kind, model, EMPTY, _prefix(), 1, 30, eos=EOS)
    assert ended is True and got.shape == (1, 0) and got.dtype == torch.long


def _greedy_rows(n, last):
    """Whatever came before: token 0 (.6), the end (.25), token 1 (.15)."""
    logits = torch.full((TINY_VOCAB_SIZE,), -1e4)
    logits[0], logits[EOS], logits[1] = math.log(.6), math.log(.25), math.log(.15)
    return logits


@pytest.mark.parametrize("kind", KINDS)
def test_an_eos_that_ranks_second_with_a_beam_of_one_is_dropped_not_taken(kind):
    # beam 1: the EOS is in the top 2 * beam but not within the first `beam`, so it ends nothing and joins no beam.
    model = tiny_decoder()
    with scripted_decoder(model, _greedy_rows):
        got, ended = _eos(kind, model, EMPTY, _prefix(), 1, 9, eos=EOS)
        want = _published(kind, model, EMPTY, _prefix(), 1, 9)
    assert ended is False and torch.equal(got, want) and got[0].tolist() == [0] * 9


# ---- which hypothesis is returned: finished against live, by normalised score ------------------------------------

def _poor(n, last):
    """Twenty near-equally likely tokens (p of about 0.05 each) and no end: each token costs about 3 nats."""
    logits = torch.full((TINY_VOCAB_SIZE,), -1e4)
    logits[:20] = torch.randn(20, generator=torch.Generator().manual_seed(11 * n + (0 if last is None else last + 1))) * 0.3
    return logits


def _strong(n, last):
    """Token 4 with p = .99 (a hundredth of a nat), never the end."""
    logits = torch.full((TINY_VOCAB_SIZE,), -1e4)
    logits[4], logits[5] = math.log(.99), math.log(.01)
    return logits


def _ends_early_then(later):
    """Beam 2. After token 0 (p .6) the report may end (p .5); after token 1 (p .4) it cannot. From the third token on, `later`."""
    table = {
        (0, None): {0: .6, 1: .4},
        (1, 0): {EOS: .5, 2: .3, 3: .2},
        (1, 1): {2: .52, 3: .48},
    }
    return markov_script(table, default=later)


@pytest.mark.parametrize("kind", KINDS)
def test_a_finished_hypothesis_beats_a_longer_live_one_when_its_normalised_score_is_higher(kind):
    # finished: [0] + EOS, ln(.6 * .5) / 2 = -0.60; the live beams are 8 tokens long and cost about 3 nats a token: about -2.4
    script, seen = _ends_early_then(_poor), []
    model = tiny_decoder()
    with scripted_decoder(model, script):
        got, ended = _eos(kind, model, EMPTY, _prefix(), 2, 8, on_step=lambda step, ids: seen.append(step), eos=EOS)
    assert (got[0].tolist(), ended) == ([0], True)
    assert seen == list(range(8))        # one hypothesis ended and two were needed: the search ran to its budget
    assert (got[0].tolist(), ended) == _oracle(script, EOS, 2, 8)


@pytest.mark.parametrize("kind", KINDS)
def test_a_live_beam_beats_a_finished_hypothesis_when_its_normalised_score_is_higher(kind):
    # the same finished hypothesis (-0.60), against a live beam that costs a hundredth of a nat a token: [1, 2, 4, 4, ...], about -0.20
    script = _ends_early_then(_strong)
    model = tiny_decoder()
    with scripted_decoder(model, script):
        got, ended = _eos(kind, model, EMPTY, _prefix(), 2, 8, eos=EOS)
    assert (got[0].tolist(), ended) == ([1, 2, 4, 4, 4, 4, 4, 4], False)
    assert (got[0].tolist(), ended) == _oracle(script, EOS, 2, 8)


@pytest.mark.parametrize("kind", KINDS)
def test_when_beam_hypotheses_have_ended_a_live_beam_that_still_scores_higher_is_returned(kind):
    # "the best of finished + live" holds when the pool is full too: two hypotheses end by step 2 (-0.65 and -0.74) and the search
    # stops there, but the best live beam, [0, 0, 0], scores -0.21. It is returned, and it did not end in EOS.
    table = {
        (0, None): {0: .55, 1: .45},
        (1, 0): {0: .99, 7: .01}, (1, 1): {EOS: .6, 1: .4},
        (2, 0): {0: .99, 8: .01}, (2, 1): {EOS: .6, 1: .4},
    }
    script, seen = markov_script(table, default=_poor), []
    model = tiny_decoder()
    with scripted_decoder(model, script):
        got, ended = _eos(kind, model, EMPTY, _prefix(), 2, 20, on_step=lambda step, ids: seen.append(step), eos=EOS)
    assert (got[0].tolist(), ended) == ([0, 0, 0], False)
    assert seen == [0, 1, 2]             # stopped at step 2, long before its budget of 20
    assert (got[0].tolist(), ended) == _oracle(script, EOS, 2, 20)


def _tie_rows(n, last):
    """Two equally likely first tokens (0, 1); after 0 the report ends for certain, after 1 it goes on to token 5 for certain.
    Every other token is impossible (-inf). Both two-token candidates then score exactly -ln 2."""
    logits = torch.full((TINY_VOCAB_SIZE,), float("-inf"))
    if n == 0:
        logits[0] = logits[1] = 0.0
    elif last == 0:
        logits[EOS] = 0.0
    else:
        logits[5] = 0.0
    return logits


@pytest.mark.parametrize("kind", KINDS)
def test_a_tie_between_a_finished_hypothesis_and_a_live_one_goes_to_the_finished_one(kind):
    # [0] + EOS and [1, 5] both have the normalised score -ln 2 / 2 to the last bit, at the end of a budget of two.
    model = tiny_decoder()
    with scripted_decoder(model, _tie_rows):
        got, ended = _eos(kind, model, EMPTY, _prefix(), 2, 2, eos=EOS)
    assert (got[0].tolist(), ended) == ([0], True)      # not [1, 5]: finished first, then live beams in rank order


# ---------------------------------------------------------------------------------------------------------
# Cached == uncached == the slow oracle, with EOS in play
# ---------------------------------------------------------------------------------------------------------

FAMILIES = {   # the EOS logit, against N(0, 2^2) noise on the other 96 tokens
    "plausible": lambda n, last: 4.0,                  # the EOS is among the likeliest tokens at every step: reports end early or never
    "creeping": lambda n, last: -3.0 + 0.4 * n,        # it starts below the noise and climbs 0.4 a token: the end comes late, or not at all
}
GRID = [(family, beam, seed) for family in FAMILIES for beam in (1, 2, 3, 4) for seed in range(6)]
BUDGET = 20


@pytest.mark.parametrize("family,beam,seed", GRID)
def test_the_twins_and_the_oracle_agree_when_the_eos_competes(family, beam, seed):
    script = noise_script(seed, EOS, FAMILIES[family])
    model, got = tiny_decoder(), {}
    with scripted_decoder(model, script):
        for kind in KINDS:
            ids, ended = _eos(kind, model, EMPTY, _prefix(), beam, BUDGET, eos=EOS)
            got[kind] = (ids[0].tolist(), ended)
    want = _oracle(script, EOS, beam, BUDGET)
    assert got["uncached"] == want, "uncached"
    assert got["cached"] == want, "cached"


def test_that_grid_ends_every_way_a_search_can_end():
    outcomes = [_oracle(noise_script(seed, EOS, FAMILIES[family]), EOS, beam, BUDGET) for family, beam, seed in GRID]
    assert any(ended and len(ids) < BUDGET for ids, ended in outcomes)       # a report that ended in EOS, before the budget
    assert any(not ended and len(ids) == BUDGET for ids, ended in outcomes)  # one cut at the budget
    assert any(not ended and len(ids) < BUDGET for ids, ended in outcomes)   # beam hypotheses ended, yet a live beam still scored higher


@pytest.mark.parametrize("length_penalty", [0.0, 2.0])
@pytest.mark.parametrize("beam", [2, 3])
def test_the_length_penalty_scores_finished_and_live_alike(beam, length_penalty):
    script = noise_script(3, EOS, FAMILIES["plausible"])
    model = tiny_decoder()
    with scripted_decoder(model, script):
        for kind in KINDS:
            ids, ended = _eos(kind, model, EMPTY, _prefix(), beam, BUDGET, eos=EOS, length_penalty=length_penalty)
            assert (ids[0].tolist(), ended) == _oracle(script, EOS, beam, BUDGET, length_penalty), kind


@pytest.mark.parametrize("beam", [1, 3])
def test_on_the_real_tiny_weights_cached_and_uncached_agree_once_a_token_the_model_likes_is_the_eos(beam):
    model = tiny_decoder()
    published = _published("uncached", model, EMPTY, _prefix(), beam, 24)
    eos = published[0, 5].item()                                  # the published decode uses it at step 5: the model likes it
    unc = _eos("uncached", model, EMPTY, _prefix(), beam, 24, eos=eos)
    cac = _eos("cached", model, EMPTY, _prefix(), beam, 24, eos=eos)
    assert torch.equal(unc[0], cac[0]) and unc[1] == cac[1]
    assert eos not in unc[0][0].tolist()                          # a returned report never holds the EOS ...
    assert unc[1] is True and unc[0].shape[1] < 24                # ... and this one did end in it, before the budget


# ---------------------------------------------------------------------------------------------------------
# on_step
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_raising_from_on_step_stops_decoding_as_in_the_published_decoders(kind):
    class Stop(Exception):
        pass
    calls = []

    def cb(step, ids):
        calls.append(step)
        if step == 2:
            raise Stop()

    with pytest.raises(Stop):
        _eos(kind, tiny_decoder(), EMPTY, _prefix(), 3, 8, on_step=cb)
    assert calls == [0, 1, 2]


@pytest.mark.parametrize("kind", KINDS)
def test_a_raise_from_the_step_that_ends_the_report_is_not_swallowed(kind):
    class Stop(Exception):
        pass
    script, calls = noise_script(5, EOS, lambda n, last: 40.0 if n == 3 else -1e4), []

    def cb(step, ids):
        calls.append(step)
        if step == 3:                   # the step at which every beam ends
            raise Stop()

    model = tiny_decoder()
    with scripted_decoder(model, script):
        with pytest.raises(Stop):
            _eos(kind, model, EMPTY, _prefix(), 3, 30, on_step=cb, eos=EOS)
    assert calls == [0, 1, 2, 3]


# ---------------------------------------------------------------------------------------------------------
# scripts/evaluate_report_generation.py --stop-at-eos
# ---------------------------------------------------------------------------------------------------------

def _main(monkeypatch, *argv):
    import scripts.evaluate_report_generation as erg
    monkeypatch.setattr("sys.argv", ["evaluate_report_generation.py"] + list(argv))
    erg.main()


def test_stop_at_eos_is_off_unless_asked_and_reaches_the_inspection_run_when_it_is(monkeypatch):
    import scripts.evaluate_report_generation as erg
    seen = []
    monkeypatch.setattr(erg, "run_checkpoint_inspection", seen.append)
    base = ["--checkpoint", "c.ckpt", "--parquet", "p.parquet", "--decode", "beam"]
    _main(monkeypatch, *base)
    _main(monkeypatch, *base, "--stop-at-eos")
    assert [a.stop_at_eos for a in seen] == [False, True]


@pytest.mark.parametrize("decode", [[], ["--decode", "greedy"]])    # greedy is the script's default
def test_stop_at_eos_needs_beam_decoding(monkeypatch, capsys, decode):
    import scripts.evaluate_report_generation as erg
    monkeypatch.setattr(erg, "run_checkpoint_inspection", lambda args: pytest.fail("the run started"))
    with pytest.raises(SystemExit) as stop:
        _main(monkeypatch, "--checkpoint", "c.ckpt", "--parquet", "p.parquet", "--stop-at-eos", *decode)
    assert stop.value.code == 2 and "--stop-at-eos needs --decode beam" in capsys.readouterr().err


def _patch_grid_module():
    return SimpleNamespace(prefix_mapper=tiny_prefix_mapper(), decoder=tiny_decoder())


def _grid():
    return torch.randn(1, 197, 32, generator=torch.Generator().manual_seed(3))


@pytest.mark.parametrize("cached", [False, True])
def test_the_patch_grid_twin_is_the_published_function_while_no_eos_is_in_reach(cached):
    from scripts.evaluate_report_generation import generate_from_patch_grid, generate_from_patch_grid_eos
    module, grid = _patch_grid_module(), _grid()
    want = generate_from_patch_grid(module, grid, decode="beam", beam_size=3, max_new_tokens=16, cached=cached)
    got, ended = generate_from_patch_grid_eos(module, grid, beam_size=3, max_new_tokens=16, cached=cached)
    assert torch.equal(got, want) and ended is False


@pytest.mark.parametrize("cached", [False, True])
def test_the_patch_grid_twin_selects_the_eos_decoders_cached_or_not(cached):
    from scripts.evaluate_report_generation import generate_from_patch_grid_eos
    module, grid = _patch_grid_module(), _grid()
    script = noise_script(5, EOS, lambda n, last: 40.0 if n == N else -1e4)
    with scripted_decoder(module.decoder, script):
        got, ended = generate_from_patch_grid_eos(module, grid, beam_size=3, max_new_tokens=30, cached=cached, eos_token_id=EOS)
    assert ended is True and got.shape == (1, N)


def test_the_patch_grid_twin_refuses_the_cache_for_a_stack_that_has_none():
    from scripts.evaluate_report_generation import generate_from_patch_grid_eos
    module = _patch_grid_module()
    with mock.patch.object(module.decoder, "supports_cached_decode", return_value=False):
        with pytest.raises(RuntimeError, match="no O\\(1\\) step path"):
            generate_from_patch_grid_eos(module, _grid(), cached=True)


class _EosTokenizer(TinyTokenizer):
    eos_token_id = EOS


def _inspect(tmp_path, monkeypatch, capsys, script=None, **flags):
    """run_checkpoint_inspection over three synthetic images, the tiny decoder (scripted when `script` is given) and the tiny tower:
    -> (what it printed, the lines it dumped to hyps.txt)."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import transformers
    import scripts.evaluate_report_generation as erg

    tower, module = TinyTower(), _patch_grid_module()
    module._patch_grid = lambda pixel_values: tower.trunk.forward_features(pixel_values)
    images = []
    for i in range(3):
        path = tmp_path / "x{}.png".format(i)
        path.write_bytes(png_bytes(64 + 32 * i, 96))
        images.append(str(path))
    parquet = tmp_path / "validate.parquet"
    pd.DataFrame({"image": images, "study_id": [1, 2, 3], "findings": ["SYNTHETIC findings"] * 3,
                  "impression": ["SYNTHETIC impression"] * 3}).to_parquet(parquet)
    monkeypatch.setattr(erg, "load_report_generation_module", lambda checkpoint, model_config, device="cpu", **kw: module)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name, **kw: _EosTokenizer()))
    options = dict(checkpoint="c.ckpt", model_config="m", parquet=str(parquet), num_samples=3, decode="beam", beam_size=3,
                   max_new_tokens=30, chexbert=False, dump_dir=str(tmp_path / "dump"), prefix_k=None, scan_impl=None,
                   tfla_impl=None, chunk_size=None, compile_decoder=False, cached_decode=False)
    args = argparse.Namespace(**dict(options, **flags))
    capsys.readouterr()
    with scripted_decoder(module.decoder, script) if script else ExitStack():
        erg.run_checkpoint_inspection(args)
    out = capsys.readouterr().out
    return out, (tmp_path / "dump" / "hyps.txt").read_text().splitlines()


@pytest.mark.parametrize("cached", [False, True])
def test_the_inspection_run_stops_at_the_eos_with_the_flag_and_counts_the_reports_that_did(tmp_path, monkeypatch, capsys, cached):
    script = noise_script(5, EOS, lambda n, last: 40.0 if n == N else -1e4)
    on, hyps_on = _inspect(tmp_path, monkeypatch, capsys, script, stop_at_eos=True, cached_decode=cached)
    assert len(hyps_on) == 3 and all(len(h.split()) == N for h in hyps_on)         # N words each: the report, without its EOS
    assert "EOS stop: 3/3 reports ended at the end-of-report token" in on
    off, hyps_off = _inspect(tmp_path, monkeypatch, capsys, script, stop_at_eos=False, cached_decode=cached)
    assert all(len(h.split()) > N + 15 for h in hyps_off)                           # the published protocol runs on past the end
    assert "EOS stop" not in off and "stops at EOS" not in off                      # and says nothing of it


def test_the_inspection_run_without_the_flag_is_what_it_was(tmp_path, monkeypatch, capsys):
    from scripts.evaluate_report_generation import generate_from_patch_grid
    plain, hyps_plain = _inspect(tmp_path, monkeypatch, capsys)                       # the args carry no stop_at_eos at all
    flagged, hyps_flagged = _inspect(tmp_path, monkeypatch, capsys, stop_at_eos=False)
    assert hyps_plain == hyps_flagged and len(hyps_plain) == 3
    strip = lambda text: "\n".join(l for l in text.splitlines() if not l.startswith(("Loaded", "  Missing")))
    assert strip(plain) == strip(flagged)
    assert "EOS" not in plain
