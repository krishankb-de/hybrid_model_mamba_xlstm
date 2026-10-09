"""CHAT_UI_PLAN.md P2-B/P2-D: engine stages and the on_step callback (tiny model, CPU)."""
import dataclasses
import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pydantic import ValidationError

from app.engine import Cancelled, build_engine, file_sha256, git_provenance, repeat_started, stream_view, tensor_sha256
from app.schemas import Options
from app.tiny import (TINY_VOCAB, TinyTokenizer, TinyTower, tiny_decoder, tiny_decoder_config,
                      tiny_prefix_mapper)
from scripts.repair_generations import repair_report, split_sentences
from tests.app_helpers import noise_script, png_bytes, scripted_decoder

EMPTY = torch.zeros((1, 0), dtype=torch.long)   # report generation seeds with no BOS


def _prefix(dim=64, k=4):
    return torch.randn(1, k, dim, generator=torch.Generator().manual_seed(1))


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _m6d_model():
    """The model tests/test_mamba3_numerics.py proves cached == uncached on (its helper seeds the global RNG)."""
    from tests.test_mamba3_numerics import _cached_lm
    with torch.random.fork_rng(devices=[]):
        return _cached_lm()


def test_tiny_vocab_matches_the_m6d_config():
    m6d = _m6d_model().config
    tiny = tiny_decoder_config()
    assert len(TINY_VOCAB) == 97 and len(set(TINY_VOCAB)) == 97
    assert tiny.vocab_size == m6d.vocab_size == len(TINY_VOCAB)
    a, b = dataclasses.asdict(tiny), dataclasses.asdict(m6d)
    assert a.keys() == b.keys()
    differing = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    # only the (never built) position table and the attention blocks' RoPE tables read this field
    assert set(differing) <= {"max_position_embeddings"}, differing
    assert not tiny_decoder().embeddings.use_pos_embedding and "attention" not in tiny.layer_pattern


def test_tiny_decoder_weights_are_the_m6d_weights():
    assert _same_weights(tiny_decoder(), _m6d_model())


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


def test_uncached_on_step_reports_steps_0_to_n_minus_1_with_growing_snapshots():
    from scripts.evaluate_report_generation import beam_search_decode
    seen = []
    out = beam_search_decode(tiny_decoder(), EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=6,
                             on_step=lambda step, ids: seen.append((step, ids)))
    assert [s for s, _ in seen] == list(range(6))
    assert [len(ids) for _, ids in seen] == list(range(1, 7))
    assert seen[-1][1] == out[0].tolist()


def test_exception_from_on_step_stops_uncached_decoding():   # the 13D engine has no cached path to cancel on
    from scripts.evaluate_report_generation import beam_search_decode

    class Stop(Exception):
        pass
    calls = []

    def cb(step, ids):
        calls.append(step)
        if step == 2:
            raise Stop()

    with pytest.raises(Stop):
        beam_search_decode(tiny_decoder(), EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=8, on_step=cb)
    assert calls == [0, 1, 2]


def test_tiny_builders_leave_the_callers_rng_alone_and_are_deterministic():
    with torch.random.fork_rng(devices=[]):              # this test seeds the global RNG; the session must not see it
        torch.manual_seed(1234)
        seed, state = torch.initial_seed(), torch.get_rng_state()
        first = (tiny_decoder(), TinyTower(), tiny_prefix_mapper())
        assert torch.initial_seed() == seed and torch.equal(torch.get_rng_state(), state)
        assert all(_same_weights(a, b) for a, b in zip(first, (tiny_decoder(), TinyTower(), tiny_prefix_mapper())))


# ---------------------------------------------------------------------------------------------------------
# P2-D: the engine (tiny model, CPU)
# ---------------------------------------------------------------------------------------------------------

def _noop(step, text):
    pass


def _encoded(eng):
    _, prep = eng.preprocess(png_bytes(320, 320))
    _, enc = eng.encode(prep)
    return enc


def _white_png(side=320):
    buf = io.BytesIO()
    Image.new("L", (side, side), 255).save(buf, "PNG")
    return buf.getvalue()


def test_tiny_engine_runs_three_stages_and_streams_snapshots():
    eng = build_engine("tiny")
    res_p, prep = eng.preprocess(png_bytes(320, 320))
    assert prep.pixel_values.shape == (1, 3, 224, 224) and res_p.detail["resized_to"] == [224, 224]
    res_e, enc = eng.encode(prep)
    assert enc.prefix.shape == (1, 4, 64) and abs(float(enc.pooled.norm()) - 1.0) < 1e-5
    assert res_e.detail["one_pass"] is True
    snaps = []
    res_g, gen = eng.generate(enc, Options(max_new_tokens=16), lambda s, t: snaps.append((s, t)),
                              threading.Event())
    assert [s for s, _ in snaps] == list(range(16))
    assert snaps[-1][1] == gen.report and len(gen.token_ids) == 16
    assert res_g.detail["stopped"] == "budget" and res_g.detail["cached_decode"] is True
    assert gen.report == " ".join(gen.report.split())          # sanitised like the dumps


def test_cancel_stops_generation_within_one_step():
    eng = build_engine("tiny")
    enc = _encoded(eng)
    cancel, seen = threading.Event(), []

    def snap(step, text):
        seen.append(step)
        if step == 3:
            cancel.set()

    with pytest.raises(Cancelled):
        eng.generate(enc, Options(max_new_tokens=50), snap, cancel)
    assert seen == [0, 1, 2, 3]


def test_uncached_and_greedy_paths_equal_the_published_functions():
    from scripts.evaluate_report_generation import beam_search_decode, greedy_decode
    eng = build_engine("tiny")
    enc = _encoded(eng)
    _, unc = eng.generate(enc, Options(max_new_tokens=16, cached_decode=False), _noop, threading.Event())
    ref = beam_search_decode(eng.decoder, EMPTY, prefix_embeds=enc.prefix, beam_size=3, max_new_tokens=16)
    assert unc.token_ids == ref[0].tolist()
    _, grd = eng.generate(enc, Options(max_new_tokens=16, decode="greedy"), _noop, threading.Event())
    assert grd.token_ids == greedy_decode(eng.decoder, EMPTY, prefix_embeds=enc.prefix, max_new_tokens=16)[0].tolist()


def test_display_repair_only_changes_the_display_copy():
    eng = build_engine("tiny")
    enc = _encoded(eng)
    _, raw = eng.generate(enc, Options(max_new_tokens=16), _noop, threading.Event())
    _, rep = eng.generate(enc, Options(max_new_tokens=16, display_repair=True), _noop, threading.Event())
    assert raw.report == rep.report and raw.display_report == raw.report


def test_streamed_sha256_matches_hashlib_and_is_cached(tmp_path):
    blob = tmp_path / "w.bin"
    blob.write_bytes(b"x" * (3 * 1024 * 1024 + 7))
    cache = tmp_path / "sha256.json"
    assert file_sha256(blob, cache) == hashlib.sha256(blob.read_bytes()).hexdigest()
    assert str(blob.resolve()) in cache.read_text()


def test_one_tower_pass_gives_the_published_patch_grid_and_pooled_vector():
    open_clip = pytest.importorskip("open_clip")
    try:
        model, _ = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    except Exception as exc:   # no network and no local cache
        pytest.skip("BiomedCLIP unavailable: {}".format(exc))
    v = model.visual.eval()
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        feats = v.trunk.forward_features(x)                    # == ReportGenerationLightningModule._patch_grid
        assert feats.shape == (1, 197, 768)
        assert torch.equal(v.head(v.trunk.forward_head(feats)), v(x))


# ---- beyond the brief: what the engine promises its callers ---------------------------------------------

def test_options_are_the_published_protocol_by_default_and_bounded():
    o = Options()
    assert (o.decode, o.beam_size, o.max_new_tokens, o.cached_decode, o.compile) == ("beam", 3, 100, True, False)
    assert (o.k_images, o.k_reports, o.label, o.display_repair) == (4, 3, True, False)
    assert o.stop_on_repeat is False   # P4-G: the API's default is the published protocol, which decodes the whole budget
    assert o.model is None and o.reference is None and o.test_row is None
    for edge in ({"beam_size": 1}, {"beam_size": 8}, {"max_new_tokens": 16}, {"max_new_tokens": 200},
                 {"k_images": 0}, {"k_images": 12}, {"k_reports": 0}, {"k_reports": 10}, {"test_row": 0}):
        Options(**edge)
    for bad in ({"beam_size": 0}, {"beam_size": 9}, {"max_new_tokens": 15}, {"max_new_tokens": 201},
                {"k_images": 13}, {"k_images": -1}, {"k_reports": 11}, {"test_row": -1}, {"decode": "sample"},
                {"reference": "x" * 20001}, {"retrieval_k": 5}):   # D10: an unknown key is an error, not ignored
        with pytest.raises(ValidationError):
            Options(**bad)


def test_stage_details_carry_the_fields_the_events_promise():          # CHAT_UI_PLAN.md 6.3
    eng = build_engine("tiny")
    data = png_bytes(320, 240)
    res_p, prep = eng.preprocess(data)
    assert set(res_p.detail) == {"format", "mode", "input_px", "exif_transposed", "resized_to", "grayscale_to_3ch",
                                 "normalize", "image_sha256"}
    assert res_p.detail["image_sha256"] == prep.sha256 == hashlib.sha256(data).hexdigest()
    assert res_p.detail["input_px"] == [320, 240] and res_p.detail["format"] == "PNG"
    assert res_p.detail["grayscale_to_3ch"] is True and res_p.detail["normalize"] == "biomedclip_clip_mean_std"
    res_e, enc = eng.encode(prep)
    assert res_e.detail == {"patch_grid": [197, 32], "pooled_dim": 16, "prefix_tokens": 4, "device": "cpu",
                            "one_pass": True}
    res_g, _ = eng.generate(enc, Options(max_new_tokens=16), _noop, threading.Event())
    assert set(res_g.detail) == {"decode", "beam_size", "tokens", "stopped", "cached_decode", "compiled",
                                 "prefill_ms", "per_token_ms", "device", "threads", "drift_note"}
    assert (res_g.detail["decode"], res_g.detail["beam_size"], res_g.detail["tokens"]) == ("beam", 3, 16)
    assert res_g.detail["compiled"] is False and res_g.detail["drift_note"] == eng.drift_note
    assert res_g.detail["threads"] == torch.get_num_threads() and res_g.detail["device"] == "cpu"
    res_gr, _ = eng.generate(enc, Options(max_new_tokens=16, decode="greedy"), _noop, threading.Event())
    assert (res_gr.detail["decode"], res_gr.detail["beam_size"]) == ("greedy", 1)   # greedy is the beam of one
    assert all(isinstance(r.ms, float) and r.ms >= 0 for r in (res_p, res_e, res_g))
    json.dumps([res_p.detail, res_e.detail, res_g.detail])             # the events carry these verbatim


def test_preprocess_is_the_published_transform_and_refuses_what_load_upload_refuses():
    from app.imaging import UploadError, load_upload, model_transform
    eng = build_engine("tiny")
    data = png_bytes(320, 240)
    _, prep = eng.preprocess(data)
    img, facts = load_upload(data)
    assert prep.image.size == (320, 240) and prep.model_input.size == (224, 224)
    assert torch.equal(prep.pixel_values, model_transform()(img).unsqueeze(0))
    assert prep.facts == dict(facts, image_sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(UploadError):
        eng.preprocess(b"not an image")


def test_encode_runs_the_trunk_once_and_equals_the_tower_surface():
    eng = build_engine("tiny")
    _, prep = eng.preprocess(png_bytes(320, 320))
    trunk = eng.tower.trunk
    with mock.patch.object(trunk, "forward_features", wraps=trunk.forward_features) as feats_spy, \
            mock.patch.object(eng.tower, "forward", side_effect=AssertionError("a second tower pass")), \
            mock.patch.object(trunk, "forward", side_effect=AssertionError("a second trunk pass")):
        _, enc = eng.encode(prep)
    assert feats_spy.call_count == 1
    px = prep.pixel_values
    with torch.no_grad():
        grid = trunk.forward_features(px)
        assert torch.equal(enc.patch_grid, grid)
        assert torch.equal(enc.pooled, F.normalize(eng.tower(px).float(), dim=-1)[0])   # == the full tower, normalised
        assert torch.equal(enc.prefix, eng.prefix_mapper(grid))


def test_tiny_tower_responds_to_the_image():
    eng = build_engine("tiny")
    noise = _encoded(eng)
    _, prep = eng.preprocess(_white_png())
    _, white = eng.encode(prep)
    assert float(noise.pooled @ white.pooled) < 0.999     # a constant pooled vector would make every query identical
    assert not torch.allclose(noise.prefix, white.prefix)


def test_tensor_sha256_fingerprints_the_weights():
    a, b = TinyTower(), TinyTower()
    assert tensor_sha256(a) == tensor_sha256(b) and len(tensor_sha256(a)) == 64
    with torch.no_grad():
        b.head.bias[0] += 1.0
    assert tensor_sha256(a) != tensor_sha256(b)
    x, y = nn.Linear(2, 2), nn.Linear(2, 2)
    assert tensor_sha256(nn.ModuleDict({"a": x, "b": y})) == tensor_sha256(nn.ModuleDict({"b": y, "a": x}))
    assert build_engine("tiny").tower_sha256() == tensor_sha256(TinyTower())    # the seeded tower, not a fresh draw


def test_file_sha256_cache_is_consulted_and_a_changed_file_misses_it(tmp_path):
    blob, cache = tmp_path / "w.bin", tmp_path / "sub" / "sha256.json"
    blob.write_bytes(b"a" * 1000)
    first = file_sha256(blob, cache)
    assert first == hashlib.sha256(b"a" * 1000).hexdigest()
    (key,) = json.loads(cache.read_text())
    cache.write_text(json.dumps({key: "f" * 64}))
    assert file_sha256(blob, cache) == "f" * 64                         # a hit never re-reads the file
    blob.write_bytes(b"b" * 2000)                                       # new size: a different key, a fresh hash
    fresh = hashlib.sha256(b"b" * 2000).hexdigest()
    assert file_sha256(blob, cache) == fresh
    cache.write_text("{not json")                                       # a damaged cache is ignored, then rewritten
    assert file_sha256(blob, cache) == fresh and fresh in cache.read_text()
    assert file_sha256(blob) == fresh                                   # no cache file given: nothing is written


def test_build_engine_knows_two_kinds():
    assert build_engine("tiny").name == "tiny"
    with pytest.raises(ValueError, match="tiny"):
        build_engine("gpu")


def test_the_disclaimer_is_the_plans_copy():
    from app.engine import DISCLAIMER
    assert DISCLAIMER == "Research prototype; not for clinical use."


def test_two_tiny_engines_agree_and_neither_touches_the_callers_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        seed, state = torch.initial_seed(), torch.get_rng_state()
        reports = []
        for _ in range(2):
            eng = build_engine("tiny")
            enc = _encoded(eng)
            for opts in (Options(max_new_tokens=16), Options(max_new_tokens=16, cached_decode=False),
                         Options(max_new_tokens=16, decode="greedy")):
                eng.generate(enc, opts, _noop, threading.Event())
            reports.append(eng.generate(enc, Options(max_new_tokens=16), _noop, threading.Event())[1].token_ids)
        assert torch.initial_seed() == seed and torch.equal(torch.get_rng_state(), state)
        assert reports[0] == reports[1]


def test_cancel_stops_uncached_generation_within_one_step():           # the 13D engine has no other path
    eng = build_engine("tiny")
    enc = _encoded(eng)
    cancel, seen = threading.Event(), []

    def snap(step, text):
        seen.append(step)
        if step == 2:
            cancel.set()

    with pytest.raises(Cancelled):
        eng.generate(enc, Options(max_new_tokens=50, cached_decode=False), snap, cancel)
    assert seen == [0, 1, 2]


def test_a_cancel_set_before_generate_starts_does_no_decoding():
    eng = build_engine("tiny")
    enc = _encoded(eng)
    cancel = threading.Event()
    cancel.set()
    with mock.patch.object(eng.decoder, "beam_search_cached", side_effect=AssertionError("decoded")):
        with pytest.raises(Cancelled):
            eng.generate(enc, Options(), _noop, cancel)


def test_cached_decode_is_refused_for_a_stack_without_a_cache():
    eng = build_engine("tiny")
    enc = _encoded(eng)
    with mock.patch.object(eng.decoder, "supports_cached_decode", return_value=False):
        with pytest.raises(ValueError, match="cached_decode"):
            eng.generate(enc, Options(max_new_tokens=16), _noop, threading.Event())
        _, unc = eng.generate(enc, Options(max_new_tokens=16, cached_decode=False), _noop, threading.Event())
    assert len(unc.token_ids) == 16                                     # the uncached path still serves it


def test_uncached_greedy_equals_the_published_greedy():
    from scripts.evaluate_report_generation import greedy_decode
    eng = build_engine("tiny")
    enc = _encoded(eng)
    _, grd = eng.generate(enc, Options(max_new_tokens=16, decode="greedy", cached_decode=False), _noop,
                          threading.Event())
    assert grd.token_ids == greedy_decode(eng.decoder, EMPTY, prefix_embeds=enc.prefix, max_new_tokens=16)[0].tolist()


class _FixedText:
    """A tokenizer that always decodes to one string, to drive the display repair deterministically."""

    def __init__(self, text):
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return self.text


@pytest.mark.parametrize("text,truncated,display", [
    ("The heart is normal. The lungs are clear.", False, "The heart is normal. The lungs are clear."),
    ("The heart is normal.  The lungs\nare", True, "The heart is normal."),      # cut mid-sentence: the repair drops it
    ("The lungs are", True, "The lungs are"),     # no complete sentence at all: nothing to repair to, flag stays true
    ("", False, ""),
    # P4-F: the display copy drops every repeated sentence (dedup="all"), and the flag still says only "cut off"
    ("The heart is normal. The heart is normal. The lungs are clear.", False,
     "The heart is normal. The lungs are clear."),                                  # consecutive repeats
    ("The heart is normal. The lungs are clear. The heart is normal. The lungs are clear.", False,
     "The heart is normal. The lungs are clear."),                                  # an A B A B loop: consecutive dedup cannot see it
    ("The heart is normal. the HEART is   normal.\nThe lungs are clear.", False,
     "The heart is normal. The lungs are clear."),                                  # same key as the repair: case and whitespace
    ("Findings: The heart is normal. Findings: The heart is normal. Impression: Clear.", False,
     "Findings: The heart is normal. Impression: Clear."),                          # a repeat after a header
    ("heart size Findings: no effusion. heart size Findings: no effusion. heart size Findings: no effusion.", False,
     "heart size Findings: no effusion."),                                          # the tiny model's own loop: a header inside the sentence
    ("The heart is normal. The heart is normal. The lungs", True, "The heart is normal."),   # repeats and a cut-off fragment
    ("The heart is normal. The lungs are clear. The heart is normal. The lungs", True,
     "The heart is normal. The lungs are clear."),                                  # ... whose fragment starts the next repeat
    ("The heart is normal. The heart is normal.", False, "The heart is normal."),   # repeats only: nothing cut off
])
def test_truncated_flag_and_display_repair_follow_the_text(text, truncated, display):
    eng = build_engine("tiny")
    eng.tokenizer = _FixedText(text)
    enc = _encoded(eng)
    _, plain = eng.generate(enc, Options(max_new_tokens=16), _noop, threading.Event())
    _, rep = eng.generate(enc, Options(max_new_tokens=16, display_repair=True), _noop, threading.Event())
    assert plain.report == rep.report == " ".join(text.split())
    assert plain.display_report == plain.report and rep.display_report == display
    assert plain.truncated_mid_sentence is truncated and rep.truncated_mid_sentence is truncated


def test_the_display_copy_is_the_published_repair_with_every_repeat_dropped_and_the_raw_report_is_untouched():
    text = "Findings: The heart is normal. Impression: The lungs are clear. Findings: The heart is normal. Impression: The lungs"
    eng = build_engine("tiny")
    eng.tokenizer = _FixedText(text)
    _, rep = eng.generate(_encoded(eng), Options(max_new_tokens=16, display_repair=True), _noop, threading.Event())
    assert rep.report == text                                                       # R2: the decoder's text, byte for byte
    assert rep.display_report == repair_report(text, dedup="all", truncate=True)[0]
    assert rep.display_report == "Findings: The heart is normal. Impression: The lungs are clear."


# ---- P4-F: the stream view, what the live card shows of a snapshot ----------------------------------------------------------

def _repeats(text):
    """How many of the text's sentences repeat an earlier one (the repair's key: case-insensitive, whitespace-normalised)."""
    keys = [s.lower() for s in split_sentences(text)]
    return len(keys) - len(set(keys))


def test_stream_view_keeps_a_text_with_no_repeats_whole_including_its_fragment():
    for text in ("The heart is normal. The lungs are clear. No pleural",
                 "Findings: The heart is normal. Impression: Clear.",
                 "The heart", "The heart is normal.", "Findings:"):
        assert stream_view(text) == text, text
    assert stream_view("") == "" and stream_view("  \n ") == ""
    assert stream_view("The heart is   normal.\nThe lungs") == "The heart is normal. The lungs"   # whitespace normalised, as the dumps are


def test_stream_view_drops_a_sentence_that_repeats_one_already_shown():
    assert stream_view("The heart is normal. The lungs are clear. The heart is normal.") == "The heart is normal. The lungs are clear."
    assert stream_view("The heart is normal. the HEART is   normal.") == "The heart is normal."      # the repair's key
    assert stream_view("The heart is normal. The heart is normal. The heart is normal. The lungs are clear.") == (
        "The heart is normal. The lungs are clear.")
    loop = "The heart is normal. The lungs are clear. The heart is normal. The lungs are clear."       # period 2: A B A B
    assert stream_view(loop) == "The heart is normal. The lungs are clear."
    assert stream_view("Findings: The heart is normal. Findings: The heart is normal.") == "Findings: The heart is normal."


def test_stream_view_drops_a_growing_repeat_so_that_a_repeat_never_flickers_into_view():
    shown = "The heart is normal. The lungs are clear."
    repeat = "The heart is normal."
    for n in range(1, len(repeat) + 1):         # every prefix of the repeat, mid-word ones included (a token can end inside a word)
        assert stream_view(shown + " " + repeat[:n]) == shown, repeat[:n]
    header = "Findings: The heart is normal."
    for n in range(1, len(header) + 1):
        assert stream_view(header + " " + header[:n]) == header, header[:n]


def test_stream_view_keeps_a_new_fragment_and_shows_a_repeat_that_turns_out_to_differ():
    shown = "The heart is normal."
    assert stream_view(shown + " The lungs") == shown + " The lungs"           # not a start of a repeat: it grows token by token
    assert stream_view(shown + " The heart is") == shown                         # still a start of one
    assert stream_view(shown + " The heart is mildly") == shown + " The heart is mildly"   # it differs now: shown at once
    assert stream_view(shown + " The heart is normal and") == shown + " The heart is normal and"   # longer than the sentence it began as
    assert stream_view("Findings: The heart is normal. Findings: The lungs") == "Findings: The heart is normal. Findings: The lungs"


def test_stream_view_agrees_with_the_display_copy_once_the_report_is_whole():
    for text in ("The heart is normal. The lungs are clear. The heart is normal. The lungs are clear.",
                 "Findings: no effusion. Findings: no effusion. Impression: Clear. Findings: no effusion.",
                 "The heart is normal. The heart is normal. The heart is normal."):
        assert stream_view(text) == repair_report(text, dedup="all", truncate=True)[0], text
        assert stream_view(stream_view(text)) == stream_view(text)               # nothing left to drop


def _views(text):
    words = text.split()
    return [stream_view(" ".join(words[:n])) for n in range(1, len(words) + 1)]


def test_stream_view_only_ever_grows_by_appending_as_a_repeating_report_is_written():
    text = ("and heart and heart without. catheter acute right large heart without. catheter acute right large heart without. "
            "catheter acute stable is bilateral Findings: endotracheal there clear and heart without. "
            "catheter acute stable is bilateral Findings: endotracheal there clear and heart without. "
            "catheter acute stable is bilateral Findings: endotracheal there clear and heart")
    views = _views(text)
    assert all(_repeats(v) == 0 for v in views)                                 # not one frame of it repeats
    assert all(b.startswith(a) for a, b in zip(views, views[1:]))               # and a frame never takes back what an earlier one said
    assert views[-1] == ("and heart and heart without. catheter acute right large heart without. "
                         "catheter acute stable is bilateral Findings: endotracheal there clear and heart without.")


def test_stream_view_does_not_flash_a_list_marker_at_the_start_of_a_repeat():
    # "1." is not the end of a sentence (a list marker), so it is the start of the next one: held back like any other start of
    # a repeat. Taken as a finished sentence of its own it would show for one step and then vanish.
    text = "1. Heart size is normal. 2. Lungs are clear. 1. Heart size is normal. 2. Lungs are clear. 1. Heart"
    views = _views(text)
    assert all(b.startswith(a) for a, b in zip(views, views[1:])) and all(_repeats(v) == 0 for v in views)
    assert views[-1] == "1. Heart size is normal. 2. Lungs are clear."
    assert stream_view("Seen by Dr. Smith. Seen by Dr.") == "Seen by Dr. Smith."   # an abbreviation does not end a sentence either


class _GrowingText:
    """A tokenizer for a report written one word per step: n ids decode to the first n words of one text."""

    def __init__(self, text):
        self.words = text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[:len(ids)])


def test_the_stream_is_the_stream_view_with_repair_on_and_the_raw_snapshot_without_it():
    text = ("The heart is normal. The lungs are clear. The heart is normal. The lungs are clear. "
            "Findings: no effusion. Findings: no effusion. Findings: no")
    words = text.split()
    assert len(words) >= 16                                                      # Options: at least 16 tokens
    eng = build_engine("tiny")
    eng.tokenizer = _GrowingText(text)
    enc = _encoded(eng)
    raw, shown = [], []
    _, plain = eng.generate(enc, Options(max_new_tokens=len(words)), lambda step, t: raw.append(t), threading.Event())
    _, rep = eng.generate(enc, Options(max_new_tokens=len(words), display_repair=True), lambda step, t: shown.append(t),
                          threading.Event())
    assert raw == [" ".join(words[:n]) for n in range(1, len(words) + 1)]       # repair off: the snapshots are exactly as before
    assert shown == [stream_view(t) for t in raw]                                # repair on: each is the stream view of the raw one
    assert any(a != b for a, b in zip(raw, shown)) and all(_repeats(t) == 0 for t in shown)
    assert all(b.startswith(a) for a, b in zip(shown, shown[1:]))                # it grows by appending, never flickers
    assert plain.report == rep.report == text                                    # R2: the same tokens, the same text
    assert plain.display_report == plain.report
    assert rep.display_report == repair_report(text, dedup="all", truncate=True)[0] == (
        "The heart is normal. The lungs are clear. Findings: no effusion.")
    assert rep.truncated_mid_sentence is True and plain.truncated_mid_sentence is True


def test_the_users_case_200_tokens_with_repair_on_shows_no_repeated_sentence_anywhere():
    eng = build_engine("tiny")
    enc = _encoded(eng)
    raw_snaps, shown = [], []
    _, plain = eng.generate(enc, Options(max_new_tokens=200), lambda step, t: raw_snaps.append(t), threading.Event())
    _, rep = eng.generate(enc, Options(max_new_tokens=200, display_repair=True), lambda step, t: shown.append(t),
                          threading.Event())
    assert len(plain.token_ids) == len(rep.token_ids) == 200                     # decoding still runs the whole budget
    assert plain.token_ids == rep.token_ids and plain.report == rep.report       # R2: byte-identical to the published decoder
    assert _repeats(plain.report) >= 5 and max(_repeats(t) for t in raw_snaps) >= 5   # the user's case: it does repeat
    assert _repeats(rep.display_report) == 0                                     # the settled card
    assert len(shown) == 200 and all(_repeats(t) == 0 for t in shown)            # and not one snapshot of the stream
    assert 0 < len(rep.display_report.split()) < len(plain.report.split()) / 2   # a clean report, not an empty one
    assert plain.display_report == plain.report                                  # repair off changes nothing


def test_tiny_step_delay_paces_the_stream_and_a_cancel_cuts_the_pause_short():
    enc = _encoded(build_engine("tiny"))
    t0 = time.perf_counter()
    build_engine("tiny", step_delay_s=0.05).generate(enc, Options(max_new_tokens=16), _noop, threading.Event())
    assert time.perf_counter() - t0 >= 16 * 0.05 * 0.9
    slow, cancel, seen = build_engine("tiny", step_delay_s=30.0), threading.Event(), []

    def snap(step, text):
        seen.append(step)
        threading.Timer(0.05, cancel.set).start()                       # the user presses stop during the pause

    t0 = time.perf_counter()
    with pytest.raises(Cancelled):
        slow.generate(enc, Options(max_new_tokens=50), snap, cancel)
    assert seen == [0] and time.perf_counter() - t0 < 10


# ---- P4-G: stop decoding when the report starts repeating --------------------------------------------------------------------

def test_repeat_started_is_false_while_nothing_has_been_said_twice():
    for text in ("", "  \n ", "The heart is normal.", "The heart is normal. The lungs are clear.",
                 "The heart is normal. The lungs are clear. No pleural",
                 "Findings: The heart is normal. Impression: Clear."):
        assert repeat_started(text) is False, text


def test_repeat_started_is_true_once_a_finished_sentence_repeats_an_earlier_one():
    assert repeat_started("The heart is normal. The heart is normal.") is True                            # a consecutive repeat
    assert repeat_started("The heart is normal. The lungs are clear. The heart is normal.") is True       # A B A
    assert repeat_started("The heart is normal. The lungs are clear. The heart is normal. The lungs are clear.") is True   # A B A B
    assert repeat_started("Findings: no effusion. Impression: Clear. Findings: no effusion.") is True     # after a header


def test_repeat_started_waits_for_the_terminal_of_a_repeat_and_is_not_undone_by_what_follows_it():
    shown, repeat = "The heart is normal. The lungs are clear.", "The heart is normal."
    for n in range(1, len(repeat)):          # every prefix of the repeat, mid-word ones included: still being written
        assert repeat_started(shown + " " + repeat[:n]) is False, repeat[:n]
    assert repeat_started(shown + " " + repeat) is True                  # its terminal has arrived
    assert repeat_started(shown + " " + repeat + " The") is True         # the best beam can already be into the next sentence
    header = "Findings: The heart is normal."
    for n in range(1, len(header)):
        assert repeat_started(header + " " + header[:n]) is False, header[:n]
    assert repeat_started(header + " " + header) is True


def test_repeat_started_reads_case_and_whitespace_as_the_repair_does():
    assert repeat_started("The heart is normal. the HEART is   normal.") is True
    assert repeat_started("The heart is normal.\n\nthe heart\tis normal.") is True
    assert repeat_started("The heart is normal. The heart is normal!") is False   # the terminal is part of the key, as in repair_report


def test_a_header_alone_is_not_a_sentence_and_so_not_a_repeat():
    for text in ("Findings:", "Findings: Impression:", "Findings: The heart is normal. Findings:",
                 "Findings: The heart is normal. Impression: The heart is normal."):   # the header is part of the sentence it starts
        assert repeat_started(text) is False, text
    assert repeat_started("Findings: Findings:") is False                               # two headers and no sentence between them


def test_repeat_started_is_about_the_last_finished_sentence_and_uses_the_repairs_splitter():
    loop = "The heart is normal. The lungs are clear. The heart is normal. The lungs are clear. Impression: no effusion."
    words = loop.split()
    fired = [n for n in range(1, len(words) + 1) if repeat_started(" ".join(words[:n]))]
    assert fired == list(range(12, 19))      # from the terminal of the third sentence until a new sentence is finished
    # the repair's list-marker rule ("1." ends no sentence) decides what a sentence is, not a second splitter of this file's own
    assert repeat_started("1. Heart size is normal. 2. Heart size is normal.") is False
    assert repeat_started("1. Heart size is normal. 2. Lungs are clear. 1. Heart size is normal.") is True
    assert repeat_started("Seen by Dr. Smith. Seen by Dr. Jones.") is False       # an abbreviation ends no sentence either
    assert repeat_started("The mass is 4.5 cm. The mass is 4.5 cm.") is True       # nor does a decimal point


class _Scripted:
    """A tokenizer whose text depends on how many ids it is given: the best beam of a beam search can change between two steps, and
    then its texts are not prefixes of one another."""

    def __init__(self, default, at):
        self.default, self.at = default, at

    def decode(self, ids, skip_special_tokens=True):
        return self.at.get(len(ids), self.default)


def test_stop_on_repeat_ends_decoding_at_the_step_where_a_repeat_is_finished_and_not_before():
    text = "The heart is normal. The lungs are clear. The heart is normal. The lungs are clear. Impression: no effusion."
    words = text.split()
    assert len(words) >= 16
    eng = build_engine("tiny")
    eng.tokenizer = _GrowingText(text)
    enc = _encoded(eng)
    snaps = []
    res, gen = eng.generate(enc, Options(max_new_tokens=len(words), stop_on_repeat=True, display_repair=True),
                            lambda step, t: snaps.append((step, t)), threading.Event())
    assert len(gen.token_ids) == res.detail["tokens"] == 12                   # the third sentence ends at the 12th word
    assert res.detail["stopped"] == "repeat"
    assert gen.report == " ".join(words[:12])                                  # the raw text includes the one repeated sentence
    assert [step for step, _ in snaps] == list(range(12))                      # one snapshot per step, the stopping step's included
    assert snaps[-1][1] == stream_view(gen.report)                             # and the last one is the answer, as it is for a budget run
    assert gen.display_report == "The heart is normal. The lungs are clear."   # the display copy has no repeat
    assert gen.truncated_mid_sentence is False
    _, full = eng.generate(enc, Options(max_new_tokens=len(words)), _noop, threading.Event())
    assert (len(full.token_ids), full.report) == (len(words), text)           # switched off, the whole budget is decoded
    assert full.truncated_mid_sentence is False


def test_stop_on_repeat_with_repair_off_streams_and_stores_the_raw_text():
    text = "The heart is normal. The lungs are clear. The heart is normal. The lungs are clear. Impression: no effusion."
    eng = build_engine("tiny")
    eng.tokenizer = _GrowingText(text)
    snaps = []
    res, gen = eng.generate(_encoded(eng), Options(max_new_tokens=19, stop_on_repeat=True), lambda step, t: snaps.append(t), threading.Event())
    assert snaps[-1] == gen.report == gen.display_report == " ".join(text.split()[:12])   # the repeat is in all three: repair is off
    assert res.detail["stopped"] == "repeat" and gen.truncated_mid_sentence is False


def test_a_repeat_stop_is_a_decision_and_not_a_cut_off_even_when_the_best_beam_is_already_into_the_next_sentence():
    # The tiny model, beam 3, token 21: 'and heart and heart without. catheter acute right large heart without. catheter acute
    # right large heart without. catheter'. The beam that finished the repeat was not the best one a step earlier.
    repeated = "The heart is normal. The lungs are clear. The heart is normal. The lungs"
    eng = build_engine("tiny")
    eng.tokenizer = _Scripted("The heart is normal. The lungs are clear.", {17: repeated})
    res, gen = eng.generate(_encoded(eng), Options(max_new_tokens=40, stop_on_repeat=True, display_repair=True), _noop, threading.Event())
    assert len(gen.token_ids) == 17 and res.detail["stopped"] == "repeat"
    assert gen.report == repeated                                              # the decoder's text, fragment and all
    assert gen.display_report == "The heart is normal. The lungs are clear."   # the display copy drops the repeat and the fragment
    assert gen.truncated_mid_sentence is False                                 # the budget did not cut it off: the turn chose to stop


def test_a_text_that_does_not_repeat_runs_to_the_budget_with_the_switch_on():
    eng = build_engine("tiny")
    eng.tokenizer = _GrowingText("The heart is normal. The lungs are clear. Impression: no effusion. Findings: the heart is mildly enlarged. "
                                 "No pleural effusion or pneumothorax is seen.")
    res, gen = eng.generate(_encoded(eng), Options(max_new_tokens=16, stop_on_repeat=True), _noop, threading.Event())
    assert res.detail["stopped"] == "budget" and len(gen.token_ids) == res.detail["tokens"] == 16
    assert gen.report.endswith("Findings: the heart is mildly")
    assert gen.truncated_mid_sentence is True          # cut off by the budget: the flag keeps its meaning there


def test_cancel_is_looked_at_before_the_repeat():
    at = {n: "The heart is normal. The heart is normal." for n in range(4, 40)}   # from the 4th step on, the text repeats
    eng = build_engine("tiny")
    eng.tokenizer = _Scripted("The heart is normal.", at)
    enc = _encoded(eng)
    cancel, seen = threading.Event(), []

    def snap(step, text):
        seen.append(step)
        if step == 2:
            cancel.set()

    with pytest.raises(Cancelled):
        eng.generate(enc, Options(max_new_tokens=40, stop_on_repeat=True), snap, cancel)
    assert seen == [0, 1, 2]                                                   # step 3 was cancelled before its text was read for a repeat


@pytest.mark.parametrize("cached,decode", [(True, "beam"), (False, "beam"), (True, "greedy"), (False, "greedy")])
def test_the_users_case_stops_at_its_first_repeat_and_what_it_stopped_with_is_the_published_decoder_at_that_length(cached, decode):
    eng = build_engine("tiny")
    enc = _encoded(eng)
    snaps = []
    res, stopped = eng.generate(enc, Options(max_new_tokens=200, decode=decode, cached_decode=cached, display_repair=True, stop_on_repeat=True),
                                lambda step, t: snaps.append(t), threading.Event())
    n = len(stopped.token_ids)
    assert 16 <= n < 200                                                       # the tiny model repeats within a few dozen tokens, not at 200
    assert (res.detail["stopped"], res.detail["tokens"]) == ("repeat", n)
    assert repeat_started(stopped.report) and _repeats(stopped.report) >= 1    # it stopped because the report repeated itself ...
    assert _repeats(stopped.display_report) == 0                               # ... and the card shows no repeat
    assert stopped.truncated_mid_sentence is False                             # no "stopped at the token budget" for a turn that did not reach it
    assert len(snaps) == n and all(_repeats(t) == 0 for t in snaps)            # not one frame of the stream repeats a sentence
    _, raw = eng.generate(enc, Options(max_new_tokens=200, decode=decode, cached_decode=cached, stop_on_repeat=True),
                          lambda step, t: snaps.append(t), threading.Event())
    assert raw.token_ids == stopped.token_ids and raw.display_report == raw.report == stopped.report   # repair off: the raw text is the display
    seen = snaps[n:]
    assert len(seen) == n and repeat_started(seen[-1]) and not any(repeat_started(t) for t in seen[:-1])   # the first step that repeats, no later
    ran, published = eng.generate(enc, Options(max_new_tokens=n, decode=decode, cached_decode=cached), _noop, threading.Event())
    assert published.token_ids == stopped.token_ids and published.report == stopped.report   # the published run, cut at that length
    assert ran.detail["stopped"] == "budget"                                                   # which is where only it says "budget"


def test_without_stop_on_repeat_the_users_case_is_the_published_decoder_to_the_last_token():
    eng = build_engine("tiny")
    enc = _encoded(eng)
    res, gen = eng.generate(enc, Options(max_new_tokens=200, display_repair=True), _noop, threading.Event())
    explicit, same = eng.generate(enc, Options(max_new_tokens=200, display_repair=True, stop_on_repeat=False), _noop, threading.Event())
    published = eng.decoder.beam_search_cached(EMPTY, prefix_embeds=enc.prefix, beam_size=3, max_new_tokens=200)
    assert gen.token_ids == same.token_ids == published[0].tolist() and len(gen.token_ids) == 200
    assert (res.detail["stopped"], explicit.detail["stopped"]) == ("budget", "budget")
    assert (res.detail["tokens"], explicit.detail["tokens"]) == (200, 200)
    assert gen.report == same.report and gen.display_report == same.display_report and gen.truncated_mid_sentence == same.truncated_mid_sentence


# ---- P9-G2: a model trained to end its reports ---------------------------------------------------------------------------------

EOS_ID = len(TINY_VOCAB) - 1   # the last id of the tiny vocab: in reach of the scripted decoders, unlike the real engine's 50256
REPORT = ("The heart is normal. The lungs are clear. No pleural effusion. Impression: no acute disease. Findings: the heart "
          "is mildly enlarged and there is a small effusion.")   # 27 words: ends in sentences 4, 8, 11, 15 and 27
WORDS = REPORT.split()


@contextmanager
def _eos_model(end_at=None, text=REPORT, eos_trained=True):
    """The tiny engine as a model whose card says it was trained to end its reports, with a scripted decoder that ends every
    beam's report after `end_at` tokens (never, with None) and a tokenizer that writes `text` a word per token."""
    eng = build_engine("tiny")
    eng._card["eos_trained"], eng.eos_token_id, eng.tokenizer = eos_trained, EOS_ID, _GrowingText(text)
    script = noise_script(5, EOS_ID, lambda n, last: 40.0 if n == end_at else -1e4)
    with scripted_decoder(eng.decoder, script):
        yield eng


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("decode", ["beam", "greedy"])
def test_an_eos_trained_model_stops_where_its_report_ends_and_says_so(cached, decode):
    with _eos_model(end_at=11) as eng:
        snaps = []
        res, gen = eng.generate(_encoded(eng), Options(max_new_tokens=40, decode=decode, cached_decode=cached),
                                lambda step, text: snaps.append((step, text)), threading.Event())
    assert res.detail["stopped"] == "eos" and res.detail["tokens"] == len(gen.token_ids) == 11
    assert gen.report == " ".join(WORDS[:11]) == "The heart is normal. The lungs are clear. No pleural effusion."
    assert gen.truncated_mid_sentence is False and gen.display_report == gen.report
    assert [step for step, _ in snaps] == list(range(12))                     # one snapshot per step, the step that ended it too
    assert [text for _, text in snaps[:11]] == [" ".join(WORDS[:n]) for n in range(1, 12)]
    json.dumps(res.detail)
    assert set(res.detail) == {"decode", "beam_size", "tokens", "stopped", "cached_decode", "compiled", "prefill_ms",
                               "per_token_ms", "device", "threads", "drift_note"}   # the stop reason is a value, not a new field


def test_an_eos_stop_is_never_a_cut_off_whatever_the_text_ends_in_and_the_display_copy_follows_p4f():
    with _eos_model(end_at=13) as eng:                                         # "... Impression: no", the 13th word ends no sentence
        _, plain = eng.generate(_encoded(eng), Options(max_new_tokens=40), _noop, threading.Event())
        res, shown = eng.generate(_encoded(eng), Options(max_new_tokens=40, display_repair=True), _noop, threading.Event())
    assert res.detail["stopped"] == "eos" and plain.report == shown.report == " ".join(WORDS[:13])
    assert plain.truncated_mid_sentence is False and shown.truncated_mid_sentence is False
    assert plain.display_report == plain.report                                # repair off: the raw text is the display
    assert shown.display_report == repair_report(shown.report, dedup="all", truncate=True)[0] == " ".join(WORDS[:11])
    with _eos_model(end_at=None) as eng:                                       # the same text cut by the budget is a cut-off
        res, cut = eng.generate(_encoded(eng), Options(max_new_tokens=18), _noop, threading.Event())
    assert res.detail["stopped"] == "budget" and cut.truncated_mid_sentence is True


def test_the_eos_decoders_run_for_a_model_whose_card_says_so_and_only_then():
    from scripts import evaluate_report_generation as erg
    with _eos_model(end_at=None, eos_trained=False) as eng:
        with mock.patch.object(erg, "beam_search_decode_eos", side_effect=AssertionError("EOS decoder")), \
                mock.patch.object(eng.decoder, "beam_search_cached_eos", side_effect=AssertionError("EOS decoder")):
            for cached in (True, False):
                res, _ = eng.generate(_encoded(eng), Options(max_new_tokens=16, cached_decode=cached), _noop, threading.Event())
                assert res.detail["stopped"] == "budget"
            del eng._card["eos_trained"]                                      # a card with no such key (an old one) is not trained to stop
            for cached in (True, False):
                eng.generate(_encoded(eng), Options(max_new_tokens=16, cached_decode=cached), _noop, threading.Event())
    with _eos_model(end_at=None) as eng:
        with mock.patch.object(erg, "beam_search_decode", side_effect=AssertionError("published decoder")), \
                mock.patch.object(eng.decoder, "beam_search_cached", side_effect=AssertionError("published decoder")):
            for cached in (True, False):
                eng.generate(_encoded(eng), Options(max_new_tokens=16, cached_decode=cached), _noop, threading.Event())


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("decode", ["beam", "greedy"])
def test_a_card_that_says_eos_trained_changes_no_token_of_a_model_that_never_emits_one(cached, decode):
    eng = build_engine("tiny")                                                 # the real engine's id, 50256, is no token of the tiny vocab
    enc = _encoded(eng)
    opts = Options(max_new_tokens=24, decode=decode, cached_decode=cached, display_repair=True)
    _, plain = eng.generate(enc, opts, _noop, threading.Event())
    eng._card["eos_trained"] = True
    res, eos = eng.generate(enc, opts, _noop, threading.Event())
    assert eos.token_ids == plain.token_ids and len(eos.token_ids) == 24
    assert (eos.report, eos.display_report, eos.truncated_mid_sentence) == (plain.report, plain.display_report, plain.truncated_mid_sentence)
    assert res.detail["stopped"] == "budget" and res.detail["tokens"] == 24


@pytest.mark.parametrize("cached", [True, False])
def test_stop_on_repeat_is_still_a_backstop_and_the_first_stop_to_fire_wins(cached):
    loop = "The heart is normal. The lungs are clear. The heart is normal. The lungs are clear. Impression: no effusion."
    opts = Options(max_new_tokens=40, cached_decode=cached, stop_on_repeat=True)
    with _eos_model(end_at=30, text=loop) as eng:                              # the report repeats (word 12) before the model would end it
        res, gen = eng.generate(_encoded(eng), opts, _noop, threading.Event())
    assert res.detail["stopped"] == "repeat" and len(gen.token_ids) == 12 and gen.truncated_mid_sentence is False
    with _eos_model(end_at=8, text=loop) as eng:                               # the model ends it (word 8) first: the switch is on, and it is EOS
        res, gen = eng.generate(_encoded(eng), opts, _noop, threading.Event())
    assert res.detail["stopped"] == "eos" and gen.report == " ".join(loop.split()[:8])
    with _eos_model(end_at=8, text=loop) as eng:                               # and the same model with the switch off
        res, gen = eng.generate(_encoded(eng), Options(max_new_tokens=40, cached_decode=cached), _noop, threading.Event())
    assert res.detail["stopped"] == "eos" and len(gen.token_ids) == 8


@pytest.mark.parametrize("cached", [True, False])
def test_cancel_stops_an_eos_decode_within_one_step(cached):
    with _eos_model(end_at=30) as eng:
        enc, cancel, seen = _encoded(eng), threading.Event(), []

        def snap(step, text):
            seen.append(step)
            if step == 3:
                cancel.set()

        with pytest.raises(Cancelled):
            eng.generate(enc, Options(max_new_tokens=50, cached_decode=cached), snap, cancel)
    assert seen == [0, 1, 2, 3]


def test_a_cancel_set_before_an_eos_decode_starts_does_no_decoding():
    with _eos_model(end_at=11) as eng:
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(eng.decoder, "beam_search_cached_eos", side_effect=AssertionError("decoded")):
            with pytest.raises(Cancelled):
                eng.generate(_encoded(eng), Options(), _noop, cancel)


def test_cached_decode_is_still_refused_for_a_stack_without_a_cache_on_the_eos_path():
    with _eos_model(end_at=11) as eng:
        with mock.patch.object(eng.decoder, "supports_cached_decode", return_value=False):
            with pytest.raises(ValueError, match="cached_decode"):
                eng.generate(_encoded(eng), Options(max_new_tokens=16), _noop, threading.Event())
            res, gen = eng.generate(_encoded(eng), Options(max_new_tokens=16, cached_decode=False), _noop, threading.Event())
    assert res.detail["stopped"] == "eos" and len(gen.token_ids) == 11          # the uncached EOS search serves it


def test_the_engines_eos_id_is_gpt2s_and_the_tiny_engine_is_not_trained_to_stop():
    from app.engine import Engine
    assert Engine.eos_token_id == 50256 and build_engine("tiny").card()["eos_trained"] is False


# ---- the card and its provenance -------------------------------------------------------------------------

def test_tiny_engine_card_is_complete_and_json_safe():
    eng = build_engine("tiny")
    card = eng.card()
    json.dumps(card)
    assert (card["name"], card["checkpoint"], card["prefix_k"]) == ("tiny", None, 4)
    assert card["cached_decode_available"] is True and card["drift_note"] == "tiny random-init model"
    assert {"checkpoint_sha256", "scan_impl", "tfla_impl", "mamba3_chunk_size", "layer_pattern", "train_experiment",
            "torch", "device", "threads", "cpu", "git_sha", "git_dirty", "git_source", "eos_trained"} <= set(card)
    assert card["eos_trained"] is False                                 # P9-G2: no model that ships today was trained to end a report
    card["layer_pattern"].append("x")                                   # a copy: callers cannot edit the record
    assert "x" not in eng.card()["layer_pattern"]


def _git(root, *args):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="t",
               GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(root)] + list(args), check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _checkout(root):
    """A throwaway git checkout with one committed file under app/."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / "app").mkdir(exist_ok=True)
    (root / "app" / "x.py").write_text("x = 1\n")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "c")
    return _git(root, "rev-parse", "HEAD")


STAMP_SHA = "0123456789abcdef0123456789abcdef01234567"


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@needs_git
def test_provenance_comes_from_git_where_there_is_a_checkout(tmp_path):
    sha = _checkout(tmp_path / "repo")
    got = git_provenance(tmp_path / "repo")
    assert got == {"git_sha": sha, "git_dirty": False, "git_source": "git"}
    (tmp_path / "repo" / "notes.ipynb").write_text("{}")               # unrelated untracked files are not "dirty"
    assert git_provenance(tmp_path / "repo")["git_dirty"] is False
    (tmp_path / "repo" / "app" / "x.py").write_text("x = 2\n")          # a code change is
    assert git_provenance(tmp_path / "repo") == {"git_sha": sha, "git_dirty": True, "git_source": "git"}


@pytest.mark.parametrize("word,dirty", [("clean", False), ("dirty", True)])
def test_provenance_falls_back_to_the_sync_stamp_without_git(tmp_path, word, dirty):
    (tmp_path / ".sync_stamp").write_text("2026-10-01T16:23:43Z {} {}\n".format(STAMP_SHA, word))
    assert git_provenance(tmp_path) == {"git_sha": STAMP_SHA, "git_dirty": dirty, "git_source": "sync_stamp"}


@needs_git
def test_a_checkout_beats_a_stale_sync_stamp(tmp_path):
    sha = _checkout(tmp_path)
    (tmp_path / ".sync_stamp").write_text("2026-01-01T00:00:00Z {} dirty\n".format(STAMP_SHA))
    assert git_provenance(tmp_path) == {"git_sha": sha, "git_dirty": False, "git_source": "git"}


@needs_git
def test_a_parent_repos_head_is_never_this_trees_provenance(tmp_path):
    _checkout(tmp_path)                                                  # e.g. a home directory that is a git repo
    child = tmp_path / "hybrid_chat_ui"                                  # the rsynced tree has no .git of its own
    child.mkdir()
    (child / ".sync_stamp").write_text("2026-10-01T16:23:43Z {} clean\n".format(STAMP_SHA))
    assert git_provenance(child) == {"git_sha": STAMP_SHA, "git_dirty": False, "git_source": "sync_stamp"}
    (child / ".sync_stamp").unlink()
    assert git_provenance(child) == {"git_sha": None, "git_dirty": None, "git_source": None}


@pytest.mark.parametrize("stamp", ["", "garbage\n", "t sha clean\n", "t {} maybe\n".format(STAMP_SHA),
                                   "t {} clean extra\n".format(STAMP_SHA)])
def test_unknown_provenance_is_none_never_clean(tmp_path, stamp):
    (tmp_path / ".sync_stamp").write_text(stamp)
    assert git_provenance(tmp_path) == {"git_sha": None, "git_dirty": None, "git_source": None}
    assert git_provenance(tmp_path / "missing") == {"git_sha": None, "git_dirty": None, "git_source": None}


# ---- RealEngine ------------------------------------------------------------------------------------------

@pytest.fixture
def keep_threads():
    """RealEngine.__init__ calls torch.set_num_threads, which is process-wide."""
    threads = torch.get_num_threads()
    yield threads
    torch.set_num_threads(threads)


def _fake_run(tmp_path, experiment="h100_report_gen_m3_tower13d_s42", prefix_k=4, dataset=None):
    """outputs/<run>/{run_metadata.json, checkpoints/last.ckpt}: the layout resolve_prefix_k reads. `dataset`, when given, is the
    resolved config's dataset block (where P9-G1's report_eos_target lands)."""
    run = tmp_path / "outputs" / "run_x"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "last.ckpt"
    ckpt.write_bytes(b"not a real checkpoint" * 100)
    resolved = {"experiment_name": experiment, "model": {"prefix_k": prefix_k}}
    if dataset is not None:
        resolved["dataset"] = dataset
    (run / "run_metadata.json").write_text(json.dumps({"git_sha": "x", "resolved_config": resolved}))
    return ckpt


def _stub_loader(monkeypatch, prefix_k=4):
    """Replace the heavy parts of RealEngine.__init__ (the published loader, the GPT-2 download)."""
    import scripts.evaluate_report_generation as erg
    import transformers
    module = SimpleNamespace(image_encoder=TinyTower(), prefix_mapper=tiny_prefix_mapper(), decoder=tiny_decoder(),
                             prefix_k=prefix_k)
    seen = {}

    def fake_load(checkpoint, model_config, device="cpu", **kw):
        seen.update(checkpoint=checkpoint, model_config=model_config, device=device, kw=kw)
        return module

    monkeypatch.setattr(erg, "load_report_generation_module", fake_load)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name, **kw: TinyTokenizer()))
    return module, seen


def test_real_engine_uses_the_published_loader_and_records_where_the_answer_came_from(tmp_path, monkeypatch,
                                                                                      keep_threads):
    ckpt = _fake_run(tmp_path)
    module, seen = _stub_loader(monkeypatch)
    eng = build_engine("real", checkpoint=str(ckpt), model_config="hybrid_150m_m3_rrg", device="cpu", threads=2,
                       cache_dir=tmp_path / "cache", drift_note="CPU, fp32")
    assert torch.get_num_threads() == 2
    assert seen["checkpoint"] == str(ckpt) and seen["model_config"] == "hybrid_150m_m3_rrg"
    assert seen["device"] == "cpu" and seen["kw"] == {}               # no operator overrides: the published protocol
    assert (eng.tower, eng.prefix_mapper, eng.decoder) == (module.image_encoder, module.prefix_mapper, module.decoder)
    assert eng.name == "hybrid_150m_m3_rrg" and eng.tower_sha256() == tensor_sha256(module.image_encoder)
    card = eng.card()
    json.dumps(card)
    assert card["name"] == "hybrid_150m_m3_rrg" and card["checkpoint"] == str(ckpt)
    assert card["checkpoint_sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert (tmp_path / "cache" / "sha256.json").exists()
    assert card["prefix_k"] == 4 and card["train_experiment"] == "h100_report_gen_m3_tower13d_s42"
    assert card["scan_impl"] == module.decoder.config.scan_impl and card["cached_decode_available"] is True
    assert card["layer_pattern"] == list(module.decoder.config.layer_pattern) and card["drift_note"] == "CPU, fp32"
    # the shared stages run over the loader's parts
    _, gen = eng.generate(_encoded(eng), Options(max_new_tokens=16), _noop, threading.Event())
    assert len(gen.token_ids) == 16


def test_real_engine_card_survives_missing_or_damaged_run_metadata(tmp_path, monkeypatch, keep_threads):
    ckpt = _fake_run(tmp_path)
    _stub_loader(monkeypatch)
    meta = ckpt.parent.parent / "run_metadata.json"
    meta.write_text("{broken")
    assert build_engine("real", checkpoint=str(ckpt), model_config="m")._card["train_experiment"] is None
    meta.unlink()
    card = build_engine("real", checkpoint=str(ckpt), model_config="m")._card
    assert card["train_experiment"] is None and card["eos_trained"] is False    # unknown is not "trained to stop"


@pytest.mark.parametrize("dataset,trained", [
    ({"report_eos_target": True}, True),
    ({"report_eos_target": False}, False),
    ({"report_eos_target": None}, False),
    ({"report_eos_target": "true"}, False),     # P9-G1 insists on a bool: anything else is not "trained to stop"
    ({"report_eos_target": 1}, False),
    ({}, False),                                # a run from before the flag existed
    (None, False),                              # no dataset block at all
    ([], False),                                # a block of the wrong shape
])
def test_real_engine_card_says_a_run_was_trained_to_end_its_reports_only_when_its_config_says_true(
        tmp_path, monkeypatch, keep_threads, dataset, trained):
    ckpt = _fake_run(tmp_path, dataset=dataset)
    _stub_loader(monkeypatch)
    card = build_engine("real", checkpoint=str(ckpt), model_config="m").card()
    assert card["eos_trained"] is trained and card["train_experiment"] == "h100_report_gen_m3_tower13d_s42"
    json.dumps(card)


def test_real_engine_of_an_eos_trained_run_decodes_with_the_eos_search_and_a_published_run_does_not(tmp_path, monkeypatch, keep_threads):
    _stub_loader(monkeypatch)
    trained = build_engine("real", checkpoint=str(_fake_run(tmp_path, dataset={"report_eos_target": True})), model_config="m")
    with mock.patch.object(trained.decoder, "beam_search_cached_eos", wraps=trained.decoder.beam_search_cached_eos) as eos_search:
        _, gen = trained.generate(_encoded(trained), Options(max_new_tokens=16), _noop, threading.Event())
    assert eos_search.call_count == 1 and eos_search.call_args.kwargs["eos_token_id"] == 50256 and len(gen.token_ids) == 16
    other = tmp_path / "other"
    other.mkdir()
    published = build_engine("real", checkpoint=str(_fake_run(other)), model_config="m")
    with mock.patch.object(published.decoder, "beam_search_cached_eos", side_effect=AssertionError("EOS search")):
        published.generate(_encoded(published), Options(max_new_tokens=16), _noop, threading.Event())


def test_real_engine_card_takes_git_fields_from_the_repo_root_stamp(tmp_path, monkeypatch, keep_threads):
    import app.engine as engine_module
    ckpt = _fake_run(tmp_path)
    _stub_loader(monkeypatch)
    tree = tmp_path / "hybrid_chat_ui"
    tree.mkdir()
    (tree / ".sync_stamp").write_text("2026-10-01T16:23:43Z {} dirty\n".format(STAMP_SHA))
    monkeypatch.setattr(engine_module, "REPO_ROOT", tree)               # an rsynced tree: no .git, only the stamp
    card = build_engine("real", checkpoint=str(ckpt), model_config="m").card()
    assert (card["git_sha"], card["git_dirty"], card["git_source"]) == (STAMP_SHA, True, "sync_stamp")


def test_real_engine_end_to_end_equals_the_published_functions_on_the_real_tower(tmp_path, monkeypatch,
                                                                                keep_threads):
    """The published loader, the real BiomedCLIP tower and the real transform over a random-init tiny decoder.

    A checkpoint is written from a reference module; the engine loads it through load_report_generation_module
    and must reproduce the published functions on the same weights (R2 at laptop scale: P2-E repeats it on the
    cluster with the real checkpoint). Skips when BiomedCLIP is not in the local cache.
    """
    pytest.importorskip("open_clip")
    import transformers
    import yaml
    import scripts.evaluate_report_generation as erg
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        ref = ReportGenerationLightningModule(decoder_config=tiny_decoder_config(), image_patch_dim=768, prefix_k=8)
        try:
            ref.load_image_encoder()                                   # stock BiomedCLIP from the local HF cache
        except Exception as exc:   # no network and no local cache
            pytest.skip("BiomedCLIP unavailable: {}".format(exc))
    ref.eval()

    (tmp_path / "configs" / "model").mkdir(parents=True)
    # the yaml says k=32; the run was trained with 8, which only run_metadata.json knows
    yaml_cfg = dict(dataclasses.asdict(tiny_decoder_config()), prefix_k=32, image_patch_dim=768)
    (tmp_path / "configs" / "model" / "tiny_rrg.yaml").write_text(yaml.safe_dump(yaml_cfg))
    monkeypatch.setattr(erg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name, **kw: TinyTokenizer()))
    ckpt = _fake_run(tmp_path, prefix_k=8)
    try:
        torch.save({"state_dict": ref.state_dict()}, ckpt)             # 344 MB: removed below (pytest keeps old dirs)
        eng = build_engine("real", checkpoint=str(ckpt), model_config="tiny_rrg", threads=keep_threads,
                           cache_dir=tmp_path / "cache")
        card = eng.card()
        assert card["prefix_k"] == eng.module.prefix_k == 8            # run_metadata.json beats the yaml's 32
        assert card["train_experiment"] == "h100_report_gen_m3_tower13d_s42" and len(card["checkpoint_sha256"]) == 64
        for mine, theirs in ((eng.decoder, ref.decoder), (eng.prefix_mapper, ref.prefix_mapper),
                             (eng.tower, ref.image_encoder)):
            assert tensor_sha256(mine) == tensor_sha256(theirs)         # loaded from the checkpoint, not a fresh init
        assert eng.tower_sha256() == tensor_sha256(ref.image_encoder)

        _, prep = eng.preprocess(png_bytes(320, 320))
        res_e, enc = eng.encode(prep)
        assert res_e.detail == {"patch_grid": [197, 768], "pooled_dim": 512, "prefix_tokens": 8, "device": "cpu",
                                "one_pass": True}
        with torch.no_grad():
            grid = ref._patch_grid(prep.pixel_values)                  # the published patch grid
            assert torch.equal(enc.patch_grid, grid)
            assert torch.equal(enc.pooled, F.normalize(ref.image_encoder(prep.pixel_values), dim=-1)[0])
        for cached in (True, False):
            _, gen = eng.generate(enc, Options(max_new_tokens=16, cached_decode=cached), _noop, threading.Event())
            published = erg.generate_from_patch_grid(ref, grid, decode="beam", beam_size=3, max_new_tokens=16,
                                                     cached=cached)
            assert gen.token_ids == published[0].tolist()
    finally:
        ckpt.unlink()
