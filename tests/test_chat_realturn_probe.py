"""CHAT_UI_PLAN.md P7-D, the laptop part: scripts/chat_realturn_probe.py, the real-turn gate from the laptop through the tunnel.

The probe runs three passes on a private chat server, over test-split rows picked by number: (1) the gate, the published model with the
published protocol: does the live report equal the published line, are the five image features served, how long does a turn take; (2) the
stop measurement, the same rows with `stop_on_repeat` on at 100 and at 200 tokens; (3) the EOS model on the same rows and on one picture
it draws itself. A private server streams MIMIC-derived text (reports, matched reports), and under rule R7 no agent may see any of it, so
the probe prints only numbers, booleans and fixed words. Here every pass runs against a real uvicorn over the tiny engines, the tiny
gallery and published dumps of synthetic lines (the P5-E fixtures of tests/test_app_retrieval_stages.py): synthetic data only, so an
assertion may print whatever it compares.
"""
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

from app.engine import _finished
from app.labels import CHEXBERT_14
from app.pipeline import PUBLISHED_MODEL, PUBLISHED_PROTOCOL as APP_PROTOCOL, STAGES as APP_STAGES, PublishedDumps
from app.schemas import Options
from app.server import CLIENT_ID as SERVER_CLIENT_ID, HOME_MODELS, MODEL_CHECKPOINTS, create_app
from app.store import Store
from scripts import chat_load_drill as drill
from scripts import chat_realturn_probe as probe
from scripts import repair_generations as rg
from scripts.build_retrieval_gallery import TINY_TEST, build_tiny
from tests.app_helpers import EOS_ID, REPORT, GrowingText, iter_sse, noise_script, scripted_decoder, start_live_server, wait_until
from tests.test_app_retrieval_stages import _published_app
from tests.test_chat_load_drill import NoNetwork, Unreachable, leaves, refusal_line, write_token_file

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRET = "s3cr3t-T0KEN-value"
TOKEN = "t0k-for-the-probe"
LOOP = "The heart is normal. The lungs are clear. The heart is normal. No pleural effusion."   # 15 words; the 12th ends a repeat of the 4th


@pytest.fixture(autouse=True)
def no_ambient_token(monkeypatch):
    """Whoever runs the real probe has CHAT_TOKEN exported in that shell: no test may send it, to a local test server or anywhere."""
    monkeypatch.delenv("CHAT_TOKEN", raising=False)


@pytest.fixture
def offline(monkeypatch):
    """A refusal must come before anything is opened: httpx's client and its own request functions fail the test when called."""
    monkeypatch.setattr(httpx, "Client", NoNetwork)
    for name in ("get", "post", "delete", "request", "stream"):
        monkeypatch.setattr(httpx, name, NoNetwork)


def run_probe(args, expect_code=None):
    """main() with its own two streams, so what it writes is told apart from the in-process server's request log. -> (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = probe.main(list(args), out, err)
    if expect_code is not None:
        assert code == expect_code, (code, out.getvalue(), err.getvalue())
    return code, out.getvalue(), err.getvalue()


# ---- what the probe asks for: the published model and protocol, the EOS model, the rows --------------------------------------------

def test_the_probe_names_the_two_models_the_server_serves_and_the_published_protocol_the_app_holds():
    """The probe imports nothing of the app (it must start with the server's code absent, as the drill), so the app's names are restated in
    the script and held equal here: the published model and its protocol (app/pipeline.py), the EOS entry of the registry, the stages."""
    assert probe.PUBLISHED_MODEL == PUBLISHED_MODEL == "hybrid_150m_m3_rrg" and probe.PUBLISHED_MODEL in MODEL_CHECKPOINTS
    assert probe.EOS_MODEL == "hybrid_150m_m3_rrg_eos" and HOME_MODELS[probe.EOS_MODEL]["eos"] is True
    assert probe.PUBLISHED_PROTOCOL == APP_PROTOCOL == {"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "stop_on_repeat": False,
                                                        "compile": False}
    assert probe.STAGES == APP_STAGES


def test_the_client_id_is_one_fixed_value_the_server_takes():
    assert probe.CLIENT_ID == "chat-realturn-probe" != drill.CLIENT_ID and SERVER_CLIENT_ID.fullmatch(probe.CLIENT_ID)


def test_each_pass_sends_the_options_the_plan_names_and_the_server_validates_them():
    """Pass 1: the published model, named, under the published protocol. Pass 2: the same with the stop on, at 100 and at 200 tokens. Pass 3:
    the EOS model, the same options at its own budget, and for the picture the probe draws, no test row. The page's retrieval defaults are
    sent as numbers, so that "the counts as asked" is what the probe asked. Options forbids an unknown key, so a typo here is a failure."""
    base = dict(APP_PROTOCOL, k_images=4, k_reports=3)
    assert (probe.K_IMAGES, probe.K_REPORTS) == (4, 3)
    assert probe.gate_options(3) == dict(base, model="hybrid_150m_m3_rrg", test_row=3)
    assert probe.stop_options(3, 100) == dict(base, model="hybrid_150m_m3_rrg", test_row=3, stop_on_repeat=True, max_new_tokens=100)
    assert probe.stop_options(4, 200) == dict(base, model="hybrid_150m_m3_rrg", test_row=4, stop_on_repeat=True, max_new_tokens=200)
    assert probe.eos_options(200, 3) == dict(base, model="hybrid_150m_m3_rrg_eos", test_row=3, max_new_tokens=200)
    assert probe.eos_options(160) == dict(base, model="hybrid_150m_m3_rrg_eos", max_new_tokens=160) and "test_row" not in probe.eos_options(160)
    for options in (probe.gate_options(0), probe.stop_options(2662, 200), probe.eos_options(200, 7), probe.eos_options(16)):
        Options.model_validate(options)
    assert probe.gate_options(0)["stop_on_repeat"] is False and probe.eos_options(200, 0)["stop_on_repeat"] is False


def test_rows_are_numbers_ranges_or_lists_of_them_in_the_order_given():
    assert probe.parse_rows("0-4") == [0, 1, 2, 3, 4] and probe.DEFAULT_ROWS == "0-4"
    assert probe.parse_rows("3") == [3] and probe.parse_rows("0,2,5") == [0, 2, 5] and probe.parse_rows("5,1-3,9") == [5, 1, 2, 3, 9]
    assert probe.parse_rows(" 2 , 4-5 ") == [2, 4, 5] and probe.parse_rows("2662") == [2662]
    assert len(probe.parse_rows("0-{}".format(probe.MAX_ROWS - 1))) == probe.MAX_ROWS


@pytest.mark.parametrize("text", ["", " ", "x", "4-1", "-1", "0-", "-", "1,,2", "1,1", "0-3,2", "0-{}".format(probe.MAX_ROWS),
                                  "99999999", "1.5", "0x3", "٣", "1;2", "1 2", "0--3", "0-9999999", "1-1000000", "0-30,31-32", "9999999-10000000"])
def test_rows_that_are_no_numbers_or_ranges_are_refused_in_one_line_without_being_quoted(text, offline):
    code, out, err = run_probe(["http://127.0.0.1:18080", "--rows", text])
    assert code == 2 and out == ""
    assert refusal_line(err) == "argument --rows: " + probe.ROWS_REFUSAL, "one fixed sentence: it never holds what was given"


def test_by_default_the_rows_are_0_to_4_and_the_eos_budget_is_the_one_its_note_was_measured_at():
    args = probe.build_parser().parse_args(["http://127.0.0.1:18080"])
    assert args.rows == [0, 1, 2, 3, 4]
    assert args.eos_tokens == probe.DEFAULT_EOS_TOKENS == 200   # app/server.py EOS_MODEL_NOTE: "at a 200-token budget, 2,662 of 2,663 ... end"
    assert args.timeout == probe.DEFAULT_TIMEOUT_S and args.token_file is None


# ---- what the script refuses: it never connects ------------------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://127.0.0.1:8000/", "http://localhost:8000", "http://[::1]:8000/",
                                 "127.0.0.1:8000", "localhost:8000", "https://example.org:8000", "HTTP://LOCALHOST:8000",
                                 "http://127.0.0.1:08000", "http://127.0.0.1:+8000", "http://127.0.0.1:８０００", "http://user@127.0.0.1:8000"])
def test_the_script_refuses_port_8000_in_one_line_with_exit_2_and_connects_to_nothing(url, offline):
    code, out, err = run_probe([url])
    assert code == 2 and out == ""
    assert "8000" in refusal_line(err) and "dev server" in err


@pytest.mark.parametrize("url", ["http://127.0.0.1:18000", "http://127.0.0.1:8001", "http://127.0.0.1:80", "http://127.0.0.1:18080",
                                 "http://localhost", "https://example.org", "http://127.0.0.1:8010"])
def test_only_port_8000_itself_is_refused_for_being_the_dev_server(url):
    """8010 is the tunnel's own port, the one the controller runs the probe on."""
    assert drill.check_url(url).startswith("http")


@pytest.mark.parametrize("url", ["", "   ", "http://", "ftp://127.0.0.1:2121", "http://127.0.0.1:0", "http://127.0.0.1:70000",
                                 "http://user:pass@127.0.0.1:1234", "http://tok@127.0.0.1:1234", "http://127.0.0.1:1234/?token=abc",
                                 "http://127.0.0.1:1234/#frag"])
def test_a_url_that_is_not_plain_scheme_host_and_port_is_refused_without_being_quoted(url, offline):
    code, out, err = run_probe([url])
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert "dev server" not in line and "127.0.0.1" not in line and "pass" not in line and "tok@" not in line and "abc" not in line


@pytest.mark.parametrize("argv", [
    ["http://127.0.0.1:18080", "--token", SECRET],                  # an option the script does not have
    ["http://127.0.0.1:18080", "--token=" + SECRET],
    ["http://127.0.0.1:18080", "--tok", SECRET],                    # an abbreviation of --token-file must not be taken for it
    ["http://127.0.0.1:18080", "--token-fil", SECRET],
    ["--token", SECRET, "http://127.0.0.1:18080"],                  # the secret in the URL's place
    ["http://127.0.0.1:18080", SECRET],                             # a second positional: URL TOKEN
    ["http://127.0.0.1:18080", "-t", SECRET],
    ["http://127.0.0.1:18080", "--bearer", SECRET, "--password=" + SECRET],
])
def test_the_script_refuses_a_token_on_the_command_line_and_never_echoes_it(argv, offline):
    code, out, err = run_probe(argv)
    assert code == 2 and out == ""
    assert refusal_line(err) == drill.ARGV_REFUSAL, "one fixed sentence, which names no argument"
    assert "CHAT_TOKEN" in drill.ARGV_REFUSAL and "--token-file" in drill.ARGV_REFUSAL
    assert SECRET not in err


@pytest.mark.parametrize("url", ["http://example.org:18080", "http://10.0.0.5:18080", "http://192.168.1.20", "http://[2001:db8::1]:18080",
                                 "http://[::ffff:10.0.0.5]:18080", "http://0.0.0.0:18080", "http://localhost.example.org:18080",
                                 "http://127.1:18080", "http://2130706433:18080", "http://node-07.internal", "example.org:18080"])
def test_a_token_is_not_sent_over_plain_http_to_a_host_that_is_not_loopback(url, offline, monkeypatch):
    """One fixed line, exit 2, nothing sent: the drill's guard (P8-C), reused. What cannot be shown to be loopback is not."""
    monkeypatch.setenv("CHAT_TOKEN", SECRET)
    code, out, err = run_probe([url])
    assert code == 2 and out == "" and refusal_line(err) == drill.CLEARTEXT_REFUSAL
    assert SECRET not in err and "example" not in err and "10.0.0.5" not in err and "internal" not in err


def test_the_refusal_to_send_a_token_in_the_clear_holds_for_a_token_file_too(tmp_path, offline):
    path = write_token_file(tmp_path, TOKEN + "\n")
    code, out, err = run_probe(["http://example.org:18080", "--token-file", str(path)])
    assert code == 2 and out == "" and refusal_line(err) == drill.CLEARTEXT_REFUSAL and TOKEN not in err


def test_a_refused_token_file_says_why_without_naming_the_file_or_what_is_in_it(tmp_path, offline):
    wide = write_token_file(tmp_path, SECRET, 0o644, "wide.token")
    code, out, err = run_probe(["http://127.0.0.1:18080", "--token-file", str(wide)])
    assert code == 2 and out == "" and "0600" in refusal_line(err)
    assert SECRET not in err and "wide.token" not in err and str(tmp_path) not in err
    code, out, err = run_probe(["http://127.0.0.1:18080", "--token-file", str(tmp_path / "nope.token")])
    assert code == 2 and refusal_line(err) and "nope.token" not in err


def test_a_chat_token_that_is_not_visible_ascii_is_refused_without_being_quoted(monkeypatch, offline):
    for bad in ("has a space", "tokén", "tab\there", SECRET + " " + SECRET):
        monkeypatch.setenv("CHAT_TOKEN", bad)
        code, out, err = run_probe(["http://127.0.0.1:18080"])
        assert code == 2 and out == "" and "CHAT_TOKEN" in refusal_line(err)
        assert SECRET not in err and bad not in err


@pytest.mark.parametrize("argv", [[], ["--rows", "0-4"], ["--timeout", "5"]])
def test_a_missing_url_is_a_usage_refusal_in_one_line(argv, offline):
    code, out, err = run_probe(argv)
    assert code == 2 and out == ""
    assert "URL" in refusal_line(err)


@pytest.mark.parametrize("argv", [["--help"], ["-h"], ["http://127.0.0.1:18080", "--help"]])
def test_help_goes_to_stdout_with_exit_0_needs_no_url_and_connects_to_nothing(argv, offline):
    code, out, err = run_probe(argv)
    assert code == 0 and err == "" and out.startswith("usage: ")
    for word in ("--token-file", "CHAT_TOKEN", "8000", "--rows", "--eos-tokens", "--timeout", "RESULT"):
        assert word in out, word


@pytest.mark.parametrize("flag, value", [("--eos-tokens", "15"), ("--eos-tokens", "201"), ("--eos-tokens", "many"), ("--eos-tokens", "-1"),
                                         ("--timeout", "0"), ("--timeout", "-5"), ("--timeout", "soon"), ("--timeout", "nan"),
                                         ("--timeout", "86401")])
def test_a_number_out_of_range_is_refused_in_one_line_before_anything_is_sent(flag, value, offline):
    code, out, err = run_probe(["http://127.0.0.1:18080", flag, value])
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert flag in line
    if not value.lstrip("-").isdigit():
        assert value not in line, "a refusal does not quote the value it was given"


# ---- the reader: what is kept of a stream (R7) -------------------------------------------------------------------------------------

def sse(frames):
    """(event, data) pairs as the server writes them: `event: x`, then `data:` and the JSON on one line, then a blank line."""
    return "".join("event: {}\ndata: {}\n\n".format(event, json.dumps(data, separators=(",", ":"), ensure_ascii=False)) for event, data in frames)


def named(value=0):
    return {name: value for name in CHEXBERT_14}


def stage_data(frames, stage):
    """The data of a stage's stage_end in a list of (event, data) frames."""
    return next(data for event, data in frames if event == "stage_end" and data.get("stage") == stage)


def private_turn(report="SECRET-report one. SECRET-report two.", stopped="budget", tokens=100, neighbor_rows=(10, 158, 63, 74), matches=3,
                 agree=(11, 9, 12, 10), live=True, skip=None, eos=False, final="done", own=(28, 40, 27)):
    """The frames of one private turn as the server sends them, the stages in order, each with its detail, and a SECRET marker in every text
    that is MIMIC-derived or could be: the generated report and its snapshots, a matched report, the reference report the turn was given
    (options.reference), a study id, an image url, the sha, the file name, a label name, and every string of a warning frame (its code and its
    message), which the probe has no branch for and so never reads. `skip` maps a stage to the reason it ended skipped (its normal end is then
    not sent); `own` is the own report's (rank, of, rank_dedup)."""
    skip, seq, out = skip or {}, iter(range(1, 1000)), []

    def add(event, **data):
        out.append((event, dict(data, seq=next(seq))))

    def stage(name, ms, detail):
        if name in skip:
            add("stage_end", stage=name, skipped=skip[name])
        else:
            add("stage_start", stage=name, index=probe.STAGES.index(name))
            add("stage_end", stage=name, ms=ms, detail=detail)

    neighbors = [{"rank": i + 1, "similarity": 0.9 - i / 10, "gallery_row": row, "image_url": "/v1/gallery/images/{}".format(row),
                  "study_id": "SECRET-study-{}".format(i), "txt_row": 1000 + i, "labels": named()} for i, row in enumerate(neighbor_rows)]
    reports = [{"rank": i + 1, "similarity": 0.8, "group": i, "group_size": 3, "txt_row": i, "report": "SECRET-matched Findings: {}".format(i),
                "labels": named()} for i in range(matches)]
    rank = {"rank": own[0], "of": own[1], "rank_dedup": own[2], "n_tied": 0, "hit_at_10": False, "protocol": "SECRET-protocol"}
    add("message_start", message_id="m_assistant0000000000000", user_message_id="m_user0000000000000000", session_id="s_session",
        mode="private", model={"name": "SECRET-model", "eos_trained": eos, "checkpoint": "/SECRET/path.ckpt"},
        options={"model": "SECRET-model", "test_row": 0, "reference": "SECRET-reference Findings: the heart is normal."},
        image={"sha256": "SECRET-sha", "filename": "SECRET.png", "source": "test_split", "urls": {"thumb": "/v1/messages/m_user/image?variant=thumb"}})
    stage("preprocess", 3.24, {"test_row": 0, "source": "test_split", "image_sha256": "SECRET-sha"})
    stage("encode", 120.46, {"patch_grid": [197, 768], "device": "cpu"})
    stage("retrieve", 5.06, {"image_neighbors": neighbors, "report_matches": reports, "true_report_rank": rank,
                             "gallery": {"build_id": "SECRET-build", "images": 191462}})
    add("stage_start", stage="generate", index=3)
    add("content_block_start", index=0, content_block={"type": "report", "text": ""})
    add("content_block_delta", index=0, delta={"type": "beam_snapshot", "step": 1, "text": "SECRET-snapshot"})
    add("content_block_stop", index=0)
    add("stage_end", stage="generate", ms=3900.04, detail={"decode": "beam", "beam_size": 3, "tokens": tokens, "stopped": stopped,
                                                           "drift_note": "SECRET-note", "device": "cpu"})
    stage("label", 80.0, {"chexbert_14": named(), "positives": ["SECRET-label"], "neighbor_agreement": [
        {"rank": i + 1, "agree": n, "of": 14, "both_positive": ["SECRET-label"], "neighbor_only": [], "generated_only": []}
        for i, n in enumerate(agree)]})
    add("warning", code="SECRET-warning-code", message="SECRET-warning-message: the reference Findings are not used.")   # where the app sends its own
    stage("score", 90.0, {"rouge_l": 0.2, "reference_source": "test_split", "reference_chexbert_14": named(), "published": {
        "model_report": "SECRET-published line.", "floor_report": "SECRET-floor line.", "live_equals_published": live}})
    add("message_stop", message_id="m_assistant0000000000000", status=final, total_ms=4123.44, report=report, display_report=report,
        truncated_mid_sentence=False, disclaimer="Research prototype; not for clinical use.")
    return out


def read(text, size=None, started=10.0, ticks=(10.25, 14.75)):
    """read_into over `text` in chunks of `size` characters, with a clock that reads `ticks` in turn. -> (Facts, Handles)."""
    chunks = [text] if size is None else [text[i:i + size] for i in range(0, len(text), size)]
    times = iter(ticks)
    facts, handles = probe.Facts(), probe.Handles()
    probe.read_into(facts, handles, iter(chunks), started, lambda: next(times))
    return facts, handles


GOOD_FACTS = dict(
    final="done", seq_contiguous=True, first_byte_ms=250.0, total_ms=4750.0, server_ms=4123.4,
    stages_ms={"preprocess": 3.2, "encode": 120.5, "retrieve": 5.1, "generate": 3900.0, "label": 80.0, "score": 90.0},
    eos_trained=False, stopped="budget", tokens=100, neighbors=4, matches=3, own_rank=28, own_rank_dedup=27, own_of=40,
    agreement=[11, 9, 12, 10], agreement_pending=False, live_equals_published=True, words=4, complete_sentence=True,
    last_sentence_terminated=True)


def test_a_private_turn_is_read_down_to_numbers_booleans_and_fixed_words():
    facts, handles = read(sse(private_turn()))
    assert facts == probe.Facts(**GOOD_FACTS)
    assert handles == probe.Handles(message_id="m_user0000000000000000", neighbor_rows=[10, 158, 63, 74])


@pytest.mark.parametrize("crlf", [False, True])
@pytest.mark.parametrize("size", [1, 7, 64, 1000])
def test_the_reading_does_not_depend_on_where_the_chunks_are_cut_or_on_the_line_ends(size, crlf):
    text = sse(private_turn())
    facts, handles = read(text.replace("\n", "\r\n") if crlf else text, size)
    assert facts == probe.Facts(**GOOD_FACTS) and handles.neighbor_rows == [10, 158, 63, 74]


def test_first_byte_is_the_first_chunk_and_total_is_the_end_of_the_stream_both_from_the_request_start():
    facts, _ = read(sse(private_turn()), size=40, started=100.0, ticks=(100.0123, 104.5))
    assert (facts.first_byte_ms, facts.total_ms) == (12.3, 4500.0)
    empty, times = probe.Facts(), iter([7.5])
    probe.read_into(empty, probe.Handles(), iter([]), 5.0, lambda: next(times))
    assert (empty.first_byte_ms, empty.total_ms, empty.final, empty.seq_contiguous) == (None, 2500.0, None, None)


def secreted(value):
    """`value` with every string in it made a SECRET (a number, a bool and a None stay as they are): a stage name, a stop word, a text."""
    if isinstance(value, dict):
        return {k: secreted(v) for k, v in value.items()}
    if isinstance(value, list):
        return [secreted(v) for v in value]
    return "SECRET-{}".format(value) if isinstance(value, str) else value


def test_every_string_the_server_sends_is_dropped_whatever_it_says_and_only_the_fixed_words_get_through():
    """Every string of every frame is a SECRET: the stage names, the stop word, the skip reason, the statuses, the error kind, the code and the
    message of a warning. A stage the probe does not know is not read; a stop word, a status or an error kind that is not one of the fixed words
    is `unknown`, and a skip reason is `other`. Nothing of a SECRET reaches the facts, which are all the probe prints of a turn."""
    frames = [(event, secreted(data)) for event, data in private_turn()]
    frames.append(("error", {"seq": len(frames) + 1, "type": "error", "error": {"type": "SECRET-kind", "message": "SECRET-message"}}))
    facts, handles = read(sse(frames))
    assert "SECRET" not in json.dumps(asdict(facts)), asdict(facts)
    assert facts.stages_ms == {} and facts.skipped == {} and facts.stopped is None and facts.neighbors is None, "no stage it does not know is read"
    assert facts.final == "unknown" and facts.stream_error == "unknown"
    turn = private_turn(stopped="SECRET-stop")
    next(data for event, data in turn if event == "message_stop")["status"] = "SECRET-status"
    facts, _ = read(sse(turn))
    assert (facts.final, facts.stopped) == ("unknown", "unknown") and "SECRET" not in json.dumps(asdict(facts))
    skipped, _ = read(sse(private_turn(skip={"label": "SECRET-reason", "score": "no_reference"})))
    assert skipped.skipped == {"label": "other", "score": "no_reference"} and "SECRET" not in json.dumps(asdict(skipped))


def test_a_stage_that_ended_skipped_has_no_numbers_and_says_why_in_a_fixed_word():
    facts, handles = read(sse(private_turn(skip={"retrieve": "gallery_unavailable", "label": "labeler_unavailable", "score": "no_reference"})))
    assert facts.skipped == {"retrieve": "gallery_unavailable", "label": "labeler_unavailable", "score": "no_reference"}
    assert set(facts.stages_ms) == {"preprocess", "encode", "generate"}
    assert (facts.neighbors, facts.matches, facts.own_rank, facts.agreement, facts.live_equals_published) == (None, None, None, [], None)
    assert facts.agreement_pending is None and handles.neighbor_rows == [] and facts.final == "done"


def test_the_stages_and_skip_reasons_it_knows_are_those_the_pipeline_has():
    """app/pipeline.py ends a stage skipped with a reason written in the code: the probe's list holds no other and misses none."""
    source = (REPO_ROOT / "app" / "pipeline.py").read_text(encoding="utf-8")
    written = set(re.findall(r'self\._skip\(turn, "[a-z]+", "([a-z_]+)"\)', source)) | set(re.findall(r'reason = "([a-z_]+)"', source))
    written |= set(re.findall(r'_Skipped\("([a-z_]+)"\)', source))
    assert written == set(probe.SKIP_REASONS), (sorted(written), sorted(probe.SKIP_REASONS))
    assert set(probe.STOPPED) == {"budget", "repeat", "eos"} and set(probe.ERROR_KINDS) == {
        "validation_error", "model_error", "overloaded_error", "server_restart", "store_failed"}


@pytest.mark.parametrize("kind", ["validation_error", "model_error", "overloaded_error", "server_restart", "store_failed"])
def test_an_error_event_keeps_its_kind_when_it_is_one_the_server_has_and_never_its_message(kind):
    frames = [("message_start", {"seq": 1, "user_message_id": "m_u", "model": {"eos_trained": False}}),
              ("error", {"seq": 2, "type": "error", "error": {"type": kind, "message": "SECRET Internal error"}}),
              ("message_stop", {"seq": 3, "status": "error", "total_ms": 5.0, "report": None, "display_report": None})]
    facts, _ = read(sse(frames))
    assert (facts.stream_error, facts.final, facts.words, facts.server_ms) == (kind, "error", None, 5.0) and facts.seq_contiguous is True
    assert "SECRET" not in json.dumps(asdict(facts))


def test_a_malformed_frame_a_wrong_type_a_bad_number_or_a_bad_id_is_dropped_and_the_rest_is_still_read():
    turn = private_turn()
    text = sse(turn[:3]) + "event: stage_end\ndata: {not json}\n\nevent: stage_end\ndata: [1, 2]\n\nevent: stage_end\ndata: \"text\"\n\n"
    text += sse([
        ("stage_end", {"seq": 9, "stage": "generate", "ms": "12", "detail": {"tokens": True, "stopped": ["budget"]}}),
        ("stage_end", {"seq": 10, "stage": "retrieve", "ms": float("nan"), "detail": {
            "image_neighbors": "SECRET", "report_matches": 3, "true_report_rank": {"rank": True, "of": -1, "rank_dedup": 2.5}}}),
        ("stage_end", {"seq": 11, "stage": "label", "ms": float("inf"), "detail": {"neighbor_agreement": [
            {"agree": 15, "of": 14}, {"agree": -1, "of": 14}, {"agree": True, "of": 14}, {"agree": 7, "of": 13}, "SECRET", {"agree": 14, "of": 14}]}}),
        ("stage_end", {"seq": 12, "stage": "score", "detail": {"published": {"live_equals_published": "yes"}}}),
        ("message_stop", {"seq": 13, "status": "done", "total_ms": -5, "report": 12})])
    facts, _ = read(text)
    assert facts.stages_ms == {"preprocess": 3.2}, "a string, a NaN and an infinity are no milliseconds"
    assert (facts.tokens, facts.neighbors, facts.matches, facts.own_rank, facts.own_rank_dedup, facts.own_of) == (None,) * 6
    assert facts.stopped == "unknown", "a stop word that is no fixed word"
    assert facts.agreement == [14] and facts.live_equals_published is None and facts.server_ms is None and facts.words is None
    assert facts.seq_contiguous is False and facts.final == "done"


def with_ms(value):
    """A private turn whose retrieve and generate stages, and its message_stop, carry `value` as their milliseconds."""
    turn = private_turn()
    for stage in ("retrieve", "generate"):
        stage_data(turn, stage)["ms"] = value
    next(data for event, data in turn if event == "message_stop")["total_ms"] = value
    return turn


WITHOUT_THOSE_MS = dict(GOOD_FACTS, stages_ms={"preprocess": 3.2, "encode": 120.5, "label": 80.0, "score": 90.0}, server_ms=None)
NO_MS = {"a 400-digit int": 10 ** 400, "minus a 400-digit int": -(10 ** 400), "a trillion": 10 ** 12, "a trillion as a float": 1e12, "1e300": 1e300,
         "infinity": float("inf"), "minus infinity": float("-inf"), "NaN": float("nan"), "negative": -1, "negative float": -0.5,
         "True": True, "False": False, "a string": "12", "null": None, "a list": [5], "an object": {"ms": 5}}


@pytest.mark.parametrize("value", list(NO_MS.values()), ids=list(NO_MS))
def test_a_number_that_is_no_milliseconds_is_dropped_and_the_rest_of_its_frame_is_still_read(value):
    """A number is taken as milliseconds only when it is a plain int or float from 0 to below the limit. A 400-digit integer is JSON like any
    other, and `math.isfinite` of one raises OverflowError: it must be compared with the limit and dropped, never converted. The stages and the
    message_stop that carried it still give every other fact they hold (the tokens, the neighbours, the stop word, the comparison)."""
    facts, handles = read(sse(with_ms(value)))
    assert facts == probe.Facts(**WITHOUT_THOSE_MS)
    assert handles.neighbor_rows == [10, 158, 63, 74]


@pytest.mark.parametrize("value, kept", [(0, 0.0), (7, 7.0), (3.24, 3.2), (0.04, 0.0), (86_400_000, 86400000.0), (10 ** 12 - 1, 999999999999.0)])
def test_milliseconds_are_any_plain_number_from_0_to_below_the_limit_to_a_tenth(value, kept):
    assert probe.NUMBER_LIMIT == 10 ** 12, "a trillion milliseconds is 31 years: no turn is that long, and no count is that big"
    facts, _ = read(sse(with_ms(value)))
    assert (facts.server_ms, facts.stages_ms["generate"], facts.stages_ms["retrieve"]) == (kept, kept, kept)


def test_a_count_of_400_digits_is_dropped_as_milliseconds_are_so_that_no_line_meets_a_number_it_cannot_average():
    turn = private_turn()
    stage_data(turn, "generate")["detail"]["tokens"] = 10 ** 400
    stage_data(turn, "retrieve")["detail"]["true_report_rank"].update(rank=10 ** 400, rank_dedup=10 ** 12)
    facts, _ = read(sse(turn))
    assert (facts.tokens, facts.own_rank, facts.own_rank_dedup, facts.own_of) == (None, None, None, 40), "the rank's `of` is a count and stays"
    assert facts.final == "done" and facts.stopped == "budget" and facts.neighbors == 4


def test_ids_that_could_not_stand_in_a_path_are_not_kept_and_a_row_is_a_whole_number_from_0():
    for bad in ("", "../x", "m_a/b", "m a", "m?x=1", "x" * 65, 12, None, ["m_x"]):
        frames = [("message_start", {"seq": 1, "user_message_id": bad, "message_id": bad, "model": {}})]
        assert read(sse(frames))[1].message_id is None, bad
    frames = [("message_start", {"seq": 1, "user_message_id": 7, "message_id": "m_fallback", "model": {}})]
    assert read(sse(frames))[1].message_id == "m_fallback", "the assistant's id stands in for a bad user id: either names the turn's image"
    rows = [3, 0, -1, True, 2.0, "4", None, 10**6]
    turn = private_turn(neighbor_rows=(0, 1))
    stage_data(turn, "retrieve")["detail"]["image_neighbors"] = [{"rank": i, "gallery_row": row} for i, row in enumerate(rows)]
    facts, handles = read(sse(turn))
    assert handles.neighbor_rows == [3, 0, 10**6] and facts.neighbors == len(rows), "the count is of the neighbours sent, whatever their rows"


@pytest.mark.parametrize("seqs, contiguous", [((1, 2, 3, 4), True), ((1, 2, 4), False), ((2, 3), False), ((1, 1, 2), False), ((1, None, 3), False)])
def test_seq_is_contiguous_from_1_or_it_is_not(seqs, contiguous):
    frames = [("content_block_delta", {} if s is None else {"seq": s}) for s in seqs]
    assert read(sse(frames))[0].seq_contiguous is contiguous


def test_frames_are_split_on_blank_lines_in_any_chunking_and_comments_are_no_frames():
    text = (": ping\n\n"
            'event: message_start\r\ndata: {"seq":1}\r\n\r\n'
            "event: a\ndata: line one\ndata: line two\n\n"
            ": ping\n\n"
            "data: no event name\n\n"
            "event: only an event\n\n")
    for size in (1, 5, 64, len(text)):
        got = list(probe.frames(iter([text[i:i + size] for i in range(0, len(text), size)])))
        assert got == [("message_start", '{"seq":1}'), ("a", "line one\nline two"), ("message", "no event name")], size


# ---- the sentence rule: the app's own, restated ------------------------------------------------------------------------------------

TOKENS = ["normal.", "normal", "?", "!", ".", "...", "1.", "12.", "J.", "j.", "Dr.", "dr.", "no.", "4.5", "cm.)", "e.g.", "i.e.", "p.m.", "end.\"",
          "end.)", "end?!", "end;", "Findings:", "___.", "a.m.", "St.", "etc.", "al.", "effusion.,", ""]
REPORTS = ["", "Findings: The heart is normal. The lungs are clear.", "The heart is normal. The lungs", "No acute disease",
           "Findings: Impression:", "1. Interval change. 2. No effusion.", "Dr. Smith saw it. J. Smith agreed", "Heart size 4.5 cm. Normal.",
           "no. no. no.", "...", "Done?", "Findings: no acute cardiopulmonary process.", "The heart is normal. The lungs are clear. No pleural"]


def test_the_sentence_rule_is_scripts_repair_generations_own():
    assert probe.ABBREVIATIONS == rg.ABBREVIATIONS and (probe.TERMINALS, probe.TRAILING) == (rg._TERMINALS, rg._TRAILING)
    for token in TOKENS:
        assert probe.is_sentence_end(token) is rg.is_sentence_end(token), repr(token)


def test_words_complete_sentence_and_terminated_end_follow_the_apps_split_of_a_report():
    for text in REPORTS:
        sentences = rg.split_sentences(text)
        words, complete, terminated = probe.sentence_facts(text)
        assert words == len(text.split()), text
        assert complete is any(_finished(s) for s in sentences), text
        assert terminated is (bool(sentences) and _finished(sentences[-1])), text
    assert probe.sentence_facts("") == (0, False, False)
    assert probe.sentence_facts("The heart is normal. The lungs") == (6, True, False)
    assert probe.sentence_facts("The heart is normal. The lungs are clear.") == (8, True, True)
    assert probe.sentence_facts("The heart is normal The lungs are clear") == (8, False, False)


# ---- the judging: passes written by hand, no server --------------------------------------------------------------------------------

def facts_of(**changes):
    """A turn that went as it should, with `changes` on top."""
    base = dict(status=200, final="done", seq_contiguous=True, first_byte_ms=20.0, total_ms=900.0, server_ms=880.0, stages_ms={"generate": 800.0},
                eos_trained=False, stopped="budget", tokens=100, neighbors=4, matches=3, own_rank=5, own_rank_dedup=4, own_of=40,
                agreement=[10, 11, 12, 13], live_equals_published=True, words=15, complete_sentence=True, last_sentence_terminated=True)
    base.update(changes)
    return probe.Facts(**base)


def served_file(size=4096):
    return probe.Fetched(200, size)


def features_of(neighbors=4, **changes):
    base = dict(thumb=served_file(), model_input=served_file(), gallery=[served_file() for _ in range(neighbors)])
    base.update(changes)
    return probe.Features(**base)


def runs_of(rows=(0, 1)):
    """The three passes as they go when all is well, for `rows`: pass 1's turns with their image checks; pass 2's at the two budgets, the stop
    firing at 12 tokens and 100 ms of generating against pass 1's 100 tokens and 800 ms; pass 3's EOS turns and the picture's, ended by EOS."""
    one = probe.PassRun(1, "gate", [probe.Turn(r, facts_of(), features_of()) for r in rows], len(rows), True)
    two = probe.PassRun(2, "stop", [probe.Turn(r, facts_of(stopped="repeat", tokens=12, stages_ms={"generate": 100.0}, total_ms=300.0, first_byte_ms=30.0,
                                                           live_equals_published=None), None, b) for r in rows for b in probe.STOP_BUDGETS],
                        2 * len(rows), True)
    eos = dict(stopped="eos", tokens=70, words=44, eos_trained=True, live_equals_published=None, total_ms=5000.0, first_byte_ms=500.0)
    three = probe.PassRun(3, "eos", [probe.Turn(r, facts_of(**eos), None, 200) for r in rows] + [probe.Turn(None, facts_of(own_rank=None, **eos), None, 200)],
                          len(rows) + 1, True)
    return [one, two, three]


def test_a_gate_whose_three_passes_went_as_they_should_holds_and_its_numbers_are_pass_1s():
    gate = probe.gate_payload(runs_of(), None)
    assert gate == {"published_equal": True, "eos_stops": True, "features_ok": True, "first_byte_ms_max": 20.0, "turn_ms_max": 900.0,
                    "eos_turn_ms_max": 5000.0, "sessions_deleted": True, "ok": True}, \
        "pass 2 and pass 3 are slower and later: the first two maxima are of the gate's own pass, the third is the EOS pass's"


def test_the_timing_maxima_are_the_slowest_of_pass_1_and_none_when_it_has_none():
    runs = runs_of()
    runs[0].turns[1].facts.first_byte_ms, runs[0].turns[1].facts.total_ms = 310.7, 4500.2
    runs[0].turns[0].facts.total_ms = None
    gate = probe.gate_payload(runs, None)
    assert (gate["first_byte_ms_max"], gate["turn_ms_max"]) == (310.7, 4500.2)
    for turn in runs[0].turns:
        turn.facts.first_byte_ms = turn.facts.total_ms = None
    gate = probe.gate_payload(runs, None)
    assert (gate["first_byte_ms_max"], gate["turn_ms_max"]) == (None, None)


def test_the_slowest_eos_turn_is_in_the_gate_over_the_rows_and_the_upload_and_it_is_not_judged():
    """`eos_turn_ms_max` is the default model's turn time for the reader to set against a target: a number, or None when no EOS turn has one. It
    is no clause of the gate: whatever it says, and whether or not it is there, `ok` is what the passes made it."""
    runs = runs_of(rows=(2, 5))
    for turn, ms in zip(runs[2].turns, (5120.7, 6100.1, 5300.0)):   # two rows and the upload: the slowest is a row
        turn.facts.total_ms = ms
    gate = probe.gate_payload(runs, None)
    assert gate["eos_turn_ms_max"] == 6100.1 and gate["ok"] is True
    runs[2].turns[-1].facts.total_ms = 9.0e9   # the upload is the slowest: it counts as the rows do
    gate = probe.gate_payload(runs, None)
    assert gate["eos_turn_ms_max"] == 9.0e9 and gate["ok"] is True, "however slow: it is not judged"
    for turn in runs[2].turns:
        turn.facts.total_ms = None
    gate = probe.gate_payload(runs, None)
    assert gate["eos_turn_ms_max"] is None and gate["ok"] is True, "no number: None, and the gate is as it was"
    assert probe.gate_payload(runs[:2], None)["eos_turn_ms_max"] is None, "no EOS pass at all"
    runs = runs_of()
    runs[0].turns[0].facts.total_ms = 99999.0
    assert probe.gate_payload(runs, None)["eos_turn_ms_max"] == 5000.0, "pass 1's slowest turn is not pass 3's"


def first_turn(run, **changes):
    for name, value in changes.items():
        setattr(run.turns[0].facts, name, value)


GATE_CASES = [   # (what went wrong, how, the clauses that fail beside ok)
    ("a row differs from its published line", lambda r: first_turn(r[0], live_equals_published=False), {"published_equal"}),
    ("a row has no comparison", lambda r: first_turn(r[0], live_equals_published=None), {"published_equal"}),
    ("a row never ran", lambda r: r[0].turns.pop(), {"published_equal", "features_ok"}),
    ("the thumbnail is not served", lambda r: setattr(r[0].turns[0].features, "thumb", probe.Fetched(404, 120)), {"features_ok"}),
    ("the model input is empty", lambda r: setattr(r[0].turns[0].features, "model_input", probe.Fetched(200, 0)), {"features_ok"}),
    ("an image fetch failed", lambda r: setattr(r[0].turns[0].features, "thumb", probe.Fetched(None, None, "ReadTimeout")), {"features_ok"}),
    ("a gallery image is not served", lambda r: r[0].turns[0].features.gallery.__setitem__(2, probe.Fetched(500, 80)), {"features_ok"}),
    ("a neighbour's image was not asked for", lambda r: r[0].turns[0].features.gallery.pop(), {"features_ok"}),
    ("a row has no image checks", lambda r: setattr(r[0].turns[0], "features", None), {"features_ok"}),
    ("fewer neighbours than asked", lambda r: first_turn(r[0], neighbors=3), {"features_ok"}),
    ("fewer matching reports than asked", lambda r: first_turn(r[0], matches=2), {"features_ok"}),
    ("no own-report rank", lambda r: first_turn(r[0], own_rank=None), {"features_ok"}),
    ("a rank beyond the test split", lambda r: first_turn(r[0], own_rank=41), {"features_ok"}),
    ("a rank of 0", lambda r: first_turn(r[0], own_rank=0), {"features_ok"}),
    ("a label agreement is missing", lambda r: first_turn(r[0], agreement=[10, 11, 12]), {"features_ok"}),
    ("the labels are skipped", lambda r: first_turn(r[0], agreement=[], skipped={"label": "labeler_unavailable"}), {"features_ok"}),
    ("a turn's seq has a gap", lambda r: first_turn(r[0], seq_contiguous=False), {"features_ok"}),
    ("a turn ended in error", lambda r: first_turn(r[0], final="error"), {"features_ok"}),
    ("a turn was refused", lambda r: first_turn(r[0], status=422, final=None, refusal="validation_error"), {"features_ok"}),
    ("an EOS turn ran to its budget", lambda r: first_turn(r[2], stopped="budget"), {"eos_stops"}),
    ("the picture's turn ran to its budget", lambda r: setattr(r[2].turns[-1].facts, "stopped", "budget"), {"eos_stops"}),
    ("the picture's turn never ran", lambda r: r[2].turns.pop(), {"eos_stops", "features_ok"}),
    ("an EOS turn found no similar X-rays", lambda r: first_turn(r[2], neighbors=0, matches=0), {"features_ok"}),
    ("an EOS turn's retrieval was skipped", lambda r: first_turn(r[2], neighbors=None, matches=None, skipped={"retrieve": "gallery_unavailable"}),
     {"features_ok"}),
]


@pytest.mark.parametrize("what, wreck, failing", GATE_CASES, ids=[c[0] for c in GATE_CASES])
def test_each_clause_of_the_gate_fails_on_its_own_and_takes_the_exit_status_with_it(what, wreck, failing):
    runs = runs_of()
    wreck(runs)
    gate = probe.gate_payload(runs, None)
    clauses = {name: gate[name] for name in ("published_equal", "eos_stops", "features_ok")}
    assert clauses == {name: name not in failing for name in clauses}, what
    assert gate["ok"] is False, what


OTHER_CASES = [   # what fails the exit status without failing one of the three clauses: the run itself
    ("a pass 2 turn failed", lambda r: first_turn(r[1], status=500, final=None)),
    ("a pass 2 turn ended in error", lambda r: first_turn(r[1], final="error", stream_error="model_error")),
    ("a pass 2 turn timed out", lambda r: first_turn(r[1], error="timeout", final=None)),
    ("a pass 2 turn is missing", lambda r: r[1].turns.pop()),
    ("a session was not deleted", lambda r: setattr(r[1], "session_deleted", False)),
    ("a session was never made", lambda r: setattr(r[2], "session_deleted", None)),
    ("a pass ended early", lambda r: setattr(r[1], "error", "timeout")),
    ("a pass is missing", lambda r: r.pop()),
    ("no pass ran", lambda r: r.clear()),
]


@pytest.mark.parametrize("what, wreck", OTHER_CASES, ids=[c[0] for c in OTHER_CASES])
def test_a_run_that_did_not_finish_or_left_a_session_behind_does_not_hold_whatever_the_clauses_say(what, wreck):
    runs = runs_of()
    wreck(runs)
    gate = probe.gate_payload(runs, None)
    assert gate["ok"] is False, what
    if what in ("a pass 2 turn failed", "a pass 2 turn ended in error", "a pass 2 turn timed out", "a pass 2 turn is missing"):
        assert (gate["published_equal"], gate["eos_stops"], gate["features_ok"]) == (True, True, True), "the stop measurement is no clause"
    if what == "a pass ended early":
        assert gate["error"] == "timeout"


def test_an_error_of_the_run_itself_is_the_gates_and_names_a_class_or_a_code_never_a_text():
    gate = probe.gate_payload(runs_of(), "interrupted")
    assert gate["ok"] is False and gate["error"] == "interrupted"
    runs = runs_of()
    runs[0].error = "session_create_http_401"
    assert probe.gate_payload(runs, None)["error"] == "session_create_http_401"
    assert "error" not in probe.gate_payload(runs_of(), None)


def test_pass_1_prints_each_row_with_its_timings_stages_counts_rank_agreement_and_image_checks():
    runs = runs_of(rows=(3, 7))
    runs[0].turns[1].facts.skipped = {"score": "no_reference"}
    runs[0].turns[1].features = features_of(model_input=probe.Fetched(404, 90), gallery=[served_file(1), probe.Fetched(None, None, "ReadTimeout")])
    row = {"row": 3, "status": 200, "final": "done", "seq_contiguous": True, "first_byte_ms": 20.0, "total_ms": 900.0, "server_ms": 880.0,
           "stages_ms": {"generate": 800.0}, "skipped": {}, "live_equals_published": True, "stopped": "budget", "tokens": 100,
           "neighbors": 4, "matches": 3, "own_rank": 5, "own_rank_dedup": 4, "own_of": 40, "agreement": [10, 11, 12, 13],
           "images": {"thumb": [200, 4096], "model_input": [200, 4096]}, "gallery": [[200, 4096]] * 4, "features_ok": True}
    line = probe.pass_payload(runs[0])
    assert line["rows"][0] == row
    assert line["rows"][1] == dict(row, row=7, skipped={"score": "no_reference"}, images={"thumb": [200, 4096], "model_input": [404, 90]},
                                   gallery=[[200, 1], [None, None]], features_ok=False)
    assert {k: v for k, v in line.items() if k != "rows"} == {
        "pass": 1, "name": "gate", "model": "hybrid_150m_m3_rrg", "max_new_tokens": 100, "published_equal": True, "features_ok": False,
        "first_byte_ms_max": 20.0, "turn_ms_max": 900.0, "session_deleted": True, "ran": True}
    assert list(line)[:3] == ["pass", "name", "model"]


def test_a_row_that_was_refused_or_failed_says_so_by_status_and_a_word_or_a_class_and_has_no_numbers_it_never_had():
    runs = runs_of(rows=(0,))
    runs[0].turns[0] = probe.Turn(0, probe.Facts(status=422, refusal="validation_error"), None)
    runs[0].turns.append(probe.Turn(1, probe.Facts(error="ReadTimeout", first_byte_ms=12.0, stages_ms={"preprocess": 3.0}, final=None), None))
    first, second = probe.pass_payload(runs[0])["rows"]
    assert first["status"] == 422 and first["refusal"] == "validation_error" and first["final"] is None and first["features_ok"] is False
    assert first["neighbors"] is None and first["agreement"] == [] and first["images"] == {"thumb": [None, None], "model_input": [None, None]}
    assert first["gallery"] == []
    assert second["error"] == "ReadTimeout" and second["stages_ms"] == {"preprocess": 3.0} and second["first_byte_ms"] == 12.0
    assert "error" not in first and "refusal" not in second and probe.pass_payload(runs[0])["ran"] is False


def test_pass_2_says_per_row_and_budget_whether_the_stop_fired_and_what_it_saved_against_pass_1():
    runs = runs_of(rows=(0, 1))
    runs[1].turns[2].facts.stopped, runs[1].turns[2].facts.tokens = "budget", 100   # row 1 at 100 tokens: the stop did not fire
    runs[1].turns[2].facts.stages_ms, runs[1].turns[2].facts.total_ms = {"generate": 790.0}, 905.0
    fired = {"status": 200, "final": "done", "stopped": "repeat", "fired": True, "tokens": 12, "ms": 100.0, "turn_ms": 300.0, "complete_sentence": True}
    line = probe.pass_payload(runs[1], runs[0])
    assert line["rows"][0] == {"row": 0, "b100": dict(fired, tokens_saved=88, ms_saved=700.0, turn_ms_saved=600.0), "b200": fired}
    assert line["rows"][1] == {"row": 1, "b100": dict(fired, stopped="budget", fired=False, tokens=100, ms=790.0, turn_ms=905.0, tokens_saved=0,
                                                      ms_saved=10.0, turn_ms_saved=-5.0), "b200": fired}
    assert {k: v for k, v in line.items() if k != "rows"} == {
        "pass": 2, "name": "stop", "model": "hybrid_150m_m3_rrg", "budgets": [100, 200], "fired": {"100": 1, "200": 2}, "of": 2,
        "tokens_saved_100": {"total": 88, "mean": 44.0}, "ms_saved_100": {"total": 710.0, "mean": 355.0},
        "turn_ms_saved_100": {"total": 595.0, "mean": 297.5}, "complete_sentence_all": True, "session_deleted": True, "ran": True}


def test_pass_2_says_when_a_report_was_cut_before_its_first_complete_sentence_and_has_no_savings_without_a_baseline():
    runs = runs_of(rows=(0,))
    runs[1].turns[1].facts.complete_sentence = False
    assert probe.pass_payload(runs[1], runs[0])["complete_sentence_all"] is False
    runs[0].turns.clear()
    line = probe.pass_payload(runs[1], runs[0])
    assert line["rows"][0]["b100"]["tokens_saved"] is None and line["rows"][0]["b100"]["ms_saved"] is None
    assert line["tokens_saved_100"] is None and line["ms_saved_100"] is None and line["turn_ms_saved_100"] is None
    runs[1].turns.pop()   # the 200-token turn never ran
    line = probe.pass_payload(runs[1], None)
    assert line["rows"][0]["b200"] is None and line["fired"] == {"100": 1, "200": 0} and line["ran"] is False


def test_pass_3_says_per_row_and_for_the_picture_how_the_eos_model_stopped_and_how_much_it_wrote():
    runs = runs_of(rows=(2, 5))
    runs[2].turns[1].facts.last_sentence_terminated = False
    runs[2].turns[-1].facts.neighbors = 0
    entry = {"status": 200, "final": "done", "first_byte_ms": 500.0, "total_ms": 5000.0, "server_ms": 880.0, "stages_ms": {"generate": 800.0},
             "eos_trained": True, "stopped": "eos", "tokens": 70, "words": 44, "last_sentence_terminated": True, "neighbors": 4, "matches": 3,
             "skipped": {}}
    line = probe.pass_payload(runs[2])
    assert line == {"pass": 3, "name": "eos", "model": "hybrid_150m_m3_rrg_eos", "max_new_tokens": 200,
                    "rows": [dict(entry, row=2), dict(entry, row=5, last_sentence_terminated=False)],
                    "upload": dict(entry, neighbors=0), "eos_stops": True, "counts_ok": False,
                    "first_byte_ms_max": 500.0, "turn_ms_max": 5000.0, "generate_ms_max": 800.0, "session_deleted": True, "ran": True}


TIMINGS = ("first_byte_ms", "total_ms", "server_ms", "stages_ms")   # what pass 1 records of a turn's time, and pass 3 records of each of its turns


def test_pass_3_records_for_each_eos_turn_the_timings_pass_1_records_and_the_slowest_of_them_over_the_rows_and_the_upload():
    runs = runs_of(rows=(2, 5))
    first, second, upload = (t.facts for t in runs[2].turns)
    timed = [(410.2, 5120.7, 5100.3, {"preprocess": 3.0, "generate": 4000.5, "score": 90.0}),
             (380.0, 6100.1, 6080.0, {"generate": 4900.2}),     # the slowest turn and the slowest generate stage: a row
             (760.9, 5300.0, 5250.5, {"generate": 4100.0, "label": 70.0})]   # the slowest first byte: the upload
    for facts, (first_byte, total, server, stages) in zip((first, second, upload), timed):
        facts.first_byte_ms, facts.total_ms, facts.server_ms, facts.stages_ms = first_byte, total, server, stages
    line = probe.pass_payload(runs[2])
    entries = line["rows"] + [line["upload"]]
    for entry, (first_byte, total, server, stages) in zip(entries, timed):
        assert [entry[k] for k in TIMINGS] == [first_byte, total, server, stages]
    assert (line["first_byte_ms_max"], line["turn_ms_max"], line["generate_ms_max"]) == (760.9, 6100.1, 4900.2)
    assert probe.gate_payload(runs, None)["eos_turn_ms_max"] == line["turn_ms_max"] == 6100.1
    one = probe.pass_payload(runs[0])["rows"][0]
    for entry in entries:   # the same four fields as a pass 1 row, under the same names and in the same order
        assert [k for k in entry if k in TIMINGS] == [k for k in one if k in TIMINGS] == list(TIMINGS)


def test_pass_3_has_no_slowest_for_a_timing_none_of_its_turns_has_and_a_turn_that_failed_still_says_what_it_timed():
    runs = runs_of(rows=(2,))
    for turn in runs[2].turns:
        turn.facts.first_byte_ms = turn.facts.total_ms = turn.facts.server_ms = None
        turn.facts.stages_ms = {"encode": 12.5}   # no generate stage
    line = probe.pass_payload(runs[2])
    assert (line["first_byte_ms_max"], line["turn_ms_max"], line["generate_ms_max"]) == (None, None, None)
    assert [line["rows"][0][k] for k in TIMINGS] == [None, None, None, {"encode": 12.5}]
    runs[2].turns[0] = probe.Turn(2, probe.Facts(error="ReadTimeout", first_byte_ms=12.0, stages_ms={"preprocess": 3.0}), None, 200)
    line = probe.pass_payload(runs[2])
    assert line["rows"][0]["error"] == "ReadTimeout" and line["rows"][0]["first_byte_ms"] == 12.0 and line["rows"][0]["stages_ms"] == {"preprocess": 3.0}
    assert (line["first_byte_ms_max"], line["turn_ms_max"], line["generate_ms_max"]) == (12.0, None, None) and line["ran"] is False


def test_no_line_the_probe_prints_holds_a_word_of_what_a_hostile_stream_says():
    """The sentinel at the printing end, with no network: every turn of the three passes is read from a private turn whose texts, its warning's and
    its reference's among them, are SECRETs, and then from one in which every string is. The lines the probe would print for them, the three
    passes' and the gate's, are looked at: nothing of a stream is in them, only the numbers and the fixed words (unknown, other) it came to."""
    for stream, said in ((private_turn(), '"stopped":"budget"'), ([(event, secreted(data)) for event, data in private_turn()], '"final":"unknown"')):
        runs = runs_of()
        for run in runs:
            for turn in run.turns:
                turn.facts = read(sse(stream))[0]
        lines = [probe.result_line(probe.pass_payload(run, runs[0])) for run in runs] + [probe.result_line({"gate": probe.gate_payload(runs, None)})]
        printed = "\n".join(lines)
        assert len(lines) == 4 and said in printed, "the lines say what the stream came to"
        assert "SECRET" not in printed, printed
        for name in CHEXBERT_14:
            assert name not in printed, name


# ---- the requests that follow a turn: images, for their status and length only ------------------------------------------------------

class Answer:
    """What a fake client's stream hands out: a status and a body in three chunks, as a context manager."""

    def __init__(self, status, chunks):
        self.status_code, self.chunks = status, chunks

    def iter_bytes(self):
        yield from self.chunks

    def read(self):
        raise AssertionError("an image's body is counted as it arrives, never read whole")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Recorder:
    """In place of httpx.Client for fetch_features alone: it writes down what was asked and answers with a body of three chunks of ten bytes."""

    def __init__(self, status=200, fail=()):
        self.asked, self.status, self.fail = [], status, set(fail)

    def stream(self, method, path, **kwargs):
        self.asked.append((method, path, kwargs))
        if path in self.fail:
            raise httpx.ReadTimeout("a thing with SECRET text in it")
        return Answer(self.status, [b"x" * 10] * 3)


def test_the_follow_up_requests_ask_for_the_two_image_variants_and_each_neighbours_gallery_image_and_keep_only_status_and_length():
    client = Recorder()
    features = probe.fetch_features(client, probe.Handles("m_user000", [10, 158, 63, 74]))
    assert [(method, path) for method, path, _ in client.asked] == [
        ("GET", "/v1/messages/m_user000/image?variant=thumb"), ("GET", "/v1/messages/m_user000/image?variant=model_input"),
        ("GET", "/v1/gallery/images/10"), ("GET", "/v1/gallery/images/158"), ("GET", "/v1/gallery/images/63"), ("GET", "/v1/gallery/images/74")]
    assert features == probe.Features(probe.Fetched(200, 30), probe.Fetched(200, 30), [probe.Fetched(200, 30)] * 4)
    assert all(kwargs == {} for _, _, kwargs in client.asked)


def test_an_image_request_that_fails_is_a_class_name_and_one_that_is_refused_is_its_status_and_the_length_of_what_came():
    features = probe.fetch_features(Recorder(fail={"/v1/gallery/images/63"}), probe.Handles("m_user000", [10, 63]))
    assert features.gallery == [probe.Fetched(200, 30), probe.Fetched(None, None, "ReadTimeout")] and "SECRET" not in repr(features)
    features = probe.fetch_features(Recorder(status=404), probe.Handles("m_user000", [10]))
    assert features == probe.Features(probe.Fetched(404, 30), probe.Fetched(404, 30), [probe.Fetched(404, 30)])


def test_with_no_id_for_the_turns_message_there_is_no_image_request_and_the_features_say_so():
    client = Recorder()
    features = probe.fetch_features(client, probe.Handles(None, [5]))
    assert [path for _, path, _ in client.asked] == ["/v1/gallery/images/5"]
    assert features.thumb == features.model_input == probe.Fetched(None, None, "no_id") and features.gallery == [probe.Fetched(200, 30)]


def test_a_stream_that_is_still_open_when_the_deadline_passes_is_cut_at_its_next_chunk():
    ticks, got = iter([1.0, 2.0, 3.0]), []
    with pytest.raises(probe.Timeout):
        for chunk in probe.watched(iter(["a", "b", "c", "d"]), 2.5, lambda: next(ticks)):
            got.append(chunk)
    assert got == ["a", "b", "c"], "the chunk that was in hand is read, then the stream is let go"
    assert list(probe.watched(iter(["a", "b"]), 99.0, lambda: 1.0)) == ["a", "b"]


# ---- live: the three passes against a real uvicorn over the tiny engines, the tiny gallery and dumps of synthetic lines ---------------

@dataclass
class Served:
    base: str
    app: Any
    n_test: int


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """One live server for the module, built as the P5-E tests build theirs: tiny engines named as the registry names the real ones (the EOS
    model first, as deployed, then the published one), the tiny gallery opened against their tower, the keyword labeller, and published dumps
    of synthetic lines. A test that wants `live_equals_published` true writes the live reports into the dumps (published_lines)."""
    root = tmp_path_factory.mktemp("realturn")
    build_tiny(root / "gallery")
    with pytest.MonkeyPatch.context() as patch:
        app, n_test = _published_app(root, root / "gallery", patch, (probe.EOS_MODEL, probe.PUBLISHED_MODEL))
    base, stop = start_live_server(app)
    try:
        yield Served(base, app, n_test)
    finally:
        stop()


def paced(script, seconds):
    """`script` with a pause before every step, for a turn that must still be running when a deadline passes."""
    def slow(n, last):
        time.sleep(seconds)
        return script(n, last)
    return slow if seconds else script


@contextmanager
def decoding(served, loop=LOOP, eos_at=15, eos_text=REPORT, step_s=0.0):
    """The tiny engines write what a test says (the scripted decoders of P9-G2). The published one writes `loop`, a word per token, and never
    ends: its turn runs to its budget, and its report is `loop` whole once the budget is above its words. The EOS one writes `eos_text` and
    ends after `eos_at` tokens, or never when that is None."""
    published, eos = served.app.state.engines[probe.PUBLISHED_MODEL], served.app.state.engines[probe.EOS_MODEL]
    ends = lambda n, last: 40.0 if n == eos_at else -1e4   # noqa: E731
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(published, "tokenizer", GrowingText(loop)))
        stack.enter_context(scripted_decoder(published.decoder, paced(noise_script(7, EOS_ID, lambda n, last: -1e4), step_s)))
        stack.enter_context(mock.patch.object(eos, "tokenizer", GrowingText(eos_text)))
        stack.enter_context(mock.patch.object(eos, "eos_token_id", EOS_ID))   # an id the tiny vocabulary has
        stack.enter_context(scripted_decoder(eos.decoder, paced(noise_script(5, EOS_ID, ends), step_s)))
        yield


@contextmanager
def published_lines(served, lines):
    """The server's published dump of the default model, with `lines` (one per test row) in place of its synthetic ones."""
    pipeline = served.app.state.worker.pipeline
    before = pipeline.published
    pipeline.published = PublishedDumps(list(lines), before.floor_hyps)
    try:
        yield
    finally:
        pipeline.published = before


def split_result(out):
    """The RESULT lines of a run: -> (the pass lines, the gate's content)."""
    def strict(constant):   # NaN and Infinity are not JSON: a reader that is strict about it must be able to take every line
        raise AssertionError("not JSON: " + constant)

    lines = out.splitlines()
    assert lines and all(line.startswith("RESULT ") for line in lines), lines
    parsed = [json.loads(line[len("RESULT "):], parse_constant=strict) for line in lines]
    assert set(parsed[-1]) == {"gate"}, parsed[-1]
    return parsed[:-1], parsed[-1]["gate"]


@dataclass
class Ran:
    code: int
    out: str
    err: str
    passes: list
    gate: dict


def probe_rows(served, rows="0", *extra):
    code, out, err = run_probe([served.base, "--rows", rows] + list(extra))
    passes, gate = split_result(out)
    return Ran(code, out, err, passes, gate)


def session_ids(served):
    return {s["id"] for s in httpx.get(served.base + "/v1/sessions").json()["sessions"]}


@pytest.fixture(scope="module")
def full_run(served):
    """The whole probe on rows 0 and 1 of a server whose published dump holds what the published engine writes: every clause holds."""
    with decoding(served), published_lines(served, [LOOP] * served.n_test):
        return probe_rows(served, "0-1")


def test_a_full_run_prints_one_result_line_per_pass_then_the_gate_and_exits_0(full_run):
    assert full_run.code == 0 and full_run.err == ""
    assert [p["pass"] for p in full_run.passes] == [1, 2, 3] and [p["name"] for p in full_run.passes] == ["gate", "stop", "eos"]
    assert full_run.out.count("\n") == 4 and full_run.out.endswith("\n")
    assert all(p["ran"] is True and p["session_deleted"] is True for p in full_run.passes)
    one = full_run.passes[0]
    assert full_run.gate == {"published_equal": True, "eos_stops": True, "features_ok": True, "first_byte_ms_max": one["first_byte_ms_max"],
                             "turn_ms_max": one["turn_ms_max"], "eos_turn_ms_max": full_run.passes[2]["turn_ms_max"], "sessions_deleted": True,
                             "ok": True}


def test_pass_1_runs_each_row_on_the_published_model_under_the_published_protocol_and_checks_what_the_page_would_show(full_run):
    one = full_run.passes[0]
    assert (one["model"], one["max_new_tokens"], [r["row"] for r in one["rows"]]) == ("hybrid_150m_m3_rrg", 100, [0, 1])
    for row in one["rows"]:
        assert (row["status"], row["final"], row["seq_contiguous"]) == (200, "done", True)
        assert row["live_equals_published"] is True and (row["stopped"], row["tokens"]) == ("budget", 100)
        assert (row["neighbors"], row["matches"]) == (4, 3) and row["skipped"] == {}
        assert 1 <= row["own_rank"] <= row["own_of"] == TINY_TEST and 1 <= row["own_rank_dedup"] <= row["own_rank"]
        assert len(row["agreement"]) == 4 and all(0 <= n <= 14 for n in row["agreement"])
        assert list(row["stages_ms"]) == list(probe.STAGES) and all(ms >= 0 for ms in row["stages_ms"].values())
        assert 0 < row["first_byte_ms"] <= row["total_ms"] and 0 < row["server_ms"] <= row["total_ms"]
        assert row["images"]["thumb"][0] == row["images"]["model_input"][0] == 200 and min(row["images"]["thumb"][1], row["images"]["model_input"][1]) > 0
        assert [status for status, _ in row["gallery"]] == [200] * 4 and all(size > 0 for _, size in row["gallery"])
        assert row["features_ok"] is True and "error" not in row and "refusal" not in row
    assert one["published_equal"] is True and one["features_ok"] is True
    assert one["first_byte_ms_max"] == max(r["first_byte_ms"] for r in one["rows"]) and one["turn_ms_max"] == max(r["total_ms"] for r in one["rows"])


def test_pass_2_measures_the_stop_at_100_and_at_200_tokens_against_pass_1(full_run):
    two, one = full_run.passes[1], full_run.passes[0]
    assert (two["model"], two["budgets"], two["of"]) == ("hybrid_150m_m3_rrg", [100, 200], 2)
    for row in two["rows"]:
        for budget in ("b100", "b200"):   # the looping text repeats a sentence at its 12th word: the stop fires there, at either budget
            entry = row[budget]
            assert (entry["status"], entry["final"], entry["stopped"], entry["fired"], entry["tokens"]) == (200, "done", "repeat", True, 12)
            assert entry["complete_sentence"] is True and entry["ms"] >= 0 and entry["turn_ms"] > 0
        assert row["b100"]["tokens_saved"] == 100 - 12
        base = next(r for r in one["rows"] if r["row"] == row["row"])
        assert row["b100"]["ms_saved"] == round(base["stages_ms"]["generate"] - row["b100"]["ms"], 1)
        assert row["b100"]["turn_ms_saved"] == round(base["total_ms"] - row["b100"]["turn_ms"], 1)
        assert "tokens_saved" not in row["b200"]
    assert two["fired"] == {"100": 2, "200": 2} and two["tokens_saved_100"] == {"total": 176, "mean": 88.0} and two["complete_sentence_all"] is True


def test_pass_3_runs_the_eos_model_on_the_same_rows_and_on_a_picture_it_draws_and_says_how_each_stopped(full_run):
    three = full_run.passes[2]
    assert (three["model"], three["max_new_tokens"], [r["row"] for r in three["rows"]]) == ("hybrid_150m_m3_rrg_eos", 200, [0, 1])
    for entry in three["rows"] + [three["upload"]]:   # it ends after 15 tokens, which are the first 15 words of the report it writes
        assert (entry["status"], entry["final"], entry["eos_trained"], entry["stopped"], entry["tokens"]) == (200, "done", True, "eos", 15)
        assert (entry["words"], entry["last_sentence_terminated"], entry["neighbors"], entry["matches"]) == (15, True, 4, 3)
    assert three["eos_stops"] is True and three["counts_ok"] is True and "row" not in three["upload"]


def test_pass_3_records_the_timings_of_every_eos_turn_the_rows_and_the_upload_alike_and_they_are_numbers_only(full_run):
    """The default model's turn time, for the plan's targets (P8-F): the same four fields pass 1 records, on each of the three EOS turns, and the
    slowest of the turns in the pass's line and, as a number that is not judged, in the gate's."""
    one, three, gate = full_run.passes[0], full_run.passes[2], full_run.gate
    entries = three["rows"] + [three["upload"]]
    assert len(entries) == 3, "two test rows and the picture the probe draws"
    for entry in entries:
        assert [k for k in entry if k in TIMINGS] == list(TIMINGS) == [k for k in one["rows"][0] if k in TIMINGS], "pass 1's fields, in its order"
        for key in ("first_byte_ms", "total_ms", "server_ms"):
            assert type(entry[key]) is float and entry[key] > 0, (key, entry[key])
        assert entry["first_byte_ms"] <= entry["total_ms"] and entry["server_ms"] <= entry["total_ms"]
        assert type(entry["stages_ms"]) is dict and "generate" in entry["stages_ms"] and set(entry["stages_ms"]) <= set(probe.STAGES)
        assert all(type(ms) is float and ms >= 0 for ms in entry["stages_ms"].values()), "a stage is a number of ms, never a text"
    assert [list(e["stages_ms"]) for e in three["rows"]] == [list(probe.STAGES)] * 2, "a test row runs all six stages"
    assert list(three["upload"]["stages_ms"]) == [s for s in probe.STAGES if s != "score"] and three["upload"]["skipped"] == {"score": "no_reference"}
    assert three["first_byte_ms_max"] == max(e["first_byte_ms"] for e in entries) and three["turn_ms_max"] == max(e["total_ms"] for e in entries)
    assert three["generate_ms_max"] == max(e["stages_ms"]["generate"] for e in entries) > 0
    assert gate["eos_turn_ms_max"] == three["turn_ms_max"] and gate["ok"] is True


# ---- the gate, failing: a live server that is wrong in one way at a time ----------------------------------------------------------

def test_a_live_report_that_is_not_the_published_line_fails_the_gate_with_exit_1_and_the_rest_of_the_run_still_holds(served):
    with decoding(served):   # the server's own dump: synthetic lines, which no report equals
        ran = probe_rows(served)
    assert ran.code == 1 and ran.err == ""
    assert ran.passes[0]["rows"][0]["live_equals_published"] is False and ran.passes[0]["published_equal"] is False
    assert ran.gate["published_equal"] is False and ran.gate["eos_stops"] is True and ran.gate["features_ok"] is True and ran.gate["ok"] is False


def test_a_server_with_no_published_dump_says_none_for_the_comparison_and_fails_the_gate(served):
    pipeline = served.app.state.worker.pipeline
    before, pipeline.published = pipeline.published, None
    try:
        with decoding(served):
            ran = probe_rows(served)
    finally:
        pipeline.published = before
    assert ran.code == 1 and ran.passes[0]["rows"][0]["live_equals_published"] is None and ran.gate["published_equal"] is False


def test_a_gallery_image_the_server_cannot_serve_fails_the_features_with_exit_1(served):
    gallery = served.app.state.worker.pipeline.gallery
    with decoding(served), published_lines(served, [LOOP] * served.n_test), mock.patch.object(gallery, "image_path", side_effect=IndexError):
        ran = probe_rows(served)
    row = ran.passes[0]["rows"][0]
    assert [status for status, _ in row["gallery"]] == [404] * 4 and row["features_ok"] is False
    assert row["images"]["thumb"][0] == 200, "the other image features were served"
    assert ran.code == 1 and ran.gate["features_ok"] is False and ran.gate["published_equal"] is True and ran.gate["eos_stops"] is True


def test_an_eos_model_that_runs_to_its_budget_fails_the_gate_with_exit_1_and_says_how_it_stopped(served):
    with decoding(served, eos_at=None), published_lines(served, [LOOP] * served.n_test):
        ran = probe_rows(served, "0", "--eos-tokens", "16")
    three = ran.passes[2]
    assert [(e["stopped"], e["tokens"]) for e in three["rows"] + [three["upload"]]] == [("budget", 16)] * 2
    assert three["eos_stops"] is False and three["max_new_tokens"] == 16
    assert ran.code == 1 and ran.gate["eos_stops"] is False and ran.gate["published_equal"] is True and ran.gate["features_ok"] is True


def test_an_eos_report_that_ends_in_the_middle_of_a_sentence_is_said_so_and_does_not_fail_the_gate(served):
    with decoding(served, eos_at=13), published_lines(served, [LOOP] * served.n_test):
        ran = probe_rows(served)
    entry = ran.passes[2]["rows"][0]
    assert (entry["stopped"], entry["words"], entry["last_sentence_terminated"]) == ("eos", 13, False)   # "... Impression: no"
    assert ran.code == 0 and ran.gate["ok"] is True


def test_a_stop_that_never_fires_saves_nothing_and_says_so(served):
    with decoding(served, loop=REPORT), published_lines(served, [REPORT] * served.n_test):   # 27 words, no sentence twice
        ran = probe_rows(served)
    two = ran.passes[1]
    row = two["rows"][0]
    assert (row["b100"]["stopped"], row["b100"]["fired"], row["b100"]["tokens"], row["b100"]["tokens_saved"]) == ("budget", False, 100, 0)
    assert (row["b200"]["stopped"], row["b200"]["fired"], row["b200"]["tokens"]) == ("budget", False, 200)
    assert two["fired"] == {"100": 0, "200": 0} and two["tokens_saved_100"] == {"total": 0, "mean": 0.0} and ran.code == 0


def test_a_picked_row_beyond_the_test_split_is_a_refusal_the_server_makes_and_the_probe_reports_by_status(served):
    with decoding(served):
        ran = probe_rows(served, "{}".format(served.n_test))
    row = ran.passes[0]["rows"][0]
    assert (row["status"], row["refusal"], row["final"], row["features_ok"]) == (422, "validation_error", None, False)
    assert ran.code == 1 and ran.gate["ok"] is False


# ---- one session per pass, deleted whatever happened ------------------------------------------------------------------------------

def test_each_pass_makes_one_session_of_its_own_and_deletes_it(served):
    before = session_ids(served)
    with decoding(served), published_lines(served, [LOOP] * served.n_test):
        with mock.patch.object(Store, "create_session", autospec=True, side_effect=Store.create_session) as created:
            ran = probe_rows(served)
    assert created.call_count == 3, "one per pass"
    assert session_ids(served) == before, "and they are gone"
    assert ran.gate["sessions_deleted"] is True and [p["session_deleted"] for p in ran.passes] == [True] * 3


def test_a_pass_that_fails_inside_still_deletes_its_session_and_the_error_is_a_class_name(served, monkeypatch):
    before = session_ids(served)

    def broken(*args, **kwargs):
        raise RuntimeError("boom, with text that must not be printed")
    monkeypatch.setattr(probe, "fetch_features", broken)
    with decoding(served), published_lines(served, [LOOP] * served.n_test):
        code, out, err = run_probe([served.base, "--rows", "0"])
    passes, gate = split_result(out)
    assert code == 1 and err == "" and gate["ok"] is False and gate["error"] == "internal_RuntimeError" and "boom" not in out
    assert [p["pass"] for p in passes] == [1] and passes[0]["session_deleted"] is True and len(passes[0]["rows"]) == 1, "what was known is printed"
    assert session_ids(served) == before


def test_an_interrupted_probe_still_deletes_its_session_prints_what_it_knew_and_exits_1(served, monkeypatch):
    before = session_ids(served)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(probe, "fetch_features", interrupted)
    with decoding(served), published_lines(served, [LOOP] * served.n_test):
        code, out, err = run_probe([served.base, "--rows", "0-1"])
    passes, gate = split_result(out)
    assert code == 1 and err == "" and gate["error"] == "interrupted" and gate["ok"] is False
    assert [p["pass"] for p in passes] == [1] and passes[0]["error"] == "interrupted" and passes[0]["session_deleted"] is True
    assert len(passes[0]["rows"]) == 1, "the second row never ran"
    assert session_ids(served) == before


def test_a_session_that_cannot_be_deleted_is_reported_and_fails_the_gate_though_the_turns_were_fine(served):
    before = session_ids(served)
    try:
        with decoding(served), published_lines(served, [LOOP] * served.n_test):
            with mock.patch.object(Store, "delete_session", autospec=True, return_value=False):   # the server answers 404
                ran = probe_rows(served)
        assert [p["session_deleted"] for p in ran.passes] == [False] * 3 and ran.gate["sessions_deleted"] is False
        assert ran.gate["published_equal"] is True and ran.gate["eos_stops"] is True and ran.gate["features_ok"] is True
        assert ran.code == 1 and ran.gate["ok"] is False
    finally:
        for sid in session_ids(served) - before:   # nothing the patched server kept may stay for the next test
            httpx.delete(served.base + "/v1/sessions/" + sid)


def test_a_probe_that_runs_out_of_time_deletes_its_session_which_stops_the_turn_it_started(served):
    before = session_ids(served)
    with decoding(served, step_s=0.05):   # 100 steps of 50 ms: a turn of 5 seconds against a deadline of 1.5
        ran = probe_rows(served, "0", "--timeout", "1.5")
        assert session_ids(served) == before
        wait_until(lambda: httpx.get(served.base + "/healthz").json()["turns_in_flight"] == 0, timeout=15)   # the server ends a deleted chat's turn (P4-H)
    one = ran.passes[0]
    assert ran.code == 1 and ran.err == "" and ran.gate["error"] == "timeout" and ran.gate["ok"] is False
    assert [p["pass"] for p in ran.passes] == [1] and one["error"] == "timeout" and one["session_deleted"] is True
    assert one["rows"][0]["error"] == "timeout" and one["rows"][0]["total_ms"] is None and one["rows"][0]["stages_ms"]


# ---- the token, the proxy, the client id: against servers that cannot serve the models, so the answer is a refusal --------------------

@pytest.fixture
def protected(tmp_path):
    """A live tiny server that wants a token. It has one engine, `tiny`, so the probe's turns are refused at once (422: no such model): what
    these tests look at is whether a request got through the token, the proxy and the client scope, and that is the session's own."""
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path / "home"), token=TOKEN, queue_cap=4))
    yield base
    stop()


def test_the_token_in_chat_token_goes_out_as_a_bearer_with_every_request(protected, monkeypatch):
    monkeypatch.setenv("CHAT_TOKEN", TOKEN)
    code, out, err = run_probe([protected, "--rows", "0"])
    passes, gate = split_result(out)
    assert err == "" and TOKEN not in out and code == 1 and gate["ok"] is False
    assert [p["pass"] for p in passes] == [1, 2, 3] and all("error" not in p and p["session_deleted"] is True for p in passes)
    assert passes[0]["rows"][0]["status"] == 422 and passes[0]["rows"][0]["refusal"] == "validation_error", "past the token: the model is not there"


def test_the_token_in_a_token_file_wins_over_chat_token(protected, monkeypatch, tmp_path):
    path = write_token_file(tmp_path, TOKEN + "\n")
    monkeypatch.setenv("CHAT_TOKEN", "the-wrong-token")
    code, out, err = run_probe([protected, "--rows", "0", "--token-file", str(path)])
    passes, _ = split_result(out)
    assert err == "" and passes[0]["rows"][0]["status"] == 422 and "error" not in passes[0]


def test_a_server_that_wants_a_token_the_probe_was_not_given_answers_401_and_the_probe_fails_with_exit_1(protected, monkeypatch):
    for token in (None, "the-wrong-token"):
        if token:
            monkeypatch.setenv("CHAT_TOKEN", token)
        code, out, err = run_probe([protected, "--rows", "0"])
        passes, gate = split_result(out)
        assert code == 1 and err == "" and [p["pass"] for p in passes] == [1], "no session: the later passes are not tried"
        assert passes[0]["error"] == "session_create_http_401" and passes[0]["session_deleted"] is None and passes[0]["rows"] == []
        assert gate["error"] == "session_create_http_401" and gate["ok"] is False and "the-wrong-token" not in out


def test_in_public_mode_every_session_is_made_for_the_probes_fixed_client_id(tmp_path, monkeypatch):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), mode="public", token=TOKEN, queue_cap=4))
    monkeypatch.setenv("CHAT_TOKEN", TOKEN)
    try:
        with mock.patch.object(Store, "create_session", autospec=True, side_effect=Store.create_session) as created:
            code, out, err = run_probe([base, "--rows", "0"])
    finally:
        stop()
    passes, _ = split_result(out)
    assert err == "" and [p["pass"] for p in passes] == [1, 2, 3] and all(p["session_deleted"] is True for p in passes)
    assert [call.args[2] for call in created.call_args_list] == [probe.CLIENT_ID] * 3
    assert passes[0]["rows"][0]["status"] == 422 and code == 1


def test_the_probe_connects_straight_to_the_url_whatever_proxy_the_environment_names(protected, monkeypatch):
    """httpx trusts HTTP_PROXY and ALL_PROXY by default, loopback included, and the token and the stream would pass through that proxy. Here the
    environment names a proxy that is not there, and the probe must not notice: it goes straight to the URL it was given."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = "http://127.0.0.1:{}".format(s.getsockname()[1])   # a free port, and nothing listens on it once the socket is closed
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, dead)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHAT_TOKEN", TOKEN)
    code, out, err = run_probe([protected, "--rows", "0"])
    passes, _ = split_result(out)
    assert err == "" and "error" not in passes[0] and passes[0]["rows"][0]["status"] == 422


@pytest.mark.parametrize("url", ["http://localhost:18080", "http://127.0.0.1:18080", "http://127.9.9.9:18080", "http://[::1]:18080",
                                 "https://example.org", "https://10.0.0.5:18080", "https://localhost:18080"])
def test_a_token_goes_to_loopback_over_http_and_to_any_host_over_https(url, monkeypatch):
    """Past the check, the probe asks for its session, and nothing answers (Unreachable): that is how this test sees it was not refused."""
    monkeypatch.setenv("CHAT_TOKEN", SECRET)
    monkeypatch.setattr(httpx, "Client", Unreachable)
    code, out, err = run_probe([url])
    passes, gate = split_result(out)
    assert code == 1 and err == "" and gate["error"] == "session_create_ConnectError" and passes[0]["error"] == "session_create_ConnectError"
    assert SECRET not in out and "secret" not in out


def test_a_server_that_cannot_be_reached_ends_in_a_pass_line_and_a_gate_with_the_class_of_the_failure_only(monkeypatch):
    monkeypatch.setattr(httpx, "Client", Unreachable)
    code, out, err = run_probe(["http://127.0.0.1:18080"])
    passes, gate = split_result(out)
    assert code == 1 and err == "" and out.count("\n") == 2 and passes[0]["rows"] == [] and passes[0]["session_deleted"] is None
    assert gate["ok"] is False and gate["sessions_deleted"] is False and gate["published_equal"] is False and gate["first_byte_ms_max"] is None
    assert gate["eos_turn_ms_max"] is None


# ---- R7: nothing of the stream is printed or saved --------------------------------------------------------------------------------

def test_the_probe_prints_and_saves_nothing_of_what_the_server_streams(served, tmp_path, monkeypatch):
    """The tiny gallery makes the streams carry what a private server's do: the generated report, the matched reports, the neighbours' study ids
    and image urls, the label names. The test reads the very bytes the probe read, finds every such string in them, and looks for each in all
    the probe wrote (stdout and stderr) and in the directory it ran in (nothing was created)."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    seen, real = [], probe.read_into

    def tee(facts, handles, chunks, *rest, **named_args):
        def copying():
            for chunk in chunks:
                seen.append(chunk)
                yield chunk
        return real(facts, handles, copying(), *rest, **named_args)
    monkeypatch.setattr(probe, "read_into", tee)
    with decoding(served), published_lines(served, [LOOP] * served.n_test):
        code, out, err = run_probe([served.base, "--rows", "0-1"])
    assert code == 0, (out, err)
    frames = list(iter_sse(["".join(seen)]))
    assert len(frames) > 60, "the probe read all three passes"
    texts = {s for f in frames for s in leaves(f["data"]) if len(s) >= 12 and " " in s}
    texts |= {s for f in frames if f["event"] == "stage_end" for n in f["data"].get("detail", {}).get("image_neighbors", []) for s in leaves(n)}
    ids = {str(n["study_id"]) for f in frames if f["event"] == "stage_end" for n in f["data"].get("detail", {}).get("image_neighbors", [])}
    labels = {s for f in frames if f["event"] == "stage_end" and f["data"]["stage"] == "label" for s in f["data"]["detail"]["positives"]}
    assert LOOP in texts and REPORT[:40] in " ".join(texts) and len(ids) >= 4, "the streams carried reports and ids to look for"
    assert any(len(s) >= 60 for s in texts) and any(s.startswith("/v1/gallery/images/") for s in texts) and labels
    everything = out + err
    for secret in texts | ids | labels:
        assert secret not in everything, secret[:30]
    for word in ("Findings", "Impression", "heart", "lungs", "effusion", "pleural", "SYNTHETIC", "image_url", "study_id"):
        assert word not in everything, word
    for name in CHEXBERT_14:
        assert name not in everything, name
    assert list(workdir.iterdir()) == [], "the probe wrote no file"


def test_a_probe_that_fails_names_its_exceptions_by_class_and_never_by_their_text(served, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("SECRET Findings: report text in an exception")
    monkeypatch.setattr(probe, "sentence_facts", broken)
    with decoding(served):
        code, out, err = run_probe([served.base, "--rows", "0"])
    assert code == 1 and "SECRET" not in out + err and "Findings" not in out + err
    passes, gate = split_result(out)
    assert passes[0]["rows"][0]["error"] == "ValueError" and gate["ok"] is False


# ---- the script as the controller runs it ------------------------------------------------------------------------------------------

def streamed_strings(served, rows):
    """What the server streams for these rows under pass 1's options, read by the test itself, in a session of its own that it deletes: every
    string of every frame that is a report or could be (long, with a space), every neighbour's strings, and the study ids."""
    texts, ids = set(), set()
    sid = httpx.post(served.base + "/v1/sessions", json={}).json()["id"]
    try:
        for row in rows:
            with httpx.stream("POST", served.base + "/v1/sessions/{}/messages".format(sid), timeout=120,
                              data={"text": "", "options": json.dumps(probe.gate_options(row))}) as response:
                frames = list(iter_sse(response.iter_text()))
            texts |= {s for f in frames for s in leaves(f["data"]) if len(s) >= 12 and " " in s}
            neighbors = [n for f in frames if f["event"] == "stage_end" for n in f["data"].get("detail", {}).get("image_neighbors", [])]
            texts |= {s for n in neighbors for s in leaves(n)}
            ids |= {str(n["study_id"]) for n in neighbors}
    finally:
        httpx.delete(served.base + "/v1/sessions/" + sid)
    return texts, ids


def test_the_script_runs_as_a_script_from_any_directory_and_prints_nothing_but_the_result_lines(served, tmp_path):
    """`venv/bin/python scripts/chat_realturn_probe.py URL`, in a process of its own, from a directory that is not the repository: the script
    finds the drill beside it, the server is the in-process one, and what the process printed, on both streams, is read for every string
    the server streamed. A stray print would show here, which the in-process runs (they hand main their own streams) cannot see."""
    script = REPO_ROOT / "scripts" / "chat_realturn_probe.py"
    env = {k: v for k, v in os.environ.items() if k != "CHAT_TOKEN"}
    before = session_ids(served)
    with decoding(served), published_lines(served, [LOOP] * served.n_test):
        texts, ids = streamed_strings(served, (0, 1))
        done = subprocess.run([sys.executable, str(script), served.base, "--rows", "0-1"], cwd=str(tmp_path), env=env, capture_output=True,
                              text=True, timeout=300)
        refused = subprocess.run([sys.executable, str(script), "http://localhost:8000"], cwd=str(tmp_path), env=env, capture_output=True,
                                 text=True, timeout=60)
    assert done.returncode == 0 and done.stderr == "", (done.stdout, done.stderr)
    passes, gate = split_result(done.stdout)
    assert [p["pass"] for p in passes] == [1, 2, 3] and gate["ok"] is True
    assert len(texts) >= 6 and len(ids) >= 4 and any(len(s) >= 60 for s in texts) and LOOP in texts
    for secret in texts | ids:
        assert secret not in done.stdout + done.stderr, secret[:30]
    assert list(tmp_path.iterdir()) == [] and session_ids(served) == before
    assert refused.returncode == 2 and refused.stdout == "" and refused.stderr.startswith("ERROR ") and "8000" in refused.stderr
    assert refused.stderr.count("\n") == 1


class Refusal:
    """A refused answer a fake client hands out, as a context manager: a status and a JSON body (or none)."""

    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def read(self):
        return b""

    def json(self):
        if self.body is None:
            raise ValueError("no JSON")
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Streaming:
    """A 200 answer a fake client hands out, as a context manager: `text` as the stream's chunks of `size` characters."""

    def __init__(self, text, size=700):
        self.status_code, self.text, self.size = 200, text, size

    def iter_text(self):
        for start in range(0, len(self.text), self.size):
            yield self.text[start:start + self.size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Fake:
    """In place of httpx.Client for run_turn alone: its stream is `answer`, or raises it when that is an exception."""

    def __init__(self, answer):
        self.answer = answer

    def stream(self, method, path, **kwargs):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def spec_of(**changes):
    return probe.Spec(**dict(dict(row=0, budget=None, options=probe.gate_options(0)), **changes))


@pytest.mark.parametrize("status, body, word", [
    (429, {"error": {"type": "overloaded_error", "message": "busy"}}, "overloaded_error"),
    (401, {"error": {"type": "authentication_error", "message": "SECRET"}}, "authentication_error"),
    (422, {"error": {"type": "validation_error", "message": "SECRET Unknown model"}}, "validation_error"),
    (500, {"error": {"type": "internal_error", "message": "SECRET"}}, "internal_error"),
    (429, {"error": {"type": "rate_limited", "message": "SECRET"}}, "rate_limited"),
    (503, {"error": {"type": "secretword", "message": "SECRET"}}, "unknown"),      # a short lower-case word that is none of the app's
    (400, {"error": {"type": "Findings: no acute", "message": "SECRET"}}, "unknown"),
    (500, None, "unknown"), (502, ["SECRET"], "unknown"), (404, {"error": "SECRET"}, "unknown"),
])
def test_a_refused_turn_is_its_status_and_a_word_the_app_has_and_nothing_else(status, body, word):
    facts, handles = probe.run_turn(Fake(Refusal(status, body)), "s_abc", spec_of(), 1e18)
    assert (facts.status, facts.refusal, facts.error, facts.final, facts.first_byte_ms, facts.total_ms) == (status, word, None, None, None, None)
    assert handles == probe.Handles() and "SECRET" not in json.dumps(asdict(facts))


def test_the_refusal_words_are_the_kinds_the_app_gives_its_statuses_and_its_rate_limit():
    from app.hardening import RATE_LIMITED
    from app.server import ERROR_KINDS
    assert set(probe.REFUSAL_KINDS) == set(ERROR_KINDS.values()) | {RATE_LIMITED}


def test_a_turn_that_fails_in_transport_is_named_by_the_class_of_the_failure_and_never_by_its_text():
    for failure in (httpx.ConnectError("SECRET http://10.0.0.5/Findings"), httpx.ReadTimeout("SECRET"), httpx.RemoteProtocolError("SECRET"),
                    RuntimeError("SECRET Findings: report text")):
        facts, handles = probe.run_turn(Fake(failure), "s_abc", spec_of(), 1e18)
        assert (facts.status, facts.error) == (None, type(failure).__name__) and "SECRET" not in json.dumps(asdict(facts))


@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400)], ids=["a 400-digit int", "minus a 400-digit int"])
def test_a_stream_with_a_400_digit_number_in_its_milliseconds_is_read_to_its_end_and_the_turn_has_no_error(value):
    """Before the fix the turn ended where the number was read, with `error` "OverflowError" and the stream's later frames unread: the retrieve
    stage's counts, the generate stage's stop word, the label and score stages, message_stop. The number is dropped, and the turn is as good."""
    facts, handles = probe.run_turn(Fake(Streaming(sse(with_ms(value)))), "s_abc", spec_of(), 1e18)
    assert (facts.status, facts.error, facts.refusal, facts.final, facts.seq_contiguous) == (200, None, None, "done", True)
    assert facts.stages_ms == WITHOUT_THOSE_MS["stages_ms"] and facts.server_ms is None
    assert (facts.tokens, facts.stopped, facts.neighbors, facts.matches, facts.live_equals_published) == (100, "budget", 4, 3, True)
    assert facts.agreement == [11, 9, 12, 10] and facts.words == 4 and handles.neighbor_rows == [10, 158, 63, 74]


def test_a_turn_sends_its_options_as_the_form_the_page_sends_and_a_picture_only_when_it_has_one():
    sent = []

    class Spy(Fake):
        def stream(self, method, path, **kwargs):
            sent.append((method, path, kwargs))
            return Refusal(500, None)

    probe.run_turn(Spy(None), "s_abc", spec_of(row=3, options=probe.gate_options(3)), 1e18)
    probe.run_turn(Spy(None), "s_abc", spec_of(row=None, options=probe.eos_options(200), png=b"PNG-BYTES"), 1e18)
    (method, path, kwargs), (_, _, with_picture) = sent
    assert (method, path) == ("POST", "/v1/sessions/s_abc/messages") and kwargs["files"] is None
    assert json.loads(kwargs["data"]["options"]) == probe.gate_options(3) and kwargs["data"]["text"] == ""
    assert with_picture["files"] == {"image": ("probe.png", b"PNG-BYTES", "image/png")}
    assert json.loads(with_picture["data"]["options"]) == probe.eos_options(200)
