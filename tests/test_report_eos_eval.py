"""P9-G4 (CHAT_UI_PLAN.md): the EOS evaluation jobs. Two layers, both CPU-only, offline and synthetic (R7: no MIMIC data; the
fixtures are made-up reports that merely look like the real thing).

* scripts/report_eos_stats.py, the numbers-only summaries the jobs print: unit tests on synthetic files, and its bootstrap
  parser against the real producer, scripts/bootstrap_compare.py, run on a few made-up studies.
* The three wrappers (scripts/eval_report_eos_h100.sh, eval_report_eos_chexbert_h100.sh, eval_report_eos_compare_h100.sh), run
  for real under the oldest bash this repo supports (the Mac's 3.2) in a throwaway tree that stands in for the cluster repo:
  the real helper script, a stub `python` that plays the GPU probe, fake evaluator / scorer / bootstrap scripts whose output
  looks like MIMIC text and paths, and made-up dumps. What is asserted is what each job prints and what it writes.

The static pins on the wrappers (SLURM header, the decode command against the published one, redirects) are in
tests/test_willi_parity.py.
"""
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from scripts import report_eos_stats as stats
from tests.test_chat_remote import STAMP_RE, Sandbox

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
STATS_SCRIPT = SCRIPTS / "report_eos_stats.py"
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"     # the Mac's /bin/bash is 3.2: the oldest shell to support
USER = "krishankumar.bhushan"
DUMP = "results/chat_report_eos_test_split_s42"
PUBLISHED = "results/report_gen_m3_test_split_s42"
LABEL_NAMES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
               "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
FIVE = [1, 4, 5, 7, 9]
MAIN_METRICS = ["rouge_l", "bleu_1", "bleu_4", "chexbert_14_micro", "chexbert_14_macro", "chexbert_5_micro", "chexbert_5_macro",
                "exact_match_accuracy_14", "exact_match_accuracy_5"]
EOS_LINE = "EOS stop: {}/{} reports ended at the end-of-report token, {} were cut at max_new_tokens={}"

# The shape scripts/chat_remote.sh sync writes to .sync_stamp: "<UTC time> <40-hex commit> <clean|dirty>".
STAMP = "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean"
LINE_OK = re.compile(r"^(=== |RESULT |ERROR)")
# What a job may write that is not the repo, the thesis checkout or CHAT_HOME: the test's own stubs, and the places a library
# caches into (the Hugging Face cache sits under SCRATCH_ROOT).
AWAY = ("stubs", "scratch", "home")


# ── the result line ───────────────────────────────────────────────────────────

def test_result_line_is_compact_json_of_numbers_flags_null_and_allowlisted_names():
    line = stats.result_line({"decode": "done", "n": 5, "share": 0.2, "ok": True, "gone": None, "worse": ["rouge_l", "Lung Lesion"]})
    assert line == 'RESULT {"decode":"done","n":5,"share":0.2,"ok":true,"gone":null,"worse":["rouge_l","Lung Lesion"]}'


# Every string a RESULT line may carry (R7), written out here on purpose: the wrapper's own constants, the metric names
# bootstrap_compare emits and the 14 CheXbert labels. A name is printed only if it is in this set, whatever file or log it came from.
ALLOWED = {"done", "paired", "pass", "fail", "eos_s42", "m3_s42", "refs"} | set(MAIN_METRICS) | set(LABEL_NAMES)


def test_the_allowlist_is_exactly_the_wrapper_constants_the_bootstrap_metrics_and_the_14_labels():
    assert stats.ALLOWED_NAMES == frozenset(ALLOWED) and len(ALLOWED) == 7 + 9 + 14


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_result_line_accepts_each_allowlisted_name_alone_and_in_a_list(name):
    assert stats.result_line({"x": name}) == 'RESULT {"x":"%s"}' % name
    assert stats.result_line({"x": [name, name]}) == 'RESULT {"x":["%s","%s"]}' % (name, name)


def test_the_label_allowlist_is_the_apps_canonical_chexbert_14():
    from app.labels import CHEXBERT_14
    assert list(stats.CHEXBERT_14) == CHEXBERT_14 == LABEL_NAMES


def test_the_metric_allowlist_is_what_bootstrap_compare_emits():
    bc = bootstrap_module()
    refs, y_true = synthetic_studies(12)
    cache = {"rouge": [0.5] * 12, "hyp_toks": [["a", "b"]] * 12, "ref_toks": [["a", "c"]] * 12, "y_true": y_true, "y_pred": y_true,
             "five_idx": FIVE, "label_names": LABEL_NAMES}
    emitted = list(bc.evaluate_subset(range(12), cache, per_label=True))
    assert [m for m in emitted if not m.startswith(bc.PER_LABEL_PREFIX)] == list(stats.BOOTSTRAP_METRICS) == MAIN_METRICS
    assert [m[len(bc.PER_LABEL_PREFIX):] for m in emitted if m.startswith(bc.PER_LABEL_PREFIX)] == list(stats.CHEXBERT_14)


@pytest.mark.parametrize("bad", [
    "No acute cardiopulmonary process.",                     # report text: what a free-name rule let through
    "s87654321",                                             # a study id
    "p12345678",                                             # a subject id
    "results/chat_report_eos_test_split_s42/hyps.txt",       # a path
    "/sc/home/someone",
    "Findings: no acute disease.",
    "line\nbreak",
    "", "x" * 41, "12345678",
    "eos_s42\n", "eos_s42 ", " eos_s42", "Eos_s42", "EOS_S42", "rouge_l2", "lung lesion", "No Finding\n",   # near misses: exact only
    {"nested": 1},
    float("nan"), float("inf"),
    87654321, -87654321, 12345678.0, 10 ** 7,                # a number of 8 digits or more could be an id
])
def test_result_line_refuses_anything_not_on_the_allowlist_and_any_number_an_id_could_hide_in(bad):
    with pytest.raises(ValueError):
        stats.result_line({"x": bad})
    with pytest.raises(ValueError):
        stats.result_line({"x": [bad]})


def test_the_module_docstring_describes_each_kind_of_line_the_module_prints():
    """R7's claims, in the module's own words, are the ones the code keeps: a RESULT line holds allowlisted names and bounded
    numbers; an ERROR line holds literals and numbers parsed as digits or computed (the EOS line's counts, a dump's line counts),
    never text; and the literal === notes are printed as well, so "and nothing else" would be false."""
    doc = " ".join(stats.__doc__.split())
    assert "numbers it parsed as digits or computed, never text" in doc
    assert "never a value read out of a log or a file" not in doc, "the ERROR lines that print the EOS line's counts do"
    assert "and nothing else" not in doc and "`=== ...` notes" in doc


def test_result_line_keeps_the_numbers_a_job_prints():
    assert stats.result_line({"n": 2663, "wall_s": 14400, "big": 9999999, "m": -5.5, "tiny": 1e-05}) == (
        'RESULT {"n":2663,"wall_s":14400,"big":9999999,"m":-5.5,"tiny":1e-05}')


def test_result_line_stays_under_the_300_characters_the_summary_keeps():
    assert len(stats.result_line({"worse": MAIN_METRICS})) < 300
    assert len(stats.result_line({"label_worse": LABEL_NAMES})) < 300, "all 14 labels fit alone: a split gate line always fits"
    with pytest.raises(ValueError):
        stats.result_line({"worse": MAIN_METRICS, "label_worse": LABEL_NAMES})


# ── hyps: mean length, repeats, unterminated endings ──────────────────────────

SYNTHETIC_HYPS = [
    "The heart is normal. The lungs are clear.",                       # 8 words, clean
    "No pneumothorax. No pneumothorax. no pneumothorax. Stable.",      # 7 words, 2 repeats: case-insensitive, any position
    "The heart is normal. The lungs are cl",                           # 8 words, cut mid-sentence
    "",                                                                # empty
    "Findings: ok. Impression:",                                       # 3 words, a dangling header is cut as well
]


def test_hyp_stats_counts_words_empties_repeats_and_unterminated_endings():
    got = stats.hyp_stats(SYNTHETIC_HYPS, count_tokens=lambda text: 2 * len(text.split()))
    assert got == {"n": 5, "mean_words": 5.2, "mean_tokens": 10.4, "empty": 1, "repeats_per_report": 0.4,
                   "share_repeat": 0.2, "share_unterminated": 0.4}


def test_the_token_mean_is_null_without_a_tokenizer():
    assert stats.hyp_stats(["A b c."])["mean_tokens"] is None


def test_a_report_that_is_one_unterminated_sentence_is_unterminated_not_lost():
    """repair_generations keeps the original when truncation would delete everything, and counts a fallback instead of a
    truncation. That is still a report whose last sentence was cut."""
    assert stats.hyp_stats(["The heart"])["share_unterminated"] == 1.0
    assert stats.hyp_stats([""])["share_unterminated"] == 0.0, "an empty report is counted under empty, not as cut"


def test_a_repeat_is_counted_every_time_it_recurs_and_per_report_not_per_distinct_sentence():
    got = stats.hyp_stats(["Stable. Stable. Stable. Stable.", "Stable."])
    assert got["repeats_per_report"] == 1.5 and got["share_repeat"] == 0.5


def test_no_reports_is_an_error_not_a_division_by_zero():
    with pytest.raises(stats.ResultError):
        stats.hyp_stats([])


def test_the_token_counter_is_none_when_the_tokenizer_cannot_be_loaded(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)             # `from transformers import ...` then raises ImportError
    assert stats.load_token_counter("gpt2") is None
    assert stats.load_token_counter("none") is None


def test_the_token_counter_counts_what_the_tokenizer_encodes_and_asks_for_the_named_one(monkeypatch):
    import types
    asked = []

    class Tokenizer:
        def encode(self, text):
            return list(range(2 * len(text.split())))

    def from_pretrained(name):
        asked.append(name)
        return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained)))
    counter = stats.load_token_counter("gpt2")
    assert asked == ["gpt2"] and counter("a b c") == 6
    assert stats.hyp_stats(["a b.", "c d e."], counter)["mean_tokens"] == 5.0


# ── the decoder's log: the EOS line and the aggregate ─────────────────────────

def test_the_decoder_cli_still_prints_the_lines_this_parser_reads():
    """The parser is written against these two prints in evaluate_report_generation.py. If either moves, this fails first."""
    src = (SCRIPTS / "evaluate_report_generation.py").read_text()
    assert ('print(f"EOS stop: {ended_by_eos}/{n} reports ended at the end-of-report token, {n - ended_by_eos} were cut at "'
            in src)
    assert 'f"max_new_tokens={args.max_new_tokens}")' in src
    assert 'print(f"=== Aggregate over {n} samples ===")' in src and "print(json.dumps(metrics, indent=2))" in src


def test_parse_eos_stop_reads_the_line_as_the_cli_prints_it():
    assert stats.parse_eos_stop("noise\n" + EOS_LINE.format(2400, 2663, 263, 200) + "\nmore noise\n") == (2400, 2663, 263, 200)
    assert stats.parse_eos_stop("no such line\nEOS stop: lots\n") is None
    assert stats.parse_eos_stop("  " + EOS_LINE.format(1, 2, 1, 200)) is None, "the line starts at column 0"


def test_parse_eos_stop_takes_the_last_line_of_a_log_that_holds_two_attempts():
    log = EOS_LINE.format(1, 5, 4, 200) + "\n--- sample 0 ---\n" + EOS_LINE.format(4, 5, 1, 200) + "\n"
    assert stats.parse_eos_stop(log) == (4, 5, 1, 200)


def real_aggregate_block(hyps: List[str], refs: List[str]) -> Tuple[str, Dict[str, Any]]:
    """(the block the CLI prints after its dump, the metrics in it), from the CLI's own compute_all_metrics."""
    pytest.importorskip("torch")
    from scripts import evaluate_report_generation as erg
    metrics = erg.compute_all_metrics(hyps, refs, chexbert=False)
    return "=== Aggregate over {} samples ===\n{}\n".format(len(hyps), json.dumps(metrics, indent=2)), metrics


def test_parse_aggregate_reads_the_block_the_real_metrics_function_prints_and_ignores_what_follows():
    block, metrics = real_aggregate_block(["The heart is normal."] * 3, ["The heart is clear."] * 3)
    got = stats.parse_aggregate("GENERATED: x\n" + block + "Some trailing warning {not json}\n")
    assert got == metrics and got["meteor"] is None and got["num_examples"] == 3
    assert stats.parse_aggregate("no aggregate here\n") is None
    assert stats.parse_aggregate("=== Aggregate over 3 samples ===\nnot json\n") is None


# Importing transformers pulls in a SWIG-wrapped library whose types lack __module__: a third-party advisory, not ours.
@pytest.mark.filterwarnings("ignore:builtin type .* has no __module__ attribute:DeprecationWarning")
def test_the_real_evaluator_cli_output_and_dump_are_what_the_parser_reads(tmp_path, monkeypatch, capsys):
    """The parser, fed by the real run_checkpoint_inspection instead of a transcription of its prints: three made-up studies, the
    real image loading, transforms, text decoding, dump writing and metrics; only the model, the tokenizer and the beam search are
    replaced. Its stdout is what the job sends to eval.log, so this is also the proof that stdout carries report text and study
    ids, which is why the job log must never see it."""
    import argparse
    pd = pytest.importorskip("pandas")
    image = pytest.importorskip("PIL.Image")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from scripts import evaluate_report_generation as erg

    rows = []
    for i in range(3):
        path = tmp_path / ("img%d.png" % i)
        image.new("L", (8, 8), 40 * i).save(str(path))
        rows.append({"image": str(path), "study_id": 50000000 + i, "findings": "The heart is normal.",
                     "impression": "No acute process."})
    parquet = tmp_path / "test.parquet"
    pd.DataFrame(rows).to_parquet(str(parquet))

    class FakeModule:
        def _patch_grid(self, pixel_values):
            return pixel_values

    class FakeTokenizer:
        eos_token_id = 50256

        def decode(self, ids, skip_special_tokens=True):
            return "The heart is normal.\n\n" + " ".join("w%d" % t for t in ids)         # a paragraph break, as MIMIC reports have

    ended = iter([True, False, True])
    asked = []

    def fake_generate(module, patch_grid, beam_size, max_new_tokens, cached, eos_token_id):
        asked.append((beam_size, max_new_tokens, cached, eos_token_id))
        return torch.tensor([[1, 2, 3]]), next(ended)

    monkeypatch.setattr(erg, "load_report_generation_module", lambda *args, **kwargs: FakeModule())
    monkeypatch.setattr(erg, "generate_from_patch_grid_eos", fake_generate)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *args, **kwargs: FakeTokenizer()))
    args = argparse.Namespace(checkpoint="x.ckpt", model_config="hybrid_150m_m3_rrg", prefix_k=32, scan_impl=None, tfla_impl=None,
                              chunk_size=None, compile_decoder=False, cached_decode=True, stop_at_eos=True, parquet=str(parquet),
                              num_samples=999999, decode="beam", beam_size=3, max_new_tokens=200, dump_dir=str(tmp_path / "dump"),
                              chexbert=False)
    erg.run_checkpoint_inspection(args)
    out = capsys.readouterr().out
    assert asked == [(3, 200, True, 50256)] * 3
    assert "study_id=50000000" in out and "GENERATED: The heart is normal." in out, "the stdout does carry text and study ids"

    result, notes = stats.decode_result(out, tmp_path / "dump", budget=200, wall_s=12)
    assert (result["n"], result["ended_by_eos"], result["cut_at_budget"], result["share_cut"], result["budget"]) == (3, 2, 1, 0.3333, 200)
    assert set(result) >= {"rouge_l", "bleu_1", "bleu_4"} and notes == []
    assert stats.count_lines(tmp_path / "dump" / "hyps.txt") == 3, "a paragraph break inside a report did not split its line"


def write_dump(directory: Path, n: int, hyp_lines: Optional[Sequence[str]] = None, ref_lines: Optional[Sequence[str]] = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    hyps = list(hyp_lines) if hyp_lines is not None else ["The heart is normal. Report %d." % i for i in range(n)]
    refs = list(ref_lines) if ref_lines is not None else ["Findings: clear. Impression: none %d." % i for i in range(n)]
    (directory / "hyps.txt").write_text("\n".join(hyps) + "\n")
    (directory / "refs.txt").write_text("\n".join(refs) + "\n")


def decode_log(n: int = 40, ended: int = 36, budget: int = 200, aggregate: bool = True) -> Tuple[str, Dict[str, Any]]:
    hyps = ["The heart is normal. Report %d." % i for i in range(n)]
    refs = ["Findings: clear. Impression: none %d." % i for i in range(n)]
    log = "Loaded checkpoint: /sc/home/x\n" + "".join("--- sample %d ---\nGENERATED: x\nREFERENCE: y\n" % i for i in range(n))
    log += EOS_LINE.format(ended, n, n - ended, budget) + "\n  Dumped %d hyp/ref pairs\n" % n
    metrics: Dict[str, Any] = {}
    if aggregate:
        block, metrics = real_aggregate_block(hyps, refs)
        log += block
    return log, metrics


def test_decode_result_has_the_counts_the_share_the_wall_time_and_the_text_metrics(tmp_path):
    log, metrics = decode_log(n=40, ended=36)
    write_dump(tmp_path, 40)
    result, notes = stats.decode_result(log, tmp_path, budget=200, wall_s=3300)
    assert result == {"decode": "done", "n": 40, "ended_by_eos": 36, "cut_at_budget": 4, "share_cut": 0.1, "budget": 200,
                      "wall_s": 3300, "rouge_l": round(metrics["rouge_l"], 4), "bleu_1": round(metrics["bleu_1"], 4),
                      "bleu_4": round(metrics["bleu_4"], 4)}
    assert notes == [] and "meteor" not in result, "a null METEOR is left out, not printed"


def test_decode_result_without_an_aggregate_leaves_the_text_metrics_out_and_says_so(tmp_path):
    log, _ = decode_log(aggregate=False)
    write_dump(tmp_path, 40)
    result, notes = stats.decode_result(log, tmp_path, budget=200, wall_s=1)
    assert "rouge_l" not in result and result["n"] == 40
    assert len(notes) == 1 and notes[0].startswith("=== ") and notes[0].endswith(" ===") and "/" not in notes[0]


@pytest.mark.parametrize("what", ["no_eos_line", "wrong_budget", "counts_do_not_add_up", "hyps_short", "refs_long", "hyps_missing",
                                  "refs_missing", "aggregate_for_another_n", "no_reports"])
def test_decode_result_refuses_a_dump_it_cannot_tie_to_the_log(tmp_path, what):
    log, _ = decode_log(n=40, ended=36)
    write_dump(tmp_path, 40)
    if what == "no_eos_line":
        log = "\n".join(l for l in log.splitlines() if not l.startswith("EOS stop:"))
    elif what == "wrong_budget":
        log = log.replace("max_new_tokens=200", "max_new_tokens=100")
    elif what == "counts_do_not_add_up":
        log = log.replace("36/40 reports ended at the end-of-report token, 4 were", "36/40 reports ended at the end-of-report token, 9 were")
    elif what == "hyps_short":
        write_dump(tmp_path, 40, hyp_lines=["a b."] * 39)
    elif what == "refs_long":
        write_dump(tmp_path, 40, ref_lines=["a b."] * 41)
    elif what == "hyps_missing":
        (tmp_path / "hyps.txt").unlink()
    elif what == "refs_missing":
        (tmp_path / "refs.txt").unlink()
    elif what == "aggregate_for_another_n":
        log = log.replace('"num_examples": 40', '"num_examples": 41')
    elif what == "no_reports":
        log = "\n".join(l for l in log.splitlines() if not l.startswith("EOS stop:")) + "\n" + EOS_LINE.format(0, 0, 0, 200) + "\n"
    with pytest.raises(stats.ResultError) as raised:
        stats.decode_result(log, tmp_path, budget=200, wall_s=1)
    assert "/" not in str(raised.value) and len(str(raised.value)) < 200, "the message is printed: literals and numbers only"


@pytest.mark.parametrize("poison", ['"Findings: no acute cardiopulmonary process. s87654321"', "87654321.5", "true", "[87654321]",
                                    '{"id": 87654321}', "null", "41", "40.0"])        # 40.0 == 40: only the type check refuses it
def test_an_aggregate_count_that_is_not_the_integer_n_is_refused_with_a_message_that_holds_none_of_it(tmp_path, poison):
    """ERROR lines carry literals and numbers the script validated, never a value read out of the log. The count in the aggregate
    block is whatever JSON the log holds: it must be an int (not a bool, float or text) before it is compared, and it is not echoed."""
    log, _ = decode_log(n=40, ended=36)
    write_dump(tmp_path, 40)
    poisoned = log.replace('"num_examples": 40', '"num_examples": ' + poison)
    assert poisoned != log
    with pytest.raises(stats.ResultError) as raised:
        stats.decode_result(poisoned, tmp_path, budget=200, wall_s=1)
    message = str(raised.value)
    assert not any(bit in message for bit in ("87654321", "Findings", "acute", "41", "id")), message
    (tmp_path / "eval.log").write_text(poisoned)
    done = run_stats("decode", "--log", str(tmp_path / "eval.log"), "--dump-dir", str(tmp_path), "--budget", "200", "--wall-s", "1")
    assert done.returncode == 2 and done.stdout.startswith("ERROR ") and "RESULT" not in done.stdout
    assert not any(bit in done.stdout + done.stderr for bit in ("87654321", "Findings", "acute")), done.stdout


# ── the CheXbert scorer's files ───────────────────────────────────────────────

def write_chexbert_files(directory: Path, n: int, labels_n: Optional[int] = None, micro_14: float = 0.4) -> None:
    def avg(value):
        return {"precision": value, "recall": value, "f1-score": value, "support": n}
    rows = [[0] * 14] * (n if labels_n is None else labels_n)
    (directory / "chexbert_metrics.json").write_text(json.dumps({
        "accuracy": 0.2, "chexbert_14": {"micro avg": avg(micro_14), "macro avg": avg(0.25)},
        "chexbert_5": {"micro avg": avg(0.45), "macro avg": avg(0.3)}, "num_examples": n}))
    (directory / "chexbert_labels.json").write_text(json.dumps({"y_true": rows, "y_pred": rows, "label_names": LABEL_NAMES}))


@pytest.mark.parametrize("poison", ['"87654321"', "87654321.0", '"Findings: no acute cardiopulmonary process."', "true", "[87654321]"])
def test_a_chexbert_example_count_that_is_not_an_int_is_refused_with_a_message_that_holds_none_of_it(tmp_path, poison):
    write_dump(tmp_path, 7)
    write_chexbert_files(tmp_path, 7)
    metrics = (tmp_path / "chexbert_metrics.json").read_text()
    (tmp_path / "chexbert_metrics.json").write_text(metrics.replace('"num_examples": 7', '"num_examples": ' + poison))
    assert (tmp_path / "chexbert_metrics.json").read_text() != metrics
    with pytest.raises(stats.ResultError) as raised:
        stats.chexbert_result(tmp_path, 1)
    assert not any(bit in str(raised.value) for bit in ("87654321", "Findings", "acute")), str(raised.value)


def test_chexbert_result_has_the_four_f1_headlines_and_the_example_count(tmp_path):
    write_dump(tmp_path, 7)
    write_chexbert_files(tmp_path, 7, micro_14=0.47361)
    assert stats.chexbert_result(tmp_path, 1800) == {"chexbert": "done", "n": 7, "micro_14": 0.4736, "macro_14": 0.25,
                                                      "micro_5": 0.45, "macro_5": 0.3, "wall_s": 1800}


@pytest.mark.parametrize("what", ["no_metrics", "no_labels", "labels_for_another_n", "metrics_for_another_n", "wrong_shape",
                                  "not_json"])
def test_chexbert_result_refuses_files_that_do_not_belong_together(tmp_path, what):
    write_dump(tmp_path, 7)
    write_chexbert_files(tmp_path, 7)
    if what == "no_metrics":
        (tmp_path / "chexbert_metrics.json").unlink()
    elif what == "no_labels":
        (tmp_path / "chexbert_labels.json").unlink()
    elif what == "labels_for_another_n":
        write_chexbert_files(tmp_path, 7, labels_n=6)
    elif what == "metrics_for_another_n":
        write_chexbert_files(tmp_path, 8)
    elif what == "wrong_shape":
        (tmp_path / "chexbert_metrics.json").write_text(json.dumps({"num_examples": 7}))
    else:
        (tmp_path / "chexbert_metrics.json").write_text("{")
    with pytest.raises(stats.ResultError) as raised:
        stats.chexbert_result(tmp_path, 1)
    assert "/" not in str(raised.value)


# ── the paired bootstrap: parsed from bootstrap_compare's own output ──────────

def bootstrap_module():
    pytest.importorskip("torch")
    from scripts import bootstrap_compare
    return bootstrap_compare


def synthetic_studies(n: int = 40, seed: int = 0) -> Tuple[List[str], List[List[int]]]:
    rng = random.Random(seed)
    vocab = "heart lungs pleural effusion normal clear opacity stable size mild".split()
    refs = [" ".join(rng.choice(vocab) for _ in range(rng.randint(8, 20))) + "." for _ in range(n)]
    return refs, [[rng.randint(0, 1) for _ in range(14)] for _ in range(n)]


def make_system(directory: Path, hyps: List[str], refs: List[str], y_true: List[List[int]], y_pred: List[List[int]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    write_dump(directory, len(hyps), hyp_lines=hyps, ref_lines=refs)
    (directory / "chexbert_labels.json").write_text(json.dumps({
        "y_true": y_true, "y_pred": y_pred, "label_names": LABEL_NAMES, "five_label_indices": FIVE}))
    return directory


def real_bootstrap(tmp_path: Path, a: str, b: str, name_a: str = "eos_s42", name_b: str = "m3_s42") -> Tuple[str, Dict]:
    """bootstrap_compare's own paired_bootstrap + render on 40 made-up studies where each system is 'good' (the references
    and their true labels) or 'bad' (a junk sentence and no findings). Returns (the markdown, the results it rendered)."""
    bc = bootstrap_module()
    refs, y_true = synthetic_studies()
    systems = {"good": (list(refs), y_true), "bad": (["zzz qqq."] * len(refs), [[0] * 14 for _ in refs])}
    caches = []
    for role, kind in (("a", a), ("b", b)):
        hyps, y_pred = systems[kind]
        directory = make_system(tmp_path / role, hyps, refs, y_true, y_pred)
        caches.append(bc.build_cache(hyps, refs, str(directory / "chexbert_labels.json")))
    results, meta = bc.paired_bootstrap(caches[0], caches[1], 20, 0, per_label=True)
    return bc.render(results, meta, name_a, name_b), results


def test_parse_bootstrap_round_trips_every_number_of_the_real_report(tmp_path):
    text, results = real_bootstrap(tmp_path, "bad", "good")
    parsed = stats.parse_bootstrap(text, "eos_s42", "m3_s42")
    assert (parsed["n"], parsed["samples"], parsed["seed"]) == (40, 20, 0)
    assert [row["name"] for row in parsed["metrics"]] == MAIN_METRICS
    rendered_labels = [m[len("label::"):] for m in results if m.startswith("label::")]
    assert sorted(row["name"] for row in parsed["labels"]) == sorted(rendered_labels) and len(rendered_labels) == 14
    for row in parsed["metrics"] + parsed["labels"]:
        truth = results[row["name"] if row["name"] in results else "label::" + row["name"]]
        for mine, theirs in ((row["a"], truth["a"]), (row["b"], truth["b"]), (row["diff"], truth["diff"]),
                             (row["lo"], truth["ci_low"]), (row["hi"], truth["ci_high"])):
            assert abs(mine - theirs) < 5.1e-5, (row["name"], mine, theirs)


def test_the_gate_fails_and_names_every_metric_the_ci_puts_entirely_below_zero(tmp_path):
    text, results = real_bootstrap(tmp_path, "bad", "good")                      # the EOS system is worse on everything
    lines = stats.comparison_lines(stats.parse_bootstrap(text, "eos_s42", "m3_s42"), "eos_s42", "m3_s42")
    assert lines[-1] == stats.result_line({"gate": "fail", "worse": MAIN_METRICS}), "the verdict is the last line, whatever else is split off"
    assert all(results[m]["ci_high"] < 0 for m in MAIN_METRICS)


def test_the_gate_passes_when_the_eos_system_is_better_or_the_same(tmp_path):
    for a, b in (("good", "bad"), ("good", "good")):
        text, _ = real_bootstrap(tmp_path / (a + b), a, b)
        lines = stats.comparison_lines(stats.parse_bootstrap(text, "eos_s42", "m3_s42"), "eos_s42", "m3_s42")
        assert lines[-1] == 'RESULT {"gate":"pass","worse":[],"label_worse":[]}', (a, b)


def test_a_system_better_on_text_and_worse_on_labels_fails_on_the_label_metrics_only(tmp_path):
    bc = bootstrap_module()
    refs, y_true = synthetic_studies()
    zero = [[0] * 14 for _ in refs]
    a = make_system(tmp_path / "a", list(refs), refs, y_true, zero)             # perfect text, no findings
    b = make_system(tmp_path / "b", ["zzz qqq."] * len(refs), refs, y_true, y_true)   # junk text, perfect findings
    caches = [bc.build_cache(h, refs, str(d / "chexbert_labels.json"))
              for h, d in ((list(refs), a), (["zzz qqq."] * len(refs), b))]
    results, meta = bc.paired_bootstrap(caches[0], caches[1], 20, 0, per_label=True)
    parsed = stats.parse_bootstrap(bc.render(results, meta, "eos_s42", "m3_s42"), "eos_s42", "m3_s42")
    assert stats.worse_metrics(parsed["metrics"]) == ["chexbert_14_micro", "chexbert_14_macro", "chexbert_5_micro",
                                                       "chexbert_5_macro", "exact_match_accuracy_14", "exact_match_accuracy_5"]


def rendered(rows: Dict[str, Tuple[float, float, float, float, float]], labels: Optional[Dict] = None,
             names: Tuple[str, str] = ("eos_s42", "m3_s42")) -> str:
    """bootstrap_compare.render over hand-made rows: name -> (a, b, diff, ci_low, ci_high)."""
    bc = bootstrap_module()
    results = {}
    for name, (a, b, diff, lo, hi) in list(rows.items()) + [("label::" + k, v) for k, v in (labels or {}).items()]:
        results[name] = {"a": a, "b": b, "diff": diff, "ci_low": lo, "ci_high": hi,
                         "significant": lo > 0 or hi < 0, "frac_sign_flipped": 0.0}
    return bc.render(results, {"n": 2663, "bootstrap_samples": 1000, "seed": 0}, names[0], names[1])


FLAT = {name: (0.3, 0.3, 0.0, -0.01, 0.01) for name in MAIN_METRICS}


def test_the_sign_of_a_rounded_zero_still_decides_whether_a_ci_reaches_zero():
    """A CI [-0.0123, -0.00001] excludes zero but prints as [-0.0123, -0.0000]; [-0.0123, +0.00001] spans it and prints +0.0000.
    The text keeps the sign, and the gate reads it, as bootstrap_compare's own `hi < 0` does at full precision."""
    below = dict(FLAT, rouge_l=(0.19, 0.2, -0.01, -0.0123, -0.00001))
    touching = dict(FLAT, rouge_l=(0.19, 0.2, -0.01, -0.0123, 0.00001))
    assert stats.worse_metrics(stats.parse_bootstrap(rendered(below), "eos_s42", "m3_s42")["metrics"]) == ["rouge_l"]
    assert stats.worse_metrics(stats.parse_bootstrap(rendered(touching), "eos_s42", "m3_s42")["metrics"]) == []


def test_per_label_rows_are_printed_but_do_not_gate():
    """Fourteen more one-sided looks would make the gate fire by chance; bootstrap_compare's own summary does not count them."""
    text = rendered(FLAT, labels={"Lung Lesion": (0.1, 0.3, -0.2, -0.3, -0.1)})
    parsed = stats.parse_bootstrap(text, "eos_s42", "m3_s42")
    lines = stats.comparison_lines(parsed, "eos_s42", "m3_s42")
    assert lines[-1] == 'RESULT {"gate":"pass","worse":[],"label_worse":["Lung Lesion"]}', "named for the reader, not counted by the gate"
    assert stats.result_line({"label": "Lung Lesion", "a": 0.1, "b": 0.3, "diff": -0.2, "lo": -0.3, "hi": -0.1}) in lines


def test_label_worse_lists_the_labels_whose_ci_is_entirely_below_zero_by_the_same_sign_rule_and_never_changes_the_gate():
    labels = {"Lung Lesion": (0.1, 0.3, -0.2, -0.3, -0.1),            # entirely below zero
              "Edema": (0.2, 0.25, -0.05, -0.1, 0.02),                # spans zero
              "Pneumonia": (0.2, 0.25, -0.05, -0.1, -0.00001),        # stops just short of zero: prints -0.0000, still below
              "Fracture": (0.2, 0.25, -0.05, -0.1, 0.00001),          # touches zero from below: prints +0.0000, spans it
              "Cardiomegaly": (0.4, 0.3, 0.1, 0.05, 0.15)}            # better
    parsed = stats.parse_bootstrap(rendered(FLAT, labels=labels), "eos_s42", "m3_s42")
    assert stats.worse_metrics(parsed["labels"]) == ["Lung Lesion", "Pneumonia"], "in the report's own order: by size of difference"
    lines = stats.comparison_lines(parsed, "eos_s42", "m3_s42")
    assert lines[-1] == 'RESULT {"gate":"pass","worse":[],"label_worse":["Lung Lesion","Pneumonia"]}'
    failing = dict(FLAT, rouge_l=(0.19, 0.2, -0.01, -0.02, -0.005))
    lines = stats.comparison_lines(stats.parse_bootstrap(rendered(failing), "eos_s42", "m3_s42"), "eos_s42", "m3_s42")
    assert lines[-1] == 'RESULT {"gate":"fail","worse":["rouge_l"],"label_worse":[]}'


def test_a_gate_line_too_long_for_both_lists_puts_label_worse_on_its_own_line_before_the_verdict(tmp_path):
    """Every metric and every label worse: 9 + 14 names do not fit in 300 characters together, so the labels get a line of
    their own and the verdict, which `chat_remote.sh summary` readers look for last, stays last."""
    text, results = real_bootstrap(tmp_path, "bad", "good")
    lines = stats.comparison_lines(stats.parse_bootstrap(text, "eos_s42", "m3_s42"), "eos_s42", "m3_s42")
    expected_labels = [m[len("label::"):] for m, r in results.items() if m.startswith("label::") and r["ci_high"] < 0]
    assert len(expected_labels) == 14
    first = json.loads(lines[-2][len("RESULT "):])
    assert list(first) == ["label_worse"] and sorted(first["label_worse"]) == sorted(expected_labels)
    assert lines[-1] == stats.result_line({"gate": "fail", "worse": MAIN_METRICS})
    assert all(len(line) < 300 for line in lines)


def test_comparison_lines_are_the_meta_then_one_line_per_metric_then_the_gate():
    parsed = stats.parse_bootstrap(rendered(FLAT), "eos_s42", "m3_s42")
    lines = stats.comparison_lines(parsed, "eos_s42", "m3_s42")
    assert lines[0] == 'RESULT {"bootstrap":"paired","name_a":"eos_s42","name_b":"m3_s42","n":2663,"samples":1000,"seed":0}'
    assert lines[1] == 'RESULT {"metric":"rouge_l","a":0.3,"b":0.3,"diff":0.0,"lo":-0.01,"hi":0.01}'
    assert len(lines) == 1 + len(MAIN_METRICS) + 1 and all(len(l) < 300 for l in lines)


def test_parse_bootstrap_refuses_a_report_whose_columns_are_not_the_two_systems_in_that_order():
    """Swapped columns would flip the sign of every difference, and with it the gate."""
    with pytest.raises(stats.ResultError):
        stats.parse_bootstrap(rendered(FLAT), "m3_s42", "eos_s42")


def test_parse_bootstrap_refuses_a_row_whose_difference_is_not_a_minus_b():
    text = rendered(FLAT).replace("| rouge_l | 0.3000 | 0.3000 | +0.0000 |", "| rouge_l | 0.3000 | 0.3000 | -0.0500 |")
    assert text != rendered(FLAT)
    with pytest.raises(stats.ResultError):
        stats.parse_bootstrap(text, "eos_s42", "m3_s42")


def test_parse_bootstrap_refuses_a_report_without_the_label_metrics_the_gate_is_about():
    """With no label matrices bootstrap_compare compares the three text metrics only: a gate over those is not the gate."""
    text_only = rendered({k: v for k, v in FLAT.items() if k in ("rouge_l", "bleu_1", "bleu_4")})
    with pytest.raises(stats.ResultError):
        stats.parse_bootstrap(text_only, "eos_s42", "m3_s42")


def test_parse_bootstrap_refuses_a_row_name_the_job_does_not_print_and_does_not_echo_it():
    """A name read out of the report is printed only if it is on the allowlist, and each table has its own: metrics in the first,
    CheXbert labels in the second."""
    poisoned_label = rendered(FLAT, labels={"s87654321": (0.1, 0.3, -0.2, -0.3, -0.1)})
    label_in_metrics = rendered(dict(FLAT, **{"Lung Lesion": (0.1, 0.3, -0.2, -0.3, -0.1)}))
    metric_in_labels = rendered(FLAT, labels={"rouge_l": (0.1, 0.3, -0.2, -0.3, -0.1)})
    for text in (poisoned_label, label_in_metrics, metric_in_labels):
        with pytest.raises(stats.ResultError) as raised:
            stats.parse_bootstrap(text, "eos_s42", "m3_s42")
        assert not any(bit in str(raised.value) for bit in ("s87654321", "Lung Lesion", "rouge_l")), str(raised.value)


@pytest.mark.parametrize("text", ["", "# nothing\n", "| metric | eos_s42 | m3_s42 | diff | 95% CI | verdict |\n|---|---|---|---|---|---|\n| rouge_l | nan |\n"])
def test_parse_bootstrap_refuses_garbage(text):
    with pytest.raises(stats.ResultError):
        stats.parse_bootstrap(text, "eos_s42", "m3_s42")


# ── the command line ──────────────────────────────────────────────────────────

def run_stats(*args: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    base.update(env or {})
    return subprocess.run([sys.executable, str(STATS_SCRIPT)] + list(args), cwd=str(REPO_ROOT), env=base, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=120)


def only_summary_lines(text: str) -> None:
    lines = text.splitlines()
    assert lines and [l for l in lines if not LINE_OK.match(l)] == [], lines
    assert not [l for l in lines if "/" in l], "no path in a printed line"
    assert all(len(l) < 300 for l in lines)


def test_cli_decode_prints_one_result_line(tmp_path):
    log, _ = decode_log(n=40, ended=36)
    (tmp_path / "eval.log").write_text(log)
    write_dump(tmp_path, 40)
    done = run_stats("decode", "--log", str(tmp_path / "eval.log"), "--dump-dir", str(tmp_path), "--budget", "200", "--wall-s", "77")
    assert done.returncode == 0, done.stdout + done.stderr
    only_summary_lines(done.stdout)
    assert json.loads(done.stdout.splitlines()[0][len("RESULT "):])["wall_s"] == 77


def test_cli_decode_failure_prints_a_literal_error_and_exits_2_with_nothing_else(tmp_path):
    (tmp_path / "eval.log").write_text("GENERATED: FAKE REPORT TEXT study_id=12345678\n")
    write_dump(tmp_path, 3)
    done = run_stats("decode", "--log", str(tmp_path / "eval.log"), "--dump-dir", str(tmp_path), "--budget", "200", "--wall-s", "1")
    assert done.returncode == 2
    only_summary_lines(done.stdout)
    assert done.stdout.startswith("ERROR ") and "FAKE" not in done.stdout + done.stderr


def test_cli_chexbert_prints_one_result_line(tmp_path):
    write_dump(tmp_path, 4)
    write_chexbert_files(tmp_path, 4)
    done = run_stats("chexbert", "--dump-dir", str(tmp_path), "--wall-s", "9")
    assert done.returncode == 0, done.stdout + done.stderr
    only_summary_lines(done.stdout)
    assert done.stdout.startswith('RESULT {"chexbert":"done","n":4,')


def test_cli_hyps_prints_one_line_per_system_in_the_order_given(tmp_path):
    write_dump(tmp_path / "a", 3, hyp_lines=SYNTHETIC_HYPS[:3])
    write_dump(tmp_path / "b", 3, hyp_lines=["No change."] * 3)
    done = run_stats("hyps", "eos_s42=" + str(tmp_path / "a" / "hyps.txt"), "m3_s42=" + str(tmp_path / "b" / "hyps.txt"),
                     "--tokenizer", "none")
    assert done.returncode == 0, done.stdout + done.stderr
    only_summary_lines(done.stdout)
    first, second = [json.loads(l[len("RESULT "):]) for l in done.stdout.splitlines()]
    assert (first["stats"], first["n"], first["mean_tokens"]) == ("eos_s42", 3, None)
    assert (second["stats"], second["mean_words"], second["repeats_per_report"]) == ("m3_s42", 2.0, 0.0)


def test_cli_hyps_says_so_when_the_tokenizer_is_not_cached_and_still_reports_everything_else(tmp_path):
    write_dump(tmp_path, 2)
    done = run_stats("hyps", "eos_s42=" + str(tmp_path / "hyps.txt"),
                     env={"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "empty_hf"), "HF_DATASETS_OFFLINE": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    only_summary_lines(done.stdout)
    note, result = done.stdout.splitlines()
    assert note.startswith("=== ") and "mean_tokens is null" in note
    assert json.loads(result[len("RESULT "):])["mean_tokens"] is None


@pytest.mark.parametrize("name", ["results/chat_x", "my_system", "s87654321", "No acute cardiopulmonary process.", ""])
def test_cli_hyps_refuses_a_system_name_that_is_not_on_the_allowlist(tmp_path, name):
    write_dump(tmp_path, 2)
    done = run_stats("hyps", name + "=" + str(tmp_path / "hyps.txt"), "--tokenizer", "none")
    assert done.returncode == 2 and done.stdout.startswith("ERROR ") and "RESULT" not in done.stdout
    if name:
        assert name not in done.stdout


def test_cli_gate_refuses_a_system_name_that_is_not_on_the_allowlist_before_it_could_reach_a_line(tmp_path):
    """The report is rendered with the poisoned name as system A, so its header matches what is asked and the allowlist check is
    the only thing in the way: a clean refusal (2), not a crash (1) when the name reaches the first RESULT line."""
    md = tmp_path / "bootstrap.md"
    md.write_text(rendered(FLAT, names=("s87654321", "m3_s42")))
    done = run_stats("gate", "--bootstrap", str(md), "--name-a", "s87654321", "--name-b", "m3_s42")
    assert done.returncode == 2 and done.stdout.startswith("ERROR ") and "RESULT" not in done.stdout
    assert "s87654321" not in done.stdout + done.stderr


def test_cli_gate_prints_the_comparison_and_exits_0_even_when_the_gate_fails(tmp_path):
    md = tmp_path / "bootstrap.md"
    md.write_text(rendered(dict(FLAT, rouge_l=(0.19, 0.2, -0.01, -0.02, -0.005))))
    done = run_stats("gate", "--bootstrap", str(md), "--name-a", "eos_s42", "--name-b", "m3_s42")
    assert done.returncode == 0, done.stdout + done.stderr
    only_summary_lines(done.stdout)
    assert done.stdout.splitlines()[-1] == 'RESULT {"gate":"fail","worse":["rouge_l"],"label_worse":[]}', "the gate line comes last"


def test_cli_gate_that_cannot_parse_prints_no_gate_line_at_all(tmp_path):
    md = tmp_path / "bootstrap.md"
    md.write_text(rendered(FLAT))
    done = run_stats("gate", "--bootstrap", str(md), "--name-a", "m3_s42", "--name-b", "eos_s42")      # names swapped
    assert done.returncode == 2
    only_summary_lines(done.stdout)
    assert done.stdout.startswith("ERROR ") and '"gate"' not in done.stdout and "RESULT" not in done.stdout


# ── the wrappers, rehearsed in a temp tree ────────────────────────────────────

STUB_PYTHON = r"""#!/bin/bash
# Stands in for the venv's python. It records every call and the Hugging Face environment the call saw, plays the GPU probe,
# and runs everything else (the fake evaluator, scorer and bootstrap, the real stats script) with the test interpreter.
{ echo "@@"; for a in "$@"; do printf '%s\n' "$a"; done; } >> "$STUB_DIR/python.calls"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE-unset} HF_HOME=${HF_HOME-unset}" >> "$STUB_DIR/python.env"
case "$*" in
  *torch.cuda.device_count*) echo "${FAKE_GPUS:-1} NVIDIA H100 80GB HBM3"; exit 0;;
esac
exec "$REAL_PYTHON" "$@"
"""

LEAKY = ('print("Findings: FAKE REPORT TEXT study_id=12345678 /sc/home/someone/images/p10/img.jpg")\n'
         'print("Traceback (most recent call last): FAKE MIMIC TEXT in a message", file=sys.stderr)\n'
         'print("ValueError: FAKE MIMIC TEXT study_id=12345678", file=sys.stderr)\n')

# What evaluate_report_generation.py --checkpoint ... --stop-at-eos prints, in the order it prints it (its own text lines carry
# MIMIC text and ids, which is the point of keeping them out of the job log), and the two files it writes.
FAKE_EVALUATOR = r'''
import argparse, json, os, sys
from pathlib import Path

stub = Path(os.environ["STUB_DIR"])
ap = argparse.ArgumentParser()
for flag in ("--checkpoint", "--model-config", "--prefix-k", "--parquet", "--num-samples", "--decode", "--beam-size",
             "--max-new-tokens", "--dump-dir"):
    ap.add_argument(flag)
for flag in ("--cached-decode", "--stop-at-eos"):
    ap.add_argument(flag, action="store_true")
args = ap.parse_args()                      # an unknown flag fails here, as it would in the real CLI
mode = (stub / "eval.mode").read_text().strip()
n = int((stub / "eval.n").read_text())
ended = int((stub / "eval.ended").read_text())
print("Decode path: O(1) recurrent cache (M6).")
print("Loaded checkpoint: /sc/home/someone/chat_sessions/models/x/checkpoints/last.ckpt")
print("Loaded %d rows from %s; inspecting first %d\n" % (n, args.parquet, n))
for i in range(n):
    print("--- sample %d (study_id=12345678) ---" % i)
    print("GENERATED: FAKE REPORT TEXT study_id=12345678 /sc/home/someone/images/p10/img.jpg")
    print("REFERENCE: FAKE REFERENCE TEXT")
print("Traceback (most recent call last): FAKE MIMIC TEXT in a message", file=sys.stderr)
if mode == "fail":
    sys.exit(3)
if mode != "noeos":
    budget = "100" if mode == "badbudget" else args.max_new_tokens
    print("EOS stop: %d/%d reports ended at the end-of-report token, %d were cut at max_new_tokens=%s" % (ended, n, n - ended, budget))
dump = Path(args.dump_dir)
dump.mkdir(parents=True, exist_ok=True)
rows = n - 1 if mode == "short" else n
(dump / "hyps.txt").write_text("".join("The heart is normal. Report %d.\n" % i for i in range(rows)))
(dump / "refs.txt").write_text("".join("Findings: clear. Impression: none %d.\n" % i for i in range(n)))
print("  Dumped %d hyp/ref pairs" % rows)
print("=== Aggregate over %d samples ===" % n)
print(json.dumps({"rouge_l": 0.19, "bleu_1": 0.31, "bleu_4": 0.05, "meteor": None, "num_examples": n}, indent=2))
'''

FAKE_SCORER = r'''
import argparse, json, os, sys
from pathlib import Path

stub = Path(os.environ["STUB_DIR"])
ap = argparse.ArgumentParser()
for flag in ("--hyp-file", "--ref-file", "--output-dir"):
    ap.add_argument(flag, required=True)
args = ap.parse_args()
mode = (stub / "score.mode").read_text().strip()
n = len(Path(args.hyp_file).read_text().splitlines())
print("Some weights of the model were not used: FAKE REPORT TEXT study_id=12345678")
print("  CheXbert F1 (14-label) micro/macro : 0.4000 / 0.2500")
print("  Results saved to /sc/home/someone/x/chexbert_metrics.json")
print("Traceback (most recent call last): FAKE MIMIC TEXT in a message", file=sys.stderr)
print("ValueError: FAKE MIMIC TEXT study_id=12345678", file=sys.stderr)
if mode == "fail":
    sys.exit(1)
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
def avg(value):
    return {"precision": value, "recall": value, "f1-score": value, "support": n}
(out / "chexbert_metrics.json").write_text(json.dumps({
    "accuracy": 0.2, "chexbert_14": {"micro avg": avg(0.4), "macro avg": avg(0.25)},
    "chexbert_5": {"micro avg": avg(0.45), "macro avg": avg(0.3)}, "num_examples": n}))
if mode != "nolabels":
    rows = [[0] * 14 for _ in range(n)]
    (out / "chexbert_labels.json").write_text(json.dumps({"y_true": rows, "y_pred": rows}))
'''

# bootstrap_compare.py reads the real scripts' modules, which the temp tree does not have: this stands in for it, accepts the
# real CLI's flags, leaks like it (it prints the whole report and the study text of a crash) and writes a prepared report.
FAKE_BOOTSTRAP = r'''
import argparse, os, shutil, sys
from pathlib import Path

stub = Path(os.environ["STUB_DIR"])
ap = argparse.ArgumentParser()
for flag in ("--hyps-a", "--hyps-b", "--refs", "--labels-a", "--labels-b", "--name-a", "--name-b", "--bootstrap-samples",
             "--seed", "--output"):
    ap.add_argument(flag)
ap.add_argument("--per-label", action="store_true")
args = ap.parse_args()
print("Findings: FAKE REPORT TEXT study_id=12345678 /sc/home/someone/images/p10/img.jpg")
print("Traceback (most recent call last): FAKE MIMIC TEXT in a message", file=sys.stderr)
if (stub / "boot.mode").read_text().strip() == "fail":
    sys.exit(4)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
shutil.copy(str(stub / "bootstrap.md"), args.output)
'''


def snapshot(root: Path, skip: Sequence[str] = ()) -> List[tuple]:
    """Every path under `root`, minus the relative prefixes in `skip`: a file with its size and mtime, a link with its target, a
    directory by name only (its own mtime moves when a child is added, and a child that appears or goes is a line of its own).
    What a job that writes nowhere else leaves exactly as it was (R8)."""
    found = []
    for base, dirs, files in os.walk(str(root)):
        for name in dirs + files:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, str(root))
            if any(rel == s or rel.startswith(s + os.sep) for s in skip):
                continue
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                found.append((rel, "link", os.readlink(full)))
            elif stat.S_ISDIR(st.st_mode):
                found.append((rel, "dir"))
            else:
                found.append((rel, st.st_size, st.st_mtime_ns))
    return sorted(found)


class Box:
    """A throwaway tree standing in for the cluster: CLUSTER_REPO (repo/, the directory the job cds into), the thesis checkout
    (main/) behind repo/results, CHAT_HOME (chat/), and a stub python."""

    def __init__(self, root: Path, wrapper: str, stamp: Optional[str] = STAMP):
        self.root, self.wrapper = root, wrapper
        self.repo, self.main, self.chat = root / "repo", root / "main", root / "chat"
        self.stubs, self.bin = root / "stubs", root / "bin"
        (self.repo / "scripts").mkdir(parents=True)
        for name in ("__init__.py", "report_eos_stats.py", "repair_generations.py", wrapper):
            shutil.copy(str(SCRIPTS / name), str(self.repo / "scripts" / name))
        for venv in (".venv", ".venv_chexbert"):
            (self.repo / venv / "bin").mkdir(parents=True)
            (self.repo / venv / "bin" / "activate").write_text("")
        (self.repo / "logs").mkdir()
        if stamp is not None:
            (self.repo / ".sync_stamp").write_text(stamp + "\n")
        (self.main / "results").mkdir(parents=True)
        (self.repo / "results").symlink_to(self.main / "results", target_is_directory=True)
        for directory in (self.chat, self.stubs, self.bin):
            directory.mkdir()
        python = self.bin / "python"
        python.write_text(STUB_PYTHON)
        python.chmod(0o755)

    @property
    def dump(self) -> Path:
        return self.main / "results" / "chat_report_eos_test_split_s42"

    @property
    def published(self) -> Path:
        return self.main / "results" / "report_gen_m3_test_split_s42"

    def script(self, name: str, source: str) -> None:
        (self.repo / "scripts" / name).write_text(source)

    def stub(self, name: str, value: Any) -> None:
        (self.stubs / name).write_text(str(value) + "\n")

    def run(self, **extra: str) -> subprocess.CompletedProcess:
        env = {"PATH": "{}:/usr/bin:/bin".format(self.bin), "HOME": str(self.root / "home"), "USER": USER,
               "SLURM_SUBMIT_DIR": str(self.repo), "SLURM_JOB_ID": "1234567", "CHAT_HOME": str(self.chat),
               "SCRATCH_ROOT": str(self.root / "scratch"), "STUB_DIR": str(self.stubs), "REAL_PYTHON": sys.executable,
               "PYTHONDONTWRITEBYTECODE": "1"}
        env.update(extra)
        # stdout and stderr share one pipe, like the single SLURM log: whatever bash itself complains about counts too.
        return subprocess.run([BASH, str(self.repo / "scripts" / self.wrapper)], cwd=str(self.root), env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)

    def calls(self) -> List[List[str]]:
        log = self.stubs / "python.calls"
        return [rec.splitlines() for rec in log.read_text().split("@@\n") if rec.strip()] if log.exists() else []

    def calls_of(self, script: str) -> List[List[str]]:
        return [c for c in self.calls() if c and c[0] == "scripts/" + script]

    def env_of(self, script: str) -> str:
        """The Hugging Face environment line the (single) call of `script` saw."""
        envs = (self.stubs / "python.env").read_text().splitlines()
        (index,) = [i for i, c in enumerate(self.calls()) if c and c[0] == "scripts/" + script]
        return envs[index]


def lines_of(done: subprocess.CompletedProcess) -> List[str]:
    return done.stdout.splitlines()


def results_of(lines: List[str]) -> List[dict]:
    return [json.loads(l[len("RESULT "):]) for l in lines if l.startswith("RESULT ")]


def assert_job_log_is_clean(lines: List[str]) -> None:
    """R7: only wrapper-authored lines, no path, nothing the fake tools leaked."""
    assert [l for l in lines if not LINE_OK.match(l)] == [], "a line that is not ===, RESULT or ERROR"
    assert not [l for l in lines if "/" in l], "a path in a printed line"
    assert not [l for l in lines if "FAKE" in l or "12345678" in l or "Traceback" in l or "ValueError" in l], lines
    assert all(len(l) < 300 for l in lines)


# ── job 1: the decode ─────────────────────────────────────────────────────────

DECODE = "eval_report_eos_h100.sh"


def decode_box(tmp_path: Path, n: int = 5, ended: int = 4, mode: str = "ok", stamp: Optional[str] = STAMP) -> Box:
    box = Box(tmp_path, DECODE, stamp=stamp)
    box.script("evaluate_report_generation.py", FAKE_EVALUATOR)
    run_dir = box.chat / "models" / "report_gen_m3_eos_s42"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").write_bytes(b"")
    (run_dir / "DONE").write_text("2026-10-09T20:00:00Z job=1\n")
    (box.root / "data").mkdir()
    (box.root / "data" / "test.parquet").write_bytes(b"")
    for name, value in (("eval.mode", mode), ("eval.n", n), ("eval.ended", ended)):
        box.stub(name, value)
    box.published.mkdir(parents=True)                   # the published dump: job 1 reads its refs.txt, the same studies in the same order
    (box.published / "refs.txt").write_text("".join("Findings: clear. Impression: none %d.\n" % i for i in range(n)))
    return box


def run_decode(box: Box, **extra: str) -> subprocess.CompletedProcess:
    return box.run(PARQUET=str(box.root / "data" / "test.parquet"), **extra)


def expected_decode_argv(box: Box, ckpt: Optional[str] = None, parquet: Optional[str] = None, dump: str = DUMP) -> List[str]:
    ckpt = ckpt or str(box.chat / "models" / "report_gen_m3_eos_s42" / "checkpoints" / "last.ckpt")
    parquet = parquet or str(box.root / "data" / "test.parquet")
    return ["scripts/evaluate_report_generation.py", "--checkpoint", ckpt, "--model-config", "hybrid_150m_m3_rrg", "--prefix-k", "32",
            "--cached-decode", "--stop-at-eos", "--parquet", parquet, "--num-samples", "999999", "--decode", "beam",
            "--beam-size", "3", "--max-new-tokens", "200", "--dump-dir", dump]


def test_a_clean_decode_prints_only_wrapper_lines_and_keeps_the_raw_output_in_eval_log(tmp_path):
    box = decode_box(tmp_path)
    before = snapshot(box.root, skip=AWAY + ("main/" + DUMP,))
    done = run_decode(box)
    lines = lines_of(done)
    assert done.returncode == 0, done.stdout
    assert_job_log_is_clean(lines)
    assert lines[0] == "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ===", "the provenance comes first"
    assert lines[-1].startswith("=== END ")
    (result,) = results_of(lines)
    assert set(result) == {"decode", "n", "ended_by_eos", "cut_at_budget", "share_cut", "budget", "wall_s", "rouge_l", "bleu_1", "bleu_4"}
    assert {k: v for k, v in result.items() if k != "wall_s"} == {
        "decode": "done", "n": 5, "ended_by_eos": 4, "cut_at_budget": 1, "share_cut": 0.2, "budget": 200,
        "rouge_l": 0.19, "bleu_1": 0.31, "bleu_4": 0.05}
    assert isinstance(result["wall_s"], int) and 0 <= result["wall_s"] < 600
    log = (box.dump / "eval.log").read_text()
    assert "FAKE REPORT TEXT" in log and "EOS stop: 4/5" in log and "FAKE MIMIC TEXT" in log, "the raw output, stderr included"
    assert len((box.dump / "hyps.txt").read_text().splitlines()) == 5 and (box.dump / "refs.txt").is_file()
    assert snapshot(box.root, skip=AWAY + ("main/" + DUMP,)) == before, "R8: nothing outside the dump directory was touched"


def test_the_decode_command_is_exactly_the_published_protocol_plus_the_eos_flags(tmp_path):
    box = decode_box(tmp_path)
    assert run_decode(box).returncode == 0
    (call,) = box.calls_of("evaluate_report_generation.py")
    assert call == expected_decode_argv(box)
    (stats_call,) = box.calls_of("report_eos_stats.py")
    assert stats_call[:2] == ["scripts/report_eos_stats.py", "decode"]
    assert stats_call[stats_call.index("--budget") + 1] == "200" and stats_call[stats_call.index("--dump-dir") + 1] == DUMP
    assert stats_call[stats_call.index("--log") + 1] == DUMP + "/eval.log"
    order = [c[0] for c in box.calls() if c[0].startswith("scripts/")]
    assert order == ["scripts/evaluate_report_generation.py", "scripts/report_eos_stats.py"]


def test_the_decode_offline_environment_is_the_other_chat_jobs(tmp_path):
    box = decode_box(tmp_path)
    assert run_decode(box).returncode == 0
    assert box.env_of("evaluate_report_generation.py") == "HF_HUB_OFFLINE=1 HF_HOME={}".format(box.root / "scratch" / ".hf")


def test_the_command_the_wrapper_builds_is_accepted_by_the_evaluators_own_parser(tmp_path):
    """The fake evaluator accepts what the test says it should. This checks the command against the real thing: the argparse
    parser of evaluate_report_generation.py, taken out of its main(), and what the EOS path needs from it."""
    pytest.importorskip("torch")
    import argparse
    from unittest import mock
    from scripts import evaluate_report_generation as erg

    class Grabbed(Exception):
        pass

    grabbed = []

    def grab(self, args=None, namespace=None):
        grabbed.append(self)
        raise Grabbed

    with mock.patch.object(argparse.ArgumentParser, "parse_args", grab), mock.patch.object(sys, "argv", ["evaluate_report_generation.py"]):
        with pytest.raises(Grabbed):
            erg.main()
    box = decode_box(tmp_path)
    assert run_decode(box).returncode == 0
    (call,) = box.calls_of("evaluate_report_generation.py")
    args = grabbed[0].parse_args(call[1:])
    assert (args.stop_at_eos, args.cached_decode, args.decode, args.beam_size, args.max_new_tokens, args.prefix_k,
            args.num_samples, args.model_config) == (True, True, "beam", 3, 200, 32, 999999, "hybrid_150m_m3_rrg")
    assert args.checkpoint and args.parquet and args.dump_dir == DUMP and not args.chexbert and not args.compile_decoder
    assert not (args.smoke_test or args.retrieval_baseline), "--stop-at-eos would be refused otherwise"


def test_a_stray_environment_variable_cannot_change_the_decode(tmp_path):
    """sbatch exports the submitting shell: a BEAM_SIZE or MAX_NEW_TOKENS left in it must not reach the evaluator."""
    box = decode_box(tmp_path)
    done = run_decode(box, MAX_NEW_TOKENS="5", BEAM_SIZE="1", NUM_SAMPLES="3", MODEL_CONFIG="hybrid_150m_v2_rrg", PREFIX_K="8",
                      DECODE="greedy", CACHED_DECODE="false", STOP_AT_EOS="false", COMPILE="true")
    assert done.returncode == 0, done.stdout
    (call,) = box.calls_of("evaluate_report_generation.py")
    assert call == expected_decode_argv(box)


def test_the_default_checkpoint_is_the_eos_run_under_chat_home(tmp_path):
    box = decode_box(tmp_path)
    assert run_decode(box).returncode == 0
    (call,) = box.calls_of("evaluate_report_generation.py")
    assert call[call.index("--checkpoint") + 1] == "{}/models/report_gen_m3_eos_s42/checkpoints/last.ckpt".format(box.chat)


def test_the_checkpoint_parquet_and_dump_are_levers_and_the_done_marker_follows_the_checkpoint(tmp_path):
    box = decode_box(tmp_path)
    other = box.root / "elsewhere" / "run_b"
    (other / "checkpoints").mkdir(parents=True)
    (other / "checkpoints" / "last.ckpt").write_bytes(b"")
    (other / "DONE").write_text("x\n")
    done = run_decode(box, CKPT=str(other / "checkpoints" / "last.ckpt"), DUMP_DIR="results/chat_report_eos_b")
    assert done.returncode == 0, done.stdout
    (call,) = box.calls_of("evaluate_report_generation.py")
    assert call == expected_decode_argv(box, ckpt=str(other / "checkpoints" / "last.ckpt"), dump="results/chat_report_eos_b")
    assert (box.main / "results" / "chat_report_eos_b" / "hyps.txt").is_file()


def test_a_training_run_without_done_is_never_decoded(tmp_path):
    box = decode_box(tmp_path)
    (box.chat / "models" / "report_gen_m3_eos_s42" / "DONE").unlink()
    before = snapshot(box.root, skip=AWAY)
    done = run_decode(box)
    lines = lines_of(done)
    assert done.returncode == 1 and [l for l in lines if l.startswith("ERROR")] and any("DONE" in l for l in lines)
    assert_job_log_is_clean(lines)
    assert box.calls_of("evaluate_report_generation.py") == [] and not box.dump.exists()
    assert snapshot(box.root, skip=AWAY) == before


def test_an_existing_dump_is_never_overwritten(tmp_path):
    for existing in ("hyps.txt", "refs.txt"):
        box = decode_box(tmp_path / existing)
        box.dump.mkdir(parents=True)
        (box.dump / existing).write_text("first run\n")
        before = snapshot(box.root, skip=AWAY)
        done = run_decode(box)
        assert done.returncode == 1 and [l for l in lines_of(done) if l.startswith("ERROR")], existing
        assert box.calls_of("evaluate_report_generation.py") == []
        assert (box.dump / existing).read_text() == "first run\n" and snapshot(box.root, skip=AWAY) == before


@pytest.mark.parametrize("dump", ["outputs/chat_x", "results/not_chat_x", "results/chat_x/../../escape", "/tmp/chat_x", "results/chat_",
                                  "results"])
def test_a_dump_directory_outside_results_chat_is_refused_before_anything_is_created(tmp_path, dump):
    box = decode_box(tmp_path)
    before = snapshot(box.root, skip=AWAY)
    done = run_decode(box, DUMP_DIR=dump)
    assert done.returncode == 1 and [l for l in lines_of(done) if l.startswith("ERROR")], dump
    assert box.calls_of("evaluate_report_generation.py") == []
    assert snapshot(box.root, skip=AWAY) == before


def test_a_dump_whose_refs_differ_from_the_published_dump_fails_the_job_after_its_result(tmp_path):
    """Job 3 refuses two dumps that are not the same studies; job 1 says so first, so that afterok stops the chain before the
    CheXbert hour. The decode's own numbers are still printed, before the verdict, and the finished dump is not touched (R8)."""
    box = decode_box(tmp_path)
    (box.published / "refs.txt").write_text("a different study.\n" * 5)
    before = snapshot(box.root, skip=AWAY + ("main/" + DUMP,))
    done = run_decode(box)
    lines = lines_of(done)
    assert done.returncode == 1, done.stdout
    assert lines[-1] == "ERROR refs differ from the published dump", lines[-3:]
    (result,) = results_of(lines)
    assert result["n"] == 5 and result["ended_by_eos"] == 4, "the decode's numbers come first"
    assert lines.index("ERROR refs differ from the published dump") > lines.index([l for l in lines if l.startswith("RESULT ")][0])
    assert not [l for l in lines if l.startswith("=== END")]
    assert_job_log_is_clean(lines)
    assert (box.dump / "hyps.txt").is_file() and (box.dump / "refs.txt").is_file(), "the dump stays: nothing is deleted"
    assert snapshot(box.root, skip=AWAY + ("main/" + DUMP,)) == before


def test_refs_that_differ_by_one_byte_or_by_a_trailing_line_are_refused_too(tmp_path):
    for variant in ("one_byte", "extra_line"):
        box = decode_box(tmp_path / variant)
        original = (box.published / "refs.txt").read_text()
        (box.published / "refs.txt").write_text(original.replace("clear", "Clear", 1) if variant == "one_byte" else original + "\n")
        done = run_decode(box)
        assert done.returncode == 1 and lines_of(done)[-1] == "ERROR refs differ from the published dump", variant


@pytest.mark.parametrize("what", ["dump_missing", "refs_missing"])
def test_a_published_dump_without_refs_stops_the_job_before_the_decode_spends_any_gpu_time(tmp_path, what):
    box = decode_box(tmp_path)
    if what == "dump_missing":
        shutil.rmtree(str(box.published))
    else:
        (box.published / "refs.txt").unlink()
    before = snapshot(box.root, skip=AWAY)
    done = run_decode(box)
    lines = lines_of(done)
    assert done.returncode == 1 and any(l.startswith("ERROR") and "published dump" in l for l in lines), lines
    assert_job_log_is_clean(lines)
    assert box.calls_of("evaluate_report_generation.py") == [] and not box.dump.exists()
    assert snapshot(box.root, skip=AWAY) == before


def test_the_published_dump_is_a_lever_like_the_other_paths(tmp_path):
    box = decode_box(tmp_path)
    other = box.main / "results" / "report_gen_other"
    box.published.rename(other)
    assert run_decode(box).returncode == 1, "the default published dump is gone"
    done = run_decode(box, PUBLISHED_DIR="results/report_gen_other")
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize("missing", ["checkpoint", "parquet", "venv"])
def test_a_missing_input_stops_the_job_before_the_decoder_runs(tmp_path, missing):
    box = decode_box(tmp_path)
    if missing == "checkpoint":
        (box.chat / "models" / "report_gen_m3_eos_s42" / "checkpoints" / "last.ckpt").unlink()
    elif missing == "parquet":
        (box.root / "data" / "test.parquet").unlink()
    else:
        (box.repo / ".venv" / "bin" / "activate").unlink()
    done = run_decode(box)
    assert done.returncode == 1 and [l for l in lines_of(done) if l.startswith("ERROR")], missing
    assert_job_log_is_clean(lines_of(done))
    assert box.calls_of("evaluate_report_generation.py") == [] and not box.dump.exists()


def test_without_a_gpu_the_decode_does_not_start(tmp_path):
    """The evaluator falls back to the CPU without a word: 2,663 reports at 3 s each would take longer than the job's limit."""
    box = decode_box(tmp_path)
    done = run_decode(box, FAKE_GPUS="0")
    assert done.returncode == 1 and any(l.startswith("ERROR") and "0 GPU" in l for l in lines_of(done)), lines_of(done)
    assert box.calls_of("evaluate_report_generation.py") == [] and not box.dump.exists()


def test_a_failed_decoder_prints_only_its_exit_code_and_leaves_no_dump(tmp_path):
    box = decode_box(tmp_path, mode="fail")
    done = run_decode(box)
    lines = lines_of(done)
    assert done.returncode == 3, done.stdout
    assert "ERROR decode exit=3" in lines
    assert_job_log_is_clean(lines)
    assert "FAKE REPORT TEXT" in (box.dump / "eval.log").read_text()
    assert not (box.dump / "hyps.txt").exists() and box.calls_of("report_eos_stats.py") == []


@pytest.mark.parametrize("mode", ["noeos", "badbudget", "short"])
def test_a_dump_the_log_cannot_vouch_for_fails_the_job_and_stops_the_chain(tmp_path, mode):
    """Exit 0 from the decoder is not enough: with no EOS line the decoder did not stop at the EOS, with another budget it was
    not asked for this run, and a short dump is not the test split. The job must exit non-zero, so that afterok stops the chain."""
    box = decode_box(tmp_path, mode=mode)
    done = run_decode(box)
    lines = lines_of(done)
    assert done.returncode == 1, done.stdout
    assert any(l.startswith("ERROR") for l in lines) and not results_of(lines)
    assert_job_log_is_clean(lines)


def test_a_crashing_stats_script_leaves_its_text_in_result_err_and_none_of_it_in_the_job_log(tmp_path):
    box = decode_box(tmp_path)
    box.script("report_eos_stats.py", 'import sys\n' + LEAKY + 'raise RuntimeError("cannot read study_id=12345678 FAKE REPORT TEXT")\n')
    done = run_decode(box)
    lines = lines_of(done)
    assert done.returncode == 1, done.stdout
    assert any(l.startswith("ERROR") and "result exit=1" in l for l in lines)
    assert_job_log_is_clean(lines)
    err = (box.dump / "result.err").read_text()
    assert "Traceback" in err and "12345678" in err and "FAKE REPORT TEXT" in err, "the text landed in the file"


def test_a_requeued_decode_announces_the_restart_and_appends_to_eval_log(tmp_path):
    """An attempt that was cut short left eval.log and no dump (the dump is written after the whole decode)."""
    box = decode_box(tmp_path)
    box.dump.mkdir(parents=True)
    (box.dump / "eval.log").write_text("earlier attempt line\n" + EOS_LINE.format(1, 5, 4, 200) + "\n")
    done = run_decode(box)
    lines = lines_of(done)
    assert done.returncode == 0, done.stdout
    assert any("earlier attempt" in l and l.startswith("=== ") for l in lines), "the restart is announced"
    log = (box.dump / "eval.log").read_text()
    assert log.startswith("earlier attempt line\n") and "FAKE REPORT TEXT" in log, "appended, not overwritten"
    (result,) = results_of(lines)
    assert result["ended_by_eos"] == 4, "the counts come from the last attempt, not the first"


# ── job 2: the CheXbert scoring ───────────────────────────────────────────────

CHEXBERT = "eval_report_eos_chexbert_h100.sh"


def chexbert_box(tmp_path: Path, n: int = 5, mode: str = "ok", stamp: Optional[str] = STAMP) -> Box:
    box = Box(tmp_path, CHEXBERT, stamp=stamp)
    box.script("score_chexbert_standalone.py", FAKE_SCORER)
    write_dump(box.dump, n)
    box.stub("score.mode", mode)
    return box


def test_a_clean_scoring_prints_one_result_line_and_keeps_the_scorers_output_in_chexbert_log(tmp_path):
    box = chexbert_box(tmp_path)
    before = snapshot(box.root, skip=AWAY + ("main/" + DUMP,))
    done = box.run()
    lines = lines_of(done)
    assert done.returncode == 0, done.stdout
    assert_job_log_is_clean(lines)
    assert lines[0] == "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ==="
    assert lines[-1].startswith("=== END ")
    (result,) = results_of(lines)
    assert {k: v for k, v in result.items() if k != "wall_s"} == {"chexbert": "done", "n": 5, "micro_14": 0.4, "macro_14": 0.25,
                                                                   "micro_5": 0.45, "macro_5": 0.3}
    assert isinstance(result["wall_s"], int)
    log = (box.dump / "chexbert.log").read_text()
    assert "FAKE REPORT TEXT" in log and "FAKE MIMIC TEXT" in log and "0.4000 / 0.2500" in log
    assert (box.dump / "chexbert_metrics.json").is_file() and (box.dump / "chexbert_labels.json").is_file()
    assert snapshot(box.root, skip=AWAY + ("main/" + DUMP,)) == before


def test_the_scorer_runs_as_score_chexbert_h100_runs_it_in_the_same_environment(tmp_path):
    """Same script, same three arguments; and D24's environment: the CheXbert weights live in the DEFAULT Hugging Face cache, so
    HF_HOME is left alone, and HF_HUB_OFFLINE defaults to 0 as in the thesis wrapper (CHAT_UI_PLAN.md D24)."""
    box = chexbert_box(tmp_path)
    assert box.run().returncode == 0
    (call,) = box.calls_of("score_chexbert_standalone.py")
    assert call == ["scripts/score_chexbert_standalone.py", "--hyp-file", DUMP + "/hyps.txt", "--ref-file", DUMP + "/refs.txt",
                    "--output-dir", DUMP]
    assert box.env_of("score_chexbert_standalone.py") == "HF_HUB_OFFLINE=0 HF_HOME=unset"
    box2 = chexbert_box(tmp_path / "again")
    assert box2.run(HF_HUB_OFFLINE="1").returncode == 0
    assert box2.env_of("score_chexbert_standalone.py") == "HF_HUB_OFFLINE=1 HF_HOME=unset", "the thesis wrapper's lever"


def test_a_finished_scoring_is_never_overwritten(tmp_path):
    box = chexbert_box(tmp_path)
    (box.dump / "chexbert_labels.json").write_text("{}")
    before = snapshot(box.root, skip=AWAY)
    done = box.run()
    assert done.returncode == 1 and [l for l in lines_of(done) if l.startswith("ERROR")]
    assert box.calls_of("score_chexbert_standalone.py") == [] and snapshot(box.root, skip=AWAY) == before


def test_a_scoring_cut_short_after_its_first_file_is_run_again(tmp_path):
    """The scorer writes chexbert_metrics.json first and chexbert_labels.json last: only the second marks a finished scoring."""
    box = chexbert_box(tmp_path)
    (box.dump / "chexbert_metrics.json").write_text("{}")
    done = box.run()
    assert done.returncode == 0, done.stdout
    assert results_of(lines_of(done))[0]["micro_14"] == 0.4


@pytest.mark.parametrize("what", ["no_hyps", "no_refs", "line_counts_differ", "no_venv", "bad_dump_dir"])
def test_a_scoring_with_nothing_to_score_stops_before_the_scorer_runs(tmp_path, what):
    box = chexbert_box(tmp_path)
    extra = {}
    if what == "no_hyps":
        (box.dump / "hyps.txt").unlink()
    elif what == "no_refs":
        (box.dump / "refs.txt").unlink()
    elif what == "line_counts_differ":
        (box.dump / "hyps.txt").write_text("a b.\n")
    elif what == "no_venv":
        (box.repo / ".venv_chexbert" / "bin" / "activate").unlink()
    else:
        extra["DUMP_DIR"] = "outputs/chat_x"
    done = box.run(**extra)
    assert done.returncode == 1 and [l for l in lines_of(done) if l.startswith("ERROR")], what
    assert_job_log_is_clean(lines_of(done))
    assert box.calls_of("score_chexbert_standalone.py") == []


def test_a_failed_scorer_prints_only_its_exit_code(tmp_path):
    box = chexbert_box(tmp_path, mode="fail")
    done = box.run()
    lines = lines_of(done)
    assert done.returncode == 1, done.stdout
    assert "ERROR chexbert exit=1" in lines
    assert_job_log_is_clean(lines)
    assert "FAKE MIMIC TEXT" in (box.dump / "chexbert.log").read_text()
    assert box.calls_of("report_eos_stats.py") == []


def test_a_scorer_that_exits_0_without_the_label_matrices_fails_the_job(tmp_path):
    """bootstrap_compare needs chexbert_labels.json: a scoring that wrote only the aggregate is not finished."""
    box = chexbert_box(tmp_path, mode="nolabels")
    done = box.run()
    lines = lines_of(done)
    assert done.returncode == 1, done.stdout
    assert any(l.startswith("ERROR") for l in lines) and not results_of(lines)
    assert_job_log_is_clean(lines)


# ── job 3: the comparison ─────────────────────────────────────────────────────

COMPARE = "eval_report_eos_compare_h100.sh"


def compare_box(tmp_path: Path, worse: bool = False, stamp: Optional[str] = STAMP) -> Box:
    box = Box(tmp_path, COMPARE, stamp=stamp)
    box.script("bootstrap_compare.py", FAKE_BOOTSTRAP)
    refs = ["Findings: clear. Impression: none %d." % i for i in range(5)]
    write_dump(box.dump, 5, hyp_lines=["The heart is normal. Report %d." % i for i in range(5)], ref_lines=refs)
    write_dump(box.published, 5, hyp_lines=["No change. Stable findings. No change."] * 5, ref_lines=refs)
    for directory in (box.dump, box.published):
        (directory / "chexbert_labels.json").write_text(json.dumps({"y_true": [[0] * 14] * 5, "y_pred": [[0] * 14] * 5}))
    rows = dict(FLAT)
    if worse:
        rows["rouge_l"] = (0.19, 0.2, -0.01, -0.02, -0.005)
        rows["bleu_4"] = (0.04, 0.05, -0.01, -0.02, -0.0001)
    (box.stubs / "bootstrap.md").write_text(rendered(rows, labels={"Lung Lesion": (0.1, 0.3, -0.2, -0.3, -0.1)}))
    box.stub("boot.mode", "ok")
    return box


def test_a_clean_comparison_prints_the_stats_the_metrics_and_the_gate_last(tmp_path):
    box = compare_box(tmp_path)
    before = snapshot(box.root, skip=AWAY + ("main/" + DUMP,))
    done = box.run()
    lines = lines_of(done)
    assert done.returncode == 0, done.stdout
    assert_job_log_is_clean(lines)
    assert lines[0] == "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ==="
    assert lines[-1].startswith("=== END ")
    found = results_of(lines)
    assert [r.get("stats") for r in found[:3]] == ["eos_s42", "m3_s42", "refs"], "the stats of A, then B, then the references, first"
    assert found[0]["n"] == 5 and (found[1]["repeats_per_report"], found[1]["share_repeat"], found[1]["mean_words"]) == (1.0, 1.0, 6.0)
    assert (found[2]["n"], found[2]["mean_words"], found[2]["share_unterminated"]) == (5, 5.0, 0.0)
    assert found[3] == {"bootstrap": "paired", "name_a": "eos_s42", "name_b": "m3_s42", "n": 2663, "samples": 1000, "seed": 0}
    assert [r["metric"] for r in found[4:4 + len(MAIN_METRICS)]] == MAIN_METRICS
    assert found[4 + len(MAIN_METRICS)]["label"] == "Lung Lesion"
    assert found[-1] == {"gate": "pass", "worse": [], "label_worse": ["Lung Lesion"]} and [r for r in found if "gate" in r] == [found[-1]]
    assert (box.dump / "bootstrap_eos_s42_vs_m3_s42.md").is_file() and (box.dump / "bootstrap.log").is_file()
    assert "FAKE REPORT TEXT" in (box.dump / "bootstrap.log").read_text()
    assert snapshot(box.root, skip=AWAY + ("main/" + DUMP,)) == before, "R8: the published dump and everything else are untouched"


def test_the_bootstrap_runs_as_the_thesis_wrapper_runs_it_on_the_two_dumps(tmp_path):
    box = compare_box(tmp_path)
    assert box.run().returncode == 0
    (call,) = box.calls_of("bootstrap_compare.py")
    assert call == ["scripts/bootstrap_compare.py", "--hyps-a", DUMP + "/hyps.txt", "--hyps-b", PUBLISHED + "/hyps.txt",
                    "--refs", DUMP + "/refs.txt", "--name-a", "eos_s42", "--name-b", "m3_s42", "--bootstrap-samples", "1000",
                    "--seed", "0", "--output", DUMP + "/bootstrap_eos_s42_vs_m3_s42.md",
                    "--labels-a", DUMP + "/chexbert_labels.json", "--labels-b", PUBLISHED + "/chexbert_labels.json", "--per-label"]
    assert box.env_of("bootstrap_compare.py") == "HF_HUB_OFFLINE=1 HF_HOME={}".format(box.root / "scratch" / ".hf")
    (gate_call,) = [c for c in box.calls_of("report_eos_stats.py") if c[1] == "gate"]
    assert gate_call[gate_call.index("--bootstrap") + 1] == DUMP + "/bootstrap_eos_s42_vs_m3_s42.md"
    assert (gate_call[gate_call.index("--name-a") + 1], gate_call[gate_call.index("--name-b") + 1]) == ("eos_s42", "m3_s42")
    order = [c[1] if c[0] == "scripts/report_eos_stats.py" else c[0] for c in box.calls() if c[0].startswith("scripts/")]
    assert order == ["scripts/bootstrap_compare.py", "hyps", "gate"]
    (hyps_call,) = [c for c in box.calls_of("report_eos_stats.py") if c[1] == "hyps"]
    assert hyps_call[2:] == ["eos_s42=" + DUMP + "/hyps.txt", "m3_s42=" + PUBLISHED + "/hyps.txt", "refs=" + DUMP + "/refs.txt"]


def test_a_failing_gate_is_a_finding_not_a_crash(tmp_path):
    box = compare_box(tmp_path, worse=True)
    done = box.run()
    lines = lines_of(done)
    assert done.returncode == 0, done.stdout
    assert_job_log_is_clean(lines)
    found = results_of(lines)
    assert found[-1] == {"gate": "fail", "worse": ["rouge_l", "bleu_4"], "label_worse": ["Lung Lesion"]}


@pytest.mark.parametrize("what", ["no_published_dump", "no_published_labels", "no_eos_labels", "no_eos_hyps", "refs_differ",
                                  "output_exists", "bad_dump_dir", "no_venv"])
def test_a_comparison_whose_inputs_do_not_pair_stops_before_anything_runs(tmp_path, what):
    box = compare_box(tmp_path)
    extra = {}
    if what == "no_published_dump":
        shutil.rmtree(str(box.published))
    elif what == "no_published_labels":
        (box.published / "chexbert_labels.json").unlink()
    elif what == "no_eos_labels":
        (box.dump / "chexbert_labels.json").unlink()
    elif what == "no_eos_hyps":
        (box.dump / "hyps.txt").unlink()
    elif what == "refs_differ":
        (box.published / "refs.txt").write_text("a different study.\n" * 5)
    elif what == "output_exists":
        (box.dump / "bootstrap_eos_s42_vs_m3_s42.md").write_text("an earlier comparison\n")
    elif what == "bad_dump_dir":
        extra["DUMP_DIR"] = "outputs/chat_x"
    else:
        (box.repo / ".venv" / "bin" / "activate").unlink()
    before = snapshot(box.root, skip=AWAY)
    done = box.run(**extra)
    assert done.returncode == 1 and [l for l in lines_of(done) if l.startswith("ERROR")], what
    assert_job_log_is_clean(lines_of(done))
    assert box.calls() == [] and snapshot(box.root, skip=AWAY) == before


def test_a_crashing_bootstrap_prints_only_its_exit_code(tmp_path):
    box = compare_box(tmp_path)
    box.stub("boot.mode", "fail")
    done = box.run()
    lines = lines_of(done)
    assert done.returncode == 4, done.stdout
    assert "ERROR bootstrap exit=4" in lines
    assert_job_log_is_clean(lines)
    assert "FAKE MIMIC TEXT" in (box.dump / "bootstrap.log").read_text()
    assert [c[1] for c in box.calls_of("report_eos_stats.py")] == [], "no stats and no gate after a failed bootstrap"


def test_a_report_the_gate_cannot_parse_gives_an_error_and_never_a_gate_line(tmp_path):
    box = compare_box(tmp_path)
    (box.stubs / "bootstrap.md").write_text(rendered(FLAT).replace("| metric | eos_s42 | m3_s42 |", "| metric | m3_s42 | eos_s42 |"))
    done = box.run()
    lines = lines_of(done)
    assert done.returncode == 1, done.stdout
    assert any(l.startswith("ERROR") for l in lines)
    assert not [r for r in results_of(lines) if "gate" in r], "no verdict without a parsed comparison"
    assert_job_log_is_clean(lines)


def test_a_crashing_stats_script_in_the_comparison_leaves_its_text_in_compare_err(tmp_path):
    box = compare_box(tmp_path)
    box.script("report_eos_stats.py", 'import sys\n' + LEAKY + 'raise RuntimeError("cannot read study_id=12345678 FAKE REPORT TEXT")\n')
    done = box.run()
    lines = lines_of(done)
    assert done.returncode == 1, done.stdout
    assert_job_log_is_clean(lines)
    err = (box.dump / "compare.err").read_text()
    assert "Traceback" in err and "12345678" in err and "FAKE REPORT TEXT" in err


# ── provenance: all three wrappers print the sync stamp first ─────────────────

def real_stamp(root: Path, dirty: bool) -> str:
    """The .sync_stamp that `scripts/chat_remote.sh sync` writes, from the real script in a throwaway git tree with ssh and
    rsync stubbed: the wrappers have to read what the producer writes."""
    box = Sandbox(root)
    if dirty:
        script = box.repo / "scripts" / "chat_remote.sh"
        script.write_text(script.read_text() + "\n# touched\n")
    done = box.run("sync")
    assert done.returncode == 0, done.stdout + done.stderr
    return (box.repo / ".sync_stamp").read_text().strip()


WRAPPERS = [DECODE, CHEXBERT, COMPARE]


@pytest.mark.parametrize("dirty", [False, True])
def test_the_first_line_of_every_job_log_names_the_commit_and_cleanliness_the_tree_was_synced_with(tmp_path, dirty):
    stamp = real_stamp(tmp_path / "sync", dirty)
    assert STAMP_RE.match(stamp), stamp
    _, sha, flag = stamp.split()
    for wrapper in WRAPPERS:
        box = Box(tmp_path / wrapper, wrapper, stamp=stamp)           # no inputs at all: the job stops at its first guard
        done = box.run()
        assert lines_of(done)[0] == "=== sync {} {} ===".format(sha, "dirty" if dirty else "clean"), (wrapper, lines_of(done)[:2])
        assert done.returncode == 1


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_a_missing_sync_stamp_is_reported_as_unknown_and_the_job_goes_on(tmp_path, wrapper):
    box = Box(tmp_path, wrapper, stamp=None)
    done = box.run()
    assert lines_of(done)[0] == "=== sync unknown ===" and done.returncode == 1


@pytest.mark.parametrize("stamp", [
    "",                                                                              # empty
    "rm -rf / ; echo $(whoami) FAKE_STAMP_TEXT",                                     # not the shape at all
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d4 clean",           # 39 hex digits
    "2026-10-09T20:00:00Z 3F2A9C41D7E86B05A1C4E9D3B7F60285AC9E1D47 clean",          # upper case
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 maybe",          # a flag that is neither
    "2026-10-09T20:00:00Z 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean FAKE_EXTRA",   # a field too many
])
@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_a_malformed_sync_stamp_is_reported_as_unknown_and_never_echoed(tmp_path, wrapper, stamp):
    box = Box(tmp_path, wrapper, stamp=stamp)
    lines = lines_of(box.run())
    assert lines[0] == "=== sync unknown ===", lines[:2]
    assert not [l for l in lines if "FAKE_" in l or "whoami" in l or "rm -rf" in l]
