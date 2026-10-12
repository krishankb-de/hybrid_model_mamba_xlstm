"""CHAT_UI_PLAN.md P8-F (part 1): the timing job, scripts/chat_decode_timing.py and its wrapper scripts/chat_decode_timing_h100.sh.

The live server's `generate` stage takes 7.4 to 8.1 s for a decode that job 2590184 measured at 3.05 s on the same node. The job decodes the
same five test rows seven ways, each adding one part of the live path to the one before: the bare cached beam search (A), with a no-op
`on_step` (B), through `Engine.generate` (C), in the pipeline's one worker thread (D), through the pipeline with a store on the NFS home (E)
and on node-local disk (E-local), and through HTTP (F). It prints numbers only, because the real run decodes MIMIC test images.

Everything here runs on the laptop over the tiny engine and the tiny gallery (the P5-E fixtures): CPU, offline, synthetic data only (R7), so an
assertion may print whatever it compares. Tested in layers:
  * the arguments, and what is refused before anything is loaded;
  * the pure pieces: the arms' lines, the ratio to A, the summary, the sanity check, the CPU string;
  * the arms one by one, each shown to run the way the ruling says (thread, callback, store, port);
  * the whole job in process, and as a script in a process of its own;
  * R7: SECRET text in the reports, the ids, the paths and the exceptions never reaches stdout, stderr or a file;
  * the wrapper, rehearsed in a temp tree with a stub interpreter.
The static pins on the script and the wrapper are at the end of tests/test_willi_parity.py.
"""
import functools
import io
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time as real_time
from pathlib import Path

import pytest
import torch

from app import server
from app.engine import build_engine
from app.gallery import Gallery
from app.pipeline import PUBLISHED_PROTOCOL, Pipeline
from app.schemas import Options
from app.store import Store
from scripts import chat_decode_timing as timing
from scripts import chat_realturn_probe as probe
from scripts.build_retrieval_gallery import build_tiny
from tests import wrapper_rehearsal as wr
from tests.app_helpers import decide_gate
from tests.wrapper_rehearsal import job_lines, safe_line

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "chat_decode_timing.py"
WRAPPER = "chat_decode_timing_h100.sh"
SECRET = "S3CR3T-sentinel"


def run(args):
    """main() with its own two streams. -> (exit status, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = timing.main(list(args), out, err)
    return code, out.getvalue(), err.getvalue()


def refusal_line(err):
    """The one ERROR line a refused argument writes to stderr, without the word."""
    lines = err.splitlines()
    assert len(lines) == 1 and lines[0].startswith("ERROR "), lines
    return lines[0][len("ERROR "):]


# ---- the arguments: what is refused before anything is loaded ------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path):
    (tmp_path / "home").mkdir()
    (tmp_path / "gallery").mkdir()
    return tmp_path


def base_args(root, *extra):
    return ["--home", str(root / "home"), "--gallery", str(root / "gallery"), "--job", "j1", "--engine", "tiny"] + list(extra)


def test_help_goes_to_stdout_with_exit_0_and_names_every_option_and_the_result_lines():
    code, out, err = run(["--help"])
    assert code == 0 and err == "" and out.startswith("usage: ")
    for word in ("--home", "--gallery", "--job", "--engine", "--threads", "--rows", "--local-dir", "RESULT"):
        assert word in out, word


@pytest.mark.parametrize("argv", [
    [],                                                         # no home, no gallery
    ["--home", "{home}"],                                       # no gallery
    ["--gallery", "{gallery}"],                                 # no home
    ["--bogus", SECRET],                                        # an option it does not have
    ["--token", SECRET],
    ["--hom", SECRET],                                          # no abbreviation
    ["{home}"],                                                 # a positional
])
def test_a_missing_or_unknown_argument_is_refused_in_one_line_that_quotes_nothing(argv, home):
    argv = [a.format(home=home / "home", gallery=home / "gallery") for a in argv]
    code, out, err = run(argv)
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert SECRET not in line and str(home) not in line


@pytest.mark.parametrize("flag, value", [
    ("--job", ""), ("--job", "../x"), ("--job", "a/b"), ("--job", "a b"), ("--job", "x" * 65), ("--job", "jé"), ("--job", "."), ("--job", ".."),
    ("--rows", "4-1"), ("--rows", "x"), ("--rows", "0-9"), ("--rows", "1,1"), ("--rows", ""), ("--rows", "10000"), ("--rows", "9999,10001"),
    ("--threads", "0"), ("--threads", "-1"), ("--threads", "many"), ("--threads", "1000"),
    ("--engine", "gpu"), ("--engine", SECRET),
])
def test_a_value_that_will_not_do_is_refused_in_one_line_without_being_quoted(flag, value, home):
    code, out, err = run(base_args(home, flag, value))
    assert code == 2 and out == ""
    line = refusal_line(err)
    assert flag in line or flag.lstrip("-") in line
    if value and not value.lstrip("-").isdigit():
        assert value not in line, "a refusal does not quote the value it was given"


def test_a_home_that_does_not_exist_is_refused_and_nothing_is_made(tmp_path):
    (tmp_path / "gallery").mkdir()
    code, out, err = run(["--home", str(tmp_path / "nope"), "--gallery", str(tmp_path / "gallery"), "--job", "j1", "--engine", "tiny"])
    assert code == 2 and out == "" and "home" in refusal_line(err)
    assert not (tmp_path / "nope").exists()


def test_a_gallery_that_does_not_exist_is_refused_before_anything_is_loaded(home):
    code, out, err = run(["--home", str(home / "home"), "--gallery", str(home / "nope"), "--job", "j1", "--engine", "tiny"])
    assert code == 2 and out == "" and "gallery" in refusal_line(err)
    assert list((home / "home").iterdir()) == [], "the timing directory is made only after every argument has passed"


def test_a_home_inside_the_repository_is_refused_as_the_server_refuses_it(home):
    inside = REPO_ROOT / "app"   # an existing directory of the checkout: CHAT_HOME is never inside one (the DUA)
    code, out, err = run(["--home", str(inside), "--gallery", str(home / "gallery"), "--job", "j1", "--engine", "tiny"])
    assert code == 2 and out == "" and "home" in refusal_line(err)
    assert not (inside / "timing").exists()


def test_a_local_directory_that_does_not_exist_is_refused(home):
    code, out, err = run(base_args(home, "--local-dir", str(home / "nope")))
    assert code == 2 and out == "" and "local" in refusal_line(err)


def test_the_timing_directory_of_a_job_is_made_fresh_and_one_that_exists_is_never_reused_or_touched(home):
    """R8: additive only. `<home>/timing/<job>` holds the stores of E and F and is left in place; a second run with the same job id would
    meet the first one's stores, so it is refused, with the first one's files as they were."""
    stale = home / "home" / "timing" / "j1"
    stale.mkdir(parents=True)
    (stale / "keep.txt").write_text(SECRET)
    code, out, err = run(base_args(home))
    assert code == 2 and out == "" and "timing" in refusal_line(err)
    assert [p.name for p in stale.iterdir()] == ["keep.txt"] and (stale / "keep.txt").read_text() == SECRET


# ---- the pure pieces ---------------------------------------------------------------------------------------------------------------

def arm_of(arm, ms, total=None, reports=None, tokens=100, error=None):
    return timing.ArmResult(arm, ms=list(ms), total_ms=list(total or []), reports=list(["r{}".format(i) for i in range(len(ms))] if reports is None else reports),
                            tokens=[tokens] * len(ms), error=error)


def ladder(means):
    """The seven arms with five rows each at the given means, as a dict in ladder order."""
    out = {}
    for arm, mean in zip(timing.ARMS, means):
        has_total = arm in ("E", "E-local", "F")
        out[arm] = arm_of(arm, [mean] * 5, [mean + 400.0] * 5 if has_total else None)
    return out


def test_the_arms_are_the_rulings_seven_in_their_order_and_the_threshold_is_one_and_a_half_times_a():
    assert timing.ARMS == ("A", "B", "C", "D", "E", "E-local", "F") and timing.OVER == 1.5
    assert timing.DEFAULT_ROWS == "0-4" and timing.MODEL == "hybrid_150m_m3_rrg"


def test_an_arm_line_has_its_rows_mean_and_ratio_to_a_and_says_whether_its_reports_were_a_s():
    a, d = arm_of("A", [3000.0, 3010.5, 3020.0, 3030.0, 3040.0]), arm_of("D", [4500.04, 4600.0, 4700.0, 4800.0, 4900.06])
    base = timing.mean_ms(a)
    assert base == 3020.1
    assert timing.arm_payload(a, base, True) == {"arm": "A", "ms": [3000.0, 3010.5, 3020.0, 3030.0, 3040.0], "mean": 3020.1, "ratio": 1.0,
                                                 "same_report": True}
    line = timing.arm_payload(d, base, False)
    assert line == {"arm": "D", "ms": [4500.0, 4600.0, 4700.0, 4800.0, 4900.1], "mean": 4700.0, "ratio": 1.56, "same_report": False}
    assert list(line) == ["arm", "ms", "mean", "ratio", "same_report"]


def test_the_pipeline_and_http_arms_also_carry_the_turn_and_its_mean():
    e = arm_of("E", [7000.0, 7100.0], total=[7500.0, 7650.5])
    line = timing.arm_payload(e, 3000.0, True)
    assert line == {"arm": "E", "ms": [7000.0, 7100.0], "mean": 7050.0, "ratio": 2.35, "total_ms": [7500.0, 7650.5], "mean_total_ms": 7575.2,
                    "same_report": True}
    assert list(line) == ["arm", "ms", "mean", "ratio", "total_ms", "mean_total_ms", "same_report"]


def test_an_arm_that_failed_is_its_class_and_the_rows_it_finished_and_has_no_mean_or_ratio():
    failed = arm_of("E", [7000.0, 7100.0], total=[7500.0, 7600.0], error="RuntimeError")
    assert timing.arm_payload(failed, 3000.0, False) == {"arm": "E", "error": "RuntimeError", "rows_done": 2, "ms": [7000.0, 7100.0]}
    never = arm_of("F", [], error="OSError")
    assert timing.arm_payload(never, 3000.0, False) == {"arm": "F", "error": "OSError", "rows_done": 0, "ms": []}


def test_with_no_baseline_the_ratio_is_none_and_never_a_division_by_zero():
    assert timing.arm_payload(arm_of("B", [10.0]), None, False)["ratio"] is None
    assert timing.arm_payload(arm_of("B", [10.0]), 0.0, False)["ratio"] is None


def test_the_summary_names_the_first_arm_whose_mean_is_over_one_and_a_half_times_a():
    results = ladder([3000.0, 3050.0, 3100.0, 4000.0, 7800.0, 7700.0, 8100.0])
    assert timing.first_over(results) == "E"
    results = ladder([3000.0, 3050.0, 3100.0, 4501.0, 7800.0, 7700.0, 8100.0])
    assert timing.first_over(results) == "D"
    assert timing.first_over(ladder([3000.0, 3050.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0])) == "none"


def test_exactly_one_and_a_half_times_a_does_not_exceed_it():
    results = ladder([3000.0, 3000.0, 3000.0, 4500.0, 4500.0, 4500.0, 4500.01])
    assert timing.first_over(results) == "F"


def test_an_arm_that_failed_is_skipped_by_the_summary_and_without_a_baseline_it_cannot_say():
    results = ladder([3000.0, 3050.0, 3100.0, 9000.0, 9000.0, 9000.0, 9000.0])
    results["D"].error = "ValueError"
    assert timing.first_over(results) == "E"
    results["A"].error = "ValueError"
    assert timing.first_over(results) == "unknown", "no A, no ratio: not the same as none"
    del results["A"]
    assert timing.first_over(results) == "unknown"


def test_the_summary_line_has_the_arm_the_sanity_check_the_tokens_and_the_failed_arms():
    results = ladder([3000.0, 3050.0, 3100.0, 3200.0, 7800.0, 7700.0, 8100.0])
    assert timing.summary_payload(results, True) == {"summary": {"first_over_1_5x": "E", "same_report_all_arms": True, "tokens": [100] * 5,
                                                                 "arms_failed": []}}
    results["F"].error = "OSError"
    summary = timing.summary_payload(results, False)["summary"]
    assert summary["same_report_all_arms"] is False and summary["arms_failed"] == ["F"]


def test_the_sanity_check_compares_every_arms_report_with_as_row_by_row_in_memory():
    results = ladder([1.0] * 7)
    same = timing.same_reports(results)
    assert same == {arm: True for arm in timing.ARMS} and all(same.values())
    results["D"].reports[3] = "a different report"
    same = timing.same_reports(results)
    assert same["D"] is False and [arm for arm, ok in same.items() if not ok] == ["D"]


def test_a_missing_report_a_short_arm_or_a_failed_arm_does_not_pass_the_sanity_check():
    for wreck in (lambda r: r["C"].reports.__setitem__(1, None), lambda r: r["C"].reports.pop(), lambda r: setattr(r["C"], "error", "X"),
                  lambda r: r["C"].reports.append("one more")):
        results = ladder([1.0] * 7)
        wreck(results)
        assert timing.same_reports(results)["C"] is False
    results = ladder([1.0] * 7)
    results["A"].reports[0] = None
    assert not any(timing.same_reports(results).values()), "without A's report there is nothing to compare with"


def test_a_result_line_is_json_on_one_line_under_the_300_characters_the_summary_keeps_even_for_the_most_rows_and_the_slowest_arm():
    rows = [12345.6, 23456.7, 34567.8, 45678.9, 56789.0, 67890.1, 78901.2, 89012.3][:timing.MAX_ROWS]
    line = timing.result_line(timing.arm_payload(arm_of("E-local", rows, total=[r + 999.9 for r in rows]), 3000.0, False))
    assert line.startswith("RESULT {") and "\n" not in line and len(line) < 300, len(line)
    json.loads(line[len("RESULT "):])
    env = timing.result_line(timing.env_payload("gx14.some.long.node.name.example.org", "Intel(R) Xeon(R) Platinum 8480+ @ 2.00GHz", 8, 4,
                                                [1.0, 2.5, 3.25], 8, 8, "real", list(range(timing.MAX_ROWS))))
    assert len(env) < 300, len(env)
    widest = timing.result_line(timing.env_payload("n" * 100, "c" * 100, 1024, 1024, [1000.55, 1000.55, 1000.55], 9999, 9999, "real",
                                                   [timing.MAX_ROW] * timing.MAX_ROWS))
    assert len(widest) < 300, "the node and the CPU are cut, and the numbers are bounded: even the widest environment line is kept whole"


def test_a_non_finite_number_is_no_json_and_is_refused_rather_than_printed():
    with pytest.raises(ValueError):
        timing.result_line({"arm": "A", "mean": float("nan")})


def test_the_cpu_model_is_the_first_model_name_of_proc_cpuinfo_and_only_plain_characters_survive(tmp_path):
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz\nprocessor\t: 1\n"
                    "model name\t: another\n")
    assert timing.cpu_model(str(info)) == "Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz"
    info.write_text('model name\t: Odd "CPU" {x}\\ é 9000\n')
    assert timing.cpu_model(str(info)) == "Odd _CPU_ _x__ _ 9000"
    info.write_text("processor\t: 0\n")
    assert timing.cpu_model(str(info)) is None
    assert timing.cpu_model(str(tmp_path / "missing")) is None
    info.write_text("model name\t: " + "x" * 500 + "\n")
    assert len(timing.cpu_model(str(info))) == timing.CPU_MAX == 60


def test_the_node_name_and_the_cpu_string_that_reach_the_environment_line_are_plain_words():
    assert timing.plain("gx14", 64) == "gx14"
    assert timing.plain('a b"c\nd', 64) == "a_b_c_d"
    assert timing.plain("x" * 200, 64) == "x" * 64
    assert timing.NODE_MAX == 32 and timing.MAX_ROW == 9999


def test_the_physical_cores_of_the_affinity_are_counted_from_sysfs_and_none_where_it_cannot_be_told(tmp_path):
    """Eight logical CPUs on four cores (two hardware threads each) are four cores, on eight cores they are eight: what a job's allocation can
    change in a decode that is bound by compute, and that no thread count shows."""
    sysfs = tmp_path / "cpu"
    for cpu in range(8):
        topology = sysfs / "cpu{}".format(cpu) / "topology"
        topology.mkdir(parents=True)
        (topology / "core_id").write_text("{}\n".format(cpu % 4))
        (topology / "physical_package_id").write_text("0\n")
    assert timing.core_count(range(8), str(sysfs)) == 4
    assert timing.core_count([0, 1, 2, 3], str(sysfs)) == 4 and timing.core_count([0, 4], str(sysfs)) == 1
    (sysfs / "cpu5" / "topology" / "physical_package_id").write_text("1\n")   # a core number of another package is another core
    assert timing.core_count(range(8), str(sysfs)) == 5
    assert timing.core_count(range(9), str(sysfs)) is None, "cpu8 has no topology: a count that is a guess is none"
    assert timing.core_count([], str(sysfs)) is None
    (sysfs / "cpu5" / "topology" / "core_id").write_text("not a number\n")
    assert timing.core_count(range(8), str(sysfs)) is None
    assert timing.core_count() is None or timing.core_count() >= 1, "the real affinity and the real sysfs: a count or none"


def test_the_load_average_is_the_nodes_three_numbers_rounded_and_none_where_there_is_none(monkeypatch):
    monkeypatch.setattr(os, "getloadavg", lambda: (0.504, 1.25, 100.999), raising=False)
    assert timing.load_average() == [0.5, 1.25, 101.0]

    def broken():
        raise OSError("no load average")

    monkeypatch.setattr(os, "getloadavg", broken, raising=False)
    assert timing.load_average() is None
    monkeypatch.delattr(os, "getloadavg", raising=False)
    assert timing.load_average() is None


def test_the_environment_payload_carries_the_cores_and_the_load_beside_the_affinity():
    env = timing.env_payload("gx14", "Intel(R) Xeon(R)", 8, 4, [1.0, 2.5, 3.25], 8, 8, "real", [0, 1])["env"]
    assert list(env) == ["node", "cpu", "affinity", "cores", "load", "threads_main", "threads_worker", "engine", "rows"]
    assert (env["affinity"], env["cores"], env["load"]) == (8, 4, [1.0, 2.5, 3.25])
    none = timing.env_payload("n", None, None, None, None, 1, 1, "tiny", [0])["env"]
    assert (none["cpu"], none["affinity"], none["cores"], none["load"]) == (None, None, None, None)


# ---- the world: the engine as the server builds it, the picker's rows, the published protocol -----------------------------------------

@pytest.fixture(scope="module")
def tiny_gallery(tmp_path_factory):
    """The P5-E gallery over the tiny world, its gate decided, carrying the tiny engine's own tower hash: the job opens a gallery against its
    engine's tower, the real job's check, so that the stand-in meets it too."""
    root = tmp_path_factory.mktemp("timing_gallery") / "gallery"
    build_tiny(root)
    decide_gate(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["tower_sha256"] = build_engine("tiny").tower_sha256()
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def tiny_args(root, gallery, *extra):
    """The parsed arguments of a run in `root` (a fresh home and a fresh local directory under it)."""
    for name in ("home", "local"):
        (root / name).mkdir(exist_ok=True)
    return timing.build_parser().parse_args(["--home", str(root / "home"), "--gallery", str(gallery), "--job", "j1", "--engine", "tiny",
                                             "--local-dir", str(root / "local")] + list(extra))


def load(root, gallery, *extra):
    args = tiny_args(root, gallery, *extra)
    return timing.load_world(args, timing.check_arguments(args))


@pytest.fixture(scope="module")
def world(tmp_path_factory, tiny_gallery):
    return load(tmp_path_factory.mktemp("timing_world"), tiny_gallery)


def spy(target, name, seen, monkeypatch):
    """Replace target.name with a wrapper that notes (thread, args, kwargs) of every call in `seen` and calls through."""
    real = getattr(target, name)

    @functools.wraps(real)
    def wrapper(*args, **kwargs):
        seen.append((threading.current_thread(), args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapper)
    return seen


class Clock:
    """In place of the time module: perf_counter moves `step` seconds at every reading, so that a measurement is a number a test can say."""

    def __init__(self, step):
        self.now, self.step = 0.0, step

    def perf_counter(self):
        self.now += self.step
        return self.now

    def __getattr__(self, name):
        return getattr(real_time, name)


def test_the_world_is_the_published_protocol_on_the_cached_decode_over_the_pickers_five_rows(world, tiny_gallery):
    assert (world.kind, world.name, world.rows) == ("tiny", "tiny", [0, 1, 2, 3, 4]) and len(world.encoded) == 5
    opts = world.opts
    assert (opts.decode, opts.beam_size, opts.max_new_tokens, opts.stop_on_repeat, opts.compile, opts.cached_decode) == (
        "beam", 3, 100, False, False, True), "PUBLISHED_PROTOCOL, with the cached decode the server defaults to"
    assert all(getattr(opts, key) == value for key, value in PUBLISHED_PROTOCOL.items())
    assert opts == Options(cached_decode=True, **PUBLISHED_PROTOCOL)
    gallery = Gallery.open(tiny_gallery, world.engine.tower_sha256())
    for row, enc in zip(world.rows, world.encoded):   # the rows' images are the gallery's test images, preprocessed and encoded as an upload is
        _, prepared = world.engine.preprocess(Path(gallery.test_study(row)["image"]).read_bytes())
        assert torch.equal(enc.prefix, world.engine.encode(prepared)[1].prefix)


def test_the_rows_are_the_ones_asked_for_and_each_is_a_different_study(tmp_path, tiny_gallery):
    two = load(tmp_path, tiny_gallery, "--rows", "3,1")
    assert two.rows == [3, 1] and len(two.encoded) == 2
    assert not torch.equal(two.encoded[0].prefix, two.encoded[1].prefix), "rows 3 and 1 are two images"


def test_the_real_engine_is_built_as_the_server_builds_it_the_published_model_on_the_cpu_from_the_repo(tmp_path, tiny_gallery, monkeypatch):
    built = []

    def fake(kind, **kw):   # the checkpoint is 2.4 GB: a tiny engine named as the registry names the model
        built.append((kind, kw))
        engine = build_engine("tiny")
        engine.name = kw.get("name") or kw["model_config"]
        engine._card.update(name=engine.name, checkpoint=kw["checkpoint"])
        return engine

    monkeypatch.setattr(server, "build_engine", fake)
    world = load(tmp_path, tiny_gallery, "--engine", "real", "--threads", "3")
    assert (world.kind, world.name) == ("real", "hybrid_150m_m3_rrg") and world.name == timing.MODEL
    assert [kind for kind, _ in built] == ["real"]
    kw = built[0][1]
    assert (kw["model_config"], kw["device"], kw["threads"]) == ("hybrid_150m_m3_rrg", "cpu", 3)
    assert kw["checkpoint"] == str(REPO_ROOT / "outputs" / "h100_report_gen_m3_tower13d_s42" / "checkpoints" / "last.ckpt")


def test_the_gallery_is_opened_against_the_engines_own_tower_and_a_setup_that_fails_says_which_step_by_class(tmp_path, tiny_gallery):
    other = tmp_path / "other" / "gallery"
    shutil.copytree(str(tiny_gallery), str(other))
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["tower_sha256"] = "0" * 64
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(timing.SetupFailed) as caught:
        load(tmp_path, other)
    assert (caught.value.step, caught.value.error) == ("gallery", "GalleryMismatch")


def test_a_row_the_gallery_does_not_have_is_a_setup_failure_at_the_rows_step(tmp_path, tiny_gallery):
    with pytest.raises(timing.SetupFailed) as caught:
        load(tmp_path, tiny_gallery, "--rows", "400")   # the tiny gallery has 40 test studies
    assert (caught.value.step, caught.value.error) == ("rows", "IndexError")


def test_the_environment_is_the_node_the_cpu_the_affinity_and_torchs_threads_in_the_main_thread_and_in_a_worker(world, monkeypatch):
    seen, real = [], torch.get_num_threads

    def counting():
        seen.append(threading.current_thread())
        return real()

    monkeypatch.setattr(torch, "get_num_threads", counting)
    env = timing.environment(world)["env"]
    assert env["node"] == timing.plain(platform.node(), 64) and env["cpu"] == timing.cpu_model()
    assert env["engine"] == "tiny" and env["rows"] == [0, 1, 2, 3, 4]
    assert env["threads_main"] == env["threads_worker"] == real()
    assert any(t is threading.main_thread() for t in seen) and any(t is not threading.main_thread() for t in seen)
    assert env["affinity"] == (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    assert env["cores"] == timing.core_count() and (env["cores"] is None or env["cores"] <= env["affinity"])
    assert env["load"] is None or (len(env["load"]) == 3 and all(v >= 0 for v in env["load"]))


# ---- the warm-up and arms A to D --------------------------------------------------------------------------------------------------

def test_one_warm_up_turn_that_no_arm_counts_is_a_whole_generate_on_row_0_in_the_main_thread(world, monkeypatch):
    seen = spy(world.engine, "generate", [], monkeypatch)
    timing.warm_up(world)
    assert len(seen) == 1
    thread, args, _ = seen[0]
    assert thread is threading.main_thread() and args[0] is world.encoded[0] and args[1] == world.opts


def test_arm_a_is_the_cached_beam_search_alone_with_no_callback_in_the_main_thread(world, monkeypatch):
    searched = spy(world.engine.decoder, "beam_search_cached", [], monkeypatch)
    generated = spy(world.engine, "generate", [], monkeypatch)
    result = timing.run_arm("A", world)
    assert result.error is None and len(searched) == 5 and generated == [], "the decoder's own search, not the engine's generate"
    for (thread, args, kwargs), enc in zip(searched, world.encoded):
        assert thread is threading.main_thread()
        assert kwargs["on_step"] is None and kwargs["beam_size"] == 3 and kwargs["max_new_tokens"] == 100
        assert kwargs["prefix_embeds"] is enc.prefix and tuple(args[0].shape) == (1, 0), "no BOS: the input ids are empty, as in training"
    assert len(result.ms) == len(result.reports) == 5 and result.tokens == [100] * 5 and all(ms > 0 for ms in result.ms)
    assert result.total_ms == []


def test_arm_b_is_arm_a_with_a_callback_that_does_nothing(world, monkeypatch):
    searched = spy(world.engine.decoder, "beam_search_cached", [], monkeypatch)
    result = timing.run_arm("B", world)
    assert result.error is None and len(searched) == 5
    for thread, _, kwargs in searched:
        assert thread is threading.main_thread() and callable(kwargs["on_step"])
        assert kwargs["on_step"](0, [1, 2, 3]) is None
    assert result.tokens == [100] * 5


def test_arm_c_is_the_engines_generate_in_the_main_thread_with_a_no_op_snapshot_and_a_cancel_that_is_never_set(world, monkeypatch):
    seen = spy(world.engine, "generate", [], monkeypatch)
    result = timing.run_arm("C", world)
    assert result.error is None and len(seen) == 5
    for (thread, args, kwargs), enc in zip(seen, world.encoded):
        assert thread is threading.main_thread() and args[0] is enc and args[1] == world.opts
        assert callable(args[2]) and args[2](1, "text") is None
        assert isinstance(args[3], threading.Event) and not args[3].is_set()
    assert result.tokens == [100] * 5 and len(result.reports) == 5


def test_arm_d_is_c_inside_a_one_worker_thread_pool_with_a_warm_up_turn_in_that_thread_that_is_not_counted(world, monkeypatch):
    seen = spy(world.engine, "generate", [], monkeypatch)
    result = timing.run_arm("D", world)
    assert result.error is None and len(seen) == 6, "one uncounted turn in the new thread, then the five rows"
    threads = {thread for thread, _, _ in seen}
    assert len(threads) == 1 and threads != {threading.main_thread()} and next(iter(threads)).name.startswith("turn")
    assert seen[0][1][0] is world.encoded[0] and [args[0] for _, args, _ in seen[1:]] == list(world.encoded)
    assert len(result.ms) == len(result.reports) == 5


def test_the_four_arms_decode_the_same_report_for_every_row(world):
    results = {arm: timing.run_arm(arm, world) for arm in ("A", "B", "C", "D")}
    assert all(r.error is None for r in results.values())
    assert all(timing.same_reports(results).values()), "the same decode, however it is reached"
    assert all(len(report.split()) > 3 for report in results["A"].reports)


@pytest.mark.parametrize("arm", ["A", "B", "C", "D"])
def test_an_arm_is_timed_with_perf_counter_around_the_decode_alone(arm, world, monkeypatch):
    monkeypatch.setattr(timing, "time", Clock(0.0123))
    result = timing.run_arm(arm, world)
    assert result.ms == [pytest.approx(12.3)] * 5, "two readings of the clock per row: the start of the call and its end"


def test_an_arm_that_raises_is_named_by_its_class_and_never_by_its_text_and_keeps_the_rows_it_finished(world, monkeypatch):
    real, calls = world.engine.decoder.beam_search_cached, []

    def broken(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("an exception text with " + SECRET)
        return real(*args, **kwargs)

    monkeypatch.setattr(world.engine.decoder, "beam_search_cached", broken)
    result = timing.run_arm("A", world)
    assert result.error == "RuntimeError" and len(result.ms) == 2 and len(result.reports) == 2
    assert SECRET not in repr(result) and "exception text" not in repr(result)


# ---- arms E and E-local: the pipeline's whole turn, a real store, the real worker, the events drained as the SSE bridge drains them ------

@pytest.fixture
def fresh(tmp_path, tiny_gallery):
    """A world of this test's own: an arm with a store makes its directory under it, and a directory that exists is never reused."""
    return load(tmp_path, tiny_gallery)


def stored_turns(home):
    """The assistant messages of the one session in the store at `home`, each with its stored events, read through a second Store once the arm
    has closed its own."""
    store = Store(home)
    try:
        sessions, _ = store.list_sessions(None, 10)
        assert len(sessions) == 1, sessions
        messages = [m for m in store.get_session(sessions[0]["id"], None)["messages"] if m["role"] == "assistant"]
        return [(m, store.events_after(m["id"], 0)) for m in messages]
    finally:
        store.close()


def stage_ms(events, stage):
    return next(e["data"]["ms"] for e in events if e["event"] == "stage_end" and e["data"].get("stage") == stage)


def test_arm_e_is_the_real_pipeline_on_the_real_worker_over_a_store_on_the_home_with_a_warm_up_turn_that_is_not_counted(fresh, monkeypatch):
    ran = spy(Pipeline, "run", [], monkeypatch)
    result = timing.run_arm("E", fresh)
    assert result.error is None and len(ran) == 6, "the warm-up turn, then the five rows"
    for thread, args, _ in ran:
        pipeline, job, emit, cancel = args
        assert thread is not threading.main_thread() and thread.name.startswith("turn"), "the pipeline's one worker thread"
        assert (pipeline.mode, pipeline.default_model, pipeline.engines) == ("private", fresh.name, {fresh.name: fresh.engine})
        assert pipeline.gallery is fresh.gallery and pipeline.labeler is None and pipeline.published is None
        assert Path(pipeline.store.home) == fresh.timing / "e", "a real Store on a fresh directory under <home>/timing/<job>"
        assert job.upload is None and job.mode == "private" and job.text == "" and callable(emit)
        assert job.options.model == fresh.name and job.options.cached_decode is True
        assert all(getattr(job.options, key) == value for key, value in PUBLISHED_PROTOCOL.items())
    assert [args[1].options.test_row for _, args, _ in ran] == [0, 0, 1, 2, 3, 4]
    assert len({thread for thread, _, _ in ran}) == 1
    turns = stored_turns(fresh.timing / "e")
    assert [m["status"] for m, _ in turns] == ["done"] * 6, "every turn ended done and is in the store, left in place"
    assert [m["report"] for m, _ in turns[1:]] == result.reports and result.tokens == [100] * 5
    for (message, events), row in zip(turns, [0, 0, 1, 2, 3, 4]):
        stages = [e["data"]["stage"] for e in events if e["event"] == "stage_end"]
        assert stages == list(probe.STAGES), "the whole turn, every stage"
        skipped = {e["data"]["stage"]: e["data"]["skipped"] for e in events if e["event"] == "stage_end" and "skipped" in e["data"]}
        assert skipped == {"label": "labeler_unavailable"}, "no labeller runs in this job: the stage after generate that needs one is skipped"
        retrieve = next(e["data"] for e in events if e["event"] == "stage_end" and e["data"]["stage"] == "retrieve")
        assert len(retrieve["detail"]["image_neighbors"]) == 4 and len(retrieve["detail"]["report_matches"]) == 3, "the page's k_images and k_reports"


def test_arm_e_records_the_generate_stage_ms_and_the_turn_ms_and_reads_each_turn_with_the_probes_own_reader(fresh, monkeypatch):
    read = spy(probe, "read_into", [], monkeypatch)
    result = timing.run_arm("E", fresh)
    assert result.error is None and len(read) == 6, "one reading per turn, by the probe's read_into"
    stored = [stage_ms(events, "generate") for _, events in stored_turns(fresh.timing / "e")][1:]
    assert result.ms == stored, "the stage_end ms of the generate stage, as it was sent"
    assert len(result.total_ms) == 5 and all(total >= ms for total, ms in zip(result.total_ms, result.ms))


def test_the_events_of_arm_e_are_drained_the_way_the_sse_bridge_drains_them_from_the_workers_thread_onto_a_loop(fresh, monkeypatch):
    assert (timing.sse, timing._push, timing._frames) == (server.sse, server._push, server._frames), "the bridge's own functions, not copies"
    import asyncio
    pushed = []
    real = asyncio.base_events.BaseEventLoop.call_soon_threadsafe

    def counting(self, callback, *args, **kwargs):
        pushed.append(threading.current_thread())
        return real(self, callback, *args, **kwargs)

    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "call_soon_threadsafe", counting)
    result = timing.run_arm("E", fresh)
    assert result.error is None
    from_worker = [t for t in pushed if t.name.startswith("turn")]
    assert len(from_worker) >= 6 * 100, "every step of every turn is handed to the loop from the worker thread, as the live server does"


def test_arm_e_local_is_e_with_the_store_in_a_directory_of_the_local_dir_which_is_removed_at_the_end(fresh, monkeypatch):
    homes = []
    real = timing.Store

    def recording(home):
        homes.append(Path(home))
        return real(home)

    monkeypatch.setattr(timing, "Store", recording)
    ran = spy(Pipeline, "run", [], monkeypatch)
    result = timing.run_arm("E-local", fresh)
    assert result.error is None and len(ran) == 6 and len(result.ms) == len(result.total_ms) == 5
    assert len(homes) == 1 and homes[0].parent == fresh.local and homes[0] != fresh.timing / "e"
    assert not homes[0].exists() and list(fresh.local.iterdir()) == [], "the node-local copy of the turns is the job's own and is removed"
    assert not (fresh.timing / "e").exists(), "E's store is E's"


def test_arms_e_and_e_local_make_the_same_reports_as_the_bare_decode(fresh):
    results = {arm: timing.run_arm(arm, fresh) for arm in ("A", "E", "E-local")}
    assert all(r.error is None for r in results.values())
    assert all(timing.same_reports(results).values()), "through the pipeline, the same report"


def test_the_store_directory_of_arm_e_is_fresh_and_an_existing_one_is_never_reused(fresh):
    (fresh.timing / "e").mkdir(parents=True)
    (fresh.timing / "e" / "keep.txt").write_text(SECRET)
    result = timing.run_arm("E", fresh)
    assert result.error == "FileExistsError" and result.ms == []
    assert [p.name for p in (fresh.timing / "e").iterdir()] == ["keep.txt"]


def test_a_turn_that_does_not_end_done_stops_the_arm_with_a_fixed_word_and_its_text_goes_nowhere(fresh, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("an exception text with " + SECRET)

    monkeypatch.setattr(fresh.engine, "generate", broken)
    result = timing.run_arm("E", fresh)
    assert result.error == "stream_model_error" and result.ms == [] and SECRET not in repr(result)


def test_the_worker_thread_of_an_arm_is_gone_when_the_arm_ends_whether_it_finished_or_failed(fresh, monkeypatch):
    def turn_threads():
        return [t for t in threading.enumerate() if t.name.startswith("turn")]

    before = len(turn_threads())
    timing.run_arm("E", fresh)
    assert len(turn_threads()) == before
    monkeypatch.setattr(fresh.engine, "generate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    timing.run_arm("E-local", fresh)
    assert len(turn_threads()) == before


# ---- arm F: E through HTTP, create_app on the real engine kind, uvicorn on a free loopback port, httpx and the probe's reader ----------

def test_arm_f_is_the_real_app_served_on_a_free_loopback_port_and_asked_for_each_turn_over_http_as_the_page_asks(fresh, monkeypatch):
    import httpx
    created, clients = [], []
    real_create, real_init = timing.create_app, httpx.Client.__init__

    def recording(**kwargs):
        created.append(kwargs)
        return real_create(**kwargs)

    def init(self, *args, **kwargs):
        clients.append(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(timing, "create_app", recording)
    monkeypatch.setattr(httpx.Client, "__init__", init)
    posts = spy(httpx.Client, "stream", [], monkeypatch)
    read = spy(probe, "read_into", [], monkeypatch)
    result = timing.run_arm("F", fresh)
    assert result.error is None and len(result.ms) == len(result.total_ms) == len(result.reports) == 5 and result.tokens == [100] * 5
    assert len(created) == 1
    kwargs = created[0]
    assert (kwargs["engine"], kwargs["models"], kwargs["home"], kwargs["gallery"]) == (fresh.kind, (timing.MODEL,), str(fresh.timing / "f"), fresh.gallery)
    assert (kwargs["device"], kwargs["threads"]) == ("cpu", fresh.threads) and "token" not in kwargs and kwargs.get("mode", "private") == "private"
    assert len(clients) == 1 and clients[0]["trust_env"] is False and clients[0]["follow_redirects"] is False
    base = httpx.URL(clients[0]["base_url"])
    assert base.host == "127.0.0.1" and base.port not in (8000, 8010) and base.port > 1024, "a free ephemeral port, never the dev server's or the tunnel's"
    turns = [(args[2], kw) for _, args, kw in posts if args[1] == "POST" and args[2].endswith("/messages")]
    assert len(turns) == 6 and len({path for path, _ in turns}) == 1, "the warm-up turn, then the five rows, in one session"
    assert re.fullmatch(r"/v1/sessions/[A-Za-z0-9_-]+/messages", turns[0][0])
    for (path, kw), row in zip(turns, [0, 0, 1, 2, 3, 4]):
        assert kw.get("files") is None and kw["data"]["text"] == ""
        assert json.loads(kw["data"]["options"]) == probe.turn_options(model=fresh.name, test_row=row)
    assert len(read) == 6, "each stream is read by the probe's read_into"
    with pytest.raises(httpx.TransportError):
        httpx.get(str(base) + "/healthz", timeout=2, trust_env=False)
    assert not [t for t in threading.enumerate() if t.name == "timing-server"], "the server is stopped and its thread is gone"
    stored = stored_turns(fresh.timing / "f")
    assert [m["status"] for m, _ in stored] == ["done"] * 6 and [m["report"] for m, _ in stored[1:]] == result.reports
    assert result.ms == [stage_ms(events, "generate") for _, events in stored[1:]]


def test_arm_f_makes_the_same_reports_as_the_bare_decode_and_a_turn_takes_at_least_its_generate_stage(fresh):
    results = {arm: timing.run_arm(arm, fresh) for arm in ("A", "F")}
    assert all(r.error is None for r in results.values()) and all(timing.same_reports(results).values())
    assert all(total >= ms for total, ms in zip(results["F"].total_ms, results["F"].ms))


def test_a_turn_the_server_refuses_ends_the_arm_with_its_status_and_stops_the_server(fresh):
    fresh.name = "no-such-model"   # the server answers 422: Unknown model
    result = timing.run_arm("F", fresh)
    assert result.error == "http_422" and result.ms == []
    assert not [t for t in threading.enumerate() if t.name == "timing-server"]


def test_a_server_that_never_starts_is_a_named_failure_and_leaves_no_thread_or_open_store(fresh, monkeypatch):
    import uvicorn
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: None)
    before = len([t for t in threading.enumerate() if t.name.startswith("turn")])   # another test of the suite may have left an idle worker
    result = timing.run_arm("F", fresh)
    assert result.error == "ServerNotStarted" and result.ms == []
    assert not [t for t in threading.enumerate() if t.name == "timing-server"]
    assert len([t for t in threading.enumerate() if t.name.startswith("turn")]) == before
    Store(fresh.timing / "f").close()   # the exclusive lock was released: another store can open the file


@pytest.mark.parametrize("port", [8000, 8010])
def test_the_two_ports_that_are_not_this_jobs_are_refused_whatever_they_came_from(port):
    with pytest.raises(timing.PortRefused):
        timing.check_port(port)
    assert timing.check_port(41234) == 41234


# ---- the whole job, in process over the tiny world ----------------------------------------------------------------------------------

def tree(root):
    """Every path under `root` with its size: what a run may not change outside the stores."""
    return sorted((str(p.relative_to(root)), p.stat().st_size if p.is_file() else -1) for p in Path(root).rglob("*"))


def parsed(out):
    lines = out.splitlines()
    assert lines and all(line.startswith("RESULT ") for line in lines), lines

    def strict(constant):   # NaN and Infinity are not JSON: a strict reader must take every line
        raise AssertionError("not JSON: " + constant)

    return [json.loads(line[len("RESULT "):], parse_constant=strict) for line in lines]


class Ran:
    """One run of the job over the tiny world: what it returned and printed, and what it left in the directories it was given."""

    def __init__(self, root, gallery, *extra):
        for name in ("home", "local", "cwd"):
            (root / name).mkdir(exist_ok=True)
        self.root, self.gallery = root, gallery
        self.args = ["--home", str(root / "home"), "--gallery", str(gallery), "--job", "j1", "--engine", "tiny", "--local-dir", str(root / "local")] + list(extra)
        self.gallery_before = tree(gallery)
        out, err = io.StringIO(), io.StringIO()
        self.flags = []   # per decode: was on_step None?
        from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
        real = HybridLanguageModel.beam_search_cached

        def counting(model, *args, **kwargs):
            self.flags.append(kwargs.get("on_step") is None)
            return real(model, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(HybridLanguageModel, "beam_search_cached", counting)
            patch.chdir(root / "cwd")
            self.code = timing.main(self.args, out, err)
        self.out, self.err = out.getvalue(), err.getvalue()
        self.lines = parsed(self.out) if self.out else []


@pytest.fixture(scope="module")
def full(tmp_path_factory, tiny_gallery):
    return Ran(tmp_path_factory.mktemp("timing_full"), tiny_gallery)


def test_a_full_run_prints_the_environment_then_each_arm_in_ladder_order_then_the_summary_and_exits_0(full):
    assert full.code == 0 and full.err.count("RESULT") == 0
    assert len(full.lines) == 9 and full.out.endswith("\n") and full.out.count("\n") == 9
    assert list(full.lines[0]) == ["env"] and list(full.lines[8]) == ["summary"]
    assert [line["arm"] for line in full.lines[1:8]] == list(timing.ARMS) == ["A", "B", "C", "D", "E", "E-local", "F"]
    assert all(len(line) < 300 for line in full.out.splitlines()), [len(line) for line in full.out.splitlines()]


def test_the_environment_line_is_the_node_the_cpu_the_affinity_the_threads_the_engine_and_the_rows(full):
    env = full.lines[0]["env"]
    assert list(env) == ["node", "cpu", "affinity", "cores", "load", "threads_main", "threads_worker", "engine", "rows"]
    assert env["node"] == timing.plain(platform.node(), 64) and env["engine"] == "tiny" and env["rows"] == [0, 1, 2, 3, 4]
    assert env["threads_main"] == env["threads_worker"] == torch.get_num_threads() and env["cpu"] == timing.cpu_model()


def test_every_arm_line_has_five_rows_a_mean_a_ratio_to_a_and_the_same_report_as_a(full):
    arms = {line["arm"]: line for line in full.lines[1:8]}
    base = arms["A"]["mean"]
    for name, line in arms.items():
        assert len(line["ms"]) == 5 and all(ms > 0 for ms in line["ms"]) and line["same_report"] is True and "error" not in line, name
        assert line["mean"] == pytest.approx(sum(line["ms"]) / 5, abs=0.11)
        assert line["ratio"] == pytest.approx(line["mean"] / base, abs=0.011)
        assert ("total_ms" in line) == (name in timing.TURN_ARMS), "the whole turn is reported by E, E-local and F only"
        if name in timing.TURN_ARMS:
            assert len(line["total_ms"]) == 5 and all(t >= ms - 0.1 for t, ms in zip(line["total_ms"], line["ms"]))
            assert line["mean_total_ms"] == pytest.approx(sum(line["total_ms"]) / 5, abs=0.11)
    assert arms["A"]["ratio"] == 1.0


def test_the_summary_names_the_first_arm_over_one_and_a_half_times_a_and_says_the_reports_were_the_same(full):
    summary = full.lines[8]["summary"]
    arms = {line["arm"]: line for line in full.lines[1:8]}
    assert summary["same_report_all_arms"] is True and summary["tokens"] == [100] * 5 and summary["arms_failed"] == []
    first = summary["first_over_1_5x"]
    assert first in set(timing.ARMS[1:]) | {"none"}
    ladder_order = timing.ARMS[1:]
    before = ladder_order if first == "none" else ladder_order[:ladder_order.index(first)]
    assert all(arms[arm]["ratio"] <= 1.51 for arm in before), "no arm before the named one is over (printed ratios are rounded to 2 places)"
    if first != "none":
        assert arms[first]["ratio"] >= 1.49


def test_the_job_writes_nothing_but_the_stores_of_e_and_f_and_removes_the_node_local_one(full):
    home, local = full.root / "home", full.root / "local"
    assert sorted(p.name for p in (home / "timing").iterdir()) == ["j1"]
    assert sorted(p.name for p in (home / "timing" / "j1").iterdir()) == ["e", "f"], "the stores of E and F, left in place"
    for name in ("e", "f"):
        assert (home / "timing" / "j1" / name / "chat.db").is_file() and (home / "timing" / "j1" / name / "uploads").is_dir()
        assert len(stored_turns(home / "timing" / "j1" / name)) == 6
    assert list(local.iterdir()) == [], "E-local's directory is the job's own and is gone"
    assert list((full.root / "cwd").iterdir()) == [], "nothing is written to the directory it runs in"
    assert tree(full.gallery) == full.gallery_before, "the gallery is only read"
    assert sorted(p.name for p in home.iterdir()) == ["timing"]


def test_the_decodes_are_one_warm_up_and_then_the_arms_in_order_with_a_callback_only_where_the_arm_has_one(full):
    assert len(full.flags) == 1 + 5 + 5 + 5 + (1 + 5) * 4, "one warm-up, A, B, C, and D, E, E-local and F with an uncounted turn each"
    assert full.flags[0] is False, "the warm-up is a whole generate: it has the engine's callback"
    assert full.flags[1:6] == [True] * 5, "A: the bare search, no callback"
    assert full.flags[6:11] == [False] * 5, "B: a callback that does nothing"


def test_each_arm_is_printed_as_it_finishes_so_that_a_job_that_is_killed_keeps_what_it_measured(tmp_path, tiny_gallery, monkeypatch):
    out, err, seen = io.StringIO(), io.StringIO(), []
    real = timing.run_arm

    def watching(name, world):
        seen.append((name, len(out.getvalue().splitlines())))
        return real(name, world)

    monkeypatch.setattr(timing, "run_arm", watching)
    for name in ("home", "local"):
        (tmp_path / name).mkdir()
    code = timing.main(["--home", str(tmp_path / "home"), "--gallery", str(tiny_gallery), "--job", "j1", "--engine", "tiny",
                        "--local-dir", str(tmp_path / "local"), "--rows", "0-1"], out, err)
    assert code == 0 and seen == [("A", 1), ("B", 2), ("C", 3), ("D", 4), ("E", 5), ("E-local", 6), ("F", 7)], "the environment first, then one line per arm"


def test_a_setup_that_fails_is_one_fatal_line_with_the_step_and_the_class_and_nothing_is_made(tmp_path, tiny_gallery):
    other = tmp_path / "other"
    shutil.copytree(str(tiny_gallery), str(other))
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["tower_sha256"] = "0" * 64
    (other / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "home").mkdir()
    out, err = io.StringIO(), io.StringIO()
    code = timing.main(["--home", str(tmp_path / "home"), "--gallery", str(other), "--job", "j1", "--engine", "tiny"], out, err)
    assert code == 1 and out.getvalue() == 'RESULT {"fatal":"GalleryMismatch","step":"gallery"}\n'
    assert list((tmp_path / "home").iterdir()) == [], "the timing directory is made only once the job can begin"


def test_an_arm_that_fails_is_a_line_with_its_class_the_others_go_on_and_the_job_exits_1(tmp_path, tiny_gallery, monkeypatch):
    def broken(world, result):
        result.ms.append(1.0)
        raise ValueError("an exception text with " + SECRET)

    monkeypatch.setitem(timing.ARM_FUNCTIONS, "C", broken)
    ran = Ran(tmp_path, tiny_gallery, "--rows", "0-1")
    arms = {line["arm"]: line for line in ran.lines[1:8]}
    assert ran.code == 1 and arms["C"] == {"arm": "C", "error": "ValueError", "rows_done": 1, "ms": [1.0]}
    assert all("error" not in arms[a] for a in timing.ARMS if a != "C"), "the others ran"
    summary = ran.lines[8]["summary"]
    assert summary["arms_failed"] == ["C"] and summary["same_report_all_arms"] is False
    assert SECRET not in ran.out + ran.err


def test_a_report_that_differs_in_one_arm_fails_the_sanity_check_and_the_exit_status_but_every_line_is_printed(tmp_path, tiny_gallery, monkeypatch):
    real = timing.arm_d

    def different(world, result):
        real(world, result)
        result.reports[1] = result.reports[1] + " and one more word"

    monkeypatch.setitem(timing.ARM_FUNCTIONS, "D", different)
    ran = Ran(tmp_path, tiny_gallery, "--rows", "0-1")
    arms = {line["arm"]: line for line in ran.lines[1:8]}
    assert ran.code == 1 and len(ran.lines) == 9
    assert [a for a in timing.ARMS if arms[a]["same_report"] is False] == ["D"]
    assert ran.lines[8]["summary"]["same_report_all_arms"] is False and ran.lines[8]["summary"]["arms_failed"] == []


# ---- R7: SECRET text in the reports, the ids, the paths and the exceptions never reaches stdout, stderr or a file --------------------

REPORT_WORDS = " ".join("{}-word{}".format(SECRET, i) for i in range(120))   # what the decoder "writes": a report of 120 words, each a SECRET


def secret_gallery(root):
    """The tiny gallery built under a path that holds the SECRET (so every image path does), its report texts and its study ids replaced by
    SECRET ones: what the job reads of MIMIC, in every place the job can reach it. -> the gallery directory."""
    import pandas as pd
    gallery = root / ("study-" + SECRET) / "gallery"
    build_tiny(gallery)
    decide_gate(gallery)
    manifest = json.loads((gallery / "manifest.json").read_text())
    manifest["tower_sha256"] = build_engine("tiny").tower_sha256()
    (gallery / "manifest.json").write_text(json.dumps(manifest))
    texts = (gallery / "report_texts.txt").read_text().splitlines()
    (gallery / "report_texts.txt").write_text("".join("{}-report-{} {}\n".format(SECRET, i, t) for i, t in enumerate(texts)))
    for name, kind in (("img_meta.parquet", "train"), ("test_meta.parquet", "test")):
        frame = pd.read_parquet(str(gallery / name))
        frame["study_id"] = ["{}-study-{}-{}".format(SECRET, kind, j) for j in range(len(frame))]
        frame.to_parquet(str(gallery / name))
    return gallery


@pytest.fixture
def secret_world(tmp_path, monkeypatch):
    """A run's directories, every path of which holds the SECRET, and a decoder whose every report is made of it."""
    from app.tiny import TinyTokenizer
    root = tmp_path / ("root-" + SECRET)
    root.mkdir()
    gallery = secret_gallery(root)
    for name in ("home-" + SECRET, "local-" + SECRET, "cwd"):
        (root / name).mkdir()
    monkeypatch.setattr(TinyTokenizer, "decode", lambda self, ids, skip_special_tokens=True: " ".join(REPORT_WORDS.split()[:len(ids)]))
    monkeypatch.chdir(root / "cwd")
    return root, gallery


def secret_run(root, gallery, capfd, *extra):
    out, err = io.StringIO(), io.StringIO()
    args = ["--home", str(root / ("home-" + SECRET)), "--gallery", str(gallery), "--job", "j1-" + "x", "--engine", "tiny",
            "--local-dir", str(root / ("local-" + SECRET))] + list(extra)
    before = tree(root)
    capfd.readouterr()
    code = timing.main(args, out, err)
    seen = capfd.readouterr()
    return code, out.getvalue(), err.getvalue(), seen.out + seen.err, before


def strings_the_job_could_leak(gallery):
    """Every string of the world that is MIMIC-shaped: the SECRET marker, the reports, the study ids, the paths of the images and of the
    directories, and the generated report."""
    import pandas as pd
    texts = (gallery / "report_texts.txt").read_text().splitlines()
    ids = [str(s) for name in ("img_meta.parquet", "test_meta.parquet") for s in pd.read_parquet(str(gallery / name))["study_id"]]
    paths = [str(p) for name in ("img_meta.parquet", "test_meta.parquet") for p in pd.read_parquet(str(gallery / name))["image"]]
    return [SECRET, REPORT_WORDS[:60]] + texts[:50] + texts[-5:] + ids[:30] + paths[:30] + [str(gallery.parent), str(gallery.parent.parent)]


def test_nothing_of_the_reports_ids_or_paths_reaches_stdout_stderr_or_a_file_outside_the_stores(secret_world, capfd):
    root, gallery = secret_world
    code, out, err, fds, before = secret_run(root, gallery, capfd)
    assert code == 0 and len(parsed(out)) == 9
    for leak in strings_the_job_could_leak(gallery):
        for name, stream in (("stdout", out), ("stderr", err), ("the process's fds", fds)):
            assert leak not in stream, (name, leak[:40])
    home = root / ("home-" + SECRET)
    after = tree(root)
    changed = sorted(set(after) - set(before))
    inside = [path for path, _ in changed if path.startswith("home-{}/timing/j1-x/".format(SECRET)) or path == "home-{}/timing".format(SECRET)
              or path == "home-{}/timing/j1-x".format(SECRET)]
    assert [path for path, _ in changed if path not in inside] == [], "no file outside the stores of E and F"
    assert sorted(p.name for p in (home / "timing" / "j1-x").iterdir()) == ["e", "f"]
    for name in ("e", "f"):   # the stores do hold the report: the setup put the SECRET where the job's output must not be
        assert any(SECRET in (m["report"] or "") for m, _ in stored_turns(home / "timing" / "j1-x" / name))
    assert list((root / ("local-" + SECRET)).iterdir()) == [] and list((root / "cwd").iterdir()) == []


def test_a_failure_with_the_secret_in_its_text_is_named_by_class_or_by_a_fixed_word_and_its_traceback_is_nowhere(secret_world, capfd, monkeypatch):
    """A turn in the worker thread that fails: D raises the exception itself, and in E, E-local and F the pipeline catches it and logs its
    traceback (log.exception), which holds the text. The job's run keeps that traceback out of every stream. pytest puts its own handlers on the
    root logger, which would hide the traceback from the logging module's last resort (stderr) that a process of its own has, so they are
    taken off for the run."""
    import logging
    from app.engine import TinyEngine
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    real = TinyEngine.generate

    def broken(self, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Findings: an exception text with " + SECRET + " and a path /sc/home/x/files/p10/s" + SECRET)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(TinyEngine, "generate", broken)
    root, gallery = secret_world
    code, out, err, fds, _ = secret_run(root, gallery, capfd, "--rows", "0-1")
    arms = {line["arm"]: line for line in parsed(out)[1:8]}
    assert code == 1
    assert arms["D"] == {"arm": "D", "error": "RuntimeError", "rows_done": 0, "ms": []}
    assert all(arms[a]["error"] == "stream_model_error" for a in ("E", "E-local", "F")), [arms[a] for a in ("E", "E-local", "F")]
    for leak in (SECRET, "exception text", "Traceback", "RuntimeError: ", "/sc/home"):
        for name, stream in (("stdout", out), ("stderr", err), ("the process's fds", fds)):
            assert leak not in stream, (name, leak)


def test_a_setup_failure_names_its_class_only_even_when_the_gallery_says_why_in_words(secret_world, capfd):
    root, gallery = secret_world
    manifest = json.loads((gallery / "manifest.json").read_text())
    manifest["tower_sha256"] = "0" * 64
    (gallery / "manifest.json").write_text(json.dumps(manifest))
    code, out, err, fds, _ = secret_run(root, gallery, capfd)
    assert code == 1 and out == 'RESULT {"fatal":"GalleryMismatch","step":"gallery"}\n' and SECRET not in err + fds


def test_the_run_leaves_the_root_loggers_handlers_as_it_found_them(tmp_path, tiny_gallery):
    import logging
    before = list(logging.getLogger().handlers)
    for name in ("home", "local"):
        (tmp_path / name).mkdir()
    code, _, _ = run(["--home", str(tmp_path / "home"), "--gallery", str(tiny_gallery), "--job", "j1", "--engine", "tiny", "--local-dir",
                      str(tmp_path / "local"), "--rows", "0"])
    assert code == 0 and list(logging.getLogger().handlers) == before


def test_the_script_runs_as_a_script_from_any_directory_and_prints_nothing_to_stdout_but_the_result_lines(tmp_path, tiny_gallery):
    """A process of its own, started the way the wrapper starts it: it puts the repository on its own path, prints the nine RESULT lines and
    nothing else on stdout, and a refused argument is one ERROR line on stderr and exit status 2."""
    import subprocess
    import sys
    for name in ("home", "local", "cwd"):
        (tmp_path / name).mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, str(SCRIPT), "--home", str(tmp_path / "home"), "--gallery", str(tiny_gallery), "--job", "p1", "--engine",
                           "tiny", "--local-dir", str(tmp_path / "local"), "--rows", "0-1"], cwd=str(tmp_path / "cwd"), env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, (done.stdout, done.stderr[-2000:])
    lines = parsed(done.stdout)
    assert len(lines) == 9 and list(lines[0]) == ["env"] and [line["arm"] for line in lines[1:8]] == list(timing.ARMS)
    assert all(len(line["ms"]) == 2 for line in lines[1:8]) and "RESULT" not in done.stderr
    refused = subprocess.run([sys.executable, str(SCRIPT), "--token", SECRET], cwd=str(tmp_path / "cwd"), env=env, capture_output=True, text=True,
                             timeout=120)
    assert refused.returncode == 2 and refused.stdout == "" and refused.stderr.startswith("ERROR ") and refused.stderr.count("\n") == 1
    assert SECRET not in refused.stderr and list((tmp_path / "cwd").iterdir()) == []


# ---- the wrapper, rehearsed in a temp tree that stands in for CLUSTER_REPO ---------------------------------------------------------------

LINE_OK = safe_line("timing")
JOB_ID = "1234567"
SYNC_LINE = "=== sync 3f2a9c41d7e86b05a1c4e9d3b7f60285ac9e1d47 clean ==="

# The script the wrapper runs, as a stub: it records its arguments and environment, prints the RESULT lines it is given (real ones, from the
# job's own run over the tiny world), a stray line on stdout that is no RESULT line, and raw text on stderr, which must never reach the job
# log; it exits as it is told.
SCRIPT_STUB = '''
import json, os, sys
stub = os.environ["STUB_DIR"]
KEYS = ("PYTHONPATH", "HF_HOME", "HF_HUB_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONUNBUFFERED", "TMPDIR")


def read(name, default=""):
    try:
        return open(os.path.join(stub, name)).read()
    except OSError:
        return default


with open(os.path.join(stub, "script.call"), "w") as f:
    json.dump({"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in KEYS}, "cwd": os.path.realpath(os.getcwd())}, f)
sys.stderr.write("Findings: SECRET REPORT TEXT study_id=12345678 /sc/home/someone/img.jpg\\n")
print("Findings: a stray stdout line with SECRET REPORT TEXT")
sys.stdout.flush()
for number, line in enumerate(read("lines.txt").splitlines()):
    print(line)
    sys.stdout.flush()
    if read("hang_after.txt") == str(number + 1):   # a job that is still measuring: it says so, and waits to be killed
        open(os.path.join(stub, "hanging"), "w").write("1")
        import time
        time.sleep(600)
sys.exit(int(read("exit.txt", "0") or 0))
'''
HOSTNAME_STUB = '#!/bin/bash\necho "${STUB_HOSTNAME:-gx14.test.cluster}"\n'
PYTHON3_SHIM = '#!/bin/bash\nexec "$REAL_PYTHON" "$@"\n'


class TimingBox(wr.JobBox):
    WRAPPER = WRAPPER
    SCRIPTS = ()
    SLURM_CPUS = "8"

    def populate(self):
        gallery = self.chat / "gallery" / "g13d_m3_v1"
        gallery.mkdir(parents=True)
        (gallery / "manifest.json").write_text(json.dumps({"gate_rk": {"equal": True}}))
        (self.repo / ".chat_deps").mkdir()
        (self.repo / ".chat_deps" / ".setup_ok").write_text("")
        (self.repo / "scripts" / "chat_decode_timing.py").write_text(SCRIPT_STUB)
        for name, text in (("hostname", HOSTNAME_STUB), ("python3", PYTHON3_SHIM)):
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)

    def say(self, lines, exit_code=0):
        (self.stubs / "lines.txt").write_text("\n".join(lines) + "\n")
        (self.stubs / "exit.txt").write_text(str(exit_code))

    def call(self):
        return json.loads((self.stubs / "script.call").read_text())

    def raw(self, suffix):
        return (self.chat / "logs" / "decode_timing_{}.{}".format(JOB_ID, suffix))


@pytest.fixture
def box(tmp_path, full):
    b = TimingBox(tmp_path)
    b.say(full.out.splitlines())
    return b


def test_the_wrapper_prints_the_sync_line_a_banner_and_the_nine_result_lines_and_nothing_else_of_what_the_script_printed(box, full):
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 0, done.stdout
    assert lines[0] == SYNC_LINE and lines[1] == "=== chat decode timing: node=gx14.test.cluster job={} threads=8 gallery=g13d_m3_v1 ===".format(JOB_ID)
    assert [l for l in lines if l.startswith("RESULT ")] == full.out.splitlines()
    assert lines[-2:] == ["=== timing lines withheld: 1 ===", "=== END chat decode timing ==="], "the stray line the script printed is counted, not shown"
    assert all(LINE_OK.match(l) for l in lines), [l for l in lines if not LINE_OK.match(l)]
    assert "SECRET" not in done.stdout and "Findings" not in done.stdout and "study_id" not in done.stdout and "/sc/home" not in done.stdout


def test_the_result_lines_reach_the_job_log_as_the_script_prints_them_so_that_a_killed_job_keeps_what_it_measured(box, full):
    """The script prints each arm's line as the arm finishes. The wrapper lets the lines of known shapes through at once, not when the script
    ends: this one stops after its fourth RESULT line, still measuring, and the first four are already in the job's output when it is killed."""
    import select
    import signal
    import time
    box.say(full.out.splitlines())
    (box.stubs / "hang_after.txt").write_text("4")
    proc = subprocess.Popen([wr.BASH, str(box.repo / "scripts" / WRAPPER)], cwd=str(box.root), env=box.base_env(), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    text = ""
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and text.count("RESULT ") < 4:   # raw reads, so that a line in a buffer is never waited for
            if select.select([proc.stdout], [], [], 0.5)[0]:
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                text += chunk.decode("utf-8", "replace")
        still_running = (box.stubs / "hanging").exists() and proc.poll() is None
    finally:
        os.killpg(proc.pid, signal.SIGKILL)   # the wrapper and the script it started: nothing is left running
        proc.wait(timeout=30)
        proc.stdout.close()
    seen = text.splitlines()
    assert still_running, "the script is still running while its lines are already shown"
    assert [l for l in seen if l.startswith("RESULT ")] == full.out.splitlines()[:4] and "SECRET" not in text, seen


def test_the_whole_job_log_comes_through_chat_remote_summary_unchanged_every_line_kept_and_none_cut_or_masked(box):
    """The controller reads this job through `chat_remote.sh summary`, which keeps the lines of its fixed shapes, cuts each at 300 characters and
    masks what looks like an id: every line of a normal log must come through whole."""
    from tests.test_chat_remote import _summary_of
    lines = job_lines(box.run())
    assert len(lines) == 14 and max(len(l) for l in lines) < 300
    assert _summary_of(box.root / "summary", lines) == lines


def test_the_raw_output_of_the_script_goes_to_files_under_chat_home_and_nowhere_else(box):
    done = box.run()
    assert done.returncode == 0
    assert "SECRET REPORT TEXT" in box.raw("err").read_text() and "a stray stdout line" in box.raw("out").read_text()
    assert list(box.repo.rglob("decode_timing_*")) == [], "nothing of it in the code tree"


def test_the_wrapper_runs_the_script_with_the_servers_home_and_gallery_the_job_id_the_cpus_and_the_tmpdir_and_the_serving_environment(box):
    (box.root / "tmp").mkdir()
    assert box.run(TMPDIR=str(box.root / "tmp")).returncode == 0
    call = box.call()
    assert call["argv"] == ["--engine", "real", "--home", str(box.chat), "--gallery", str(box.chat / "gallery" / "g13d_m3_v1"), "--job", JOB_ID,
                            "--threads", "8", "--local-dir", str(box.root / "tmp")]
    assert call["cwd"] == os.path.realpath(str(box.repo))
    env = call["env"]
    assert (env["OMP_NUM_THREADS"], env["MKL_NUM_THREADS"], env["HF_HUB_OFFLINE"], env["PYTHONUNBUFFERED"]) == ("8", "8", "1", "1")
    assert env["HF_HOME"] == str(box.scratch / ".hf") and env["PYTHONPATH"] == ".chat_deps" and env["TMPDIR"] == str(box.root / "tmp")


def test_the_node_local_directory_is_tmp_without_a_tmpdir_and_the_threads_follow_the_cpus_of_the_job(box):
    assert box.run(TMPDIR=None, SLURM_CPUS_PER_TASK="4").returncode == 0
    call = box.call()
    assert call["argv"][-2:] == ["--local-dir", "/tmp"] and call["argv"][call["argv"].index("--threads") + 1] == "4"
    assert call["env"]["OMP_NUM_THREADS"] == call["env"]["MKL_NUM_THREADS"] == "4"


def test_the_gallery_lever_names_another_build_and_the_banner_shows_its_name_only(box):
    other = box.chat / "gallery" / "other_build_v2"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"gate_rk": {"equal": True}}))
    done = box.run(GALLERY=str(other))
    assert done.returncode == 0 and "gallery=other_build_v2 ===" in done.stdout and str(box.chat) not in done.stdout
    assert box.call()["argv"][box.call()["argv"].index("--gallery") + 1] == str(other)


LEVER_CASES = [   # (what is wrong, the environment, the fixed sentence the job says)
    ("NODE with a space", {"NODE": "gx 14"}, "NODE must be one node name"),
    ("NODE in capitals", {"NODE": "GX14"}, "NODE must be one node name"),
    ("NODE that is a command", {"NODE": "$(echo SECRET)"}, "NODE must be one node name"),
    ("NODE with a semicolon", {"NODE": "gx14;SECRET"}, "NODE must be one node name"),
    ("NODE too long", {"NODE": "g" * 40}, "NODE must be one node name"),
    ("NODE starting with a digit", {"NODE": "14gx"}, "NODE must be one node name"),
    ("GALLERY that is no build name", {"GALLERY": "{chat}/gallery/bad name SECRET"}, "GALLERY must end in one plain build name"),
    ("GALLERY with no manifest", {"GALLERY": "{chat}/gallery/empty_build"}, "the gallery has no manifest.json"),
    ("GALLERY whose gate is not equal", {"GALLERY": "{chat}/gallery/failed_build"}, "R@k gate is not equal"),
    ("CPUS that is zero", {"SLURM_CPUS_PER_TASK": "0"}, "SLURM_CPUS_PER_TASK must be a whole number"),
    ("CPUS that is no number", {"SLURM_CPUS_PER_TASK": "many SECRET"}, "SLURM_CPUS_PER_TASK must be a whole number"),
    ("CPUS with a leading zero", {"SLURM_CPUS_PER_TASK": "08"}, "SLURM_CPUS_PER_TASK must be a whole number"),
    ("CHAT_HOME that is missing", {"CHAT_HOME": "{root}/no/such/home"}, "CHAT_HOME does not exist"),
    ("a node-local directory that is missing", {"TMPDIR": "{root}/no/such/tmp"}, "the node-local directory"),
    ("a venv that is missing", {"VENV_ACTIVATE": "{root}/no/such/activate"}, "the venv's activate script is missing"),
]


@pytest.mark.parametrize("what, env, sentence", LEVER_CASES, ids=[c[0] for c in LEVER_CASES])
def test_a_lever_that_will_not_do_is_refused_in_one_line_that_does_not_echo_it_before_anything_runs(what, env, sentence, box):
    (box.chat / "gallery" / "empty_build").mkdir(exist_ok=True)
    (box.chat / "gallery" / "failed_build").mkdir(exist_ok=True)
    (box.chat / "gallery" / "failed_build" / "manifest.json").write_text(json.dumps({"gate_rk": {"equal": False}}))
    env = {k: v.format(chat=box.chat, root=box.root) for k, v in env.items()}
    done = box.run(**env)
    lines = job_lines(done)
    errors = [l for l in lines if l.startswith("ERROR ")]
    assert done.returncode != 0 and len(errors) == 1 and sentence in errors[0], (what, done.stdout)
    for value in env.values():   # a value of one or two characters is found in any hash: for those the one fixed sentence is the check
        assert (len(value) < 4 or value not in done.stdout) and "SECRET" not in done.stdout, what
    assert str(box.chat) not in done.stdout and not (box.stubs / "script.call").exists(), "nothing ran"
    assert all(LINE_OK.match(l) for l in lines), lines


def test_the_node_lever_holds_the_job_to_that_node_and_says_how_to_place_it_without_naming_either(box):
    assert box.run(NODE="gx14", STUB_HOSTNAME="gx14.test.cluster").returncode == 0, "the short name of the node it landed on"
    (box.stubs / "script.call").unlink()
    done = box.run(NODE="gx14", STUB_HOSTNAME="gx15.test.cluster")
    errors = [l for l in job_lines(done) if l.startswith("ERROR ")]
    assert done.returncode != 0 and len(errors) == 1 and "not on the requested node" in errors[0] and "--nodelist" in errors[0]
    assert "gx14" not in errors[0] and "gx15" not in errors[0] and not (box.stubs / "script.call").exists()


def test_the_submit_lines_the_header_documents_are_ones_chat_remote_sh_takes_and_sends_as_documented(tmp_path):
    """`[NODE=gx14]` is a plain NAME=value argument, which chat_remote.sh puts on the sbatch command as an environment variable; the node itself
    is placed by sbatch's own option, which goes after `--`."""
    from tests.test_chat_remote import Sandbox
    sandbox = Sandbox(tmp_path)
    shutil.copy(str(REPO_ROOT / "scripts" / WRAPPER), str(sandbox.repo / "scripts" / WRAPPER))
    wrapper = "'scripts/" + WRAPPER + "'"
    for args, sent in (((), "env sbatch --parsable " + wrapper),
                       (("NODE=gx14",), "env 'NODE=gx14' sbatch --parsable " + wrapper),
                       (("NODE=gx14", "--", "--nodelist=gx14"), "env 'NODE=gx14' sbatch --parsable '--nodelist=gx14' " + wrapper)):
        done = sandbox.run("submit", "scripts/" + WRAPPER, *args)
        assert done.returncode == 0, done.stderr
        assert " ".join(sandbox.calls()[-1][2].split()) == "cd '/fake/hybrid_chat_ui' && " + sent
    header = (REPO_ROOT / "scripts" / WRAPPER).read_text()
    assert "submit scripts/chat_decode_timing_h100.sh [NODE=gx14]" in header and "NODE=gx14 -- --nodelist=gx14" in header


def test_a_script_that_exits_1_after_its_lines_has_them_shown_and_the_job_ends_with_one_error_line_and_the_same_status(box, full):
    box.say(full.out.splitlines(), exit_code=1)
    done = box.run()
    lines = job_lines(done)
    assert done.returncode == 1 and len([l for l in lines if l.startswith("RESULT ")]) == 9
    errors = [l for l in lines if l.startswith("ERROR ")]
    assert errors == ["ERROR the timing script exited with code 1 (see the logs under CHAT_HOME)"] and "END chat decode timing" not in done.stdout
    assert lines.index(errors[0]) > max(i for i, l in enumerate(lines) if l.startswith("RESULT "))


def test_a_script_that_exits_0_with_the_wrong_number_of_lines_is_not_a_clean_end(box, full):
    box.say(full.out.splitlines()[:5])
    done = box.run()
    assert done.returncode != 0 and "ERROR the timing script printed 5 result lines, not 9" in done.stdout
    assert "END chat decode timing" not in done.stdout


def test_a_result_line_of_a_shape_the_script_does_not_have_is_withheld_whatever_it_says(box, full):
    lines = full.out.splitlines()
    leaks = ['RESULT {"arm":"A","ms":[1.0],"mean":1.0,"ratio":1.0,"same_report":true,"report":"Findings: SECRET REPORT TEXT"}',
             'RESULT {"env":{"node":"SECRET node name","cpu":null,"affinity":8,"cores":4,"load":null,"threads_main":8,"threads_worker":8,"engine":"real","rows":[0]}}',
             'RESULT {"summary":{"first_over_1_5x":"SECRET","same_report_all_arms":true,"tokens":[],"arms_failed":[]}}',
             'RESULT {"fatal":"Findings: SECRET REPORT TEXT","step":"gallery"}']
    box.say(lines[:1] + leaks + lines[1:])
    done = box.run()
    assert "SECRET" not in done.stdout and "=== timing lines withheld: 5 ===" in done.stdout, "four of them and the stray line"
    assert [l for l in job_lines(done) if l.startswith("RESULT ")] == lines


def shapes_of_the_wrapper():
    """The wrapper's own TIMING_SHAPES: the text between its single quotes."""
    text = (REPO_ROOT / "scripts" / WRAPPER).read_text()
    found = re.search(r"^TIMING_SHAPES='(.*?)'$", text, re.S | re.M)
    assert found, "TIMING_SHAPES is not in the wrapper"
    return found.group(1)


def matches(shapes, line):
    return subprocess.run(["grep", "-aE", shapes], input=line + "\n", capture_output=True, text=True).returncode == 0


def test_the_wrappers_allowlist_takes_every_line_the_script_prints_in_every_state_it_can_end_in(full):
    shapes = shapes_of_the_wrapper()
    assert all(line for line in shapes.splitlines()), "an empty pattern matches every line"
    for line in full.out.splitlines():
        assert matches(shapes, line), line
    base = arm_of("A", [3000.0] * 5)
    samples = [
        timing.result_line(timing.arm_payload(arm_of("E", [7000.0, 7100.5], total=[7500.0, 7650.5], error="stream_model_error"), 3000.0, False)),
        timing.result_line(timing.arm_payload(arm_of("F", [], error="http_422"), 3000.0, False)),
        timing.result_line(timing.arm_payload(arm_of("D", [4500.0] * 5), None, False)),
        timing.result_line(timing.arm_payload(arm_of("E-local", [4500.0] * 8, total=[4999.9] * 8), 3000.0, True)),
        timing.result_line(timing.arm_payload(arm_of("F", [123456.7] * 8, total=[234567.8] * 8), 3000.0, True)),
        timing.result_line(timing.summary_payload({"A": base, "D": arm_of("D", [9000.0] * 5, error="ValueError")}, False)),
        timing.result_line(timing.summary_payload({"A": arm_of("A", [], error="ValueError")}, False)),
        timing.result_line(timing.env_payload("gx14", None, None, None, None, 8, 8, "real", [0, 1, 2, 3, 4])),
        timing.result_line(timing.env_payload("gx14.long-node.example.org", "Intel(R) Xeon(R) Platinum 8480+ @ 2.00GHz", 8, 4, [0.0, 12.5, 101.25], 8, 8,
                                              "tiny", [0])),
        timing.result_line(timing.env_payload("n" * 100, "c" * 100, 1024, 1024, [1000.55, 1000.55, 1000.55], 9999, 9999, "real", [9999] * 8)),
        timing.result_line({"fatal": "GalleryMismatch", "step": "gallery"}),
        timing.result_line({"fatal": "OSError", "step": "timing"}),
        timing.result_line({"fatal": "RuntimeError", "step": "run"}),
    ]
    for line in samples:
        assert matches(shapes, line), line


@pytest.mark.parametrize("leak", [
    'RESULT {"arm":"A","ms":[1.0],"mean":1.0,"ratio":1.0,"same_report":true,"report":"x"}',
    'RESULT {"arm":"Z","ms":[1.0],"mean":1.0,"ratio":1.0,"same_report":true}',
    'RESULT {"arm":"A","ms":["x"],"mean":1.0,"ratio":1.0,"same_report":true}',
    'RESULT {"arm":"A","ms":[1.0],"mean":1.0,"ratio":1.0,"same_report":"Findings: x"}',
    'RESULT {"arm":"A","error":"has a space","rows_done":1,"ms":[]}',
    'RESULT {"arm":"A","error":"Findings: x","rows_done":1,"ms":[]}',
    'RESULT {"env":{"node":"a b","cpu":null,"affinity":1,"cores":1,"load":null,"threads_main":1,"threads_worker":1,"engine":"real","rows":[0]}}',
    'RESULT {"env":{"node":"n","cpu":"a\\"b","affinity":1,"cores":1,"load":null,"threads_main":1,"threads_worker":1,"engine":"real","rows":[0]}}',
    'RESULT {"env":{"node":"n","cpu":null,"affinity":1,"cores":1,"load":null,"threads_main":1,"threads_worker":1,"engine":"real","rows":[0],"path":"/sc/home/x"}}',
    'RESULT {"env":{"node":"n","cpu":null,"affinity":1,"cores":"Findings","load":null,"threads_main":1,"threads_worker":1,"engine":"real","rows":[0]}}',
    'RESULT {"env":{"node":"n","cpu":null,"affinity":1,"cores":1,"load":[1.0,"x",2.0],"threads_main":1,"threads_worker":1,"engine":"real","rows":[0]}}',
    'RESULT {"env":{"node":"n","cpu":null,"affinity":1,"cores":1,"load":[1.0,2.0],"threads_main":1,"threads_worker":1,"engine":"real","rows":[0]}}',
    'RESULT {"summary":{"first_over_1_5x":"E","same_report_all_arms":true,"tokens":[],"arms_failed":["Findings"]}}',
    'RESULT {"fatal":"A B","step":"gallery"}', 'RESULT {"fatal":"X","step":"somewhere"}',
    'RESULT {"gate": {"ok": true}}', 'RESULT {}', 'RESULT ', 'Findings: SECRET', '=== not a result ===', 'ERROR x',
    'RESULT {"arm":"A","ms":[1.0],"mean":1.0,"ratio":1.0,"same_report":true} trailing',
    ' RESULT {"arm":"A","ms":[1.0],"mean":1.0,"ratio":1.0,"same_report":true}',
])
def test_the_wrappers_allowlist_takes_nothing_else(leak):
    assert not matches(shapes_of_the_wrapper(), leak), leak
