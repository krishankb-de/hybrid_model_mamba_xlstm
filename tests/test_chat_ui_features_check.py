"""CHAT_UI_PLAN.md P6-F: the parts of the features check (scripts/chat_ui_features_check.py) that are plain Python: its arguments, what its
main() refuses of a --url (the refusals themselves are scripts/chat_ui_cdp.py's, tested in tests/test_chat_ui_layout_check.py), the
checklist writer and the environment it writes (read from the app), the words and numbers it expects of the page (read off the
sources of the page, so that a rename is caught here), and the consistency of its own lists. No Chrome and no app process.

The run itself needs Chrome and is a local tool (venv/bin/python scripts/chat_ui_features_check.py); validate.sh does not run it.
Synthetic data only: the tiny engine, the tiny gallery, and phantom pictures drawn for the check.
"""
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image

from app.labels import CHEXBERT_14
from scripts import chat_ui_cdp
from scripts import chat_ui_features_check as check
from tests import chat_ui_url_cases as url_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC = REPO_ROOT / "app" / "static"
SCRIPT = Path(check.__file__)


def _page_source(*names: str) -> str:
    return "\n".join((STATIC / name).read_text(encoding="utf-8") for name in names)


# ---- what the script is: five items, one evidence directory ----------------------------------------------------------------------

def test_the_items_are_the_five_features_of_the_brief_in_order():
    assert [name for name, _, _ in check.CHECKS] == ["attach", "reload", "similar", "matches", "viewer"]
    claims = [claim for _, claim, _ in check.CHECKS]
    assert len(set(claims)) == 5 and all(claims)
    assert claims[0].startswith("click, drag-drop and paste") and "refused file" in claims[0]
    assert "224" in claims[1] and "n/14 labels agree" in claims[2] and "own-rank badge" in claims[3]
    assert "zoom, pan, brightness and contrast" in claims[4] and "mouse and by keyboard" in claims[4]


def test_the_evidence_goes_to_p6f_and_the_defaults_are_the_other_checks_defaults():
    args = check.parse_args([])
    assert Path(args.out) == REPO_ROOT / "docs" / "chat_ui" / "evidence" / "p6f"
    assert (args.url, args.no_evidence, args.timeout) == (None, False, 420)
    args = check.parse_args(["--no-evidence", "--timeout", "90", "--out", "somewhere"])
    assert (args.no_evidence, args.timeout, args.out) == (True, 90, "somewhere")


def test_an_unknown_argument_is_refused(capsys):
    with pytest.raises(SystemExit) as refused:
        check.parse_args(["--nonsense"])
    assert refused.value.code == 2


# ---- a --url: the app is someone else's, so nothing of this run is evidence, and it is the tiny app or it is refused (P6-F fix 1) ---------
# The address and the app's own answer about itself are refused by scripts/chat_ui_cdp.py (url_problem, tiny_only_problem), which tests/test_chat_ui_layout_check.py
# tests as a harness; these are the cases of this check's own arguments and main().

def test_a_url_needs_no_evidence_because_the_pictures_and_the_checklist_would_be_evidence_of_an_app_the_check_did_not_start(capsys):
    with pytest.raises(SystemExit) as refused:
        check.parse_args(["--url", "http://127.0.0.1:51234/"])
    assert refused.value.code == 2
    assert chat_ui_cdp.URL_NEEDS_NO_EVIDENCE in capsys.readouterr().err
    assert check.parse_args(["--url", "http://127.0.0.1:51234/", "--no-evidence"]).url == "http://127.0.0.1:51234/"
    assert check.parse_args([]).url is None


@pytest.mark.parametrize("port", [8000, 8010])
def test_the_features_check_refuses_the_dev_servers_and_the_tunnels_port_before_it_sends_or_starts_anything(monkeypatch, capsys, port):
    url_cases.refuses_a_port(check, monkeypatch, capsys, port, ("--no-evidence",))


def test_the_features_check_refuses_an_address_that_is_not_this_machines_loopback_before_it_sends_or_starts_anything(monkeypatch, capsys):
    url_cases.refuses_a_host(check, monkeypatch, capsys, ("--no-evidence",))


def test_the_features_check_refuses_an_app_that_is_not_the_tiny_one_with_a_fixed_line_before_it_starts_anything(monkeypatch, capsys):
    url_cases.refuses_a_server_that_is_not_the_tiny_app(check, monkeypatch, capsys, ("--no-evidence",))


def test_the_features_check_goes_on_with_the_tiny_app(monkeypatch, capsys):
    url_cases.goes_on_with_the_tiny_app(check, monkeypatch, capsys, ("--no-evidence",))


def test_a_server_that_cannot_show_the_features_is_named():
    ok = {"mode": "private", "features": {"retrieval": True, "labels": True}}
    assert check.missing_features(ok) is None
    assert "gallery" in check.missing_features(dict(ok, features={"retrieval": False, "labels": True}))
    assert "labeller" in check.missing_features(dict(ok, features={"retrieval": True, "labels": False}))
    assert "private" in check.missing_features(dict(ok, mode="public"))
    assert check.missing_features({}) and check.missing_features({"mode": "private"})


# ---- the checklist ---------------------------------------------------------------------------------------------------------------------

def _entry(name: str, status: str = "PASS") -> Dict[str, Any]:
    return {"name": name, "claim": "c", "status": status, "detail": "d", "seconds": 1.0, "data": {"k": [1, 2]}, "console_errors": []}


def test_the_checklist_has_one_entry_per_item_with_its_name_status_and_measured_facts(tmp_path):
    out = tmp_path / "out"
    path = check.write_checklist(str(out), {"checks": 2, "failures": ["viewer"]}, {"engine": "tiny"},
                                 [_entry("attach"), _entry("viewer", "FAIL")], [{"file": "a_1280x900_light.png"}])
    assert Path(path) == out / "checklist.json" and sorted(p.name for p in out.iterdir()) == ["checklist.json"]
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    assert list(doc) == ["result", "environment", "checks", "screenshots"]
    assert doc["result"] == {"checks": 2, "failures": ["viewer"]}
    assert [(c["name"], c["status"], c["data"]) for c in doc["checks"]] == [("attach", "PASS", {"k": [1, 2]}), ("viewer", "FAIL", {"k": [1, 2]})]
    assert doc["screenshots"] == [{"file": "a_1280x900_light.png"}]
    assert Path(path).read_text(encoding="utf-8").endswith("}\n")


def test_old_evidence_is_cleared_by_name_and_nothing_else_is_touched(tmp_path):
    for name in check.EVIDENCE_FILES:
        (tmp_path / name).write_bytes(b"old")
    (tmp_path / "notes.txt").write_text("not ours")
    (tmp_path / "other_1280x900_light.png").write_bytes(b"not ours either")
    check.clear_evidence(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt", "other_1280x900_light.png"]
    check.clear_evidence(str(tmp_path / "missing"))   # a directory that is not there yet is nothing to clear


def test_the_evidence_files_are_pictures_of_a_size_and_a_scheme_and_a_checklist_with_dark_for_the_card_and_the_viewer():
    names = check.EVIDENCE_FILES
    assert names[-1] == "checklist.json" and len(set(names)) == len(names)
    pictures = [n for n in names if n.endswith(".png")]
    assert len(pictures) + 1 == len(names) and all(re.fullmatch(r"[a-z_]+_\d+x\d+_(light|dark)\.png", n) for n in pictures), names
    for kind in ("card", "viewer_side_by_side"):
        assert any(n.startswith(kind + "_") and n.endswith("_light.png") for n in pictures), kind
        assert any(n.startswith(kind + "_") and n.endswith("_dark.png") for n in pictures), kind
    assert all(not n.endswith(".md") for n in names)


def test_every_screenshot_the_script_takes_has_a_rule_and_a_file_it_declares():
    source = SCRIPT.read_text(encoding="utf-8")
    calls = re.findall(r'ctx\.shot\("(\w+)", "(\w+)", [^\n]*?(DESKTOP|TALL)(?:, "(\w+)")?\)', source)
    assert calls, "no ctx.shot call found: the pattern is stale"
    files = set()
    for key, name, viewport, scheme in calls:
        assert key in check.SHOT_RULES, "{} has no page rule".format(key)
        width, height = {"DESKTOP": check.DESKTOP, "TALL": check.TALL}[viewport]
        files.add("{}_{}x{}_{}.png".format(name, width, height, scheme or "light"))
    assert files == set(check.EVIDENCE_FILES) - {"checklist.json"}, sorted(files ^ set(check.EVIDENCE_FILES))
    assert set(check.SHOT_RULES) == {key for key, *_ in calls}, "a rule that no screenshot uses"


def test_each_page_rule_is_a_function_body_that_returns_and_reads_the_page_only():
    for key, rule in check.SHOT_RULES.items():
        assert "return " in rule, key
        assert "http" not in rule and "innerHTML" not in rule and "fetch" not in rule, key   # a rule looks at the page; it changes and asks nothing
        assert rule.count("(") == rule.count(")") and rule.count("'") % 2 == 0, key


def _page_item(rank: int, overflows: bool, more: Any = None) -> Dict[str, Any]:
    return {"rank": str(rank), "overflows": overflows, "more": more}


def test_what_the_matches_item_states_of_show_all_is_what_it_held_every_report_to_and_whether_it_clicked():
    """P6-F fix 1. The item's facts said 'Show all and Show less' and every show_all of the committed checklist was null: the tiny reports have
    12 to 15 words, so none overflows its three lines, and the branch that clicks Show all never ran. What is checked on every report is the rule
    (the button is offered exactly when the text overflows); the click is made only on a report that overflows, and the facts say which it was."""
    none_overflow = check.show_all_facts([{"width": 760, "items": [_page_item(1, False), _page_item(2, False), _page_item(3, False)]},
                                          {"width": 760, "items": [_page_item(1, False), _page_item(2, False), _page_item(3, False)]}], None)
    assert none_overflow["reports_checked"] == 6 and none_overflow["reports_that_overflow"] == 0 and none_overflow["reports_with_the_button"] == 0
    assert none_overflow["list_width_px"] == [760, 760] and none_overflow["clicked_show_all_and_show_less"] is False
    assert "overflow" in none_overflow["not_clicked_because"] and "three lines" in none_overflow["not_clicked_because"]
    one_overflows = check.show_all_facts([{"width": 320, "items": [_page_item(1, True, "Show all"), _page_item(2, False)]}],
                                         {"report": 1, "opened_whole": True, "closed_again": True})
    assert one_overflows["reports_that_overflow"] == 1 and one_overflows["reports_with_the_button"] == 1
    assert one_overflows["clicked_show_all_and_show_less"] is True and one_overflows["not_clicked_because"] is None


def test_the_features_check_says_no_more_of_show_all_than_it_checks_and_no_longer_calls_the_thumbnail_a_picture_and_no_more():
    doc = check.__doc__
    assert "clamped to three lines with Show all" not in doc and "Show all" in doc and "overflow" in doc   # the item checks when it is offered, not a click
    assert "tests/frontend/render.test.mjs" in doc
    assert "Show all opens a clamped report and Show less closes it" in (Path(__file__).resolve().parents[1] / "tests" / "frontend" / "render.test.mjs").read_text(encoding="utf-8")
    source = SCRIPT.read_text(encoding="utf-8")
    # P6-G: the thumbnail in the user's own message opens the viewer, so no line says it is a picture and no more, or cites the P6 fix 1 ruling that kept it one
    assert "by design" not in source and "ruled at P6 fix 1" not in source and "a picture and no more" not in source


def test_the_viewer_item_expects_the_users_thumbnail_to_open_what_the_cards_your_xray_tile_opens():
    """P6-G. The thumbnail has no data-key, so the item names it (USER_THUMB); what the viewer must say of it is what it says of the card's tile."""
    assert check.USER_THUMB not in ("image:thumb", "image:model_input") and not check.USER_THUMB.startswith("neighbor:")
    assert check.expected_viewer(check.USER_THUMB) == check.expected_viewer("image:thumb")
    assert check.expected_viewer("image:thumb") == {"label": "Image viewer: Your X-ray", "footers": ["Your X-ray · 320×320 px"], "variant": "original"}
    assert check.expected_viewer("image:model_input") == {"label": "Image viewer: What the model saw", "footers": ["What the model saw · 224×224 px"],
                                                           "variant": "model_input"}
    assert check.expected_viewer("neighbor:3") == {"label": "Image viewer: Your X-ray beside Similar X-ray #3", "variant": "original",
                                                   "footers": ["Your X-ray · 320×320 px", "Similar X-ray #3 · 320×320 px (stored gallery size)"]}
    for key in ("neighbor:", "neighbor", "", "image:other", "user:other", None):
        with pytest.raises(ValueError):
            check.expected_viewer(key)   # a key the item does not know is not given the neighbour's words
    viewer = _page_source("viewer.js")
    assert "Image viewer: " in viewer and " beside " in viewer   # the words the expectation is made of


def test_the_variant_a_fact_gives_is_read_from_a_servers_picture_path_and_never_from_a_local_one():
    assert check.variant_of("/v1/messages/m_01-x/image?variant=original") == "original"
    assert check.variant_of("/v1/messages/m_1/image?variant=model_input") == "model_input"
    for local in ("blob:http://127.0.0.1:51234/abc-123", "/tmp/chat_ui_features_check_x/xray_a.png", "file:///tmp/x.png", "/v1/gallery/images/7",
                  "/v1/messages/m_1/image", "", None, 5):
        assert check.variant_of(local) is None, local


def test_the_script_drives_the_thumbnail_button_of_the_user_turn_and_the_page_still_has_it():
    assert "thumb-button" in check.PAGE_HOOKS
    assert "thumb-button" in _page_source("render.js")


def test_the_features_check_runs_its_items_and_reads_the_console_with_the_browser_checks_own_functions():
    """P6-F fix 1. drain_console was a verbatim copy of the browser check's and run_check a near copy; one implementation now, imported."""
    from scripts import chat_ui_browser_check as browser_check
    assert check.run_check is browser_check.run_check
    for name in ("run_check", "drain_console"):
        assert getattr(check, name, None) in (None, getattr(browser_check, name)), name   # never a copy of its own
    assert not hasattr(check.Context, "drain_console") and not hasattr(browser_check.Context, "drain_console")
    run_all_source = Path(check.__file__).read_text(encoding="utf-8")
    assert "claim=claim" in run_all_source and "keep=1500" in run_all_source   # what the features check says differently: the claim, and the whole of a failure


# ---- the environment of a run: what the app said of itself, read from it (P6-F fix 1) ------------------------------------------------

class _Version:
    def call(self, method: str, **params: Any) -> Dict[str, Any]:
        assert method == "Browser.getVersion"
        return {"product": "Chrome/1.2.3"}


GALLERY = {"build_id": "b-1", "images": 200, "report_rows": 240, "report_groups": 190, "towers_identical": True}


def test_the_environment_states_what_the_app_said_of_itself_and_no_fixed_words_about_it():
    models = url_cases.tiny_models(True)
    env = check.environment(_Version(), True, models, GALLERY)
    card = models["models"][0]
    assert env["models"] == [{key: card[key] for key in check.CARD_FACTS}] and set(check.CARD_FACTS) <= set(card)
    assert env["server"] == {"started_by_the_check": True, "mode": "private", "default_model": "tiny", "features": {"retrieval": True, "labels": True}}
    assert env["gallery"] == GALLERY
    assert "engine" not in env and "synthetic tiny gallery" not in json.dumps(env) and "no checkpoint, no data" not in json.dumps(env)


def test_the_words_of_the_environment_change_with_what_the_app_says():
    models = url_cases.tiny_models(False)
    models["mode"], models["default_model"], models["features"] = "public", "other", {"retrieval": False, "labels": False}
    models["models"][0].update(name="other", checkpoint="last.ckpt", drift_note="a note", device="cuda")
    env = check.environment(_Version(), False, models, None)
    assert env["server"] == {"started_by_the_check": False, "mode": "public", "default_model": "other", "features": {"retrieval": False, "labels": False}}
    assert [(c["name"], c["checkpoint"], c["drift_note"], c["device"]) for c in env["models"]] == [("other", "last.ckpt", "a note", "cuda")]
    assert env["gallery"] is None
    assert "tiny" not in json.dumps([env["server"], env["models"], env["gallery"]])   # none of the facts about the app is a word of the script's


def test_the_gallery_facts_are_the_retrieve_stages_own_five_fields():
    from app.gallery import Gallery
    assert set(GALLERY) == {"build_id", "images", "report_rows", "report_groups", "towers_identical"}
    assert all(name in Gallery.facts.__doc__ for name in GALLERY)   # the fields the stage's `gallery` detail has


# ---- the words and numbers the page is expected to show, as the page builds them ------------------------------------------------------

def test_a_similarity_reads_as_the_pages_rank_and_three_decimals_and_its_bar_as_a_rounded_per_cent():
    assert check.score_text(3, 0.91349) == "#3 · 0.913"
    assert check.score_text(1, 1.0) == "#1 · 1.000"
    assert check.score_text(2, -0.0004) == "#2 · -0.000"
    assert (check.bar_width(0.9134), check.bar_width(0.915), check.bar_width(1.2), check.bar_width(-0.3)) == ("91%", "92%", "100%", "0%")


def test_the_agreement_and_the_own_rank_badge_read_as_the_page_words_them():
    assert check.agreement_text(9, 14) == "9/14 labels agree"
    assert check.own_rank_text({"rank": 1234, "of": 2663, "hit_at_10": True}) == "Own report: rank 1,234 of 2,663 test reports · R@10 hit"
    assert check.own_rank_text({"rank": 3, "of": 40, "hit_at_10": False}) == "Own report: rank 3 of 40 test reports · R@10 miss"
    assert check.own_rank_text({"rank": 3, "of": 40}) == "Own report: rank 3 of 40 test reports"


def test_the_words_the_script_expects_are_the_words_of_the_page():
    render, composer, app = (_page_source(n) for n in ("render.js", "composer.js", "app.js"))
    for words in ("labels agree", "Own report: rank", "test reports", "R@10 hit", "R@10 miss", "Similar X-rays (13D tower)",
                  "Matching reports (13D image and text encoders)", "What the model saw (224×224)", "Dedup-aware rank", "Protocol: ", "Show all"):
        assert words in render, words
    assert check.TITLES == {"neighbors": "Similar X-rays (13D tower)", "matches": "Matching reports (13D image and text encoders)"}
    assert check.MODEL_SAW == "What the model saw (224×224)"
    for message in (check.NOT_AN_IMAGE, check.DICOM, check.TOO_BIG):
        assert message in composer, message
    assert "Drop, paste or click to attach an X-ray" in _page_source("index.html") and "Test-split study" in app


def test_the_server_side_refusal_is_the_servers_own_words_and_the_page_shows_it_as_it_is():
    from app.imaging import TOO_SMALL_MSG
    assert check.TOO_SMALL == TOO_SMALL_MSG and check.TOO_SMALL not in (check.NOT_AN_IMAGE, check.DICOM, check.TOO_BIG)   # not a page-side refusal


def test_the_viewer_arithmetic_the_script_expects_is_the_viewers():
    viewer = _page_source("viewer.js")
    assert "export const ZOOM_STEP = {};".format(check.ZOOM_STEP) in viewer and "export const PAN_STEP = {};".format(check.PAN_STEP) in viewer
    assert "export const MIN_SCALE = {}, MAX_SCALE = {};".format(check.MIN_SCALE, check.MAX_SCALE) in viewer
    assert check.wheel_factor(-240) == pytest.approx(math.exp(0.48))
    assert check.wheel_factor(240) == pytest.approx(math.exp(-0.48))
    assert (check.wheel_factor(-100000), check.wheel_factor(100000), check.wheel_factor(0)) == (2.0, 0.5, 1.0)
    assert "Math.exp(-px * 0.002)" in viewer and "clamp(Math.exp(-px * 0.002), 0.5, 2)" in viewer


def test_every_hook_of_the_page_the_script_drives_is_still_in_the_page():
    source = _page_source("index.html", "app.js", "render.js", "composer.js", "viewer.js")
    for hook in check.PAGE_HOOKS:
        assert hook in source, hook


def test_a_pictures_brightness_and_spread_are_its_luma_mean_and_standard_deviation():
    def png(image: Any) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    flat = Image.new("L", (8, 8), 100)
    assert check.luma_stats(png(flat)) == (100.0, 0.0)
    halves = Image.new("L", (10, 10), 0)
    halves.paste(255, (5, 0, 10, 10))
    mean, std = check.luma_stats(png(halves))
    assert mean == pytest.approx(127.5) and std == pytest.approx(127.5)
    colour = Image.new("RGB", (4, 4), (255, 0, 0))   # luma, not the red channel
    assert check.luma_stats(png(colour))[0] == pytest.approx(0.299 * 255, abs=1.0)


def test_the_phantom_is_a_picture_with_structure_of_the_size_asked_for():
    for size in ((320, 320), (288, 256), (256, 224)):
        image = Image.open(io.BytesIO(check.phantom_png(*size)))
        assert image.size == size and image.mode == "L"
        mean, std = check.luma_stats(check.phantom_png(*size))
        assert 40 < mean < 200 and std > 20, (size, mean, std)   # not flat, and not clipped to black or white
    assert check.phantom_png(64, 64) == check.phantom_png(64, 64)   # the same picture every time


def test_the_stages_of_a_stored_turn_are_read_from_its_events():
    events = [{"event": "message_start", "data": {"seq": 1, "options": {"k_images": 4}, "image": {"source": "upload"}}},
              {"event": "stage_end", "data": {"seq": 2, "stage": "retrieve", "detail": {"image_neighbors": [{"rank": 1}]}}},
              {"event": "stage_end", "data": {"seq": 3, "stage": "score", "skipped": "no_reference"}},
              {"event": "message_stop", "data": {"seq": 4, "status": "done"}}]
    turn = check.stages_of(events)
    assert turn["start"]["options"] == {"k_images": 4} and turn["stop"]["status"] == "done"
    assert turn["stages"]["retrieve"]["detail"]["image_neighbors"] == [{"rank": 1}] and turn["stages"]["score"]["skipped"] == "no_reference"
    empty = check.stages_of([{"event": "ping", "data": 3}, None, "x"])
    assert empty == {"start": None, "stop": None, "stages": {}}


def test_the_positive_labels_of_a_report_are_listed_in_the_chips_order():
    labels = {name: 0 for name in CHEXBERT_14}
    labels.update({"Edema": 1, "Cardiomegaly": 1, "No Finding": 1})
    assert check.positives(labels) == [n for n in CHEXBERT_14 if n in ("Edema", "Cardiomegaly", "No Finding")]
    assert check.positives({}) == [] and check.positives(None) == []


# ---- main: what it stops, in what order, when something is interrupted -------------------------------------------------------------

@pytest.mark.parametrize("interrupted", ["browser", "app"])
def test_main_stops_the_app_and_removes_its_work_even_when_a_step_of_its_clean_up_is_interrupted(monkeypatch, tmp_path, interrupted):
    """The same rule as the other two checks' mains (tests/test_chat_ui_layout_check.py): Browser.close() lets an interrupt through once it has stopped
    Chrome, and each step of the clean-up is in the finally of the one before, so the steps after the interrupted one still run."""
    import signal
    calls = []   # type: List[str]
    work = tmp_path / "work"

    class FakeApp:
        url = "http://127.0.0.1:51234/"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append("app started")

        def stop(self) -> None:
            calls.append("app stopped")
            if interrupted == "app":
                raise KeyboardInterrupt

    class FakeBrowser:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append("browser started")

        def close(self) -> None:
            calls.append("browser closed")
            if interrupted == "browser":
                raise KeyboardInterrupt

    def make_work(prefix: str = "") -> str:
        work.mkdir()
        return str(work)

    monkeypatch.setattr(check, "find_chrome", lambda: "chrome")
    monkeypatch.setattr(check, "App", FakeApp)
    monkeypatch.setattr(check, "Browser", FakeBrowser)
    monkeypatch.setattr(check, "get_json", lambda url: {"mode": "private", "features": {"retrieval": True, "labels": True}})
    monkeypatch.setattr(check, "warm_up", lambda base: 0.0)
    monkeypatch.setattr(check, "run_all", lambda *args, **kwargs: ([], None))
    monkeypatch.setattr(check, "environment", lambda *args, **kwargs: {})
    monkeypatch.setattr(check.tempfile, "mkdtemp", make_work)
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGALRM)
    try:
        with pytest.raises(KeyboardInterrupt):
            check.main(["--no-evidence"])
    finally:
        signal.signal(signal.SIGTERM, before[0])
        signal.signal(signal.SIGALRM, before[1])
        signal.alarm(0)
    assert calls == ["app started", "browser started", "browser closed", "app stopped"], calls
    assert not work.exists(), "the work directory was left behind"


def test_main_does_not_start_a_browser_for_an_app_that_lacks_the_gallery(monkeypatch, capsys):
    """A tiny app without the gallery (a --url run against the wrong tiny app) is named and refused at once, before Chrome is started."""
    started = []
    monkeypatch.setattr(check, "find_chrome", lambda: "chrome")
    monkeypatch.setattr(check, "Browser", lambda *a, **k: started.append("browser"))
    with url_cases.stub_app(url_cases.tiny_models(False)) as stub, url_cases.keeping_signals():
        assert check.main(["--url", stub.url, "--no-evidence"]) == 2
    out = capsys.readouterr().out
    assert started == [] and out.startswith("ERROR the app cannot show these features: ") and "no retrieval gallery" in out and "no labeller" in out, out
    assert stub.requests == [("GET", "/v1/models")] * 2   # the refusal's read of the app, and the check's own


# ---- hygiene -----------------------------------------------------------------------------------------------------------------------

def test_the_script_names_no_absolute_path_of_this_machine():
    assert "/Users/" not in SCRIPT.read_text(encoding="utf-8")


def test_the_script_writes_no_markdown_and_names_no_checkpoint_or_cluster_path():
    import ast
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree, clean=False)
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != docstring]
    assert not [s for s in literals if re.search(r"\.md\b", s)], "a string of the code names a markdown file"
    assert not [n for n in check.EVIDENCE_FILES if n.endswith(".md")]
    for word in ("/sc/", "ckpt", "mimic_full", "parquet", "chat_sessions"):
        assert word not in source, word
