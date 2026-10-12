"""CHAT_UI_PLAN.md P9-G5a: the EOS-trained model in the server's model registry (name hybrid_150m_m3_rrg_eos, CLI alias m3eos).

In this order: the registry (the two published models unchanged; the EOS model beside them with its own name, the config it was trained
with and a checkpoint that is resolved against CHAT_HOME; a missing checkpoint is a one-line refusal that names no path), the CLI alias and
the defaults (nothing here makes the EOS model the default: that is the user's decision), the card (eos_trained and the training's name
are read from the run's own metadata, never written into the registry; the note's text is pinned), and the two ways an EOS entry is
refused rather than served without its stop (fix 1): its run's metadata does not say it was trained to end its reports, or its run has
no DONE marker. The loaders are stubbed and the checkpoints are made-up bytes: synthetic data only (R7).
"""
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from app import cli, server
from app.engine import REPO_ROOT, build_engine
from app.server import create_app
from app.tiny import TinyTokenizer, TinyTower, tiny_decoder, tiny_prefix_mapper
from tests.app_helpers import iter_sse, png_bytes

EOS_MODEL = "hybrid_150m_m3_rrg_eos"
PUBLISHED = ("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg")
EOS_CHECKPOINT = Path("models") / "report_gen_m3_eos_s42" / "checkpoints" / "last.ckpt"
G4_NUMBERS = ("2,662", "2,663", "44.2", "0.1948", "0.1920", "0.4464", "0.4661")   # the P9-G4 result the card's note is built from
EOS_NOTE = ("EOS-trained: at a 200-token budget, 2,662 of 2,663 test reports end by themselves (44.2 words on average). Against the published "
            "model: ROUGE-L 0.1948 vs 0.1920, CheXbert-14 micro F1 0.4464 vs 0.4661; BLEU and all four CheXbert F1 scores are significantly lower.")


@pytest.fixture
def keep_threads():
    """RealEngine.__init__ calls torch.set_num_threads, which is process-wide."""
    threads = torch.get_num_threads()
    yield threads
    torch.set_num_threads(threads)


def eos_run(home: Path, dataset=None, experiment="report_gen_m3_eos_s42", done: bool = True) -> Path:
    """<home>/models/report_gen_m3_eos_s42/{DONE, run_metadata.json, checkpoints/last.ckpt}: the layout scripts/train_report_eos_h100.sh
    leaves under CHAT_HOME (DONE only for a run that finished). `dataset`, when given, is the resolved config's dataset block, where
    P9-G1's report_eos_target lands."""
    run = home / EOS_CHECKPOINT.parent.parent
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "last.ckpt"
    ckpt.write_bytes(b"not a real checkpoint" * 50)
    resolved = {"experiment_name": experiment, "model": {"prefix_k": 4}}
    if dataset is not None:
        resolved["dataset"] = dataset
    (run / "run_metadata.json").write_text(json.dumps({"git_sha": "x", "resolved_config": resolved}))
    if done:
        (run / "DONE").write_text("2026-10-09T20:00:00Z job=1\n")
    return ckpt


def stub_loader(monkeypatch):
    """The heavy parts of RealEngine.__init__ (the published loader, the GPT-2 download) replaced; -> what the loader was asked for."""
    import scripts.evaluate_report_generation as erg
    import transformers
    module = SimpleNamespace(image_encoder=TinyTower(), prefix_mapper=tiny_prefix_mapper(), decoder=tiny_decoder(), prefix_k=4)
    seen = {}

    def fake_load(checkpoint, model_config, device="cpu", **kw):
        seen.update(checkpoint=checkpoint, model_config=model_config, device=device)
        return module

    monkeypatch.setattr(erg, "load_report_generation_module", fake_load)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name, **kw: TinyTokenizer()))
    return seen


def fake_builder(monkeypatch):
    """server.build_engine replaced by one that records the calls for real engines and answers a tiny engine called what the engine would
    be called. A tiny engine is built as it always was. An engine named for an EOS entry of the registry says what its run's metadata would
    say, eos_trained: the server refuses to start an EOS entry whose card does not (P9-G5a fix 1)."""
    calls = []
    real = server.build_engine

    def fake(kind, **kw):
        if kind == "tiny":
            return real(kind, **kw)
        calls.append((kind, kw))
        engine = real("tiny")
        engine.name = kw.get("name") or kw["model_config"]
        engine._card.update(name=engine.name, checkpoint=kw["checkpoint"])
        if (server.HOME_MODELS.get(engine.name) or {}).get("eos"):
            engine._card["eos_trained"] = True
        return engine

    monkeypatch.setattr(server, "build_engine", fake)
    return calls


def listing(app):
    with TestClient(app) as c:
        return c.get("/v1/models").json()


def never_serve(monkeypatch):
    """The CLI must stop before it serves: a start that did not stop would block its test for good, so serving fails the test instead."""
    monkeypatch.setattr(cli, "ServingServer", lambda *a, **k: pytest.fail("the server was started"))


# ---- the registry ------------------------------------------------------------------------------------------------------------------

def test_the_two_published_models_are_registered_exactly_as_before_and_the_eos_model_is_a_third():
    assert server.MODEL_CHECKPOINTS == {
        "hybrid_150m_m3_rrg": "outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt",
        "hybrid_150m_v2_rrg": "outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt"}
    assert set(server.MODEL_NAMES) == set(PUBLISHED) | {EOS_MODEL} and len(server.MODEL_NAMES) == 3


def test_the_published_models_are_built_exactly_as_before_and_the_eos_model_from_chat_home_under_its_own_name(tmp_path, monkeypatch):
    home = tmp_path / "home"
    eos_run(home)
    calls = fake_builder(monkeypatch)
    app = create_app(engine="real", home=str(home), models=PUBLISHED + (EOS_MODEL,), drift_note="CPU vs GPU: 0/20 differ")
    with TestClient(app):
        pass
    assert calls[:2] == [("real", {"checkpoint": str(REPO_ROOT / server.MODEL_CHECKPOINTS[m]), "model_config": m, "device": "cpu",
                                    "drift_note": "CPU vs GPU: 0/20 differ"}) for m in PUBLISHED]   # no name, no note: nothing new reaches them
    assert calls[2] == ("real", {"checkpoint": str(home.resolve() / EOS_CHECKPOINT), "model_config": "hybrid_150m_m3_rrg",
                                 "name": EOS_MODEL, "evaluation_note": server.EOS_MODEL_NOTE, "device": "cpu",
                                 "drift_note": "This model's CPU decode was not probed against a GPU decode."})   # P9-G5b: its own, not the flag's
    assert len(calls) == 3


def test_the_eos_models_checkpoint_follows_the_servers_home_and_never_the_repo(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    eos_run(first)
    eos_run(second)
    calls = fake_builder(monkeypatch)
    for home in (first, second):
        with TestClient(create_app(engine="real", home=str(home), models=(EOS_MODEL,))):
            pass
    paths = [kw["checkpoint"] for _, kw in calls]
    assert paths == [str(first.resolve() / EOS_CHECKPOINT), str(second.resolve() / EOS_CHECKPOINT)]
    assert not any(str(REPO_ROOT) in p for p in paths)


def test_the_first_model_listed_is_the_default_in_either_order_and_the_eos_model_is_listed_under_its_own_name(tmp_path, monkeypatch):
    eos_run(tmp_path / "home")
    fake_builder(monkeypatch)
    for models in ((EOS_MODEL, PUBLISHED[0]), (PUBLISHED[0], EOS_MODEL)):
        answer = listing(create_app(engine="real", home=str(tmp_path / "home"), models=models))
        assert answer["default_model"] == models[0] and [m["name"] for m in answer["models"]] == list(models)


def test_a_missing_eos_checkpoint_refuses_the_start_with_one_line_that_names_no_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(server, "build_engine", lambda *a, **k: pytest.fail("an engine was built for a model that has no checkpoint"))
    with pytest.raises(server.ModelUnavailable) as stop:
        create_app(engine="real", home=str(home), models=(EOS_MODEL,))
    message = str(stop.value)
    assert EOS_MODEL in message and "CHAT_HOME" in message and len(message.splitlines()) == 1
    for fragment in (str(tmp_path), str(home), "last.ckpt", "report_gen_m3_eos_s42", "/", "\\"):
        assert fragment not in message, fragment
    assert not home.exists(), "nothing was created: the database was not opened"


def test_a_missing_eos_checkpoint_is_only_refused_when_the_model_is_asked_for(tmp_path, monkeypatch):
    calls = fake_builder(monkeypatch)
    with TestClient(create_app(engine="real", home=str(tmp_path / "home"))) as c:    # the defaults: no EOS model
        assert [m["name"] for m in c.get("/v1/models").json()["models"]] == ["hybrid_150m_m3_rrg"]
    assert [kw["model_config"] for _, kw in calls] == ["hybrid_150m_m3_rrg"]
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "tiny"), models=(EOS_MODEL,))) as c:   # tiny ignores models
        assert [m["name"] for m in c.get("/v1/models").json()["models"]] == ["tiny"]


def test_a_directory_where_the_checkpoint_should_be_is_no_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "home" / EOS_CHECKPOINT).mkdir(parents=True)
    monkeypatch.setattr(server, "build_engine", lambda *a, **k: pytest.fail("an engine was built for a model that has no checkpoint"))
    with pytest.raises(server.ModelUnavailable):
        create_app(engine="real", home=str(tmp_path / "home"), models=(EOS_MODEL,))


def test_a_name_that_is_in_no_registry_is_still_refused_and_the_message_lists_all_three(tmp_path):
    with pytest.raises(ValueError, match="models") as stop:
        create_app(engine="real", home=str(tmp_path / "b"), models=("hybrid_70m",))
    assert all(name in str(stop.value) for name in server.MODEL_NAMES)


def test_the_cli_turns_a_missing_eos_checkpoint_into_one_error_line_and_exit_1_without_a_path(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    never_serve(monkeypatch)
    code = cli.main(["--engine", "real", "--mode", "private", "--home", str(home), "--models", "m3eos"])
    captured = capsys.readouterr()
    errors = [line for line in captured.out.splitlines() if line.startswith("ERROR")]
    assert code == 1 and len(errors) == 1 and EOS_MODEL in errors[0] and "no checkpoint in CHAT_HOME" in errors[0], captured.out
    assert str(tmp_path) not in captured.out and "last.ckpt" not in captured.out and "/" not in errors[0]
    assert not home.exists()


# ---- the alias and the defaults ----------------------------------------------------------------------------------------------------

def test_the_alias_m3eos_names_the_eos_model_and_fits_the_wrappers_models_pattern():
    assert cli.MODEL_ALIASES["m3eos"] == EOS_MODEL
    assert cli.MODEL_ALIASES["m3"] == "hybrid_150m_m3_rrg" and cli.MODEL_ALIASES["13d"] == "hybrid_150m_v2_rrg"
    assert set(cli.MODEL_ALIASES.values()) == set(server.MODEL_NAMES), "an alias for every registered model, and no other"
    for models in ("m3eos", "m3eos,m3", "m3,m3eos", "m3,13d,m3eos"):
        assert re.fullmatch(r"[a-z0-9,]{1,32}", models), models   # serve_chat_*_h100.sh's MODELS pattern


@pytest.mark.parametrize("value, want", [("m3eos", (EOS_MODEL,)), ("m3eos,m3", (EOS_MODEL, "hybrid_150m_m3_rrg")),
                                         ("m3,m3eos", ("hybrid_150m_m3_rrg", EOS_MODEL)),
                                         ("hybrid_150m_m3_rrg_eos,13d", (EOS_MODEL, "hybrid_150m_v2_rrg"))])
def test_models_take_the_eos_alias_or_its_registry_name_in_the_order_given(value, want):
    assert cli.parse_args(["--engine", "tiny", "--mode", "private", "--home", "H", "--models", value]).models == want


def test_the_eos_model_named_twice_is_refused_like_any_other(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["--engine", "tiny", "--mode", "private", "--home", "H", "--models", "m3eos,hybrid_150m_m3_rrg_eos"])
    assert "twice" in capsys.readouterr().out


def test_the_defaults_do_not_change_the_published_model_alone_is_what_a_server_serves_without_models():
    assert inspect.signature(create_app).parameters["models"].default == ("hybrid_150m_m3_rrg",)
    assert cli.build_parser().get_default("models") == "m3"
    assert cli.parse_args(["--engine", "tiny", "--mode", "private", "--home", "H"]).models == ("hybrid_150m_m3_rrg",)


def test_a_server_started_without_models_lists_the_published_model_alone_as_its_default(tmp_path, monkeypatch):
    fake_builder(monkeypatch)
    answer = listing(create_app(engine="real", home=str(tmp_path / "home")))
    assert answer["default_model"] == "hybrid_150m_m3_rrg" and [m["name"] for m in answer["models"]] == ["hybrid_150m_m3_rrg"]


# ---- the card ----------------------------------------------------------------------------------------------------------------------

def test_the_eos_models_card_reads_its_training_from_the_run_metadata_and_carries_the_g4_note(tmp_path, monkeypatch, keep_threads):
    home = tmp_path / "home"
    eos_run(home, dataset={"report_eos_target": True})
    seen = stub_loader(monkeypatch)
    answer = listing(create_app(engine="real", home=str(home), models=(EOS_MODEL,)))
    (card,) = answer["models"]
    assert answer["default_model"] == card["name"] == EOS_MODEL
    assert card["eos_trained"] is True and card["train_experiment"] == "report_gen_m3_eos_s42"
    assert card["checkpoint"] == str(home.resolve() / EOS_CHECKPOINT) and card["prefix_k"] == 4
    assert seen == {"checkpoint": card["checkpoint"], "model_config": "hybrid_150m_m3_rrg", "device": "cpu"}
    note = card["evaluation_note"]
    assert note == EOS_NOTE == server.EOS_MODEL_NOTE, "the text is pinned: it says the budget the numbers were measured at and a mean"
    assert all(number in note for number in G4_NUMBERS), note
    assert len(note) <= 280 and "\n" not in note and note.isascii(), "one or two lines of plain text"
    assert "at a 200-token budget" in note and "44.2 words on average" in note and "44.2 words each" not in note


def test_eos_trained_is_read_from_the_runs_metadata_and_the_registry_only_says_what_the_entry_is(tmp_path, monkeypatch, keep_threads):
    """The flag on the card still comes from the run's own metadata: the registry marks the entry as an EOS model (`eos`), which says what
    the entry is and so what its card must agree with; it never says `eos_trained` itself."""
    assert server.HOME_MODELS[EOS_MODEL]["eos"] is True and "eos_trained" not in server.HOME_MODELS[EOS_MODEL]
    assert not set(PUBLISHED) & set(server.HOME_MODELS), "the published models are not EOS entries and are not judged by that guard"
    home = tmp_path / "home"
    eos_run(home, dataset={"report_eos_target": True})
    stub_loader(monkeypatch)
    (card,) = listing(create_app(engine="real", home=str(home), models=(EOS_MODEL,)))["models"]
    assert card["eos_trained"] is True and card["name"] == EOS_MODEL and card["train_experiment"] == "report_gen_m3_eos_s42"


# ---- a mis-provisioned EOS entry is refused, not served without its stop (P9-G5a fix 1) -------------------------------------------

NOT_MARKED = "hybrid_150m_m3_rrg_eos is not marked as trained to end its reports in its run metadata: serve the run it came from."
NO_CHECKPOINT = "hybrid_150m_m3_rrg_eos has no checkpoint in CHAT_HOME: train it first, or serve without it."
FLAGGED_METADATA = {"git_sha": "x", "resolved_config": {"experiment_name": "report_gen_m3_eos_s42", "model": {"prefix_k": 4},
                                                         "dataset": {"report_eos_target": True}}}
HOWS = ["unflagged", "false", "null", "text", "no_dataset_block", "missing", "damaged"]   # how a run's metadata can fail to say it


def mis_provisioned_run(home: Path, how: str) -> Path:
    """A finished EOS run (checkpoint and DONE marker) whose metadata does not say it was trained to end its reports: it lacks the flag
    (`unflagged`, no dataset block), says otherwise (`false`, `null`, or `text`, since P9-G1 insists on a bool), or is not there (`missing`)
    or not readable (`damaged`). -> the run's metadata path."""
    datasets = {"unflagged": {}, "false": {"report_eos_target": False}, "null": {"report_eos_target": None},
                "text": {"report_eos_target": "true"}, "no_dataset_block": None}
    ckpt = eos_run(home, dataset=datasets.get(how, {"report_eos_target": True}))
    meta = ckpt.parent.parent / "run_metadata.json"
    if how == "missing":
        meta.unlink()
    elif how == "damaged":
        meta.write_text("{broken")
    return meta


@pytest.mark.parametrize("how", HOWS)
def test_an_eos_entry_whose_run_metadata_does_not_say_it_was_trained_to_end_its_reports_refuses_the_start(tmp_path, monkeypatch, keep_threads, how):
    home = tmp_path / "home"
    meta = mis_provisioned_run(home, how)
    stub_loader(monkeypatch)
    with pytest.raises(server.ModelUnavailable) as stop:
        create_app(engine="real", home=str(home), models=(EOS_MODEL,))
    assert str(stop.value) == NOT_MARKED, "one line that names the model and no path"
    for fragment in (str(tmp_path), "last.ckpt", "report_gen_m3_eos_s42", "run_metadata", "/", "\n"):
        assert fragment not in str(stop.value), fragment
    # The refusal came inside the block that closes the database: its exclusive lock is gone, and the same home starts once the run says so.
    meta.write_text(json.dumps(FLAGGED_METADATA))
    (card,) = listing(create_app(engine="real", home=str(home), models=(EOS_MODEL,)))["models"]
    assert card["eos_trained"] is True and card["name"] == EOS_MODEL


@pytest.mark.parametrize("how", HOWS)
def test_the_cli_turns_an_eos_entry_that_is_not_marked_eos_trained_into_one_error_line_and_exit_1(tmp_path, monkeypatch, capsys, keep_threads, how):
    home = tmp_path / "home"
    mis_provisioned_run(home, how)
    stub_loader(monkeypatch)
    never_serve(monkeypatch)
    code = cli.main(["--engine", "real", "--mode", "private", "--home", str(home), "--models", "m3eos"])
    captured = capsys.readouterr()
    assert code == 1 and [l for l in captured.out.splitlines() if l.startswith("ERROR")] == ["ERROR " + NOT_MARKED], captured.out
    assert str(tmp_path) not in captured.out and "last.ckpt" not in captured.out


def unmarked_eos_builder(monkeypatch):
    """fake_builder with the EOS entry's engine saying what a run whose metadata does not mark it would: eos_trained False. -> the build calls."""
    calls = fake_builder(monkeypatch)
    build = server.build_engine

    def unmarked(kind, **kw):
        engine = build(kind, **kw)
        if kw.get("name") == EOS_MODEL:
            engine._card["eos_trained"] = False
        return engine

    monkeypatch.setattr(server, "build_engine", unmarked)
    return calls


@pytest.mark.parametrize("order", ["m3eos,m3", "m3,m3eos"])
def test_a_mis_provisioned_eos_entry_is_refused_in_either_place_of_the_models_list_so_no_deployment_serves_it_without_its_stop(
        tmp_path, monkeypatch, order):
    """MODELS=m3eos,m3 makes the EOS entry the deployment's default, the case this guard is for; MODELS=m3,m3eos puts it beside the published
    model. Either way it stops the start, and the published model is not judged by the guard."""
    home = tmp_path / "home"
    mis_provisioned_run(home, "unflagged")
    calls = unmarked_eos_builder(monkeypatch)
    models = tuple(cli.MODEL_ALIASES[alias] for alias in order.split(","))
    with pytest.raises(server.ModelUnavailable) as stop:
        create_app(engine="real", home=str(home), models=models)
    assert str(stop.value) == NOT_MARKED and len(calls) == 2, "both were built, and the one that is not what its entry says stops the start"


@pytest.mark.parametrize("order", ["m3eos,m3", "m3,m3eos"])
def test_the_cli_refuses_a_mis_provisioned_eos_entry_in_either_place_of_models_with_one_error_line(tmp_path, monkeypatch, capsys, order):
    home = tmp_path / "home"
    mis_provisioned_run(home, "missing")
    unmarked_eos_builder(monkeypatch)
    never_serve(monkeypatch)
    code = cli.main(["--engine", "real", "--mode", "private", "--home", str(home), "--models", order])
    captured = capsys.readouterr()
    assert code == 1 and [l for l in captured.out.splitlines() if l.startswith("ERROR")] == ["ERROR " + NOT_MARKED], captured.out
    assert str(tmp_path) not in captured.out


def test_the_guard_judges_the_entries_marked_eos_and_no_other_entry_of_the_home_registry(tmp_path, monkeypatch):
    other = "hybrid_150m_m3_rrg_other"
    monkeypatch.setitem(server.HOME_MODELS, other, {k: v for k, v in server.HOME_MODELS[EOS_MODEL].items() if k != "eos"})
    monkeypatch.setattr(server, "MODEL_NAMES", server.MODEL_NAMES + (other,))
    eos_run(tmp_path / "home")
    fake_builder(monkeypatch)
    answer = listing(create_app(engine="real", home=str(tmp_path / "home"), models=(other, EOS_MODEL)))
    by_name = {m["name"]: m for m in answer["models"]}
    assert by_name[other]["eos_trained"] is False and by_name[EOS_MODEL]["eos_trained"] is True    # the unmarked entry starts as it is


# ---- an EOS run that did not finish is no checkpoint (P9-G5a fix 1) ------------------------------------------------------------------

def test_an_eos_run_without_its_done_marker_is_refused_like_a_missing_checkpoint(tmp_path, monkeypatch):
    """The decode job (scripts/eval_report_eos_h100.sh) decodes a run that has DONE and no other; the server serves the same one. Same one-line
    refusal as a checkpoint that is not there, before the database is opened and before anything is built."""
    home = tmp_path / "home"
    eos_run(home, dataset={"report_eos_target": True}, done=False)
    monkeypatch.setattr(server, "build_engine", lambda *a, **k: pytest.fail("an engine was built for a run that did not finish"))
    with pytest.raises(server.ModelUnavailable) as stop:
        create_app(engine="real", home=str(home), models=(EOS_MODEL,))
    assert str(stop.value) == NO_CHECKPOINT
    assert not (home / "chat.db").exists(), "the database was not opened"


def test_a_directory_named_done_is_no_marker_and_the_marker_is_checked_only_for_the_eos_entry(tmp_path, monkeypatch):
    home = tmp_path / "home"
    eos_run(home, dataset={"report_eos_target": True}, done=False)
    (home / "models" / "report_gen_m3_eos_s42" / "DONE").mkdir()
    calls = fake_builder(monkeypatch)
    with pytest.raises(server.ModelUnavailable):
        create_app(engine="real", home=str(home), models=(EOS_MODEL,))
    assert calls == [], "nothing was built for a run whose marker is a directory"
    with TestClient(create_app(engine="real", home=str(tmp_path / "bare"))):   # a published model asks nothing of CHAT_HOME: no run there at all
        pass
    assert [kw["model_config"] for _, kw in calls] == ["hybrid_150m_m3_rrg"]


def test_the_cli_turns_an_unfinished_eos_run_into_the_missing_checkpoint_line(tmp_path, monkeypatch, capsys, keep_threads):
    home = tmp_path / "home"
    eos_run(home, dataset={"report_eos_target": True}, done=False)
    stub_loader(monkeypatch)
    never_serve(monkeypatch)
    code = cli.main(["--engine", "real", "--mode", "private", "--home", str(home), "--models", "m3eos"])
    captured = capsys.readouterr()
    assert code == 1 and [l for l in captured.out.splitlines() if l.startswith("ERROR")] == ["ERROR " + NO_CHECKPOINT], captured.out
    assert str(tmp_path) not in captured.out and not (home / "chat.db").exists()


def test_a_model_given_no_note_has_no_evaluation_note_key_so_the_published_cards_are_byte_for_byte_what_they_were(tmp_path, monkeypatch,
                                                                                                                 keep_threads):
    ckpt = eos_run(tmp_path / "home", dataset={"report_eos_target": True})
    stub_loader(monkeypatch)
    plain = build_engine("real", checkpoint=str(ckpt), model_config="hybrid_150m_m3_rrg").card()
    assert "evaluation_note" not in plain and plain["name"] == "hybrid_150m_m3_rrg"
    named = build_engine("real", checkpoint=str(ckpt), model_config="hybrid_150m_m3_rrg", name=EOS_MODEL, evaluation_note="A note.")
    assert named.name == EOS_MODEL and named.card()["name"] == EOS_MODEL and named.card()["evaluation_note"] == "A note."
    assert {k: v for k, v in named.card().items() if k not in ("name", "evaluation_note")} == {
        k: v for k, v in plain.items() if k != "name"}, "the name and the note are the only differences"


def test_the_models_route_says_a_card_may_carry_an_evaluation_note_in_the_two_sentences_a_description_may_have(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        description = c.get("/openapi.json").json()["paths"]["/v1/models"]["get"]["description"]
    assert "`evaluation_note`" in description and "`eos_trained`" in description
    assert len(re.split(r"(?<=[.!?])\s+", description.strip())) <= 2    # tests/test_app_openapi.py's rule for every route


def test_the_note_is_aggregate_numbers_only_so_a_public_card_keeps_it_and_shortens_only_the_checkpoint(tmp_path, monkeypatch, keep_threads):
    from app.redact import redact_card
    home = tmp_path / "home"
    eos_run(home, dataset={"report_eos_target": True})
    stub_loader(monkeypatch)
    (card,) = listing(create_app(engine="real", home=str(home), models=(EOS_MODEL,)))["models"]
    public = redact_card(card, "public")
    assert public["evaluation_note"] == card["evaluation_note"] and public["checkpoint"] == "last.ckpt"
    assert "study" not in card["evaluation_note"].lower() and not re.search(r"\d{8,}", card["evaluation_note"])


# ---- the drift note belongs to the model it was measured for (P9-G5b, ruling 2b) --------------------------------------------------------

FLAG = "CPU decode matched the published GPU decode on 20/20 probe studies"   # the P1-D sentence as --drift-note carries it (made up here)
EOS_DRIFT = "This model's CPU decode was not probed against a GPU decode."


def published_run(root: Path, name: str) -> Path:
    """<root>/<name>/{run_metadata.json, checkpoints/last.ckpt}: a published model's files where RealEngine reads them, with no EOS flag in its
    metadata, as the real ones have none. -> the checkpoint's path. Made-up bytes: nothing here is a model."""
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "last.ckpt"
    ckpt.write_bytes(b"not a real checkpoint" * 50)
    (run / "run_metadata.json").write_text(json.dumps({"resolved_config": {"experiment_name": name, "model": {"prefix_k": 4}}}))
    return ckpt


def three_models(tmp_path, monkeypatch):
    """The two published models and the EOS one on RealEngine itself, its loader and tokenizer stubbed: -> the home the EOS run is in. The
    published checkpoints are pointed at made-up files (an absolute path replaces the repo root in REPO_ROOT / path)."""
    home = tmp_path / "home"
    eos_run(home, dataset={"report_eos_target": True})
    monkeypatch.setattr(server, "MODEL_CHECKPOINTS", {name: str(published_run(tmp_path / "published", name)) for name in PUBLISHED})
    stub_loader(monkeypatch)
    return home


def generate_detail(client, session_id: str, model: str) -> dict:
    """The generate stage's detail of one tiny turn on `model`, read from its stream."""
    r = client.post("/v1/sessions/{}/messages".format(session_id), files={"image": ("x.png", png_bytes(), "image/png")},
                    data={"text": "", "options": json.dumps({"model": model, "max_new_tokens": 16})})
    assert r.status_code == 200, r.text
    ends = [f["data"] for f in iter_sse([r.text]) if f["event"] == "stage_end" and f["data"].get("stage") == "generate"]
    assert len(ends) == 1
    return ends[0]["detail"]


def test_the_registry_gives_the_eos_model_its_own_drift_sentence_and_the_published_models_none():
    assert server.HOME_MODELS[EOS_MODEL]["drift_note"] == EOS_DRIFT == server.EOS_DRIFT_NOTE
    assert not set(PUBLISHED) & set(server.HOME_MODELS), "the published models take the flag's sentence, so the registry has none for them"


def test_the_flags_drift_note_is_built_into_the_published_models_only_and_the_eos_model_gets_its_own(tmp_path, monkeypatch):
    home = tmp_path / "home"
    eos_run(home)
    calls = fake_builder(monkeypatch)
    for flag in (FLAG, ""):   # the flag given, and left out: the EOS sentence is the model's, not the flag's
        calls.clear()
        with TestClient(create_app(engine="real", home=str(home), models=PUBLISHED + (EOS_MODEL,), drift_note=flag)):
            pass
        notes = {kw.get("name") or kw["model_config"]: kw["drift_note"] for _, kw in calls}
        assert notes == {PUBLISHED[0]: flag, PUBLISHED[1]: flag, EOS_MODEL: EOS_DRIFT}, flag


@pytest.mark.parametrize("device, sentence", [("cpu", EOS_DRIFT), ("cuda", "")])
def test_the_eos_models_sentence_says_a_cpu_decode_was_not_probed_so_only_a_cpu_server_carries_it(tmp_path, monkeypatch, device, sentence):
    """P9-G5b fix 1 (2): on `--device cuda` the card would say "CPU decode was not probed against a GPU decode" beside its own device: cuda. A
    GPU server has no CPU decode to speak of, so the model carries none there, as a published model carries none when --drift-note is empty;
    the flag's sentence is the operator's and goes to the published models on either device, as it did."""
    home = tmp_path / "home"
    eos_run(home)
    calls = fake_builder(monkeypatch)
    with TestClient(create_app(engine="real", home=str(home), models=PUBLISHED + (EOS_MODEL,), drift_note=FLAG, device=device)):
        pass
    notes = {kw.get("name") or kw["model_config"]: kw["drift_note"] for _, kw in calls}
    assert notes == {PUBLISHED[0]: FLAG, PUBLISHED[1]: FLAG, EOS_MODEL: sentence}, device
    assert {kw["device"] for _, kw in calls} == {device}


def test_a_card_and_a_generate_stage_carry_the_drift_note_of_their_own_model(tmp_path, monkeypatch, keep_threads):
    """Through RealEngine itself, so what is checked is what a page receives: the card of each model in /v1/models, and the detail of the
    generate stage of a turn on it. The sentence the flag carries was measured for the published models; the EOS model says it was not probed."""
    home = three_models(tmp_path, monkeypatch)
    with TestClient(create_app(engine="real", home=str(home), models=PUBLISHED + (EOS_MODEL,), drift_note=FLAG)) as c:
        cards = {m["name"]: m for m in c.get("/v1/models").json()["models"]}
        assert [cards[m]["drift_note"] for m in PUBLISHED] == [FLAG, FLAG] and cards[EOS_MODEL]["drift_note"] == EOS_DRIFT
        assert FLAG not in json.dumps(cards[EOS_MODEL]), "the P1-D sentence is nowhere on the card of a model it does not describe"
        assert [cards[m]["eos_trained"] for m in PUBLISHED] == [False, False] and cards[EOS_MODEL]["eos_trained"] is True
        sid = c.post("/v1/sessions", json={}).json()["id"]
        assert [generate_detail(c, sid, m)["drift_note"] for m in PUBLISHED] == [FLAG, FLAG]
        assert generate_detail(c, sid, EOS_MODEL)["drift_note"] == EOS_DRIFT


def test_without_the_flag_the_published_cards_have_no_drift_note_as_before_and_the_eos_card_still_says_it_was_not_probed(
        tmp_path, monkeypatch, keep_threads):
    home = three_models(tmp_path, monkeypatch)
    cards = {m["name"]: m for m in listing(create_app(engine="real", home=str(home), models=PUBLISHED + (EOS_MODEL,)))["models"]}
    assert [cards[m]["drift_note"] for m in PUBLISHED] == ["", ""] and cards[EOS_MODEL]["drift_note"] == EOS_DRIFT


def test_the_flag_and_the_wrappers_lever_are_kept_and_say_whose_cards_carry_the_sentence():
    help_text = " ".join(cli.build_parser().format_help().split())
    assert "--drift-note" in help_text and "every card carries" not in help_text and "published models" in help_text
    for wrapper in ("serve_chat_h100.sh", "serve_chat_gpu_h100.sh"):
        text = (REPO_ROOT / "scripts" / wrapper).read_text()
        assert '--drift-note="${DRIFT_NOTE:-}"' in text, wrapper                      # the lever, as it was
        assert "the P1-D sentence every" not in text.replace("\n# ", " "), wrapper   # and its header no longer says every card
