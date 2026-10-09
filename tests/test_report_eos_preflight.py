"""P9-G3 (CHAT_UI_PLAN.md): scripts/report_eos_preflight.py, the fail-fast gate of the EOS training job.

The job costs about 5 H100-hours and is worth nothing if it quietly trains the wrong thing. Two ways it could:
  - ImageTextDataset is imported from the thesis checkout the shared venv has installed (a copy that predates
    P9-G1) instead of the rsynced chat_ui tree: dataset.report_eos_target is then ignored without a word;
  - the composed config differs from the published run's by more than the new run's own names and paths, plus the flag.

CPU only, offline, synthetic data only (R7): fake dataset modules in temp trees, the published run's metadata composed
here from the published scripts' own override lists (tests/report_eos_recipe.py), and Hydra's real compose against the
real configs/. The wrapper that calls this script is exercised in tests/test_report_eos_job.py.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from scripts import report_eos_preflight as pre
from tests import report_eos_recipe as recipe

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "report_eos_preflight.py"
CONFIGS = REPO_ROOT / "configs"
OUT = "/sc/home/krishankumar.bhushan/chat_sessions/models/report_gen_m3_eos_s42"
FLAG = "dataset.report_eos_target"

# What the new run composes minus what the published one did: its own names and the paths derived from them, and the flag.
EXPECTED_CHANGED = ["checkpoint_dir", "experiment_name", "log_dir", "output_dir", "trainer.default_root_dir", "wandb.name"]


def job_overrides(overrides: List[str]) -> List[str]:
    return pre.split_overrides(overrides)[0]


def compose(overrides: List[str]) -> Dict:
    return pre.compose_job_config(job_overrides(overrides), CONFIGS)


def published_config() -> Dict:
    return compose(recipe.published_job_overrides())


def new_overrides() -> List[str]:
    return recipe.new_job_overrides(OUT)


# ── flatten ───────────────────────────────────────────────────────────────────

def test_flatten_gives_dotted_keys_and_keeps_lists_and_empty_containers_as_leaves():
    nested = {"a": {"b": 1, "c": {"d": None}}, "tags": [], "layers": ["x", "y"], "e": {}, "f": [{"g": 1}]}
    assert pre.flatten(nested) == {"a.b": 1, "a.c.d": None, "tags": [], "layers": ["x", "y"], "e": {}, "f": [{"g": 1}]}


# ── hydra.* overrides are not part of the job config (named risk 2) ───────────

@pytest.mark.parametrize("hydra_override", [
    "hydra.run.dir=/x/hydra", "hydra.job.chdir=true", "+hydra.verbose=true", "++hydra.job.name=n", "~hydra.run.dir",
    "hydra/job_logging=disabled", "hydra/hydra_logging=none",
])
def test_hydra_overrides_are_split_off_before_compose(hydra_override):
    job, hydra = pre.split_overrides(
        ["model=a", hydra_override, "+dataset.report_eos_target=true", "hydraulic=1", "model.hydra.depth=2"])
    assert job == ["model=a", "+dataset.report_eos_target=true", "hydraulic=1", "model.hydra.depth=2"]
    assert hydra == [hydra_override], "only a key whose FIRST segment is hydra configures Hydra"


# ── compare_recipe ────────────────────────────────────────────────────────────

def published_flat() -> Dict:
    return {
        "experiment_name": "pub", "output_dir": "./outputs/pub", "checkpoint_dir": "./outputs/pub/checkpoints",
        "log_dir": "./outputs/pub/logs", "trainer.default_root_dir": "./outputs/pub", "wandb.name": "pub",
        "seed": 42, "model.decoder_lr": 1e-05, "model.layer_pattern": ["a", "b"], "model.flag": False,
        "decoder_checkpoint": "./outputs/stage0/checkpoints/last.ckpt",
    }


def new_flat(**changes) -> Dict:
    """What the new run composes: the published config with the two roots swapped, the derived paths following, plus the flag."""
    flat = dict(published_flat())
    flat.update({
        "experiment_name": "new", "output_dir": "/chat/models/new", "checkpoint_dir": "/chat/models/new/checkpoints",
        "log_dir": "/chat/models/new/logs", "trainer.default_root_dir": "/chat/models/new", "wandb.name": "new",
        FLAG: True,
    })
    flat.update(changes)
    return flat


def test_the_expected_difference_is_ok_and_lists_every_changed_key_by_name():
    cmp = pre.compare_recipe(published_flat(), new_flat())
    assert cmp["ok"] is True and cmp["problems"] == []
    assert cmp["changed"] == sorted(EXPECTED_CHANGED)
    assert cmp["added"] == [FLAG] and cmp["removed"] == []


def test_a_key_derived_from_experiment_name_is_as_allowed_as_one_derived_from_output_dir():
    """wandb.name is `${experiment_name}` in configs/config.yaml: it changes with the run's name, and a check that
    allowed only the keys derived from output_dir would fail the real job on it every time."""
    assert pre.compare_recipe(published_flat(), new_flat())["ok"]
    assert "wandb.name" in pre.compare_recipe(published_flat(), new_flat())["changed"]


@pytest.mark.parametrize("key, value", [
    ("model.decoder_lr", 2e-05), ("seed", 43), ("model.layer_pattern", ["a", "c"]),
    ("decoder_checkpoint", "./outputs/other/checkpoints/last.ckpt"),
])
def test_any_other_changed_key_fails_and_comes_first_in_the_list(key, value):
    cmp = pre.compare_recipe(published_flat(), new_flat(**{key: value}))
    assert cmp["ok"] is False
    assert cmp["changed"][0] == key, "the culprit leads, so a truncated RESULT line can not hide it"
    assert sorted(cmp["changed"][1:]) == sorted(EXPECTED_CHANGED)
    assert any(key in line for line in cmp["problems"])


def test_a_derived_path_that_does_not_follow_the_new_roots_fails():
    """log_dir moved somewhere else: it differs from the published value, but not because the roots changed."""
    cmp = pre.compare_recipe(published_flat(), new_flat(log_dir="/somewhere/else/logs"))
    assert cmp["ok"] is False and cmp["changed"][0] == "log_dir"


def test_a_removed_key_fails():
    flat = new_flat()
    del flat["model.decoder_lr"]
    cmp = pre.compare_recipe(published_flat(), flat)
    assert cmp["ok"] is False and cmp["removed"] == ["model.decoder_lr"]


def test_an_added_key_other_than_the_flag_fails_and_leads_the_added_list():
    cmp = pre.compare_recipe(published_flat(), new_flat(**{"model.new_knob": 1}))
    assert cmp["ok"] is False and cmp["added"] == ["model.new_knob", FLAG]


def test_the_flag_must_be_added_and_be_true():
    missing = new_flat()
    del missing[FLAG]
    assert pre.compare_recipe(published_flat(), missing)["ok"] is False
    assert pre.compare_recipe(published_flat(), new_flat(**{FLAG: False}))["ok"] is False
    assert pre.compare_recipe(published_flat(), new_flat(**{FLAG: "true"}))["ok"] is False, "the string is not the bool"
    already = dict(published_flat(), **{FLAG: True})            # the published run can not have had it
    assert pre.compare_recipe(already, new_flat())["ok"] is False


@pytest.mark.parametrize("root", ["output_dir", "experiment_name"])
def test_a_run_that_reuses_the_published_runs_name_or_directory_fails(root):
    """Nothing differs from the published config but the flag: the run would write over the published checkpoints (R8)."""
    cmp = pre.compare_recipe(published_flat(), new_flat(**{root: published_flat()[root]}))
    assert cmp["ok"] is False
    assert any(line.startswith("ERROR recipe unchanged") and root in line for line in cmp["problems"]), cmp["problems"]


def test_comparison_is_type_strict_so_one_is_not_true_and_five_is_not_five_point_zero():
    assert pre.compare_recipe(published_flat(), new_flat(**{"model.flag": 0}))["changed"][0] == "model.flag"
    assert pre.compare_recipe(published_flat(), new_flat(seed=42.0))["changed"][0] == "seed"
    assert pre.compare_recipe(published_flat(), new_flat(**{"model.layer_pattern": ["a", "b"]}))["ok"], "equal lists"


def test_problem_lines_show_numbers_and_flags_but_never_strings():
    cmp = pre.compare_recipe(
        published_flat(), new_flat(**{"model.decoder_lr": 2e-05, "decoder_checkpoint": "./outputs/other/last.ckpt"}))
    text = "\n".join(cmp["problems"])
    assert "model.decoder_lr" in text and "1e-05" in text and "2e-05" in text
    assert "decoder_checkpoint" in text and "other" not in text and "outputs" not in text, "R7: no values that are paths"
    assert all(line.startswith("ERROR ") for line in cmp["problems"])


# ── the RESULT line: compact, under 300 characters, truncated with a count ────

def test_result_line_is_compact_json_and_a_short_payload_is_kept_whole():
    payload = {"preflight": "recipe", "changed": ["a", "b"], "added": [FLAG], "removed": [], "ok": True}
    line = pre.result_line(payload)
    assert line == 'RESULT {"preflight":"recipe","changed":["a","b"],"added":["dataset.report_eos_target"],"removed":[],"ok":true}'


def test_result_line_cuts_the_tail_of_the_longest_list_and_says_how_many():
    keys = ["trainer.some.long.dotted.key.number.%02d" % i for i in range(40)]
    payload = {"preflight": "recipe", "changed": keys, "added": [FLAG], "removed": [], "ok": False}
    line = pre.result_line(payload)
    assert line.startswith("RESULT {") and len(line) <= 300
    body = json.loads(line[len("RESULT "):])
    assert body["changed"][:3] == keys[:3], "the head survives"
    marker = body["changed"][-1]
    assert re.fullmatch(r"\+\d+ more", marker), marker
    assert len(body["changed"]) - 1 + int(marker[1:].split()[0]) == 40, "kept + counted = all of them"
    assert body["added"] == [FLAG] and body["removed"] == [], "short lists are not touched"
    assert body["ok"] is False and body["preflight"] == "recipe"


def test_result_line_cuts_every_long_list_not_just_the_first():
    payload = {"preflight": "recipe", "ok": False,
               "changed": ["c.%03d.%s" % (i, "x" * 20) for i in range(30)],
               "added": ["a.%03d.%s" % (i, "y" * 20) for i in range(30)],
               "removed": ["r.%03d.%s" % (i, "z" * 20) for i in range(30)]}
    line = pre.result_line(payload)
    assert len(line) <= 300
    body = json.loads(line[len("RESULT "):])
    for name in ("changed", "added", "removed"):
        assert body[name][0].startswith(name[0] + "."), name
        assert re.fullmatch(r"\+\d+ more", body[name][-1]), (name, body[name][-1])


# ── the code check (named risk 3) ─────────────────────────────────────────────

class WithFlag:
    def _eos_row(self, text):
        return None


class WithoutFlag:
    pass


def fake_module(tree: Path) -> str:
    module = tree / "scripts" / "train_contrastive.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("")
    return str(module)


def test_a_dataset_class_with_the_flag_inside_the_tree_passes(tmp_path):
    assert pre.check_code(lambda: (WithFlag, fake_module(tmp_path)), root=tmp_path) == {
        "eos_flag": True, "module_in_cwd": True}


def test_a_dataset_class_without_the_flag_is_reported(tmp_path):
    report = pre.check_code(lambda: (WithoutFlag, fake_module(tmp_path)), root=tmp_path)
    assert report == {"eos_flag": False, "module_in_cwd": True}


def test_an_attribute_that_is_not_callable_is_not_the_flag(tmp_path):
    class Shadow:
        _eos_row = None
    assert pre.check_code(lambda: (Shadow, fake_module(tmp_path)), root=tmp_path)["eos_flag"] is False


def test_a_module_outside_the_tree_is_reported(tmp_path):
    thesis, cluster = tmp_path / "thesis", tmp_path / "cluster"
    cluster.mkdir()
    report = pre.check_code(lambda: (WithFlag, fake_module(thesis)), root=cluster)
    assert report == {"eos_flag": True, "module_in_cwd": False}


def test_a_sibling_directory_sharing_the_trees_name_as_a_prefix_is_outside_it(tmp_path):
    tree, sibling = tmp_path / "repo", tmp_path / "repo2"
    tree.mkdir()
    assert pre.module_inside(fake_module(sibling), tree) is False
    assert pre.module_inside(fake_module(tree), tree) is True


def test_a_package_directory_symlinked_to_another_tree_is_outside_it(tmp_path):
    """The cluster's repo links outputs/, results/ and .venv into the thesis checkout: a scripts/ that were linked the
    same way would resolve into that checkout, and the check has to follow the link."""
    thesis, cluster = tmp_path / "thesis", tmp_path / "cluster"
    fake_module(thesis)
    cluster.mkdir()
    (cluster / "scripts").symlink_to(thesis / "scripts", target_is_directory=True)
    assert pre.module_inside(str(cluster / "scripts" / "train_contrastive.py"), cluster) is False


# The module under test is run as a script from a temp tree (recipe.stub_tree), the way the wrapper runs it from the
# cluster repo.
stub_tree = recipe.stub_tree


def run_cli(script: Path, *args: str, cwd: Path, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    full_env.pop("PYTHONPATH", None)
    full_env.update(env or {})
    return subprocess.run([sys.executable, str(script)] + list(args), cwd=str(cwd), env=full_env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)


CODE_OK = 'RESULT {"preflight":"code","eos_flag":true,"module_in_cwd":true}'


def test_cli_code_check_run_from_inside_the_tree_prints_the_result_line_and_exits_0(tmp_path):
    tree = stub_tree(tmp_path / "cluster_repo")
    done = run_cli(tree / "scripts" / "report_eos_preflight.py", "--only", "code", cwd=tree)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [CODE_OK]


def test_cli_code_check_fails_when_the_dataset_class_lacks_the_flag(tmp_path):
    tree = stub_tree(tmp_path / "cluster_repo", with_flag=False)
    done = run_cli(tree / "scripts" / "report_eos_preflight.py", "--only", "code", cwd=tree)
    assert done.returncode == 1, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == 'RESULT {"preflight":"code","eos_flag":false,"module_in_cwd":true}'
    assert any(line.startswith("ERROR code") and "_eos_row" in line for line in lines[1:])


def test_cli_code_check_fails_when_run_from_another_directory(tmp_path):
    """The script and its dataset module live in one tree, the job's cwd is another: not the tree the job trains from."""
    tree = stub_tree(tmp_path / "somewhere_else")
    cwd = tmp_path / "cluster_repo"
    cwd.mkdir()
    done = run_cli(tree / "scripts" / "report_eos_preflight.py", "--only", "code", cwd=cwd)
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stdout.splitlines()[0] == 'RESULT {"preflight":"code","eos_flag":true,"module_in_cwd":false}'
    assert any(line.startswith("ERROR code") for line in done.stdout.splitlines())


def test_cli_code_check_refuses_a_thesis_copy_that_wins_the_import_search(tmp_path):
    """A cluster tree that lacks scripts/train_contrastive.py (a partial rsync) while the thesis checkout is on
    PYTHONPATH: the import still succeeds, from the wrong tree, and the dataset silently ignores the flag."""
    thesis = stub_tree(tmp_path / "thesis")
    cluster = tmp_path / "cluster_repo"
    (cluster / "scripts").mkdir(parents=True)
    shutil.copy(str(SCRIPT), str(cluster / "scripts" / "report_eos_preflight.py"))
    done = run_cli(cluster / "scripts" / "report_eos_preflight.py", "--only", "code", cwd=cluster,
                   env={"PYTHONPATH": str(thesis)})
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stdout.splitlines()[0] == 'RESULT {"preflight":"code","eos_flag":true,"module_in_cwd":false}'


def test_cli_code_check_imports_the_real_dataset_the_way_the_trainer_does():
    """The real scripts/train_contrastive.py of this checkout (imported through load_mimic_cxr, the trainer's own
    route): it must carry P9-G1's _eos_row and resolve inside the cwd tree."""
    done = run_cli(SCRIPT, "--only", "code", cwd=REPO_ROOT)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [CODE_OK]


def test_the_trainers_own_import_route_is_the_one_the_preflight_follows():
    """The check is only about the trainer's dataset if it imports the way the trainer does."""
    trainer = (REPO_ROOT / "scripts" / "train_report_generation.py").read_text()
    preflight = SCRIPT.read_text()
    assert "from scripts.train_contrastive import load_mimic_cxr" in trainer
    assert "from scripts.train_contrastive import load_mimic_cxr" in preflight
    prelude = "project_root = Path(__file__).parent.parent\nsys.path.insert(0, str(project_root))"
    assert prelude in trainer and prelude in preflight, "the same sys.path prelude: the tree this file lives in comes first"


# ── the recipe check ──────────────────────────────────────────────────────────

def test_the_eos_wrappers_override_list_changes_only_names_paths_and_the_flag():
    """The wrapper's real override list and the published run's, both composed by Hydra against the real configs/."""
    published, new = pre.flatten(published_config()), pre.flatten(compose(new_overrides()))
    cmp = pre.compare_recipe(published, new)
    assert cmp["ok"], cmp["problems"]
    assert sorted(cmp["changed"]) == EXPECTED_CHANGED
    assert cmp["added"] == [FLAG] and cmp["removed"] == []
    assert new[FLAG] is True


def test_every_path_derived_from_output_dir_lands_in_the_new_directory_and_nothing_points_into_outputs():
    """Named risk 4: log_dir and checkpoint_dir follow output_dir (configs/config.yaml), so does the trainer's
    default_root_dir, and no other value of the composed config writes under ./outputs. The only strings that still
    name it are the two checkpoints the run READS."""
    new = pre.flatten(compose(new_overrides()))
    assert new["output_dir"] == OUT
    assert new["checkpoint_dir"] == OUT + "/checkpoints" and new["log_dir"] == OUT + "/logs"
    assert new["trainer.default_root_dir"] == OUT
    mentions = {key for key, value in new.items() if isinstance(value, str) and "outputs" in value}
    assert mentions == {"decoder_checkpoint", "image_encoder_checkpoint"}, mentions
    assert not [key for key, value in new.items() if isinstance(value, str) and value.startswith(OUT)
                and key not in ("output_dir", "checkpoint_dir", "log_dir", "trainer.default_root_dir")]
    assert not new.get("wandb.enabled"), "no W&B directory either"


HYDRA_RUN = """
import json, os, sys
from hydra import version
from hydra._internal.hydra import Hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
configs, out, overrides = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
version.setbase(None)            # what @hydra.main(version_base=None) does when the decorator runs
seen = {}
def task(cfg):
    from hydra.core.hydra_config import HydraConfig
    seen["cfg"] = OmegaConf.to_container(cfg, resolve=True)
    seen["cwd"] = os.path.realpath(os.getcwd())
    seen["run_dir"] = HydraConfig.get().run.dir
GlobalHydra.instance().clear()
Hydra.create_main_hydra_file_or_module(
    calling_file=None, calling_module="stub_trainer", config_path=configs, job_name="stub_trainer").run(
    config_name="config", task_function=task, overrides=overrides, with_log_configuration=False)
json.dump({"cfg": seen["cfg"], "cwd": seen["cwd"], "run_dir": seen["run_dir"], "left": sorted(os.listdir("."))},
          open(out, "w"))
"""


def hydra_run(tmp_path: Path, cwd: Path, overrides: List[str]) -> Dict:
    """Run Hydra's own run path on `overrides` in `cwd`; what the task function saw, as a dict."""
    out = tmp_path / "dump.json"
    done = subprocess.run([sys.executable, "-c", HYDRA_RUN, str(CONFIGS), str(out), json.dumps(overrides)],
                          cwd=str(cwd), capture_output=True, text=True, timeout=180)
    if done.returncode != 0 and re.search(r"(AttributeError|TypeError|ImportError)", done.stderr):
        pytest.skip("Hydra's internal run API moved: " + done.stderr.strip().splitlines()[-1])
    assert done.returncode == 0, done.stderr
    return json.loads(out.read_text())


def test_the_composed_config_is_what_hydra_hands_the_trainer_and_its_run_dir_stays_out_of_the_cwd(tmp_path):
    """Named risks 1 and 2. write_run_metadata records the cfg Hydra gives main(): the job config with the `hydra`
    node removed. The compose API must reproduce it exactly, with hydra.run.dir filtered out of the compose call, and
    the override itself must keep Hydra's own run directory away from the cwd (on the cluster ./outputs is a symlink
    into the thesis checkout, where Hydra's default outputs/<date>/<time> would be created). Hydra's own run path is
    driven without its argparse front end (Hydra 1.3.2's help strings are rejected by Python 3.14)."""
    job, out = tmp_path / "job", tmp_path / "run"
    job.mkdir()
    overrides = recipe.new_job_overrides(str(out))               # the wrapper's list, with a temp OUT_DIR
    assert "hydra.run.dir={}/hydra".format(out) in overrides
    ran = hydra_run(tmp_path, job, overrides)
    assert "hydra" not in ran["cfg"]
    assert ran["cfg"] == compose(overrides), "compose is not what the trainer's main() receives"
    assert ran["cfg"]["output_dir"] == str(out) and ran["cfg"]["dataset"]["report_eos_target"] is True
    assert ran["left"] == [] and ran["cwd"] == str(job.resolve()), "Hydra wrote into, or moved to, the cwd"
    assert (out / "hydra" / ".hydra").is_dir(), "Hydra's snapshot goes where hydra.run.dir says"


def test_a_ddp_child_rank_still_parses_the_override_list_with_the_runs_own_run_dir(tmp_path):
    """strategy=ddp starts every extra rank with the parent's argv plus hydra.run.dir="<dir>", hydra.job.name=... and
    hydra.output_subdir=null (pytorch_lightning's subprocess launcher, _hydra_subprocess_cmd). The wrapper sets
    hydra.run.dir itself, so the child sees that key twice. Each rank must still compose the same job config, the flag
    included (every rank builds its own dataset), on the same run directory, and write no .hydra snapshot of its own."""
    job, out = tmp_path / "job", tmp_path / "run"
    job.mkdir()
    parent = recipe.new_job_overrides(str(out))                  # the wrapper's list, with a temp OUT_DIR
    run_dir = out / "hydra"
    child = parent + ['hydra.run.dir="{}"'.format(run_dir), "hydra.job.name=train_ddp_process_1", "hydra.output_subdir=null"]
    assert sum(o.startswith("hydra.run.dir=") for o in child) == 2
    ran = hydra_run(tmp_path, job, child)
    assert ran["cfg"] == compose(parent), "a child rank composes a different job config"
    assert ran["cfg"]["dataset"]["report_eos_target"] is True
    assert ran["run_dir"] == str(run_dir) and ran["left"] == []
    assert not (run_dir / ".hydra").exists(), "output_subdir=null: only rank 0 writes the snapshot"


def cfg_mutations_before(source: str, function: str, call: str) -> List[str]:
    """Every statement of `function` up to and including the first one that calls `call` that assigns to, deletes from or
    re-binds `cfg`, or hands it to open_dict / setattr / OmegaConf.update. Empty when cfg arrives at `call` as received."""
    func = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == function)

    def root(node):
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def mutates(node) -> bool:
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            targets = []
        if any(root(t) == "cfg" for t in targets):
            return True
        return (isinstance(node, ast.Call) and ast.unparse(node.func) in ("open_dict", "setattr", "OmegaConf.update")
                and bool(node.args) and root(node.args[0]) == "cfg")

    hits = []
    for stmt in func.body:
        hits += [ast.unparse(n).splitlines()[0] for n in ast.walk(stmt) if mutates(n)]
        if any(isinstance(n, ast.Call) and ast.unparse(n.func) == call for n in ast.walk(stmt)):
            return hits
    raise AssertionError("{}() never calls {}".format(function, call))


def test_the_trainer_does_not_touch_cfg_before_it_writes_run_metadata():
    """Named risk 1. main() seeds, makes three directories and records cfg: nothing there changes it, so what the
    preflight composes is what the published run's metadata holds. A later edit that adds a derived value above
    write_run_metadata would make the recipe check fail every run, so it fails here instead."""
    source = (REPO_ROOT / "scripts" / "train_report_generation.py").read_text()
    assert cfg_mutations_before(source, "main", "write_run_metadata") == []


def test_the_cfg_mutation_detector_sees_assignment_deletion_rebinding_and_open_dict():
    template = "def main(cfg):\n    {}\n    write_run_metadata(cfg, cfg.output_dir)\n"
    for body in ("cfg.seed = 1", "cfg['seed'] = 1", "cfg.model.x += 1", "del cfg.seed", "cfg = merge(cfg)",
                 "with open_dict(cfg):\n        pass", "OmegaConf.update(cfg, 'a', 1)", "setattr(cfg, 'a', 1)"):
        assert cfg_mutations_before(template.format(body), "main", "write_run_metadata"), body
    assert cfg_mutations_before(template.format("print(cfg.seed)"), "main", "write_run_metadata") == []
    assert cfg_mutations_before(template.format("x = cfg.seed"), "main", "write_run_metadata") == []


def published_metadata(tmp_path: Path, mutate=None) -> Path:
    """A synthetic run_metadata.json of the published run: the published override list composed by Hydra."""
    resolved = published_config()
    if mutate:
        mutate(resolved)
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"timestamp_utc": "2026-09-19T00:00:00+00:00", "resolved_config": resolved}))
    return path


def recipe_cli(published: Path, *overrides: str) -> subprocess.CompletedProcess:
    return run_cli(SCRIPT, "--only", "recipe", "--published", str(published), "--", *overrides, cwd=REPO_ROOT)


def test_cli_recipe_check_passes_for_the_published_recipe_and_prints_one_short_result_line(tmp_path):
    done = recipe_cli(published_metadata(tmp_path), *new_overrides())
    assert done.returncode == 0, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("RESULT {") and len(lines[0]) < 300
    body = json.loads(lines[0][len("RESULT "):])
    assert body == {"preflight": "recipe", "changed": sorted(EXPECTED_CHANGED), "added": [FLAG], "removed": [], "ok": True}


def test_cli_recipe_check_refuses_a_run_that_would_overwrite_the_published_one(tmp_path):
    """The published override list plus the flag: the same experiment_name and output_dir, hence the same directory."""
    done = recipe_cli(published_metadata(tmp_path), *recipe.published_job_overrides(), "+dataset.report_eos_target=true")
    assert done.returncode == 1, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    body = json.loads(lines[0][len("RESULT "):])
    assert body == {"preflight": "recipe", "changed": [], "added": [FLAG], "removed": [], "ok": False}
    assert [l for l in lines[1:] if "unchanged" in l and "output_dir" in l]
    assert [l for l in lines[1:] if "unchanged" in l and "experiment_name" in l]


def test_cli_recipe_check_drops_hydra_overrides_before_it_composes(tmp_path):
    done = recipe_cli(published_metadata(tmp_path), *new_overrides(), "hydra.run.dir={}/hydra".format(OUT))
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout.splitlines()[0][len("RESULT "):])["ok"] is True


def test_cli_recipe_check_fails_on_drift_and_names_keys_and_numbers_only(tmp_path):
    def drift(cfg):
        cfg["model"]["decoder_lr"] = 2e-05                          # a number
        cfg["decoder_checkpoint"] = "./outputs/OTHER_STAGE0/checkpoints/last.ckpt"     # a string, which is a path
        del cfg["model"]["prefix_k"]                                # a key the new run has and this one lacks
    done = recipe_cli(published_metadata(tmp_path, drift), *new_overrides())
    assert done.returncode == 1, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    body = json.loads(lines[0][len("RESULT "):])
    assert body["ok"] is False and len(lines[0]) < 300
    assert set(body["changed"][:2]) == {"model.decoder_lr", "decoder_checkpoint"}
    assert body["added"][0] == "model.prefix_k", "an extra key leads the added list, the flag follows"
    assert [line for line in lines[1:] if not line.startswith("ERROR ")] == []
    text = "\n".join(lines)
    assert "OTHER_STAGE0" not in text, "R7: no string values"
    assert all("/" not in line for line in lines), "no paths anywhere in the output"


def test_cli_recipe_check_fails_on_an_unreadable_or_incomplete_published_file(tmp_path):
    missing = recipe_cli(tmp_path / "nope.json", *new_overrides())
    assert missing.returncode == 1
    assert [l for l in missing.stdout.splitlines() if l.startswith("ERROR recipe")]
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    assert recipe_cli(empty, *new_overrides()).returncode == 1


def test_cli_recipe_check_fails_when_hydra_rejects_an_override(tmp_path):
    done = recipe_cli(published_metadata(tmp_path), "model=no_such_model", "+dataset.report_eos_target=true")
    assert done.returncode == 1
    errors = [l for l in done.stdout.splitlines() if l.startswith("ERROR recipe")]
    assert errors and "Exception" in errors[0], "the exception CLASS is named, never its message"
    assert "no_such_model" not in done.stdout


def test_cli_runs_both_checks_by_default_and_exits_0_when_both_pass(tmp_path):
    done = run_cli(SCRIPT, "--published", str(published_metadata(tmp_path)), "--", *new_overrides(), cwd=REPO_ROOT)
    assert done.returncode == 0, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == CODE_OK and lines[1].startswith('RESULT {"preflight":"recipe"') and len(lines) == 2


def test_cli_exit_code_is_1_when_either_check_fails_and_both_still_report(tmp_path):
    tree = stub_tree(tmp_path / "cluster_repo", with_flag=False)
    shutil.copytree(str(CONFIGS), str(tree / "configs"))
    done = run_cli(tree / "scripts" / "report_eos_preflight.py", "--published", str(published_metadata(tmp_path)), "--",
                   *new_overrides(), cwd=tree)
    assert done.returncode == 1, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith('RESULT {"preflight":"code","eos_flag":false')
    assert any(l.startswith('RESULT {"preflight":"recipe"') and l.endswith('"ok":true}') for l in lines), lines


def test_the_default_published_run_is_the_one_the_chain_produces():
    """The preflight's default --published and the chain's experiment name are the same run."""
    assert pre.DEFAULT_PUBLISHED == "./outputs/{}/run_metadata.json".format(recipe.PUBLISHED_EXPERIMENT)
    assert recipe.chain_decoder_env()["EXPERIMENT"] == recipe.PUBLISHED_EXPERIMENT
