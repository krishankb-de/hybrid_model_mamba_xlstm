"""P9-G3 (CHAT_UI_PLAN.md): the preflight of the EOS training job. Two checks, run before about 5 H100-hours are spent.

    python scripts/report_eos_preflight.py [--only code|recipe] [--published run_metadata.json] -- <Hydra overrides>

scripts/train_report_eos_h100.sh runs it on the compute node, from the same cwd and with the same interpreter and
environment as the trainer, and hands it the very override list the trainer gets. A non-zero exit stops the job.

(a) code. The ImageTextDataset the trainer will import must support dataset.report_eos_target (P9-G1's `_eos_row`) and
    must come from the working tree. The shared venv has the thesis package installed, and the thesis checkout holds a
    copy of scripts/train_contrastive.py that predates the flag: a trainer that imported that one would ignore the flag
    without a word and train, for hours, the model that already exists. The import follows scripts/train_report_generation.py
    exactly (its sys.path prelude, then `from scripts.train_contrastive import load_mimic_cxr`), and the module file must
    resolve inside the cwd tree, which on the cluster is the rsynced chat_ui code.

(b) recipe. The job config, composed with Hydra's compose API from the same overrides (hydra.* ones dropped: they configure
    Hydra, they are not part of the job config), must equal the published run's run_metadata.json `resolved_config` apart
    from the new run's own names and paths and the flag. Changed, added and removed keys are reported by name.
      allowed changes: experiment_name and output_dir, which must both change, and any key whose published value is
                       that value with the two published names replaced by the new ones (log_dir, checkpoint_dir,
                       trainer.default_root_dir are `${output_dir}/...`, wandb.name is `${experiment_name}`);
      required:        dataset.report_eos_target added, and true;
      everything else (any other changed key, any removed or other added key) fails.
    main() in the trainer does not touch cfg before write_run_metadata, so the composed config is what that file holds
    (tests/test_report_eos_preflight.py pins both).

R7: stdout carries `RESULT {json}` and `ERROR ...` lines only, each under 300 characters (what `chat_remote.sh summary`
keeps), with key names, numbers and flags and never a string value (a value can be a path) or an exception message. A
traceback goes to stderr, which the wrapper keeps in a file on the cluster.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# The trainer's own prelude (scripts/train_report_generation.py): the tree this file lives in comes first on sys.path, so
# `scripts.train_contrastive` and `hybrid_xmamba` resolve to it however the venv's editable install points.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_PUBLISHED = "./outputs/h100_report_gen_m3_tower13d_s42/run_metadata.json"
ROOT_KEYS = ("experiment_name", "output_dir")             # the two names the new run sets itself; both must change
REQUIRED_ADDED = {"dataset.report_eos_target": True}      # the one change the job exists for
RESULT_LIMIT = 300                                        # what `chat_remote.sh summary` keeps of a line
MAX_PROBLEM_LINES = 12


# ── the recipe check ──────────────────────────────────────────────────────────

def flatten(node: Any, prefix: str = "") -> Dict[str, Any]:
    """Dotted-key view of a nested dict. A list, an empty dict and every scalar is one leaf: the report is by key name,
    and a changed list is one changed key, not one per element."""
    if isinstance(node, dict) and node:
        flat: Dict[str, Any] = {}
        for key, value in node.items():
            flat.update(flatten(value, "{}.{}".format(prefix, key) if prefix else str(key)))
        return flat
    return {prefix: node}


def is_hydra_override(override: str) -> bool:
    """`hydra.run.dir=...`, `+hydra.verbose=true`, `~hydra.run.dir`, `hydra/job_logging=disabled`: it configures Hydra's own
    node, which the trainer's @hydra.main removes before main() sees the config, and the compose API would reject."""
    key = override.lstrip("+~").split("=", 1)[0]
    return key.split(".", 1)[0] == "hydra" or key.startswith("hydra/")


def split_overrides(overrides: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(job overrides, hydra overrides), each in the order given."""
    job = [o for o in overrides if not is_hydra_override(o)]
    return job, [o for o in overrides if is_hydra_override(o)]


def compose_job_config(overrides: Sequence[str], config_dir: Path) -> Dict[str, Any]:
    """The job config Hydra hands the trainer's main(), resolved: what write_run_metadata records as resolved_config.
    `overrides` must be free of hydra.* ones (split_overrides)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
            cfg = compose(config_name="config", overrides=list(overrides))
    finally:
        GlobalHydra.instance().clear()
    return OmegaConf.to_container(cfg, resolve=True)


def same(a: Any, b: Any) -> bool:
    """Equal in value AND type: JSON text tells 1 from true and 5 from 5.0, which `==` does not."""
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def explained_by_roots(published_value: Any, new_value: Any, swaps: Sequence[Tuple[Any, Any]]) -> bool:
    """True when `new_value` is `published_value` with each published root replaced by the new one, in the order given
    (output_dir first: it contains the experiment name). That is a key that merely DERIVES from the roots; any other
    difference is a real change."""
    if not isinstance(published_value, str) or not isinstance(new_value, str):
        return False
    text = published_value
    for old, new in swaps:
        if isinstance(old, str) and old and isinstance(new, str):
            text = text.replace(old, new)
    return text == new_value


def _numbers(published: Any, new: Any) -> str:
    """' published=1e-05 new=2e-05' for numbers, flags and null; nothing for a string or a list (R7: it may be a path)."""
    plain = (bool, int, float, type(None))
    if isinstance(published, plain) and isinstance(new, plain):
        return " published={} new={}".format(json.dumps(published), json.dumps(new))
    return ""


def compare_recipe(published: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Both arguments are flattened configs. Returns the changed, added and removed keys by name (the keys behind a
    failure first, so a cut-down RESULT line can not hide them), `ok`, and one ERROR line per problem."""
    swaps = [(published.get(key), new.get(key)) for key in ("output_dir", "experiment_name")]
    changed = sorted(k for k in published if k in new and not same(published[k], new[k]))
    unexplained = [k for k in changed if k not in ROOT_KEYS and not explained_by_roots(published[k], new[k], swaps)]
    explained = [k for k in changed if k not in unexplained]
    added = sorted(k for k in new if k not in published)
    unexpected = [k for k in added if k not in REQUIRED_ADDED]
    removed = sorted(k for k in published if k not in new)
    missing = [k for k, want in REQUIRED_ADDED.items() if not (k in added and same(new[k], want))]
    unchanged = [k for k in ROOT_KEYS if k in published and k in new and same(published[k], new[k])]

    problems = ["ERROR recipe changed: {}{}".format(k, _numbers(published[k], new[k])) for k in unexplained]
    problems += ["ERROR recipe removed: {}".format(k) for k in removed]
    problems += ["ERROR recipe added: {}".format(k) for k in unexpected]
    problems += ["ERROR recipe not added as required: {}".format(k) for k in missing]
    problems += ["ERROR recipe unchanged, the new run would write over the published one: {}".format(k) for k in unchanged]
    return {
        "changed": unexplained + explained,
        "added": unexpected + [k for k in added if k in REQUIRED_ADDED],
        "removed": removed,
        "ok": not problems,
        "problems": problems,
    }


def result_line(payload: Dict[str, Any], limit: int = RESULT_LIMIT) -> str:
    """`RESULT {compact json}` of at most `limit` characters. A line that is too long loses the tail of its longest list
    at a time, replaced by a '+N more' entry, so the lists keep their heads (the culprits come first) and a count."""
    kept = {k: list(v) for k, v in payload.items() if isinstance(v, list)}
    dropped = {k: 0 for k in kept}

    def render() -> str:
        body = dict(payload)
        for name, items in kept.items():
            body[name] = items + (["+{} more".format(dropped[name])] if dropped[name] else [])
        return "RESULT " + json.dumps(body, separators=(",", ":"))

    line = render()
    while len(line) > limit:
        longest = max(kept, key=lambda name: len(json.dumps(kept[name])), default=None)
        if longest is None or not kept[longest]:
            break
        kept[longest].pop()
        dropped[longest] += 1
        line = render()
    return line


def run_recipe_check(published_path: str, overrides: Sequence[str], config_dir: Optional[Path] = None) -> bool:
    """Prints the recipe RESULT line and one ERROR line per problem; True when the recipe is the published one plus the flag."""
    try:
        published = flatten(json.loads(Path(published_path).read_text())["resolved_config"])
    except Exception as exc:     # R7: the class name only; the traceback goes to stderr
        traceback.print_exc()
        print("ERROR recipe: cannot read resolved_config of the published run ({})".format(type(exc).__name__))
        return False
    job_overrides, _ = split_overrides(overrides)
    try:
        new = flatten(compose_job_config(job_overrides, config_dir or project_root / "configs"))
    except Exception as exc:
        traceback.print_exc()
        print("ERROR recipe: Hydra could not compose the job config ({})".format(type(exc).__name__))
        return False
    cmp = compare_recipe(published, new)
    print(result_line({"preflight": "recipe", "changed": cmp["changed"], "added": cmp["added"],
                       "removed": cmp["removed"], "ok": cmp["ok"]}))
    for line in cmp["problems"][:MAX_PROBLEM_LINES]:
        print(line)
    if len(cmp["problems"]) > MAX_PROBLEM_LINES:
        print("ERROR recipe: {} more problem(s) not shown".format(len(cmp["problems"]) - MAX_PROBLEM_LINES))
    return cmp["ok"]


# ── the code check ────────────────────────────────────────────────────────────

def module_inside(module_file: str, root: Path) -> bool:
    """Whether `module_file`, symlinks resolved, lies inside the tree `root`. A directory merely sharing the tree's name as a
    prefix is outside it, and so is a package directory symlinked into another tree."""
    try:
        Path(module_file).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def load_trainer_dataset_class() -> Tuple[type, str]:
    """ImageTextDataset the way scripts/train_report_generation.py reaches it: through load_mimic_cxr in
    scripts.train_contrastive, which builds ImageTextDataset from that module's own globals. Returns that class and the
    file of the module it lives in."""
    from scripts.train_contrastive import load_mimic_cxr
    module = sys.modules[load_mimic_cxr.__module__]
    return load_mimic_cxr.__globals__["ImageTextDataset"], module.__file__


def check_code(load: Callable[[], Tuple[type, str]] = load_trainer_dataset_class,
               root: Optional[Path] = None) -> Dict[str, bool]:
    cls, module_file = load()
    return {
        "eos_flag": callable(getattr(cls, "_eos_row", None)),
        "module_in_cwd": module_inside(module_file, Path.cwd() if root is None else root),
    }


def run_code_check(load: Callable[[], Tuple[type, str]] = load_trainer_dataset_class,
                   root: Optional[Path] = None) -> bool:
    try:
        report = check_code(load, root)
    except Exception as exc:     # an import failure is a failed check, not a crash; the class name is all that is printed
        traceback.print_exc()
        print("ERROR code check raised {}".format(type(exc).__name__))
        return False
    print(result_line({"preflight": "code", "eos_flag": report["eos_flag"], "module_in_cwd": report["module_in_cwd"]}))
    if not report["eos_flag"]:
        print("ERROR code: ImageTextDataset has no _eos_row, so dataset.report_eos_target would be ignored")
    if not report["module_in_cwd"]:
        print("ERROR code: the dataset module resolves outside the working tree, so it is not the code that was synced")
    return all(report.values())


# ── command line ──────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    own, overrides = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     epilog="Everything after -- is the job's Hydra override list, as the trainer gets it.")
    parser.add_argument("--only", choices=("code", "recipe"), help="run one check (default: both)")
    parser.add_argument("--published", default=DEFAULT_PUBLISHED,
                        help="the published run's run_metadata.json (default: %(default)s)")
    args = parser.parse_args(own)
    passed = []
    if args.only in (None, "code"):
        passed.append(run_code_check())
    if args.only in (None, "recipe"):
        passed.append(run_recipe_check(args.published, overrides))
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
