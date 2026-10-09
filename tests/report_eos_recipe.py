"""P9-G3 test helpers: the published decoder recipe, read from the files that define it, the EOS wrapper's own, and the
throwaway trees the job tests run in.

The published run h100_report_gen_m3_tower13d_s42 was submitted by scripts/submit_v3_chain.sh (its decoder
submission, seed 42) around scripts/train_report_generation_h100.sh, whose own defaults supply every value the
chain does not set. Reading both here, instead of restating the recipe in a test, is what makes "the new wrapper
trains the published recipe" fail when either source moves. Every parse asserts its anchor, so a reformatted source
fails loudly instead of matching nothing.

Nothing here runs bash or touches the cluster, and no MIMIC data is involved (R7): the recipe is read as text, the temp
trees hold fakes, and the event files hold made-up losses.
"""
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_WRAPPER = REPO_ROOT / "scripts" / "train_report_generation_h100.sh"
CHAIN = REPO_ROOT / "scripts" / "submit_v3_chain.sh"
NEW_WRAPPER = REPO_ROOT / "scripts" / "train_report_eos_h100.sh"

# What $USER is on the cluster: configs/dataset/cxr_mimic_full.yaml hardcodes the same name in its paths.
CLUSTER_USER = "krishankumar.bhushan"
PUBLISHED_EXPERIMENT = "h100_report_gen_m3_tower13d_s42"
NEW_EXPERIMENT = "report_gen_m3_eos_s42"
FLAG_OVERRIDE = "+dataset.report_eos_target=true"


def expand(text: str, values: Dict[str, str]) -> str:
    """Bash-style expansion of ${NAME}, ${NAME:-default} and $NAME for the names in `values`, repeated because a
    value may name another variable. A name not in `values` is left as written."""
    def with_default(match):                                 # ${NAME:-default}
        return values.get(match.group(1)) or match.group(2)

    def plain(match):                                        # ${NAME} and $NAME
        return values.get(match.group(1), match.group(0))

    for _ in range(6):
        before = text
        text = re.sub(r"\$\{(\w+):-([^}]*)\}", with_default, text)
        text = re.sub(r"\$\{(\w+)\}", plain, text)
        text = re.sub(r"\$(\w+)", plain, text)
        if text == before:
            break
    return text


def unquote(token: str) -> str:
    """'key="value"' -> 'key=value': what Hydra receives once bash has consumed the quotes."""
    key, sep, value = token.partition("=")
    return key + sep + value.strip('"')


def published_wrapper_tokens() -> List[str]:
    """The published wrapper's Hydra overrides as written (variables unexpanded), in order: its python block, then
    the one it appends through EXTRA_ARGS."""
    text = PUBLISHED_WRAPPER.read_text()
    anchor = "python scripts/train_report_generation.py \\"
    assert anchor in text, "the published wrapper's python invocation moved: update tests/report_eos_recipe.py"
    block = text.split(anchor, 1)[1].split("\n\necho", 1)[0]
    lines = [line.strip().rstrip("\\").strip() for line in block.splitlines()]
    # "=" keeps overrides and drops `--config-name config` and the bash expansion of EXTRA_ARGS.
    tokens = [line for line in lines if line and "=" in line]
    tokens += re.findall(r'EXTRA_ARGS\+=\("([^"]+)"\)', text)
    assert len(tokens) > 20, tokens
    return tokens


def published_wrapper_defaults() -> Dict[str, str]:
    """NAME -> default for every `NAME="${NAME:-default}"` line of the published wrapper."""
    text = PUBLISHED_WRAPPER.read_text()
    found = dict(re.findall(r'(?m)^([A-Z][A-Z0-9_]*)="\$\{\1:-(.*)\}"(?:[ \t]*#.*)?$', text))
    assert {"MODEL_CONFIG", "BATCH_SIZE", "DECODER_LR", "SEED", "MIMIC_CACHE_DIR"} <= set(found), sorted(found)
    return found


def published_trainer_config(num_gpus: int) -> str:
    """The trainer= group the published wrapper picks for `num_gpus` (it decides from NUM_GPUS, before the python call)."""
    text = PUBLISHED_WRAPPER.read_text()
    single = re.search(r'(?m)^TRAINER_CFG="([^"]+)"$', text)
    multi = re.search(r'if \[ "\$\{NUM_GPUS\}" -gt 1 \]; then\s+TRAINER_CFG="([^"]+)"', text)
    assert single and multi, "the published wrapper's TRAINER_CFG choice moved: update tests/report_eos_recipe.py"
    return multi.group(1) if num_gpus > 1 else single.group(1)


def chain_decoder_env(seed: str = "42") -> Dict[str, str]:
    """The NAME=value pairs submit_v3_chain.sh sends the decoder job for `seed`, expanded the way the chain expands them."""
    chain = CHAIN.read_text()
    anchor = 'dec=$(_v3_submit "decoder s${seed}"'
    assert anchor in chain, "the chain's decoder submission moved: update tests/report_eos_recipe.py"
    block = chain.split(anchor, 1)[1].split("scripts/train_report_generation_h100.sh", 1)[0]
    pairs = re.findall(r'"([A-Z][A-Z0-9_]*)=([^"]*)"', block)
    assert len(pairs) >= 9, pairs
    names = dict(re.findall(r'local (\w+)="\$\{\1:-([^}]*)\}"', chain))      # STAGE0_EXPERIMENT, TOWER_CKPT, ...
    names["decoder_ckpt"] = re.search(r'local decoder_ckpt="([^"]+)"', chain).group(1)
    names["exp"] = re.search(r'\bexp="([^"]+)"', chain).group(1)
    names["seed"] = seed
    return {name: expand(value, names) for name, value in pairs}


def published_recipe() -> Dict[str, str]:
    """Every variable of the published s42 decoder run: the wrapper's defaults, then the chain's values over them."""
    values = dict(published_wrapper_defaults())
    values.update(chain_decoder_env())
    values["TRAINER_CFG"] = published_trainer_config(int(values["NUM_GPUS"]))
    return {name: expand(value, dict(values, USER=CLUSTER_USER)) for name, value in values.items()}


def published_job_overrides() -> List[str]:
    """The Hydra overrides of the published s42 decoder run, as Hydra received them."""
    values = dict(published_recipe(), USER=CLUSTER_USER)
    return [unquote(expand(token, values)) for token in published_wrapper_tokens()]


def new_wrapper_tokens() -> List[str]:
    """The EOS wrapper's OVERRIDES array, element by element as written."""
    text = NEW_WRAPPER.read_text()
    assert "OVERRIDES=(\n" in text, "the EOS wrapper has no OVERRIDES array"
    block = text.split("OVERRIDES=(\n", 1)[1].split("\n)\n", 1)[0]
    lines = [re.sub(r"\s+#.*$", "", line).strip() for line in block.splitlines()]
    return [line for line in lines if line]


def new_wrapper_assignments() -> Dict[str, str]:
    """NAME -> value of the EOS wrapper's top-level NAME=value lines (quotes dropped, variables left as written)."""
    found = {}
    for line in NEW_WRAPPER.read_text().splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)=(.*?)[ \t]*(?:#.*)?$", line)
        if match:
            found[match.group(1)] = match.group(2).strip('"')
    return found


def new_job_overrides(out_dir: str, chat_home: str = "/sc/home/{}/chat_sessions".format(CLUSTER_USER)) -> List[str]:
    """The EOS wrapper's Hydra overrides as Hydra receives them, with OUT_DIR = `out_dir`."""
    values = dict(new_wrapper_assignments(), USER=CLUSTER_USER, CHAT_HOME=chat_home, OUT_DIR=out_dir)
    values = {name: expand(value, values) for name, value in values.items()}
    return [unquote(expand(token, values)) for token in new_wrapper_tokens()]


def stub_tree(root: Path, with_flag: bool = True) -> Path:
    """A throwaway tree shaped like the cluster repo's scripts/: a package whose train_contrastive.py is a fake
    (ImageTextDataset with or without P9-G1's _eos_row, and load_mimic_cxr, the trainer's own route to it) and a copy of
    the preflight, which finds its tree from its own location the way the trainer does."""
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "__init__.py").write_text("")
    row = "def _eos_row(self, text):\n        return None\n" if with_flag else "pass\n"
    (scripts / "train_contrastive.py").write_text(
        "class ImageTextDataset:\n    {}\n\ndef load_mimic_cxr(cfg, split, tokenizer, teacher_tokenizer=None):\n"
        "    return ImageTextDataset()\n".format(row))
    shutil.copy(str(REPO_ROOT / "scripts" / "report_eos_preflight.py"), str(scripts / "report_eos_preflight.py"))
    return root


def write_events(log_dir: Path, versions: Dict[str, Sequence[Tuple[str, int, float]]]) -> None:
    """TensorBoard event files the way TensorBoardLogger(save_dir=log_dir, name="tensorboard") lays them out:
    log_dir/tensorboard/<version>/events.out.tfevents.*, one (tag, step, value) scalar at a time."""
    from torch.utils.tensorboard import SummaryWriter
    for version, scalars in versions.items():
        writer = SummaryWriter(str(Path(log_dir) / "tensorboard" / version))
        for tag, step, value in scalars:
            writer.add_scalar(tag, value, step)
        writer.close()
