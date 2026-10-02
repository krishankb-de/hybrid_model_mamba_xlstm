"""Inference engine for the chat app (CHAT_UI_PLAN.md P2-D).

A thin layer over code the thesis already trusts: load_report_generation_module (prefix_k from
run_metadata.json, the operator flags, the missing-key guard), the published transform, and the two
beam searches with the observe-only on_step callback (P2-B). Nothing here re-implements decoding.
"""
import copy
import hashlib
import json
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from app.imaging import load_upload, model_input_image, model_transform
from app.tiny import TinyTokenizer, TinyTower, tiny_decoder, tiny_prefix_mapper

if TYPE_CHECKING:
    from app.schemas import Options

DISCLAIMER = "Research prototype; not for clinical use."
REPO_ROOT = Path(__file__).resolve().parent.parent
# What `chat_remote.sh sync` judges clean or dirty by, so "dirty" means the same with and without a .git.
CODE_PATHS = ("app", "scripts", "hybrid_xmamba", "configs", "tests")


@dataclass
class StageResult:
    detail: Dict[str, Any]
    ms: float


@dataclass
class Prepared:
    image: "Image.Image"          # 8-bit RGB after load_upload (EXIF applied)
    model_input: "Image.Image"    # 224x224 RGB, exactly what the tower sees before ToTensor
    pixel_values: torch.Tensor    # (1, 3, 224, 224)
    sha256: str
    facts: Dict[str, Any]


@dataclass
class Encoded:
    patch_grid: torch.Tensor      # (1, 197, D_patch)
    pooled: torch.Tensor          # (D_joint,), L2-normalised: the image->image and image->report query
    prefix: torch.Tensor          # (1, k, D_model)


@dataclass
class Generated:
    token_ids: List[int]
    report: str
    display_report: str
    truncated_mid_sentence: bool


class Cancelled(Exception):
    """The turn's cancel event was set; generation stopped at a step boundary."""


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _read_hash_cache(cache_file: Optional[Path]) -> Dict[str, str]:
    if cache_file is None:
        return {}
    try:
        cache = json.loads(Path(cache_file).read_text())
    except (OSError, ValueError):   # absent or damaged: hash again and rewrite it
        return {}
    return cache if isinstance(cache, dict) else {}


def file_sha256(path: Path, cache_file: Optional[Path] = None, chunk: int = 1 << 20) -> str:
    """Streamed SHA-256 (D12), cached by (path, size, mtime) so a 2.4 GB checkpoint is hashed once."""
    path = Path(path).resolve()
    st = path.stat()
    key = "{}|{}|{}".format(path, st.st_size, int(st.st_mtime))
    cache = _read_hash_cache(cache_file)
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    digest = h.hexdigest()
    if cache_file is not None:
        cache[key] = digest
        try:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_file).write_text(json.dumps(cache, indent=1))
        except OSError:   # the cache is an optimisation: a full or read-only disk must not stop the engine
            pass
    return digest


def tensor_sha256(module: torch.nn.Module) -> str:
    """Fingerprint of a module's weights: fp32 bytes of every state-dict tensor, sorted by key."""
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().float().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(["git", "-C", str(root)] + list(args), capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):   # no git binary, or it hung on a slow filesystem
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_provenance(root: Path) -> Dict[str, Any]:
    """git_sha, git_dirty and where they came from, for the model card.

    The cluster tree has no .git (P0-G keeps it out of the rsync), so there the repo-root .sync_stamp
    ("<UTC time> <sha> <clean|dirty>", written by chat_remote.sh sync) is the record. A checkout wins over a
    stale stamp; a parent directory's repository never stands in for this tree; unknown stays None, not "clean".
    """
    root = Path(root).resolve()
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is not None and Path(top).resolve() == root:
        sha = _git(root, "rev-parse", "HEAD")
        if sha:
            changed = _git(root, "status", "--porcelain", "--", *CODE_PATHS)
            return {"git_sha": sha, "git_dirty": None if changed is None else bool(changed), "git_source": "git"}
    try:
        fields = (root / ".sync_stamp").read_text().split()
    except OSError:
        fields = []
    if len(fields) == 3 and re.fullmatch(r"[0-9a-f]{40}", fields[1]) and fields[2] in ("clean", "dirty"):
        return {"git_sha": fields[1], "git_dirty": fields[2] == "dirty", "git_source": "sync_stamp"}
    return {"git_sha": None, "git_dirty": None, "git_source": None}


class Engine:
    """Shared stage logic. Subclasses set name, device, tower, prefix_mapper, decoder, tokenizer and _card."""

    name = "engine"
    device = torch.device("cpu")
    drift_note = ""

    def preprocess(self, data: bytes) -> Tuple[StageResult, Prepared]:
        t0 = time.perf_counter()
        img, facts = load_upload(data)
        return self._prepared(img, dict(facts, image_sha256=hashlib.sha256(data).hexdigest()), t0)

    def _prepared(self, img: "Image.Image", facts: Dict[str, Any], t0: float) -> Tuple[StageResult, Prepared]:
        pixel_values = model_transform()(img).unsqueeze(0)
        detail = dict(facts, resized_to=[224, 224], grayscale_to_3ch=True, normalize="biomedclip_clip_mean_std")
        prepared = Prepared(img, model_input_image(img), pixel_values, facts["image_sha256"], facts)
        return StageResult(detail, _ms(t0)), prepared

    def encode(self, prepared: Prepared) -> Tuple[StageResult, Encoded]:
        t0 = time.perf_counter()
        with torch.no_grad():
            px = prepared.pixel_values.to(self.device)
            feats = self.tower.trunk.forward_features(px)          # == module._patch_grid(px)
            pooled = F.normalize(self.tower.head(self.tower.trunk.forward_head(feats)).float(), dim=-1)[0]
            prefix = self.prefix_mapper(feats)
        detail = {"patch_grid": list(feats.shape[1:]), "pooled_dim": int(pooled.shape[-1]),
                  "prefix_tokens": int(prefix.shape[1]), "device": str(self.device), "one_pass": True}
        return StageResult(detail, _ms(t0)), Encoded(feats, pooled.cpu(), prefix)

    def _decode_text(self, ids: List[int]) -> str:
        return " ".join(self.tokenizer.decode(ids, skip_special_tokens=True).split())

    def generate(self, enc: Encoded, opts: "Options", on_snapshot: Callable[[int, str], None],
                 cancel: threading.Event) -> Tuple[StageResult, Generated]:
        from scripts.evaluate_report_generation import beam_search_decode
        from scripts.repair_generations import repair_report

        if opts.cached_decode and not self.decoder.supports_cached_decode():
            raise ValueError("{} has no O(1) decode cache; set cached_decode=false.".format(self.name))
        if cancel.is_set():   # pressed during an earlier stage: do not pay for a prefill first
            raise Cancelled()
        t0 = time.perf_counter()
        first: List[float] = []

        def cb(step: int, ids: List[int]) -> None:
            if cancel.is_set():
                raise Cancelled()
            if not first:
                first.append(time.perf_counter())
            on_snapshot(step, self._decode_text(ids))

        beam = 1 if opts.decode == "greedy" else opts.beam_size
        empty = torch.zeros((1, 0), dtype=torch.long, device=self.device)   # no BOS, as in training
        with torch.no_grad():
            if opts.cached_decode:
                out = self.decoder.beam_search_cached(empty, prefix_embeds=enc.prefix, beam_size=beam,
                                                      max_new_tokens=opts.max_new_tokens, on_step=cb)
            else:
                out = beam_search_decode(self.decoder, empty, prefix_embeds=enc.prefix, beam_size=beam,
                                         max_new_tokens=opts.max_new_tokens, on_step=cb)
        decoded = time.perf_counter()
        ids = out[0].tolist()
        report = self._decode_text(ids)
        repaired, stats = repair_report(report, dedup="none", truncate=True)
        # With no complete sentence at all the repair keeps the original and reports a fallback, not a truncation.
        truncated = stats.get("sentences_truncated", 0) > 0 or (bool(report) and stats.get("fallbacks", 0) > 0)
        start = first[0] if first else t0
        detail = {"decode": opts.decode, "beam_size": beam, "tokens": len(ids), "stopped": "budget",
                  "cached_decode": bool(opts.cached_decode), "compiled": False,
                  "prefill_ms": round((start - t0) * 1000.0, 1),
                  "per_token_ms": round((decoded - start) * 1000.0 / max(len(ids) - 1, 1), 2),
                  "device": str(self.device), "threads": torch.get_num_threads(), "drift_note": self.drift_note}
        gen = Generated(ids, report, repaired if opts.display_repair else report, truncated)
        return StageResult(detail, _ms(t0)), gen

    def card(self) -> Dict[str, Any]:
        return copy.deepcopy(self._card)

    def tower_sha256(self) -> str:
        return tensor_sha256(self.tower)

    def _make_card(self, prefix_k: int, checkpoint: Optional[str] = None, checkpoint_sha256: Optional[str] = None,
                   train_experiment: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.decoder.config
        card = {"name": self.name, "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha256,
                "prefix_k": int(prefix_k), "scan_impl": cfg.scan_impl, "tfla_impl": cfg.tfla_impl,
                "mamba3_chunk_size": getattr(cfg, "mamba3_chunk_size", None),
                "layer_pattern": list(cfg.layer_pattern),
                "cached_decode_available": bool(self.decoder.supports_cached_decode()),
                "train_experiment": train_experiment,
                "torch": torch.__version__, "device": str(self.device), "threads": torch.get_num_threads(),
                "cpu": platform.processor() or platform.machine(), "drift_note": self.drift_note}
        card.update(git_provenance(REPO_ROOT))
        return card


class TinyEngine(Engine):
    """The real stages over random-init tiny weights: laptop work and the API tests. No checkpoint, no data."""

    name = "tiny"
    drift_note = "tiny random-init model"

    def __init__(self, step_delay_s: float = 0.0):
        self.step_delay_s = float(step_delay_s)
        self.tower = TinyTower()
        self.prefix_mapper = tiny_prefix_mapper()
        self.decoder = tiny_decoder()
        self.tokenizer = TinyTokenizer()
        self._card = self._make_card(prefix_k=self.prefix_mapper.k)

    def generate(self, enc: Encoded, opts: "Options", on_snapshot: Callable[[int, str], None],
                 cancel: threading.Event) -> Tuple[StageResult, Generated]:
        if self.step_delay_s <= 0:
            return super().generate(enc, opts, on_snapshot, cancel)

        def paced(step: int, text: str) -> None:
            on_snapshot(step, text)
            cancel.wait(self.step_delay_s)   # a stop pressed during the pause ends it at once

        return super().generate(enc, opts, paced, cancel)


class RealEngine(Engine):
    def __init__(self, checkpoint: str, model_config: str, device: str = "cpu", threads: int = 8,
                 cache_dir: Optional[Path] = None, drift_note: str = ""):
        from scripts.evaluate_report_generation import load_report_generation_module
        from transformers import AutoTokenizer
        torch.set_num_threads(threads)
        self.name, self.device, self.drift_note = model_config, torch.device(device), drift_note
        self.module = load_report_generation_module(checkpoint, model_config, device=device)
        self.tower, self.prefix_mapper, self.decoder = (self.module.image_encoder, self.module.prefix_mapper,
                                                        self.module.decoder)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._card = self._provenance(Path(checkpoint), cache_dir)

    def _provenance(self, ckpt: Path, cache_dir: Optional[Path]) -> Dict[str, Any]:
        meta = ckpt.resolve().parent.parent / "run_metadata.json"    # beside the run, where resolve_prefix_k reads k
        experiment = None
        try:   # write_run_metadata files the Hydra config under "resolved_config"
            experiment = json.loads(meta.read_text()).get("resolved_config", {}).get("experiment_name")
        except (OSError, ValueError, AttributeError):   # absent or damaged: the card just has no experiment name
            pass
        sha = file_sha256(ckpt, (Path(cache_dir) / "sha256.json") if cache_dir else None)
        return self._make_card(prefix_k=self.module.prefix_k, checkpoint=str(ckpt), checkpoint_sha256=sha,
                               train_experiment=experiment)


def build_engine(kind: str, **kw: Any) -> Engine:
    if kind == "tiny":
        return TinyEngine(**kw)
    if kind == "real":
        return RealEngine(**kw)
    raise ValueError("unknown engine kind {!r}: use 'tiny' or 'real'".format(kind))
