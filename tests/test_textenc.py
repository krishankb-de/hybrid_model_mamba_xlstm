"""TEXTENC_PLAN.md T1: the attention-free mLMamba text encoder, CPU only, no downloads.

R3: the encoder contains no attention module and never calls scaled_dot_product_attention.
T1-A: its backbone is the published mLMamba decoder backbone (same keys and shapes as
hybrid_150m_m3_rrg), so the Stage-0 mLMamba checkpoint loads into it completely.
T1-D: the Stage-1 wrapper keeps the published 13D tower recipe and refuses to write anywhere
but its own root.
"""
import os
import pathlib
import subprocess

import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG_DIR = REPO_ROOT / "configs" / "model"
WRAPPER = REPO_ROOT / "scripts" / "textenc_stage1_h100.sh"

BACKBONE_KEYS = [
    "vocab_size", "dim", "num_layers", "layer_pattern", "head_dim", "num_heads", "use_tfla",
    "proj_factor", "norm_type", "use_mlp", "mlp_ratio", "max_position_embeddings",
    "tie_word_embeddings", "norm_topology", "tfla_impl",
]


def _yaml(name):
    return OmegaConf.load(CFG_DIR / f"{name}.yaml")


def _small_encoder(pooling="mean", bidirectional=False, pattern=None):
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridTextEncoder

    torch.manual_seed(0)
    cfg = HybridConfig(vocab_size=97, dim=64, num_layers=4,
                       layer_pattern=pattern or ["mamba3", "mlstm", "mamba3", "mamba3"],
                       head_dim=16, num_heads=4, max_position_embeddings=64, tfla_impl="exact",
                       mamba3_d_state=32, mamba3_head_dim=16, mamba3_chunk_size=8,
                       pooling_strategy=pooling, bidirectional_encode=bidirectional)
    return HybridTextEncoder(cfg, embed_dim=32).eval()


# ---------------------------------------------------------------- T1-A config


def test_textenc_yaml_is_attention_free_and_exact():
    cfg = _yaml("hybrid_150m_m3_textenc")
    assert cfg.pooling_strategy == "mean"
    assert cfg.tfla_impl == "exact"
    assert cfg.bidirectional_encode is False
    assert "attention" not in list(cfg.layer_pattern)
    assert list(cfg.layer_pattern) == ["mamba3"] * 4 + ["mlstm"] * 3 + ["mamba3"] * 5


def test_textenc_backbone_matches_published_mlmamba_decoder():
    enc, dec = _yaml("hybrid_150m_m3_textenc"), _yaml("hybrid_150m_m3_rrg")
    for key in BACKBONE_KEYS + [k for k in dec if str(k).startswith("mamba3_")]:
        assert OmegaConf.select(enc, key) == OmegaConf.select(dec, key), key


def test_textenc_backbone_state_dict_matches_stage0_layout():
    """Every Stage-0 mLMamba tensor has a slot of the same shape in the encoder backbone."""
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel, HybridTextEncoder

    stage0_cfg = _yaml("hybrid_150m_m3")
    stage0_cfg.tfla_impl = "exact"                # the Stage-0 run's argv override
    with torch.device("meta"):
        stage0 = HybridLanguageModel(HybridConfig.from_hydra(stage0_cfg))
        enc = HybridTextEncoder(HybridConfig.from_hydra(_yaml("hybrid_150m_m3_textenc")))
    s0 = {k: tuple(v.shape) for k, v in stage0.state_dict().items()}
    lm = {k: tuple(v.shape) for k, v in enc.lm.state_dict().items()}
    assert s0 == lm
    assert enc.attn_pool is None


# ---------------------------------------------------------------- R3 no attention


def test_text_encoder_has_no_attention_module():
    from hybrid_xmamba.models.hybrid_lm import AttentionPooling

    enc = _small_encoder()
    banned = [AttentionPooling, torch.nn.MultiheadAttention]
    try:
        from hybrid_xmamba.layers.attention_block import AttentionBlock
        banned.append(AttentionBlock)
    except ImportError:
        pass
    for name, mod in enc.named_modules():
        assert not isinstance(mod, tuple(banned)), name
        assert "attention" not in type(mod).__name__.lower(), (name, type(mod).__name__)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_text_encoder_never_calls_sdpa(monkeypatch, bidirectional):
    def boom(*a, **k):
        raise AssertionError("scaled_dot_product_attention called")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", boom)
    enc = _small_encoder(bidirectional=bidirectional)
    ids = torch.randint(0, 97, (2, 12))
    with torch.no_grad():
        z = enc.encode(ids, attention_mask=torch.ones_like(ids))
    assert z.shape == (2, 32) and torch.isfinite(z).all()


def test_attention_pooling_control_is_detected():
    """The guard above must actually catch the published encoder's attention pooler."""
    from hybrid_xmamba.models.hybrid_lm import AttentionPooling

    enc = _small_encoder(pooling="attention")
    assert any(isinstance(m, AttentionPooling) for m in enc.modules())


# ---------------------------------------------------------------- pooling behaviour


@pytest.mark.parametrize("bidirectional", [False, True])
def test_mean_pooling_ignores_right_padding(bidirectional):
    enc = _small_encoder(bidirectional=bidirectional)
    ids = torch.randint(1, 97, (1, 10))
    padded = torch.cat([ids, torch.zeros(1, 6, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 6)], dim=1).long()
    with torch.no_grad():
        a = enc.encode(ids, attention_mask=torch.ones_like(ids))
        b = enc.encode(padded, attention_mask=mask)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_text_encoder_trains_on_cpu():
    enc = _small_encoder().train()
    ids = torch.randint(0, 97, (3, 12))
    z = enc.encode(ids, attention_mask=torch.ones_like(ids))
    (z @ z.t()).diagonal().sum().backward()
    grads = [p.grad for n, p in enc.named_parameters()
             if p.requires_grad and not n.startswith(("logit_", "lm.lm_head"))]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------- T1-D wrapper


def _run_wrapper(env_extra):
    env = {k: v for k, v in os.environ.items() if k != "TEXTENC_ROOT"}
    env.update(env_extra)
    return subprocess.run(["bash", str(WRAPPER)], env=env, capture_output=True, text=True,
                          timeout=60)


def test_wrapper_refuses_without_root():
    r = _run_wrapper({})
    assert r.returncode == 2 and "TEXTENC_ROOT is not set" in r.stdout


@pytest.mark.parametrize("root", ["/sc/home/u/hybrid_mamba_xlstm", "/sc/home/u/hybrid_isbi",
                                  "/sc/home/u/hybrid_mamba_xlstm/outputs"])
def test_wrapper_refuses_published_roots(root):
    r = _run_wrapper({"TEXTENC_ROOT": root})
    assert r.returncode == 2 and "published checkout" in r.stdout


def test_wrapper_refuses_symlinked_outputs(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "outputs").symlink_to(tmp_path / "elsewhere")
    r = _run_wrapper({"TEXTENC_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "symlink" in r.stdout


def test_wrapper_pins_the_published_tower_recipe():
    sh = WRAPPER.read_text()
    for line in ["BATCH_SIZE=64", 'MAX_STEPS="${MAX_STEPS:-6000}"', "BACKBONE_LR=1.41e-5",
                 "HEAD_LR=4.24e-4", "VIT_UNFREEZE=12", "VIT_LR=3e-6", "VIT_SCOPE=blocks",
                 "GAMMA_SIMCSE=0.1", "KD_DECAY_STEPS=0", "ALPHA_KD_FLOOR=0.0",
                 "CLIP_LOSS=infonce", "MULTIPOS=false", "DATASET_CONFIG=cxr_mimic_full",
                 "+distill=biomedclip_kd_joint_v2", "contrastive_mode=joint",
                 "model.use_gradient_checkpointing=true", "trainer.compile_model=false",
                 "MODEL_CONFIG=hybrid_150m_m3_textenc"]:
        assert line in sh, line


def test_ablation_transformer_text_encoder_runs_with_mean_pooling():
    """ABL-T (matched Transformer text encoder) uses the same pooling; it must at least run."""
    enc = _small_encoder(pattern=["attention"] * 3)
    ids = torch.randint(1, 97, (2, 10))
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    with torch.no_grad():
        z = enc.encode(ids, attention_mask=mask)
    assert z.shape == (2, 32) and torch.isfinite(z).all()
    assert _yaml("transformer_150m_baseline").get("pooling_strategy", "mean") in ("mean", "attention")


# ---------------------------------------------------------------- T3 retrieval index


def _unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def test_rerank_excludes_same_patient_and_orders_by_text_score():
    from scripts.textenc_rag_index import rerank_candidates

    torch.manual_seed(0)
    gal_img = _unit(torch.randn(8, 4))
    gal_txt = _unit(torch.randn(8, 4))
    q = gal_img[[2]].clone()                       # identical to gallery row 2
    gal_subj = [10, 11, 12, 13, 14, 15, 16, 17]
    res = rerank_candidates(q, gal_img, gal_txt, [12], gal_subj, n_candidates=5, k=3, device="cpu")
    assert 2 not in res["cand"][0].tolist()        # own patient never retrieved
    it = (q @ gal_txt.t())[0]
    cand = res["cand"][0]
    expected = cand[it[cand].argsort(descending=True)[:3]]
    assert res["topk"][0].tolist() == expected.tolist()
    assert torch.all(res["ii"][0][:-1] >= res["ii"][0][1:])   # candidates by image score
    assert res["topk"].shape == (1, 3) and res["it"].shape == (1, 5)


def test_rerank_chunking_is_exact():
    from scripts.textenc_rag_index import rerank_candidates

    torch.manual_seed(1)
    gal_img, gal_txt, q = _unit(torch.randn(30, 8)), _unit(torch.randn(30, 8)), _unit(torch.randn(7, 8))
    subj_g, subj_q = list(range(30)), [3, 3, 5, 100, 101, 7, 29]
    a = rerank_candidates(q, gal_img, gal_txt, subj_q, subj_g, 10, 4, chunk=3, device="cpu")
    b = rerank_candidates(q, gal_img, gal_txt, subj_q, subj_g, 10, 4, chunk=1000, device="cpu")
    for key in ("cand", "topk"):
        assert torch.equal(a[key], b[key]), key
    for key in ("ii", "it"):                       # matmul blocking differs by batch size
        assert torch.allclose(a[key], b[key], atol=1e-6), key


def test_report_text_matches_stage1_format():
    from scripts.textenc_rag_index import report_text

    assert report_text("a b", "c") == "Findings: a b Impression: c"
    assert report_text(None, "c") == "Findings:  Impression: c"


# ---------------------------------------------------------------- T4 RAG decoder


class _CharTok:
    """Offline stand-in for the GPT-2 tokenizer: one id per character, '\\n' -> 10."""
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text):
        return [min(ord(c), 96) for c in text]


def test_context_token_ids_budget_and_separator():
    from hybrid_xmamba.training.rag_report_generation import CONTEXT_LEAD, context_token_ids

    tok = _CharTok()
    ids = context_token_ids(tok, ["short", "x" * 500], per_report_tokens=40)
    first = tok.encode(CONTEXT_LEAD + "short") + [10]
    assert ids[: len(first)] == first
    assert len(ids) == len(first) + 40 and ids[-1] == 10


def test_build_rag_example_labels_only_the_report():
    from hybrid_xmamba.training.rag_report_generation import build_rag_example

    ex = build_rag_example([5, 6, 7], [20, 21], total_len=8, pad_id=0)
    assert ex["rag_input_ids"].tolist() == [5, 6, 7, 20, 21, 0, 0, 0]
    assert ex["rag_labels"].tolist() == [-100, -100, -100, 20, 21, -100, -100, -100]
    assert ex["rag_attention_mask"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        build_rag_example([1] * 6, [2] * 3, total_len=8, pad_id=0)


class _FakeBase(torch.utils.data.Dataset):
    max_length = 6

    def __init__(self):
        self.data = {"study_id": [101, 102, 101]}

    def __len__(self):
        return 3

    def __getitem__(self, i):
        ids = torch.tensor([30 + i, 31, 32, 0, 0, 0])
        return {"input_ids": ids, "attention_mask": (ids != 0).long(),
                "pixel_values": torch.zeros(3, 4, 4)}


def test_rag_dataset_adds_context_and_keeps_published_keys():
    from hybrid_xmamba.training.rag_report_generation import RAGDataset, context_token_ids

    tok = _CharTok()
    ds = RAGDataset(_FakeBase(), {101: [1, 0], 102: [2, 1]}, ["aa", "bb", "cc"], tok,
                    k=2, per_report_tokens=20)
    item = ds[1]
    ctx = context_token_ids(tok, ["cc", "bb"], 20)
    n = len(ctx)
    assert item["rag_input_ids"][:n].tolist() == ctx
    assert item["rag_input_ids"][n:n + 3].tolist() == [31, 31, 32]
    assert item["rag_labels"][:n].eq(-100).all() and item["rag_labels"][n:n + 3].tolist() == [31, 31, 32]
    assert item["rag_input_ids"].shape[0] == 2 * 20 + 6
    assert set(_FakeBase()[0]).issubset(item)                 # published keys untouched
    with pytest.raises(KeyError):
        RAGDataset(_FakeBase(), {101: [0]}, ["aa"], tok, k=1, per_report_tokens=10)


def _tiny_rg_module(pattern):
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.training.rag_report_generation import RAGReportGenerationLightningModule

    torch.manual_seed(0)
    cfg = HybridConfig(vocab_size=97, dim=64, num_layers=len(pattern), layer_pattern=pattern,
                       head_dim=16, num_heads=4, max_position_embeddings=256, tfla_impl="exact",
                       mamba3_d_state=32, mamba3_head_dim=16, mamba3_chunk_size=8)
    return RAGReportGenerationLightningModule(
        decoder_config=cfg, image_patch_dim=32, prefix_k=4, decoder_lr=1e-5, head_lr=3e-4,
        weight_decay=0.01, warmup_steps=2, max_steps=10, gradient_clip_val=0.5)


def test_rag_step_without_context_equals_published_step():
    """With an empty context the RAG loss is exactly the published loss on the same report."""
    from hybrid_xmamba.training.rag_report_generation import build_rag_example

    mod = _tiny_rg_module(["mamba3", "mlstm"]).eval()
    ids = torch.tensor([[11, 12, 13, 14, 0, 0], [21, 22, 23, 0, 0, 0]])
    mask = (ids != 0).long()
    grid = torch.randn(2, 9, 32)
    published = mod._step({"input_ids": ids, "attention_mask": mask, "patch_grid": grid}, "val")
    rag = [build_rag_example([], ids[b][mask[b].bool()].tolist(), 6, 0) for b in range(2)]
    batch = {key: torch.stack([r[key] for r in rag]) for key in rag[0]}
    batch["patch_grid"] = grid
    assert torch.allclose(mod._step(batch, "val"), published, atol=1e-6)


def test_rag_step_trains_and_ignores_context_in_the_loss():
    from hybrid_xmamba.training.rag_report_generation import build_rag_example

    mod = _tiny_rg_module(["mamba3", "mlstm"]).train()
    rag = [build_rag_example([40, 41, 42, 10], [11, 12, 13], 12, 0),
           build_rag_example([50, 10], [21, 22], 12, 0)]
    batch = {key: torch.stack([r[key] for r in rag]) for key in rag[0]}
    batch["patch_grid"] = torch.randn(2, 9, 32)
    loss = mod._step(batch, "train")
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in mod.prefix_mapper.parameters())


@pytest.mark.parametrize("pattern", [["mamba3", "mlstm"], ["attention", "attention"]])
def test_cached_beam_with_context_prompt_equals_uncached(pattern):
    """The RAG evaluation decodes cached after a context prompt; it must match the uncached path."""
    import scripts.evaluate_report_generation as erg

    mod = _tiny_rg_module(pattern).eval()
    ctx = torch.randint(1, 97, (1, 23))
    with torch.no_grad():
        prefix = mod.prefix_mapper(torch.randn(1, 9, 32))
        a = mod.decoder.beam_search_cached(ctx, prefix_embeds=prefix, beam_size=3, max_new_tokens=8)
        b = erg.beam_search_decode(mod.decoder, ctx, prefix_embeds=prefix, beam_size=3, max_new_tokens=8)
    assert torch.equal(a, b) and torch.equal(a[:, :23], ctx)


def test_eval_shards_cover_every_row_once(tmp_path):
    import json as _json
    from scripts.textenc_eval_rag import merge_shards, shard_rows

    n, shards = 11, 3
    seen = sorted(r for s in range(shards) for r in shard_rows(n, s, shards))
    assert seen == list(range(n))
    for s in range(shards):
        rows = shard_rows(n, s, shards)
        (tmp_path / f"shard_{s}.json").write_text(_json.dumps(
            {"rows": rows, "hyps": [f"h{r}" for r in rows], "refs": [f"r{r}" for r in rows]}))
    merge_shards(tmp_path, shards, n)
    assert (tmp_path / "hyps.txt").read_text().splitlines() == [f"h{r}" for r in range(n)]


def test_rag_wrapper_swaps_exactly_the_entrypoint_line(tmp_path):
    """The published wrapper is untouched; the RAG job runs a copy differing in one line."""
    import re
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    assert sh.count("python scripts/train_report_generation.py \\\n") == 1
    rag = (REPO_ROOT / "scripts" / "textenc_rag_train_h100.sh").read_text()
    m = re.search(r"sed '(s#.*#)' ", rag)
    assert m, "sed expression not found"
    r = subprocess.run(["sed", m.group(1), str(REPO_ROOT / "scripts" / "train_report_generation_h100.sh")],
                       capture_output=True, text=True, check=True)
    a, b = sh.splitlines(), r.stdout.splitlines()
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert diff == [("python scripts/train_report_generation.py \\", "python scripts/textenc_train_rag.py \\")]


def test_install_swaps_module_and_wraps_loader(tmp_path):
    import types
    import pandas as pd
    from torch.utils.data import DataLoader
    from hybrid_xmamba.training.rag_report_generation import RAGReportGenerationLightningModule
    from scripts import textenc_train_rag as ttr

    data = tmp_path / "data"
    rag = tmp_path / "rag"
    data.mkdir(); rag.mkdir()
    pd.DataFrame({"study_id": [1, 2, 3], "findings": ["f1", "f2", "f3"],
                  "impression": ["i1", "i2", "i3"]}).to_parquet(data / "train.parquet")
    pd.DataFrame({"study_id": [101, 102, 101], "findings": ["a", "b", "a"],
                  "impression": ["x", "y", "x"]}).to_parquet(data / "validate.parquet")
    torch.save({"topk_rows": torch.tensor([[0, 1], [2, 0], [1, 2]])}, rag / "rag_validate.pt")

    fake = types.SimpleNamespace(
        ReportGenerationLightningModule=object,
        prepare_report_gen_dataloader=lambda cfg, split, tok: DataLoader(_FakeBase(), batch_size=2))
    ttr.install(fake, rag, data, k=2, per_report_tokens=20)
    assert fake.ReportGenerationLightningModule is RAGReportGenerationLightningModule
    cfg = types.SimpleNamespace(output_dir=str(tmp_path / "out"))
    batch = next(iter(fake.prepare_report_gen_dataloader(cfg, "validation", _CharTok())))
    assert batch["rag_input_ids"].shape == (2, 2 * 20 + 6)


NEW_WRAPPERS = ["textenc_rag_index_h100.sh", "textenc_rag_train_h100.sh", "textenc_rag_eval_h100.sh"]


@pytest.mark.parametrize("name", NEW_WRAPPERS)
def test_new_wrappers_refuse_without_root_or_on_published_roots(name):
    path = REPO_ROOT / "scripts" / name
    env = {k: v for k, v in os.environ.items() if k not in ("TEXTENC_ROOT", "SLURM_SUBMIT_DIR")}
    r = subprocess.run(["bash", str(path)], env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "TEXTENC_ROOT is not set" in r.stdout
    env["TEXTENC_ROOT"] = "/sc/home/u/hybrid_mamba_xlstm"
    r = subprocess.run(["bash", str(path)], env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "published checkout" in r.stdout


def test_rag_train_wrapper_pins_the_published_stage3_recipe():
    sh = (REPO_ROOT / "scripts" / "textenc_rag_train_h100.sh").read_text()
    for s in ["NUM_GPUS=4 MAX_STEPS=12000 BATCH_SIZE=32 SAVE_TOP_K=0 AUX_LAMBDA=0.0 PREFIX_K=32",
              "DECODER_LR=1e-5 HEAD_LR=3e-4 GRAD_CLIP=0.5 VIT_LR=1e-6",
              "h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt",
              "h100_stage0_150m_m3/checkpoints/last.ckpt",
              "h100_stage0_transformer_150m/checkpoints/last.ckpt",
              "MODEL_CONFIG=hybrid_150m_m3_rrg", "MODEL_CONFIG=transformer_150m_baseline_rrg",
              'MIMIC_CACHE_DIR="${ROOT_REAL}/cache/mimic_cxr_cache"',
              'exec bash "${RUN_COPY}"']:
        assert s in sh, s


def test_rag_entrypoint_composes_the_published_config(monkeypatch):
    """Regression for jobs 2680065/2680069 ("Primary config module 'configs' not found"): the
    published main must be re-decorated with the published settings and the ABSOLUTE configs
    path, and that path must compose the published Stage-3 overrides. hydra.main's own argv
    parser does not run on the laptop's Python 3.14, so the decorator is stubbed and the config
    is composed with Hydra's compose API; the cluster smoke job runs the real path."""
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import scripts.train_report_generation as trg
    from scripts import textenc_train_rag as ttr

    assert trg.main.__wrapped__.__name__ == "main"
    calls = {}

    def fake_main(**kw):
        calls.update(kw)
        def deco(task):
            def run():
                GlobalHydra.instance().clear()
                with initialize_config_dir(config_dir=kw["config_path"], version_base=kw["version_base"]):
                    cfg = compose(config_name=kw["config_name"],
                                  overrides=["model=hybrid_150m_m3_rrg", "dataset=cxr_mimic_full",
                                             "trainer=h100_multi_ddp", "seed=43"])
                return task(cfg)
            return run
        return deco

    monkeypatch.setattr(hydra, "main", fake_main)
    seen = ttr.run_published_main(lambda cfg: (list(cfg.model.layer_pattern), cfg.seed, cfg.model.tfla_impl))
    assert calls == {"version_base": None, "config_path": str(REPO_ROOT / "configs"), "config_name": "config"}
    assert pathlib.Path(calls["config_path"]).is_absolute()
    assert seen == (["mamba3"] * 4 + ["mlstm"] * 3 + ["mamba3"] * 5, 43, "exact")


def test_rag_train_wrapper_smoke_mode_is_short():
    sh = (REPO_ROOT / "scripts" / "textenc_rag_train_h100.sh").read_text()
    assert 'if [ "${SMOKE:-false}" = "true" ]; then' in sh and "export NUM_GPUS=1 MAX_STEPS=20" in sh
