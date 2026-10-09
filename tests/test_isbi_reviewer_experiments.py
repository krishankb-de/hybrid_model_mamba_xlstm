"""ISBI_BASELINES_PLAN.md B7-B and B8-A, CPU only, no downloads.

B7-B: the attention KV-cache step must be exact, so the decode benchmark compares two correct
cached paths. B8-A: the swappable report image encoder must leave the published BiomedCLIP path
untouched and give HF towers the final-normed token grid.
"""
import json
import pathlib

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _lm(pattern, vocab=97, **kw):
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel

    torch.manual_seed(0)
    cfg = dict(vocab_size=vocab, dim=64, num_layers=len(pattern), layer_pattern=pattern,
               head_dim=16, num_heads=4, max_position_embeddings=64, tfla_impl="exact",
               mamba3_d_state=32, mamba3_head_dim=16, mamba3_chunk_size=8)
    cfg.update(kw)
    return HybridLanguageModel(HybridConfig(**cfg)).eval()


# ---------------------------------------------------------------- B7-B: attention KV cache


def test_attention_stack_now_supports_cached_decode():
    assert _lm(["attention", "attention"]).supports_cached_decode()


@pytest.mark.parametrize("with_prefix", [False, True])
def test_attention_step_logits_match_full_forward(with_prefix):
    model = _lm(["attention", "attention", "attention"])
    ids = torch.randint(0, 97, (2, 9))
    prefix = torch.randn(2, 4, 64) if with_prefix else None
    with torch.no_grad():
        hidden = model.embeddings(ids)
        if prefix is not None:
            hidden = torch.cat([prefix, hidden], dim=1)
        full = model(inputs_embeds=hidden).logits if hasattr(model(inputs_embeds=hidden), "logits") \
            else model(inputs_embeds=hidden)["logits"]
        caches = model.allocate_inference_cache(2)
        stepped = torch.stack([model.step_logits(hidden[:, t], caches)
                               for t in range(hidden.shape[1])], dim=1)
    assert torch.allclose(full, stepped, atol=1e-4, rtol=1e-4), (full - stepped).abs().max()


def test_attention_cache_grows_past_its_first_buffer():
    """The doubling buffer starts at 64 slots; decoding past it must stay exact."""
    model = _lm(["attention"], max_position_embeddings=256)
    ids = torch.randint(0, 97, (1, 150))
    with torch.no_grad():
        out = model(input_ids=ids)
        full = out.logits if hasattr(out, "logits") else out["logits"]
        caches = model.allocate_inference_cache(1)
        last = None
        for t in range(ids.shape[1]):
            last = model.step_logits(model.embeddings(ids[:, t:t + 1])[:, 0], caches)
    assert caches[0]["k"].shape[2] >= 150 and caches[0]["seen"] == 150
    assert torch.allclose(full[:, -1], last, atol=1e-4, rtol=1e-4)


def test_attention_cached_beam_search_is_token_identical_to_uncached():
    import scripts.evaluate_report_generation as erg

    model = _lm(["attention", "attention"])
    ids = torch.randint(0, 97, (1, 5))
    prefix = torch.randn(1, 3, 64)
    with torch.no_grad():
        uncached = erg.beam_search_decode(model, ids, prefix_embeds=prefix,
                                          beam_size=3, max_new_tokens=10)
        cached = model.beam_search_cached(ids, prefix_embeds=prefix,
                                          beam_size=3, max_new_tokens=10)
    assert torch.equal(uncached, cached), (uncached.tolist(), cached.tolist())


# ---------------------------------------------------------------- B8-A: report image encoder


def test_biomedclip_spec_is_exactly_the_published_transform():
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule as M

    spec = M.REPORT_IMAGE_ENCODERS["biomedclip"]
    assert spec["image_size"] == 224 and spec["patch_dim"] == 768 and spec["hf_id"] is None
    assert spec["mean"] == [0.48145466, 0.4578275, 0.40821073]
    assert spec["std"] == [0.26862954, 0.26130258, 0.27577711]


def test_report_image_transform_sizes():
    from PIL import Image
    from scripts.evaluate_report_generation import report_image_transform

    img = Image.new("RGB", (300, 250), color=(120, 120, 120))
    assert report_image_transform("biomedclip")(img).shape == (3, 224, 224)
    assert report_image_transform("xrayclip")(img).shape == (3, 512, 512)
    assert report_image_transform("medsiglip")(img).shape == (3, 448, 448)


def test_hf_patch_grid_clip_applies_post_layernorm_to_every_token():
    from transformers import CLIPVisionConfig, CLIPVisionModel
    from hybrid_xmamba.training.lightning_module import hf_vision_patch_grid

    torch.manual_seed(0)
    vm = CLIPVisionModel(CLIPVisionConfig(hidden_size=32, intermediate_size=64,
                                          num_hidden_layers=1, num_attention_heads=4,
                                          image_size=32, patch_size=8)).vision_model.eval()
    pv = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        grid = hf_vision_patch_grid(vm, pv)
        expected = vm.post_layernorm(vm(pixel_values=pv).last_hidden_state)
    assert grid.shape == (2, 1 + 16, 32)
    assert torch.allclose(grid, expected)


def test_hf_patch_grid_siglip_is_last_hidden_state():
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from hybrid_xmamba.training.lightning_module import hf_vision_patch_grid

    torch.manual_seed(0)
    vm = SiglipVisionModel(SiglipVisionConfig(hidden_size=32, intermediate_size=64,
                                              num_hidden_layers=1, num_attention_heads=4,
                                              image_size=32, patch_size=8)).vision_model.eval()
    pv = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        grid = hf_vision_patch_grid(vm, pv)
        expected = vm(pixel_values=pv).last_hidden_state
    assert grid.shape == (2, 16, 32)
    assert torch.equal(grid, expected)


def test_non_default_encoder_rejects_a_biomedclip_tower_checkpoint():
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule as M

    module = M.__new__(M)
    torch.nn.Module.__init__(module)
    module.vit_unfreeze_blocks = 0
    with pytest.raises(ValueError, match="BiomedCLIP tower"):
        module.load_image_encoder(image_encoder_checkpoint="x.ckpt", encoder_name="xrayclip")
    with pytest.raises(ValueError, match="unknown report image encoder"):
        module.load_image_encoder(encoder_name="nope")


def test_resolve_report_image_encoder_reads_run_metadata(tmp_path):
    from scripts.evaluate_report_generation import resolve_report_image_encoder

    ckpt = tmp_path / "run" / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    assert resolve_report_image_encoder(ckpt) == "biomedclip"          # no metadata
    (tmp_path / "run" / "run_metadata.json").write_text(json.dumps(
        {"resolved_config": {"model": {"prefix_k": 32}}}))
    assert resolve_report_image_encoder(ckpt) == "biomedclip"          # published runs
    (tmp_path / "run" / "run_metadata.json").write_text(json.dumps(
        {"resolved_config": {"model": {"report_image_encoder": "xrayclip"}}}))
    assert resolve_report_image_encoder(ckpt) == "xrayclip"


def test_training_wrapper_default_encoder_adds_no_arguments():
    sh = (REPO_ROOT / "scripts" / "train_report_generation_h100.sh").read_text()
    assert 'REPORT_IMAGE_ENCODER="${REPORT_IMAGE_ENCODER:-biomedclip}"' in sh
    assert "  biomedclip) ;;" in sh
    assert '"model.report_image_encoder=xrayclip"' in sh and '"dataset.image_size=512"' in sh


# ---------------------------------------------------------------- B7-C: decode benchmark helper


@pytest.mark.parametrize("pattern", [["mamba3", "mlstm"], ["attention", "attention"]])
def test_decode_bench_cache_fill_then_step_runs(pattern):
    from scripts.isbi_decode_bench import fill_cache_to

    model = _lm(pattern)
    with torch.no_grad():
        caches = model.allocate_inference_cache(6)
        fill_cache_to(caches, 40)
        logits = model.step_logits(model.embeddings(torch.randint(0, 97, (6, 1)))[:, 0], caches)
        caches = model.reorder_cache(caches, torch.tensor([2, 1, 0, 5, 4, 3]))
    assert logits.shape == (6, 97) and torch.isfinite(logits).all()
    for c in caches:
        assert c["seen"] == 41
        if "k" in c:
            assert c["k"].shape[2] >= 41


# ---------------------------------------------------------------- B9: error analysis arithmetic


def test_error_rates_count_hallucinations_and_omissions():
    from scripts.isbi_error_analysis import rates

    names = ["Pneumothorax", "Edema", "No Finding"]
    y_true = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0]]
    y_pred = [[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 1, 0]]
    r = rates(y_true, y_pred, names)
    ptx, edema = r["per_finding"]["Pneumothorax"], r["per_finding"]["Edema"]
    assert (ptx["tp"], ptx["fp"], ptx["fn"]) == (1, 1, 1)
    assert ptx["hallucination_rate"] == 0.5 and ptx["omission_rate"] == 0.5
    assert (edema["tp"], edema["fp"], edema["fn"]) == (1, 1, 1)
    assert r["mean_positive_findings_generated"] == (2 + 1 + 0 + 1) / 4     # No Finding excluded
    assert r["mean_positive_findings_reference"] == (1 + 0 + 1 + 2) / 4
    assert r["studies_with_hallucinated_acute_finding"] == 2 / 4
    assert r["studies_with_omitted_acute_finding"] == 2 / 4
