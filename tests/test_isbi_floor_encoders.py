"""ISBI_BASELINES_PLAN.md B2-B: the swappable retrieval-floor encoder.

CPU only, no network: the Hugging Face path is exercised on tiny randomly
initialised CLIP and SigLIP models built from configs, never downloaded.
"""
import argparse
import pathlib
import re

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_default_floor_encoder_is_the_published_biomedclip_floor():
    from scripts.evaluate_report_generation import DEFAULT_FLOOR_ENCODER, FLOOR_ENCODERS

    assert DEFAULT_FLOOR_ENCODER == "biomedclip"
    spec = FLOOR_ENCODERS["biomedclip"]
    assert spec["loader"] == "open_clip"
    assert spec["id"] == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def test_registry_has_at_least_four_foundation_models_besides_biomedclip():
    from scripts.evaluate_report_generation import FLOOR_ENCODERS

    others = [k for k in FLOOR_ENCODERS if k != "biomedclip"]
    assert len(others) >= 4
    assert all(FLOOR_ENCODERS[k]["loader"] == "hf" for k in others)
    assert FLOOR_ENCODERS["medsiglip"].get("gated") is True


def test_unknown_floor_encoder_is_rejected():
    from scripts.evaluate_report_generation import build_floor_embedder

    with pytest.raises(ValueError, match="unknown floor encoder"):
        build_floor_embedder("not_a_model", "cpu")


def _tiny_vision_kwargs():
    return dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, image_size=32, patch_size=8)


def test_hf_image_features_clip_uses_projection_and_is_unit_norm():
    from transformers import CLIPConfig, CLIPModel
    from scripts.evaluate_report_generation import hf_image_features

    torch.manual_seed(0)
    cfg = CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, vocab_size=99),
        vision_config=_tiny_vision_kwargs(), projection_dim=16)
    model = CLIPModel(cfg).eval()
    pv = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        feats = hf_image_features(model, pv)
        expected = model.visual_projection(model.vision_model(pixel_values=pv).pooler_output)
    assert feats.shape == (3, 16)
    assert torch.allclose(feats.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(feats, torch.nn.functional.normalize(expected, dim=-1), atol=1e-6)


def test_hf_image_features_siglip_uses_pooled_output_and_is_unit_norm():
    from transformers import SiglipConfig, SiglipModel
    from scripts.evaluate_report_generation import hf_image_features

    torch.manual_seed(0)
    cfg = SiglipConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, vocab_size=99),
        vision_config=_tiny_vision_kwargs())
    model = SiglipModel(cfg).eval()
    assert getattr(model, "visual_projection", None) is None
    pv = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        feats = hf_image_features(model, pv)
    assert feats.shape == (2, 32)
    assert torch.allclose(feats.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_cli_exposes_floor_encoder_with_biomedclip_default():
    src = (REPO_ROOT / "scripts" / "evaluate_report_generation.py").read_text()
    assert re.search(r'"--floor-encoder", type=str, default=DEFAULT_FLOOR_ENCODER', src)
    assert "choices=sorted(FLOOR_ENCODERS)" in src


def test_wrapper_passes_encoder_and_never_echoes_the_token():
    sh = (REPO_ROOT / "scripts" / "retrieval_baseline_h100.sh").read_text()
    assert 'FLOOR_ENCODER="${FLOOR_ENCODER:-biomedclip}"' in sh
    assert '--floor-encoder "${FLOOR_ENCODER}"' in sh
    for line in sh.splitlines():
        if "echo" in line:
            assert "${HF_TOKEN}" not in line and "$HF_TOKEN" not in line, line


def test_floor_cost_script_runs_as_a_file_from_the_repo_root():
    """Regression for job 2601789: `python scripts/isbi_floor_cost.py` put only
    scripts/ on sys.path, so `from scripts.evaluate_report_generation import`
    failed for every encoder. --help imports the module without a GPU."""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "scripts/isbi_floor_cost.py", "--help"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    src = (REPO_ROOT / "scripts" / "isbi_floor_cost.py").read_text()
    assert "sys.path.insert(0, str(PROJECT_ROOT))" in src
