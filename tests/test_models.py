"""Unit tests for model implementations."""

import pytest
import torch

from hybrid_xmamba.models.configuration_hybrid import HybridConfig
from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
from hybrid_xmamba.models.vision_hybrid import VisionHybridModel


class TestHybridConfig:
    """Tests for model configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = HybridConfig()
        
        assert config.dim == 768
        assert config.num_layers == 12
        assert len(config.layer_pattern) > 0
    
    def test_auto_compute_heads(self):
        """Test automatic computation of num_heads."""
        config = HybridConfig(dim=1024, head_dim=64, num_heads=None)
        
        assert config.num_heads == 16  # 1024 / 64
    
    def test_to_dict_from_dict(self):
        """Test serialization."""
        config1 = HybridConfig(dim=1024, num_layers=24)
        config_dict = config1.to_dict()
        config2 = HybridConfig.from_dict(config_dict)
        
        assert config2.dim == 1024
        assert config2.num_layers == 24


class TestHybridLanguageModel:
    """Tests for hybrid language model."""
    
    def test_model_creation(self):
        """Test model instantiation."""
        config = HybridConfig(
            vocab_size=1000,
            dim=256,
            num_layers=4,
            layer_pattern=["mamba", "mlstm"],
        )
        
        model = HybridLanguageModel(config)
        
        assert model.config.dim == 256
        assert len(model.layers) == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = HybridConfig(
            vocab_size=1000,
            dim=256,
            num_layers=4,
            layer_pattern=["mamba"],
        )
        
        model = HybridLanguageModel(config)
        input_ids = torch.randint(0, 1000, (2, 128))
        
        outputs = model(input_ids, return_dict=True)
        
        assert outputs.logits.shape == (2, 128, 1000)
        assert not torch.isnan(outputs.logits).any()
    
    def test_with_labels(self):
        """Test forward pass with labels (training mode)."""
        config = HybridConfig(
            vocab_size=1000,
            dim=256,
            num_layers=4,
        )
        
        model = HybridLanguageModel(config)
        input_ids = torch.randint(0, 1000, (2, 128))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels, return_dict=True)
        
        assert outputs.loss is not None
        assert outputs.loss.item() > 0
    
    def test_generation(self):
        """Test text generation."""
        config = HybridConfig(
            vocab_size=1000,
            dim=256,
            num_layers=2,
        )
        
        model = HybridLanguageModel(config)
        input_ids = torch.randint(0, 1000, (1, 10))
        
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
        )
        
        assert generated.shape[1] == 30  # 10 + 20
    
    def test_get_layer_types(self):
        """Test layer type retrieval."""
        config = HybridConfig(
            vocab_size=1000,
            dim=256,
            num_layers=6,
            layer_pattern=["mamba", "mlstm", "slstm"],
        )
        
        model = HybridLanguageModel(config)
        layer_types = model.get_layer_types()
        
        assert len(layer_types) == 6
        assert layer_types == ["mamba", "mlstm", "slstm"] * 2


class TestVisionHybridModel:
    """Tests for vision model."""
    
    def test_model_creation(self):
        """Test vision model instantiation."""
        config = HybridConfig(
            dim=256,
            num_layers=4,
            layer_pattern=["mamba"],
        )
        
        model = VisionHybridModel(
            config,
            img_size=224,
            patch_size=16,
            num_classes=10,
        )
        
        assert model.num_classes == 10
    
    def test_forward_pass(self):
        """Test forward pass with images."""
        config = HybridConfig(
            dim=256,
            num_layers=4,
        )
        
        model = VisionHybridModel(
            config,
            img_size=224,
            patch_size=16,
            num_classes=10,
        )
        
        images = torch.randn(2, 3, 224, 224)
        logits = model(images)
        
        assert logits.shape == (2, 10)
        assert not torch.isnan(logits).any()
    
    def test_feature_extraction(self):
        """Test feature extraction (no classification head)."""
        config = HybridConfig(
            dim=256,
            num_layers=4,
        )
        
        model = VisionHybridModel(
            config,
            img_size=224,
            patch_size=16,
            num_classes=0,  # Feature extraction mode
        )
        
        images = torch.randn(2, 3, 224, 224)
        features = model.forward_features(images)
        
        assert features.shape[0] == 2
        assert features.shape[-1] == 256
