"""Tests for src/model.py"""
from __future__ import annotations

import pytest
import torch

from src.model import DentalClassifier, ModelConfig, build_model


class TestModelConfig:
    def test_valid_config(self):
        config = ModelConfig(
            model_type="efficientnet_b3",
            variant="efficientnet_b3",
            num_classes=5,
        )
        assert config.num_classes == 5

    def test_default_device_is_cuda(self):
        config = ModelConfig(model_type="yolov8", variant="yolov8n", num_classes=5)
        assert config.device == "cuda"


class TestDentalClassifier:
    @pytest.fixture
    def classifier(self):
        return DentalClassifier(num_classes=5, dropout=0.3, pretrained=False)

    def test_output_shape(self, classifier):
        x = torch.randn(2, 3, 300, 300)
        out = classifier(x)
        assert out.shape == (2, 5)

    def test_freeze_backbone(self, classifier):
        classifier.freeze_backbone()
        for name, param in classifier.backbone.named_parameters():
            if "classifier" not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_unfreeze_backbone(self, classifier):
        classifier.freeze_backbone()
        classifier.unfreeze_backbone()
        for param in classifier.backbone.parameters():
            assert param.requires_grad

    def test_classifier_head_always_trainable_when_frozen(self, classifier):
        classifier.freeze_backbone()
        for name, param in classifier.backbone.named_parameters():
            if "classifier" in name:
                assert param.requires_grad, f"Classifier head {name} should always be trainable"


class TestBuildModel:
    def test_build_classifier(self):
        config = ModelConfig(
            model_type="efficientnet_b3",
            variant="efficientnet_b3",
            num_classes=5,
            pretrained=False,
            device="cpu",
        )
        model = build_model(config)
        assert isinstance(model, DentalClassifier)

    def test_build_invalid_model_raises(self):
        config = ModelConfig(
            model_type="resnet999",  # type: ignore
            variant="resnet999",
            num_classes=5,
            device="cpu",
        )
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model(config)
