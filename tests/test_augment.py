"""Tests for src/augment.py"""
from __future__ import annotations

import albumentations as A
from omegaconf import OmegaConf

from src.augment import (
    _build_single_transform,
    _default_pipeline,
    build_augmentation_pipeline,
)


class TestBuildAugmentationPipeline:
    def test_loads_from_config(self):
        pipeline = build_augmentation_pipeline("configs/augment.yaml", split="train")
        assert isinstance(pipeline, A.Compose)

    def test_val_split(self):
        pipeline = build_augmentation_pipeline("configs/augment.yaml", split="val")
        assert isinstance(pipeline, A.Compose)

    def test_missing_config_falls_back(self, tmp_path):
        pipeline = build_augmentation_pipeline(tmp_path / "nonexistent.yaml", split="train")
        assert isinstance(pipeline, A.Compose)


class TestBuildSingleTransform:
    def test_known_transform(self):
        cfg = OmegaConf.create({"name": "HorizontalFlip", "p": 0.5})
        t = _build_single_transform(cfg)
        assert isinstance(t, A.HorizontalFlip)

    def test_unknown_transform_returns_none(self):
        cfg = OmegaConf.create({"name": "NonExistentTransform"})
        t = _build_single_transform(cfg)
        assert t is None

    def test_bad_params_returns_none(self):
        cfg = OmegaConf.create({"name": "Resize", "invalid_param": 999})
        t = _build_single_transform(cfg)
        assert t is None


class TestDefaultPipeline:
    def test_train_pipeline(self):
        pipeline = _default_pipeline("train", 640, "yolo")
        assert isinstance(pipeline, A.Compose)

    def test_val_pipeline(self):
        pipeline = _default_pipeline("val", 640, "yolo")
        assert isinstance(pipeline, A.Compose)
