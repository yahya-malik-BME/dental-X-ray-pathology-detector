"""Tests for src/dataset.py"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.dataset import DentalXRayDataset, create_dataloaders, get_transform


class TestGetTransform:
    def test_train_transform_returns_compose(self):
        t = get_transform("train", image_size=640)
        assert t is not None

    def test_val_transform_returns_compose(self):
        t = get_transform("val", image_size=640)
        assert t is not None

    def test_unknown_split_defaults_to_val(self):
        t = get_transform("test", image_size=640)
        assert t is not None


class TestDentalXRayDataset:
    def test_dataset_length(self, synthetic_dataset_dir):
        img_dir, lbl_dir = synthetic_dataset_dir
        ds = DentalXRayDataset(img_dir, lbl_dir, split="train")
        assert len(ds) == 5

    def test_getitem_keys(self, synthetic_dataset_dir):
        img_dir, lbl_dir = synthetic_dataset_dir
        ds = DentalXRayDataset(img_dir, lbl_dir, split="train")
        item = ds[0]
        assert "image" in item
        assert "boxes" in item
        assert "labels" in item
        assert "image_id" in item
        assert "orig_size" in item

    def test_image_tensor_shape(self, synthetic_dataset_dir):
        img_dir, lbl_dir = synthetic_dataset_dir
        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        item = ds[0]
        assert item["image"].shape[0] == 3
        assert item["image"].dtype == torch.float32

    def test_boxes_tensor_dtype(self, synthetic_dataset_dir):
        img_dir, lbl_dir = synthetic_dataset_dir
        ds = DentalXRayDataset(img_dir, lbl_dir, split="train")
        item = ds[0]
        assert item["boxes"].dtype == torch.float32
        assert item["labels"].dtype == torch.int64

    def test_empty_label_file(self, tmp_path):
        """Dataset should not crash on images with no annotations."""
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        import cv2
        import numpy as np

        img = np.zeros((640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "empty.jpg"), img)
        (lbl_dir / "empty.txt").write_text("")

        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        item = ds[0]
        assert item["boxes"].shape[0] == 0
        assert item["labels"].shape[0] == 0

    def test_missing_label_file_handled_gracefully(self, tmp_path):
        """Images without label files should return empty annotations."""
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        import cv2
        import numpy as np

        img = np.zeros((640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "no_label.jpg"), img)

        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        item = ds[0]
        assert item["boxes"].shape[0] == 0
