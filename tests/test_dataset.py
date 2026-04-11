"""Tests for src/dataset.py"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from src.dataset import DentalXRayDataset, collate_fn, get_transform


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
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        img = np.zeros((640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "empty.jpg"), img)
        (lbl_dir / "empty.txt").write_text("")

        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        item = ds[0]
        assert item["boxes"].shape[0] == 0
        assert item["labels"].shape[0] == 0

    def test_missing_label_file_handled_gracefully(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        img = np.zeros((640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "no_label.jpg"), img)

        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        item = ds[0]
        assert item["boxes"].shape[0] == 0

    def test_missing_image_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DentalXRayDataset(tmp_path / "nonexistent", tmp_path, split="val")

    def test_missing_label_dir_raises(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            DentalXRayDataset(img_dir, tmp_path / "nonexistent", split="val")


class TestCollateFn:
    def test_collate_batch(self, synthetic_dataset_dir):
        img_dir, lbl_dir = synthetic_dataset_dir
        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        batch = [ds[i] for i in range(min(3, len(ds)))]
        result = collate_fn(batch)
        assert result["image"].shape[0] == len(batch)
        assert result["boxes"].shape[0] == len(batch)
        assert result["labels"].shape[0] == len(batch)
        assert len(result["image_id"]) == len(batch)

    def test_collate_mixed_box_counts(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        for i in range(2):
            img = np.random.randint(30, 200, (640, 640), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)

        (lbl_dir / "img_0.txt").write_text("1 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1")
        (lbl_dir / "img_1.txt").write_text("")

        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        batch = [ds[0], ds[1]]
        result = collate_fn(batch)
        assert result["boxes"].shape[0] == 2
        assert result["boxes"].shape[1] >= 2


class TestLoadLabels:
    def test_malformed_line_skipped(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        img = np.zeros((640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)
        (lbl_dir / "test.txt").write_text("1 0.5 0.5 0.2\nbad line\n2 0.3 0.3 0.1 0.1")

        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        item = ds[0]
        assert item["boxes"].shape[0] == 1

    def test_out_of_range_coordinates_clamped(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        img = np.zeros((640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)
        # cx=1.5 clamps to 1.0, making w=0 (degenerate), so box is skipped
        (lbl_dir / "test.txt").write_text("1 1.5 0.5 0.2 0.2")

        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        item = ds[0]
        assert item["boxes"].shape[0] == 0

    def test_slightly_out_of_range_still_valid(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        img = np.zeros((640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)
        # cx=0.8 with w=0.5 -> clamped w=min(0.5, 1.6, 0.4)=0.4, still valid
        (lbl_dir / "test.txt").write_text("1 0.8 0.5 0.5 0.2")

        ds = DentalXRayDataset(img_dir, lbl_dir, split="val")
        item = ds[0]
        assert item["boxes"].shape[0] == 1
