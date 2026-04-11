"""Tests for src/evaluate.py"""
from __future__ import annotations

import cv2
import numpy as np

from src.evaluate import (
    DetectionMetrics,
    _compute_iou,
    compute_metrics,
    visualize_predictions,
)


class TestDetectionMetrics:
    def test_str_representation(self):
        m = DetectionMetrics(map50=0.823, map50_95=0.612, precision=0.78, recall=0.81, f1=0.794)
        s = str(m)
        assert "0.8230" in s
        assert "mAP@0.50" in s

    def test_default_values_are_zero(self):
        m = DetectionMetrics()
        assert m.map50 == 0.0
        assert m.per_class_ap == {}

    def test_per_class_ap_in_str(self):
        m = DetectionMetrics(per_class_ap={"caries": 0.75, "deep_caries": 0.68})
        s = str(m)
        assert "caries" in s
        assert "deep_caries" in s


class TestComputeMetrics:
    def test_perfect_prediction(self):
        gt = [{"box": [10, 10, 100, 100], "class_id": 1}]
        pred = [{"box": [10, 10, 100, 100], "class_id": 1, "confidence": 0.9}]
        result = compute_metrics(pred, gt, iou_threshold=0.5)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_no_predictions(self):
        gt = [{"box": [10, 10, 100, 100], "class_id": 1}]
        result = compute_metrics([], gt)
        assert result["recall"] == 0.0
        assert result["fn"] == 1

    def test_no_ground_truth(self):
        pred = [{"box": [10, 10, 100, 100], "class_id": 1, "confidence": 0.9}]
        result = compute_metrics(pred, [])
        assert result["precision"] == 0.0
        assert result["fp"] == 1

    def test_both_empty(self):
        result = compute_metrics([], [])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["tp"] == 0

    def test_class_mismatch_is_fp(self):
        gt = [{"box": [10, 10, 100, 100], "class_id": 1}]
        pred = [{"box": [10, 10, 100, 100], "class_id": 2, "confidence": 0.9}]
        result = compute_metrics(pred, gt, iou_threshold=0.5)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_low_iou_is_fp(self):
        gt = [{"box": [10, 10, 50, 50], "class_id": 1}]
        pred = [{"box": [200, 200, 300, 300], "class_id": 1, "confidence": 0.9}]
        result = compute_metrics(pred, gt, iou_threshold=0.5)
        assert result["tp"] == 0
        assert result["fp"] == 1

    def test_multiple_detections(self):
        gt = [
            {"box": [10, 10, 100, 100], "class_id": 1},
            {"box": [200, 200, 300, 300], "class_id": 2},
        ]
        pred = [
            {"box": [10, 10, 100, 100], "class_id": 1, "confidence": 0.9},
            {"box": [200, 200, 300, 300], "class_id": 2, "confidence": 0.8},
            {"box": [400, 400, 500, 500], "class_id": 1, "confidence": 0.5},
        ]
        result = compute_metrics(pred, gt, iou_threshold=0.5)
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 0


class TestComputeIou:
    def test_perfect_overlap(self):
        assert _compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_no_overlap(self):
        assert _compute_iou([0, 0, 10, 10], [100, 100, 200, 200]) == 0.0

    def test_partial_overlap(self):
        iou = _compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0.1 < iou < 0.3


class TestVisualizePredictions:
    def test_saves_annotated_image(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = np.random.randint(50, 200, (640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        detections = [
            {"box": [100, 150, 250, 300], "confidence": 0.87, "class_id": 1, "class_name": "caries"},
        ]
        out_path = tmp_path / "annotated.jpg"
        visualize_predictions(image_path=img_path, detections=detections, output_path=out_path, show=False)
        assert out_path.exists()
        assert out_path.stat().st_size > 1000

    def test_filters_low_confidence(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = np.random.randint(50, 200, (640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        detections = [
            {"box": [100, 150, 250, 300], "confidence": 0.01, "class_id": 1, "class_name": "caries"},
        ]
        out_path = tmp_path / "annotated.jpg"
        visualize_predictions(
            image_path=img_path, detections=detections, output_path=out_path, show=False, conf_threshold=0.5
        )
        assert out_path.exists()

    def test_empty_detections(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = np.random.randint(50, 200, (640, 640), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        out_path = tmp_path / "annotated.jpg"
        visualize_predictions(image_path=img_path, detections=[], output_path=out_path, show=False)
        assert out_path.exists()
