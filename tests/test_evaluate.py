"""Tests for src/evaluate.py"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluate import DetectionMetrics, compute_metrics


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
