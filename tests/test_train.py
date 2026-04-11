"""Tests for src/train.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.train import EarlyStopping, _build_yolo_data_yaml, _load_yolo_results, set_seed


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(99)
        b = torch.randn(5)
        assert not torch.allclose(a, b)

    def test_numpy_reproducibility(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        assert np.allclose(a, b)


class TestEarlyStopping:
    def test_no_stop_when_improving_max(self):
        es = EarlyStopping(patience=3, mode="max")
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            assert not es.step(s)

    def test_stops_after_patience_max(self):
        es = EarlyStopping(patience=3, mode="max")
        scores = [0.80, 0.81, 0.81, 0.81, 0.81]
        stopped_at = None
        for i, s in enumerate(scores):
            if es.step(s):
                stopped_at = i
                break
        assert stopped_at == 4

    def test_stops_after_patience_min(self):
        es = EarlyStopping(patience=2, mode="min")
        scores = [0.5, 0.4, 0.4, 0.4]
        stopped_at = None
        for i, s in enumerate(scores):
            if es.step(s):
                stopped_at = i
                break
        assert stopped_at == 3

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=2, mode="max")
        assert not es.step(0.5)
        assert not es.step(0.5)  # counter=1
        assert not es.step(0.6)  # improvement, counter resets
        assert es.counter == 0

    def test_min_delta(self):
        es = EarlyStopping(patience=2, mode="max", min_delta=0.1)
        assert not es.step(0.5)
        assert not es.step(0.55)  # improvement < min_delta
        assert es.counter == 1

    def test_first_step_never_stops(self):
        es = EarlyStopping(patience=1, mode="max")
        assert not es.step(0.5)

    def test_should_stop_attribute(self):
        es = EarlyStopping(patience=1, mode="max")
        es.step(0.5)
        es.step(0.5)
        assert es.should_stop


class TestBuildYoloDataYaml:
    def test_writes_valid_yaml(self, tmp_path):
        data_cfg = OmegaConf.create({
            "root": str(tmp_path),
            "num_classes": 5,
            "classes": {0: "tooth", 1: "caries", 2: "deep_caries", 3: "periapical_lesion", 4: "impacted_tooth"},
        })
        result_path = _build_yolo_data_yaml(data_cfg)
        assert Path(result_path).exists()
        import yaml

        with open(result_path) as f:
            content = yaml.safe_load(f)
        assert content["nc"] == 5
        assert len(content["names"]) == 5


class TestLoadYoloResults:
    def test_missing_results_file(self, tmp_path):
        result = _load_yolo_results(tmp_path)
        assert result["map50"] == 0.0
        assert result["map50_95"] == 0.0

    def test_loads_csv(self, tmp_path):
        df = pd.DataFrame({
            "metrics/mAP50(B)": [0.5, 0.7, 0.82],
            "metrics/mAP50-95(B)": [0.3, 0.5, 0.61],
            "metrics/precision(B)": [0.6, 0.7, 0.78],
            "metrics/recall(B)": [0.5, 0.65, 0.81],
        })
        df.to_csv(tmp_path / "results.csv", index=False)
        result = _load_yolo_results(tmp_path)
        assert abs(result["map50"] - 0.82) < 0.01
        assert abs(result["precision"] - 0.78) < 0.01
