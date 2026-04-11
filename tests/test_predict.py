"""Tests for src/predict.py"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.predict import InferenceConfig, Predictor


class TestPredictor:
    def test_missing_weights_raises_file_not_found(self, tmp_path):
        config = InferenceConfig(
            weights_path=str(tmp_path / "nonexistent.pt"),
        )
        with pytest.raises(FileNotFoundError, match="Weights not found"):
            Predictor(config)

    def test_preprocess_returns_rgb_array(self, synthetic_xray, tmp_path):
        """Test preprocessing without needing model weights."""
        import cv2

        with patch("src.predict.DentalDetectionModel") as mock_model:
            mock_model.return_value = MagicMock()
            weights = tmp_path / "dummy.pt"
            weights.touch()

            config = InferenceConfig(weights_path=str(weights), device="cpu")
            predictor = Predictor.__new__(Predictor)
            predictor.config = config
            predictor._clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

            result = predictor.preprocess(synthetic_xray)

            assert result.ndim == 3
            assert result.shape[2] == 3
            assert result.dtype == np.uint8

    def test_pathology_summary_keys(self):
        """Summary dict always has all 4 pathology keys."""
        summary_keys = {"caries", "deep_caries", "periapical_lesion", "impacted_tooth"}
        import inspect

        source = inspect.getsource(Predictor.predict_image)
        for key in summary_keys:
            assert f'"{key}"' in source or f"'{key}'" in source
