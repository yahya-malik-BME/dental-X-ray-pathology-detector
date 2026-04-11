"""Dental X-ray pathology detection package."""

__version__ = "0.1.0"

from src.dataset import DentalXRayDataset, create_dataloaders
from src.evaluate import DetectionMetrics, evaluate_detection, visualize_predictions
from src.model import (
    DentalClassifier,
    DentalDetectionModel,
    ModelConfig,
    build_model,
)
from src.predict import InferenceConfig, Predictor

__all__ = [
    "DentalXRayDataset",
    "create_dataloaders",
    "DentalDetectionModel",
    "DentalClassifier",
    "build_model",
    "ModelConfig",
    "Predictor",
    "InferenceConfig",
    "evaluate_detection",
    "visualize_predictions",
    "DetectionMetrics",
]
