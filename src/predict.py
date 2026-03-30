from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.model import DentalDetectionModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    weights_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "cpu"
    batch_size: int = 8
    save_visualizations: bool = True
    output_dir: str = "outputs/predictions/"


class Predictor:
    """
    Inference engine for dental X-ray pathology detection.

    Handles single image prediction, batch prediction over a directory,
    CLAHE preprocessing, and result serialization to JSON.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.model = self._load_model()
        self._clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

    def _load_model(self) -> DentalDetectionModel:
        """Load model from weights path."""
        if not Path(self.config.weights_path).exists():
            raise FileNotFoundError(
                f"Weights not found: {self.config.weights_path}\n"
                f"Train the model first with: python -m src.train"
            )
        model_config = ModelConfig(
            model_type="yolov8",
            variant="yolov8m",
            num_classes=5,
            pretrained=False,
            pretrained_weights=self.config.weights_path,
            device=self.config.device,
        )
        return DentalDetectionModel(model_config)

    def preprocess(self, image_path: str | Path) -> np.ndarray:
        """
        Load and preprocess a dental X-ray image.

        Steps: load -> grayscale -> CLAHE -> 3-channel RGB.
        Returns uint8 numpy array of shape (H, W, 3).
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return rgb

    def predict_image(
        self,
        image_path: str | Path,
        save_visualization: bool | None = None,
    ) -> dict:
        """
        Run inference on a single dental X-ray image.

        Returns dict with image_path, image_size, detections, num_detections,
        and pathology_summary.
        """
        preprocessed = self.preprocess(image_path)
        detections = self.model.predict(
            preprocessed,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
        )

        # Filter to pathology classes only (exclude class 0: "tooth")
        pathologies = [d for d in detections if d["class_id"] != 0]

        summary: dict[str, int] = {
            "caries": 0,
            "deep_caries": 0,
            "periapical_lesion": 0,
            "impacted_tooth": 0,
        }
        for det in pathologies:
            name = det["class_name"]
            if name in summary:
                summary[name] += 1

        result = {
            "image_path": str(image_path),
            "image_size": list(preprocessed.shape[:2]),
            "detections": detections,
            "num_detections": len(pathologies),
            "pathology_summary": summary,
        }

        should_save = save_visualization if save_visualization is not None else self.config.save_visualizations
        if should_save:
            from src.evaluate import visualize_predictions

            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            visualize_predictions(
                image_path=image_path,
                detections=detections,
                output_path=out_dir / f"{stem}_predicted.jpg",
            )

        return result

    def predict_directory(
        self,
        image_dir: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> Iterator[dict]:
        """Run inference on all images in a directory."""
        image_dir = Path(image_dir)
        image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]

        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return

        logger.info(f"Running inference on {len(image_paths)} images in {image_dir}")

        for i, path in enumerate(image_paths, 1):
            try:
                result = self.predict_image(path)
                logger.info(
                    f"[{i}/{len(image_paths)}] {path.name}: "
                    f"{result['num_detections']} pathologies detected"
                )
                yield result
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                continue

    def save_results_json(self, results: list[dict], output_path: str | Path) -> None:
        """Serialize prediction results to JSON."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
