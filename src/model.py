from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model construction."""

    model_type: Literal["yolov8", "efficientnet_b3", "resnet50"]
    variant: str
    num_classes: int
    pretrained: bool = True
    pretrained_weights: str | None = None
    dropout: float = 0.3
    freeze_backbone_epochs: int = 0
    device: str = "cuda"
    input_size: int = 640


class DentalDetectionModel:
    """
    Wrapper around YOLOv8 for dental X-ray pathology detection.

    The underlying model is self.model (ultralytics.YOLO).
    This wrapper adds config-driven initialization, standardized save/load,
    and inference post-processing specific to dental classes.
    """

    CLASS_NAMES: dict[int, str] = {
        0: "tooth",
        1: "caries",
        2: "deep_caries",
        3: "periapical_lesion",
        4: "impacted_tooth",
    }

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = self._build_model()

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string with graceful fallback: cuda -> mps -> cpu."""
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            if device != "cpu":
                logger.warning(
                    f"Requested device '{device}' not available. Falling back to CPU."
                )
            return torch.device("cpu")

    def _build_model(self) -> YOLO:
        """Build YOLOv8 model from pretrained weights or variant string."""
        if self.config.pretrained_weights and Path(self.config.pretrained_weights).exists():
            logger.info(f"Loading weights from {self.config.pretrained_weights}")
            model = YOLO(self.config.pretrained_weights)
        else:
            logger.info(f"Loading pretrained {self.config.variant} from Ultralytics hub")
            model = YOLO(f"{self.config.variant}.pt")
        return model

    def train(self, data_yaml: str, output_dir: str, train_config: DictConfig) -> None:
        """Launch YOLOv8 training via Ultralytics trainer."""
        self.model.train(
            data=data_yaml,
            epochs=train_config.epochs,
            imgsz=self.config.input_size,
            batch=train_config.batch_size,
            device=str(self.device),
            project=output_dir,
            name=train_config.experiment_name,
            patience=train_config.patience,
            save_period=train_config.save_period,
            optimizer=train_config.optimizer.name,
            lr0=train_config.optimizer.lr,
            weight_decay=train_config.optimizer.weight_decay,
            warmup_epochs=train_config.scheduler.warmup_epochs,
            cos_lr=True if train_config.scheduler.name == "cosine" else False,
            box=train_config.loss.box,
            cls=train_config.loss.cls,
            dfl=train_config.loss.dfl,
            val=True,
            plots=True,
            exist_ok=True,
        )

    def predict(
        self,
        image: torch.Tensor | str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[dict]:
        """
        Run inference on a single image or path.

        Returns:
            List of detections, each a dict with box, confidence, class_id, class_name.
        """
        results = self.model.predict(image, conf=conf, iou=iou, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "box": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": self.CLASS_NAMES.get(int(box.cls[0]), "unknown"),
                })
        return detections

    def save(self, path: str | Path) -> None:
        """Save model weights to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, weights_path: str | Path, config: ModelConfig) -> DentalDetectionModel:
        """Load a saved model from disk."""
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}\n"
                f"Train the model first with: python -m src.train"
            )
        config.pretrained_weights = str(weights_path)
        return cls(config)


class DentalClassifier(nn.Module):
    """
    EfficientNet-B3 fine-tuned for dental pathology classification.

    Used as a secondary model to classify cropped tooth regions
    detected by DentalDetectionModel.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # Replace classifier head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen — only classifier head will be trained")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — full fine-tuning enabled")


def build_model(config: ModelConfig) -> DentalDetectionModel | DentalClassifier:
    """
    Model factory. Returns the appropriate model based on config.model_type.

    Raises:
        ValueError: if model_type is not recognized
    """
    logger.info(f"Building model: {config.model_type} | device: {config.device}")

    if config.model_type == "yolov8":
        return DentalDetectionModel(config)
    elif config.model_type in ("efficientnet_b3", "resnet50"):
        model = DentalClassifier(
            num_classes=config.num_classes,
            dropout=config.dropout,
            pretrained=config.pretrained,
        )
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        return model.to(device)
    else:
        raise ValueError(
            f"Unknown model_type: '{config.model_type}'. "
            f"Expected one of: 'yolov8', 'efficientnet_b3', 'resnet50'"
        )
