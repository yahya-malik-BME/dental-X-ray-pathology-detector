from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from src.model import DentalClassifier, DentalDetectionModel

logger = logging.getLogger(__name__)

# Color map for bounding box visualization per class
CLASS_COLORS: dict[int, str] = {
    0: "#4FC3F7",  # tooth — light blue
    1: "#FFB74D",  # caries — amber
    2: "#EF5350",  # deep_caries — red
    3: "#AB47BC",  # periapical_lesion — purple
    4: "#66BB6A",  # impacted_tooth — green
}


@dataclass
class DetectionMetrics:
    """Stores all detection evaluation results."""

    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    per_class_ap: dict[str, float] = field(default_factory=dict)
    num_images: int = 0
    num_instances: int = 0

    def __str__(self) -> str:
        lines = [
            f"{'─' * 50}",
            "  Detection Evaluation Results",
            f"{'─' * 50}",
            f"  mAP@0.50:      {self.map50:.4f}",
            f"  mAP@0.50:0.95: {self.map50_95:.4f}",
            f"  Precision:     {self.precision:.4f}",
            f"  Recall:        {self.recall:.4f}",
            f"  F1 Score:      {self.f1:.4f}",
            f"{'─' * 50}",
            "  Per-class AP@0.50:",
        ]
        for cls_name, ap in self.per_class_ap.items():
            lines.append(f"    {cls_name:<25} {ap:.4f}")
        lines.append(f"{'─' * 50}")
        return "\n".join(lines)


def evaluate_detection(
    model: DentalDetectionModel,
    data_yaml: str,
    conf: float = 0.001,
    iou: float = 0.6,
    split: str = "test",
) -> DetectionMetrics:
    """
    Evaluate YOLOv8 model using Ultralytics built-in validation.

    Uses the Ultralytics .val() method which computes full COCO metrics.
    """
    logger.info(f"Evaluating detection model on {split} split...")

    results = model.model.val(
        data=data_yaml,
        conf=conf,
        iou=iou,
        split=split,
        verbose=False,
        plots=False,
    )

    class_names = list(DentalDetectionModel.CLASS_NAMES.values())
    per_class_ap = {}
    if hasattr(results, "ap_class_index") and results.ap_class_index is not None:
        for idx, class_idx in enumerate(results.ap_class_index):
            cls_name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
            per_class_ap[cls_name] = float(results.ap50[idx])

    metrics = DetectionMetrics(
        map50=float(results.box.map50),
        map50_95=float(results.box.map),
        precision=float(results.box.mp),
        recall=float(results.box.mr),
        f1=float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-9)),
        per_class_ap=per_class_ap,
    )

    logger.info(f"\n{metrics}")
    return metrics


def evaluate_classifier(
    model: DentalClassifier,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    output_dir: str | Path | None = None,
) -> dict:
    """
    Evaluate classification model.
    Computes accuracy, per-class precision/recall/F1, and saves confusion matrix.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"][:, 0].tolist()
            logits = model(images)
            preds = logits.argmax(dim=1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Filter out padding labels (-1)
    valid = [(p, lab) for p, lab in zip(all_preds, all_labels) if lab != -1]
    preds_clean = [v[0] for v in valid]
    labels_clean = [v[1] for v in valid]

    accuracy = sum(p == lab for p, lab in zip(preds_clean, labels_clean)) / max(len(labels_clean), 1)
    report = classification_report(
        labels_clean, preds_clean, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels_clean, preds_clean)

    logger.info(f"Classification Accuracy: {accuracy:.4f}")
    logger.info(
        "\n" + classification_report(labels_clean, preds_clean, target_names=class_names, zero_division=0)
    )

    if output_dir:
        _save_confusion_matrix(cm, class_names, Path(output_dir))

    return {
        "accuracy": accuracy,
        "per_class_report": report,
        "confusion_matrix": cm,
    }


def visualize_predictions(
    image_path: str | Path,
    detections: list[dict],
    output_path: str | Path | None = None,
    show: bool = False,
    conf_threshold: float = 0.25,
) -> None:
    """Draw bounding boxes and labels on a dental X-ray image."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img_rgb, cmap="gray")
    ax.axis("off")

    for det in detections:
        if det["confidence"] < conf_threshold:
            continue

        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_id"], "#FFFFFF")
        label = f"{det['class_name']} {det['confidence']:.0%}"

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            label,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor="none"),
            verticalalignment="bottom",
        )

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Prediction visualization saved to {output_path}")
    if show:
        plt.show()
    plt.close()


def compute_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute precision, recall, and F1 from raw predictions and ground truth.

    A prediction is TP if IoU >= threshold and class matches.
    FP: unmatched predictions. FN: unmatched ground truth boxes.
    """
    if not ground_truth and not predictions:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truth)}

    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predictions), "fn": 0}

    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    # Sort predictions by confidence (descending)
    sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue

            iou = _compute_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-9)


def _save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Save confusion matrix as a PNG to output_dir/confusion_matrix.png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Pathology Classification — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
