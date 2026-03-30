from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.model import DentalClassifier, DentalDetectionModel, ModelConfig, build_model

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience: number of epochs with no improvement before stopping
        min_delta: minimum change to qualify as improvement
        mode: "min" for loss, "max" for mAP/accuracy
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, score: float) -> bool:
        """Call after each epoch. Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("EarlyStopping triggered — stopping training")

        return self.should_stop


def train_yolo(
    model: DentalDetectionModel,
    data_cfg: DictConfig,
    train_cfg: DictConfig,
) -> dict:
    """
    Train YOLOv8 model using Ultralytics built-in trainer.

    Returns:
        dict with keys: map50, map50_95, precision, recall
    """
    yolo_data_yaml = _build_yolo_data_yaml(data_cfg)

    if train_cfg.wandb.enabled:
        wandb.init(
            project=train_cfg.wandb.project,
            entity=train_cfg.wandb.entity,
            name=train_cfg.experiment_name,
            config=OmegaConf.to_container(train_cfg, resolve=True),
        )

    logger.info(f"Starting YOLOv8 training: {train_cfg.experiment_name}")
    start_time = time.time()

    model.train(
        data_yaml=yolo_data_yaml,
        output_dir=train_cfg.output_dir,
        train_config=train_cfg,
    )

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 3600:.2f}h")

    results = _load_yolo_results(Path(train_cfg.output_dir) / train_cfg.experiment_name)

    if train_cfg.wandb.enabled:
        wandb.log(results)
        wandb.finish()

    return results


def train_classifier(
    model: DentalClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: DictConfig,
    device: torch.device,
) -> dict:
    """
    Custom training loop for EfficientNet classifier.

    Implements backbone freeze/unfreeze schedule, AdamW with cosine LR,
    early stopping, W&B logging, and best model checkpointing.

    Returns:
        dict with keys: best_val_acc, best_epoch, final_train_loss
    """
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.epochs,
        eta_min=train_cfg.scheduler.min_lr,
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=train_cfg.patience, mode="max")

    freeze_epochs = getattr(train_cfg, "freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        model.freeze_backbone()

    best_val_acc = 0.0
    best_epoch = 0
    train_loss = 0.0

    if train_cfg.wandb.enabled:
        wandb.init(
            project=train_cfg.wandb.project,
            name=train_cfg.experiment_name + "-classifier",
            config=OmegaConf.to_container(train_cfg, resolve=True),
        )

    for epoch in range(1, train_cfg.epochs + 1):
        # Unfreeze backbone after N epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            logger.info(f"Epoch {epoch}: unfreezing backbone for full fine-tuning")
            model.unfreeze_backbone()
            optimizer = AdamW(
                model.parameters(),
                lr=train_cfg.optimizer.lr * 0.1,
                weight_decay=train_cfg.optimizer.weight_decay,
            )

        # Training step
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["labels"][:, 0].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()
            train_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # Validation step
        val_loss, val_acc = _validate_classifier(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d}/{train_cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if train_cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            weights_dir = Path(train_cfg.weights_dir)
            weights_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = weights_dir / f"{train_cfg.experiment_name}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"New best model saved: val_acc={best_val_acc:.4f}")

        if early_stopping.step(val_acc):
            break

    if train_cfg.wandb.enabled:
        wandb.finish()

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_loss": train_loss,
    }


def _validate_classifier(
    model: DentalClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation loop. Returns (val_loss, val_accuracy)."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"][:, 0].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return val_loss / max(len(loader), 1), correct / max(total, 1)


def _build_yolo_data_yaml(data_cfg: DictConfig) -> str:
    """Write a YOLO-format data.yaml file from our project config."""
    import yaml

    yolo_cfg = {
        "path": str(Path(data_cfg.root).absolute()),
        "train": "data/processed/train",
        "val": "data/processed/val",
        "test": "data/processed/test",
        "nc": data_cfg.num_classes,
        "names": list(data_cfg.classes.values()),
    }

    out_path = Path("configs/yolo_data.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)

    logger.info(f"YOLO data config written to {out_path}")
    return str(out_path)


def _load_yolo_results(results_dir: Path) -> dict:
    """Load best metrics from Ultralytics results.csv."""
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        logger.warning(f"Results file not found: {results_csv}")
        return {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}

    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    best_row = df.iloc[-1]
    return {
        "map50": float(best_row.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(best_row.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(best_row.get("metrics/precision(B)", 0.0)),
        "recall": float(best_row.get("metrics/recall(B)", 0.0)),
    }


def main() -> None:
    """Main training entrypoint."""
    import hydra

    @hydra.main(config_path="../configs", config_name="train", version_base="1.3")
    def _main(cfg: DictConfig) -> None:
        set_seed(cfg.training.seed)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

        model_cfg = ModelConfig(
            model_type=cfg.model.detection.name,
            variant=cfg.model.detection.variant,
            num_classes=cfg.model.detection.num_classes,
            pretrained=cfg.model.detection.pretrained,
            device=cfg.model.device,
        )

        model = build_model(model_cfg)
        results = train_yolo(model, cfg.dataset, cfg.training)
        logger.info(f"Final results: {results}")

    _main()


if __name__ == "__main__":
    main()
