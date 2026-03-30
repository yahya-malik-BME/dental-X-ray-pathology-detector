from __future__ import annotations

import logging
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pydantic import BaseModel, field_validator
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Validates dataset configuration loaded from configs/data.yaml."""

    root: str
    image_size: int = 640
    num_classes: int = 5
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    classes: dict[int, str]

    @field_validator("root")
    @classmethod
    def root_must_exist(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"Data root does not exist: {v}")
        return v


class DentalXRayDataset(Dataset):
    """PyTorch Dataset for dental X-ray images with YOLO-format annotations."""

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        transform: A.Compose | None = None,
        image_size: int = 640,
        num_classes: int = 5,
        split: str = "train",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.split = split

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}\n"
                f"Download the dataset first or check your data.yaml paths."
            )
        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir}\n"
                f"Run annotation conversion first or check your data.yaml paths."
            )

        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )

        if not self.image_paths:
            logger.warning(f"No images found in {self.image_dir}")

        self.transform = transform if transform is not None else get_transform(split, image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image_id = image_path.stem

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Corrupt image, skipping: {image_path}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        orig_h, orig_w = img.shape[:2]

        # Convert to grayscale then back to 3-channel RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Load YOLO annotations
        label_path = self.label_dir / f"{image_id}.txt"
        boxes_yolo, class_labels = self._load_labels(label_path)

        # Apply transforms with bounding box support
        if self.transform is not None and len(boxes_yolo) > 0:
            transformed = self.transform(
                image=img_rgb,
                bboxes=boxes_yolo,
                class_labels=class_labels,
            )
            img_tensor = transformed["image"]
            boxes_yolo = transformed["bboxes"]
            class_labels = transformed["class_labels"]
        elif self.transform is not None:
            transformed = self.transform(
                image=img_rgb,
                bboxes=[],
                class_labels=[],
            )
            img_tensor = transformed["image"]
            boxes_yolo = []
            class_labels = []
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # Convert YOLO cxcywh normalized -> xyxy absolute pixel coords
        if len(boxes_yolo) > 0:
            boxes_tensor = torch.tensor(boxes_yolo, dtype=torch.float32)
            boxes_xyxy = box_convert(boxes_tensor, in_fmt="cxcywh", out_fmt="xyxy")
            # Scale to image size
            boxes_xyxy[:, [0, 2]] *= self.image_size
            boxes_xyxy[:, [1, 3]] *= self.image_size
            labels_tensor = torch.tensor(class_labels, dtype=torch.int64)
        else:
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": img_tensor,
            "boxes": boxes_xyxy,
            "labels": labels_tensor,
            "image_id": image_id,
            "orig_size": (orig_h, orig_w),
        }

    def _load_labels(self, label_path: Path) -> tuple[list[list[float]], list[int]]:
        """Load YOLO-format labels from a txt file."""
        boxes: list[list[float]] = []
        class_labels: list[int] = []

        if not label_path.exists():
            return boxes, class_labels

        text = label_path.read_text().strip()
        if not text:
            return boxes, class_labels

        for line_num, line in enumerate(text.split("\n"), 1):
            parts = line.strip().split()
            if len(parts) != 5:
                logger.warning(f"Malformed line {line_num} in {label_path}: {line}")
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                logger.warning(f"Invalid values at line {line_num} in {label_path}: {line}")
                continue

            # Clamp coordinates to [0, 1]
            for val_name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < 0 or val > 1:
                    logger.warning(
                        f"Out-of-range {val_name}={val} at line {line_num} in {label_path}, clamping"
                    )
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            boxes.append([cx, cy, w, h])
            class_labels.append(cls_id)

        return boxes, class_labels


def get_transform(split: str, image_size: int = 640) -> A.Compose:
    """Build Albumentations pipeline for the given split."""
    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    )

    if split == "train":
        return A.Compose(
            [
                A.CLAHE(clip_limit=4.0, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
                A.Rotate(limit=10, border_mode=0, p=0.4),
                A.Blur(blur_limit=3, p=0.2),
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for variable-size bounding box tensors."""
    images = torch.stack([item["image"] for item in batch])
    image_ids = [item["image_id"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]

    max_boxes = max(item["boxes"].shape[0] for item in batch) if batch else 0
    max_boxes = max(max_boxes, 1)  # at least 1 to avoid zero-dim tensors

    padded_boxes = torch.zeros(len(batch), max_boxes, 4)
    padded_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.int64)

    for i, item in enumerate(batch):
        n = item["boxes"].shape[0]
        if n > 0:
            padded_boxes[i, :n] = item["boxes"]
            padded_labels[i, :n] = item["labels"]

    return {
        "image": images,
        "boxes": padded_boxes,
        "labels": padded_labels,
        "image_id": image_ids,
        "orig_size": orig_sizes,
    }


def create_dataloaders(
    data_root: str | Path,
    image_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Splits images in data_root/processed/ into train/val/test by filename.
    Saves split filenames to data_root/splits/ for reproducibility.

    Returns:
        train_loader, val_loader, test_loader
    """
    data_root = Path(data_root)
    image_dir = data_root / "processed"
    label_dir = data_root / "annotations"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n" f"Place processed images in {image_dir}"
        )

    # Discover all images
    all_images = sorted(
        [p.name for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    if not all_images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(all_images)

    n = len(all_images)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_files = all_images[:n_test]
    val_files = all_images[n_test : n_test + n_val]
    train_files = all_images[n_test + n_val :]

    # Save split lists
    splits_dir = data_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        (splits_dir / f"{name}.txt").write_text("\n".join(files))
    logger.info(
        f"Split sizes — train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}"
    )

    # Create subset datasets using symlinks or by passing file lists
    # We'll create temp directories with symlinks for each split
    for split_name, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        split_img_dir = data_root / "processed" / split_name
        split_lbl_dir = data_root / "annotations" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for fname in split_files:
            src_img = image_dir / fname
            dst_img = split_img_dir / fname
            if not dst_img.exists() and src_img.exists():
                dst_img.symlink_to(src_img.resolve())

            label_name = Path(fname).stem + ".txt"
            src_lbl = label_dir / label_name
            dst_lbl = split_lbl_dir / label_name
            if not dst_lbl.exists() and src_lbl.exists():
                dst_lbl.symlink_to(src_lbl.resolve())

    # Build datasets
    train_ds = DentalXRayDataset(
        image_dir=data_root / "processed" / "train",
        label_dir=data_root / "annotations" / "train",
        image_size=image_size,
        split="train",
    )
    val_ds = DentalXRayDataset(
        image_dir=data_root / "processed" / "val",
        label_dir=data_root / "annotations" / "val",
        image_size=image_size,
        split="val",
    )
    test_ds = DentalXRayDataset(
        image_dir=data_root / "processed" / "test",
        label_dir=data_root / "annotations" / "test",
        image_size=image_size,
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


class DENTEXDownloader:
    """
    Downloads DENTEX dataset and converts COCO JSON annotations to YOLO .txt format.

    Usage:
        downloader = DENTEXDownloader(root="data/")
        downloader.download()
        downloader.convert_to_yolo()
    """

    DENTEX_URL = "https://zenodo.org/record/7812323"

    def __init__(self, root: str = "data/") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:
        """Download DENTEX dataset files."""
        logger.info(f"Download DENTEX dataset from {self.DENTEX_URL}")
        logger.info(
            "Automatic download is not yet implemented.\n"
            "Please download manually from:\n"
            f"  {self.DENTEX_URL}\n"
            f"and place images in {self.root / 'raw/'}\n"
            f"and annotations in {self.root / 'annotations/'}"
        )

    def convert_to_yolo(self) -> None:
        """Convert COCO JSON bbox annotations to YOLO format txt files."""
        import json

        annotations_dir = self.root / "annotations"
        output_dir = self.root / "annotations"
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(annotations_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON annotation files found in {annotations_dir}")
            return

        for json_path in json_files:
            logger.info(f"Converting {json_path.name} to YOLO format...")
            with open(json_path) as f:
                coco = json.load(f)

            # Build image id -> dimensions lookup
            img_lookup: dict[int, dict] = {}
            for img in coco.get("images", []):
                img_lookup[img["id"]] = {
                    "width": img["width"],
                    "height": img["height"],
                    "file_name": img["file_name"],
                }

            # Group annotations by image
            img_annotations: dict[int, list] = {}
            for ann in coco.get("annotations", []):
                img_id = ann["image_id"]
                if img_id not in img_annotations:
                    img_annotations[img_id] = []
                img_annotations[img_id].append(ann)

            # Convert each image's annotations
            for img_id, img_info in img_lookup.items():
                w, h = img_info["width"], img_info["height"]
                stem = Path(img_info["file_name"]).stem
                label_path = output_dir / f"{stem}.txt"

                lines = []
                for ann in img_annotations.get(img_id, []):
                    # COCO format: [x_min, y_min, width, height] absolute
                    x_min, y_min, bw, bh = ann["bbox"]
                    # YOLO format: [cx, cy, w, h] normalized
                    cx = (x_min + bw / 2) / w
                    cy = (y_min + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    cls_id = ann["category_id"]
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                label_path.write_text("\n".join(lines))

        logger.info("COCO to YOLO conversion complete")

    def verify_integrity(self) -> bool:
        """Check that all images have corresponding label files."""
        raw_dir = self.root / "raw"
        ann_dir = self.root / "annotations"

        if not raw_dir.exists() or not ann_dir.exists():
            logger.error(
                f"Missing directories: raw={raw_dir.exists()}, annotations={ann_dir.exists()}"
            )
            return False

        images = [
            p.stem for p in raw_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        labels = [p.stem for p in ann_dir.iterdir() if p.suffix == ".txt"]

        missing = set(images) - set(labels)
        if missing:
            logger.warning(f"{len(missing)} images missing label files: {list(missing)[:5]}...")
            return False

        logger.info(f"Integrity check passed: {len(images)} images, {len(labels)} labels")
        return True
