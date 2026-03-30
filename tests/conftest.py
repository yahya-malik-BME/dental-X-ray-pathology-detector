"""Shared pytest fixtures for the dental CV test suite."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def synthetic_xray(tmp_path: Path) -> Path:
    """Generate a synthetic grayscale X-ray-like image (640x640, uint8)."""
    img = np.random.randint(30, 200, (640, 640), dtype=np.uint8)
    cv2.rectangle(img, (100, 200), (300, 400), 180, -1)
    cv2.ellipse(img, (400, 300), (80, 50), 0, 0, 360, 220, -1)
    path = tmp_path / "synthetic_xray.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def synthetic_label(tmp_path: Path) -> Path:
    """Generate a YOLO-format label file with 3 random boxes."""
    path = tmp_path / "synthetic_xray.txt"
    lines = []
    for _ in range(3):
        cls = random.randint(0, 4)
        cx, cy = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
        w, h = random.uniform(0.05, 0.25), random.uniform(0.05, 0.25)
        cx = min(max(cx, w / 2), 1 - w / 2)
        cy = min(max(cy, h / 2), 1 - h / 2)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path, synthetic_xray: Path, synthetic_label: Path) -> tuple[Path, Path]:
    """Create a small synthetic dataset directory with 5 images and labels."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    img = cv2.imread(str(synthetic_xray))
    for i in range(5):
        cv2.imwrite(str(img_dir / f"xray_{i:03d}.jpg"), img)
        (lbl_dir / f"xray_{i:03d}.txt").write_text(synthetic_label.read_text())

    return img_dir, lbl_dir
