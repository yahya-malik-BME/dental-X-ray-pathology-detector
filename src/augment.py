"""Albumentations pipeline builder — loads augmentation config from YAML."""
from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Registry mapping augmentation names to Albumentations classes
_AUG_REGISTRY: dict[str, type] = {
    "HorizontalFlip": A.HorizontalFlip,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "RandomGamma": A.RandomGamma,
    "GaussNoise": A.GaussNoise,
    "Rotate": A.Rotate,
    "CLAHE": A.CLAHE,
    "Blur": A.Blur,
    "ElasticTransform": A.ElasticTransform,
    "Resize": A.Resize,
    "Normalize": A.Normalize,
    "ToTensorV2": ToTensorV2,
}


def build_augmentation_pipeline(
    config_path: str | Path = "configs/augment.yaml",
    split: str = "train",
    image_size: int = 640,
    bbox_format: str = "yolo",
) -> A.Compose:
    """
    Build an Albumentations Compose pipeline from a YAML config.

    Args:
        config_path: path to augment.yaml
        split: "train" or "val"/"test"
        image_size: target image size (used for Resize)
        bbox_format: bounding box format for BboxParams

    Returns:
        A.Compose pipeline with BboxParams configured for detection
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Augmentation config not found: {config_path}. Using defaults.")
        return _default_pipeline(split, image_size, bbox_format)

    cfg = OmegaConf.load(config_path)
    aug_cfg = cfg.augmentation

    transforms: list[A.BasicTransform] = []

    # Add split-specific transforms
    split_key = "train" if split == "train" else "val"
    if split_key in aug_cfg:
        for aug_def in aug_cfg[split_key]:
            t = _build_single_transform(aug_def)
            if t is not None:
                transforms.append(t)

    # Add "always" transforms
    if "always" in aug_cfg:
        for aug_def in aug_cfg["always"]:
            t = _build_single_transform(aug_def)
            if t is not None:
                transforms.append(t)

    bbox_params = A.BboxParams(
        format=bbox_format,
        label_fields=["class_labels"],
        min_visibility=0.3,
    )

    logger.info(f"Built {split} augmentation pipeline with {len(transforms)} transforms")
    return A.Compose(transforms, bbox_params=bbox_params)


def _build_single_transform(aug_def: DictConfig) -> A.BasicTransform | None:
    """Instantiate a single Albumentations transform from a config dict."""
    name = aug_def.get("name")
    if name not in _AUG_REGISTRY:
        logger.warning(f"Unknown augmentation: {name}, skipping")
        return None

    cls = _AUG_REGISTRY[name]
    params = {k: v for k, v in aug_def.items() if k != "name"}

    # Convert OmegaConf lists to Python lists
    params = OmegaConf.to_container(OmegaConf.create(params), resolve=True)

    try:
        return cls(**params)
    except TypeError as e:
        logger.warning(f"Failed to build {name} with params {params}: {e}")
        return None


def _default_pipeline(
    split: str,
    image_size: int,
    bbox_format: str,
) -> A.Compose:
    """Fallback pipeline when no config file is available."""
    bbox_params = A.BboxParams(
        format=bbox_format,
        label_fields=["class_labels"],
        min_visibility=0.3,
    )

    if split == "train":
        return A.Compose(
            [
                A.CLAHE(clip_limit=4.0, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
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
