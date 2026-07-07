"""Dataset and data loading utilities with albumentations support."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
import logging
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .utils import log_print

logger = logging.getLogger("training.data")

cv2.setNumThreads(1)

class CachedGastroDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, data_root=None, im_size="regular", phase="train"):
        self.labels = labels
        self.transform = transform
        self.file_paths = file_paths
        self.phase = phase

        # Pre-cache resize when using albumentations
        if self.phase=="train":
            if im_size == "regular":
                pre_h, pre_w = 256, 256
            elif im_size == "large":
                pre_h, pre_w = 512, 512
            else:
                raise NotImplementedError(f"Image size {im_size} not implemented.")
        else:
            pre_h = pre_w = None

        self.cached_images = []

        data_root = Path(data_root) if data_root else None
        total = len(file_paths)
        log_print(
            f"CachedGastroDataset [{phase}]: caching {total} images "
            f"(resize={'256x256' if phase == 'train' else 'full resolution'})..."
        )

        for idx, fp in enumerate(file_paths):
            image_path = data_root / fp if (data_root and not os.path.isabs(fp)) else fp
            with Image.open(image_path) as img:
                image = img.convert('RGB').copy()
            image = np.array(image)

            if self.phase=="train":
                image = A.Resize(height=pre_h, width=pre_w)(image=image)['image']

            self.cached_images.append(image)

            if total <= 20 or (idx + 1) % max(1, total // 10) == 0 or idx == 0 or idx == total - 1:
                log_print(f"CachedGastroDataset [{phase}]: loaded {idx + 1}/{total} images")

        if len(self.cached_images) == 0:
            raise RuntimeError("No images were loaded successfully. Check your dataset paths and formats.")

        log_print(f"CachedGastroDataset [{phase}]: done — {len(self.cached_images)} images in RAM")


    def __getitem__(self, idx):
        image = self.cached_images[idx]
        label = self.labels[idx]

        if self.transform:
            if  isinstance(self.transform, A.Compose):
                img_np = image if isinstance(image, np.ndarray) else np.array(image)
                transformed = self.transform(image=img_np.copy())
                image = transformed['image']
            else:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                image = self.transform(image)

        return image, label, str(self.file_paths[idx])

    def __len__(self):
        return len(self.cached_images)


def get_transforms(phase='train', config=None):
    """Get albumentations transforms based on PRESET combinations.

    Controlled by config.aug_preset and meant to be called
    when it's not None via the wrapper function.

    Presets:
    - top1: jigsaw | blur_variations | color_variations | optical_distortion
    - top2: affine | texture_variations | elastic
    - top3: affine | blur_variations | texture_variations | optical_distortion
    - top4: affine | color_variations | optical_distortion
    """
    if config.im_size == "regular":
        im_size = 224
    elif config.im_size == "large":
        im_size = 448
    else:
        raise NotImplementedError(f"Image size {config.im_size} not implemented")

    presets = {
        "top1": {"jigsaw", "blur_variations", "color_variations", "optical_distortion"},
        "top2": {"affine", "texture_variations", "elastic"},
        "top3": {"affine", "blur_variations", "texture_variations", "optical_distortion"},
        "top4": {"affine", "color_variations", "optical_distortion"},
    }
    preset_name = getattr(config, "aug_preset", None)
    if preset_name not in presets:
        raise ValueError(
            f"Invalid or missing preset '{preset_name}'. "
            f"Expected one of {sorted(list(presets.keys()))} when use_augmentation_presets=True."
        )

    if phase == 'train':
        transforms_list = []

        # Base geometric transforms
        transforms_list.extend([
            # A.Resize(height=256, width=256),
            A.RandomResizedCrop(size=(im_size, im_size), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.8),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        ])

        selected = presets[preset_name]
        logger.info(f"Using augmentation preset '{preset_name}': {sorted(selected)}")

        if "jigsaw" in selected:
            transforms_list.append(
                A.RandomGridShuffle(
                    grid=(config.jigsaw_grid_size, config.jigsaw_grid_size),
                    p=config.jigsaw_prob,
                )
            )
            logger.info(f"Preset includes Jigsaw: grid={config.jigsaw_grid_size}x{config.jigsaw_grid_size}, p={config.jigsaw_prob}")

        if "affine" in selected:
            transforms_list.append(
                A.Affine(
                    scale=(config.affine_scale_min, config.affine_scale_max),
                    translate_percent=(-config.affine_translate_percent, config.affine_translate_percent),
                    rotate=(-config.affine_rotate, config.affine_rotate),
                    shear=(-config.affine_shear, config.affine_shear),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    fit_output=False,
                    p=config.affine_prob,
                )
            )
            logger.info(f"Preset includes Affine: scale=({config.affine_scale_min}, {config.affine_scale_max}), p={config.affine_prob}")

        if "elastic" in selected:
            transforms_list.append(
                A.ElasticTransform(
                    alpha=config.elastic_alpha,
                    sigma=config.elastic_sigma,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=config.elastic_prob,
                )
            )
            logger.info(f"Preset includes Elastic: alpha={config.elastic_alpha}, sigma={config.elastic_sigma}, p={config.elastic_prob}")

        if "optical_distortion" in selected:
            transforms_list.append(
                A.OpticalDistortion(distort_limit=0.1, p=config.optical_distortion_prob)
            )
            logger.info(f"Preset includes Optical Distortion: p={config.optical_distortion_prob}")

        if "color_variations" in selected:
            transforms_list.extend([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=config.color_variations_prob),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=config.color_variations_prob * 0.5),
            ])
            logger.info(f"Preset includes Color Variations: p={config.color_variations_prob}")

        if "texture_variations" in selected:
            transforms_list.extend([
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=config.texture_variations_prob),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=config.texture_variations_prob * 0.7),
            ])
            logger.info(f"Preset includes Texture Variations: p={config.texture_variations_prob}")

        if "blur_variations" in selected:
            transforms_list.extend([
                A.MotionBlur(blur_limit=7, p=config.blur_prob),
                A.GaussianBlur(blur_limit=(1, 3), p=config.blur_prob * 0.7),
            ])
            logger.info(f"Preset includes Blur Variations: p={config.blur_prob}")

        # Normalization and tensor conversion
        transforms_list.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return A.Compose(transforms_list)

    else:  # Validation
        return A.Compose([
            A.Resize(height=im_size, width=im_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def create_weighted_sampler(labels):
    """Create weighted sampler to handle class imbalance."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    logger.info(f"Class counts: {class_counts}")
    logger.info(f"Class weights: {class_weights}")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_data_loaders(train_paths, train_labels, val_paths, val_labels, config, data_root=None):
    """Create training and validation data loaders with augmentation support."""

    log_print(f"create_data_loaders: batch_size={config.batch_size}, num_workers={config.num_workers}, "
              f"prefetch_factor={config.prefetch_factor}")
    
    # Get transforms
    train_transform = get_transforms('train', config)
    val_transform = get_transforms('val', config)
    
    # Create datasets
    log_print("create_data_loaders: building train dataset...")
    train_dataset = CachedGastroDataset(
        train_paths, train_labels, 
        transform=train_transform, 
        data_root=data_root,
        im_size=config.im_size,
        phase="train"
    )
    log_print("create_data_loaders: building val dataset...")
    val_dataset = CachedGastroDataset(
        val_paths, val_labels, 
        transform=val_transform, 
        data_root=data_root,
        im_size=config.im_size,
        phase="val"
    )

    if config.use_weighted_sampling:
        weighted_sampler = create_weighted_sampler(train_labels)
        shuffle = False  # Cannot shuffle when using sampler
    else:
        weighted_sampler = None
        shuffle = True
        logger.info("Weighted sampling disabled - using regular shuffling")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        sampler=weighted_sampler,
        shuffle=shuffle,
        num_workers= config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size*8 if config.model_type!="dinov3" else config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
    )

    log_print(f"create_data_loaders: ready — train={len(train_dataset)}, val={len(val_dataset)}, "
              f"weighted_sampling={config.use_weighted_sampling}")
    
    return train_loader, val_loader
