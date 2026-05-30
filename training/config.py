"""Configuration and argument parsing for training."""

import argparse
from dataclasses import dataclass
import os
from typing import Optional, Dict, Any, Tuple, List

data_dir: str = os.environ.get('DATA_DIR', 'data/train')
results_dir: str = os.environ.get('RESULTS_DIR', 'results')
splits_dir: str = os.environ.get('SPLITS_DIR', 'data/splits')

LOSS_TYPES: Tuple[str, ...] = (
    "cross_entropy",
    "ppv",
    "surrogate",
)


@dataclass
class TrainingConfig:
    """Training configuration class."""

    #Rank for lora
    lora_rank: int = 8

    # Data settings
    data_dir: str = os.environ.get('DATA_DIR', 'data/train')
    results_dir: str = os.environ.get('RESULTS_DIR', 'results')
    splits_dir: str = os.environ.get('SPLITS_DIR', 'data/splits')
    huggingface_cache_dir: str = None

    # wandb
    wandb_project: str = "rare_resnet"
    wandb_group: str = "gastronet"

    # Model settings
    num_classes: int = 2
    model_type: str = "resnet50"
    model_filename: str = None
    our_weights: bool = False

    # Image settings
    im_size: str = "regular" # "regular" or "large"

    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Loss selection (single source of truth)
    loss_type: str = "cross_entropy"  # one of LOSS_TYPES

    # Loss-specific parameters
    # ppv (CE + ranking for PPV@Recall)
    ppv_lambda: float = 0.5
    ppv_margin: float = 0.1
    ppv_beta: float = 0.99
    # surrogate (binary/multi automatically chosen by num_classes)
    surrogate_lambda: float = 0.1

    # Imbalance handling
    use_class_weights: bool = True
    use_weighted_sampling: bool = True

    #Dinov3 specific settings
    backbone_lr_multiplier: float = 0.1   # LR for unfrozen backbone layers
    dinov3_unfreeze_blocks: int = 2        # How many last blocks to unfreeze

    # Augmentations settings
    aug_preset: str = "top1" # "top1", "top2", "top3", "top4"

    # Augmentation parameters
    jigsaw_grid_size: int = 3
    jigsaw_prob: float = 0.5
    erasing_prob: float = 0.5
    erasing_scale_min: float = 0.02
    erasing_scale_max: float = 0.33
    hide_seek_prob: float = 0.5
    hide_seek_ratio: float = 0.5
    hide_seek_unit_size_min: int = 10
    hide_seek_unit_size_max: int = 80
    elastic_alpha: float = 120.0
    elastic_sigma: float = 6.0
    elastic_prob: float = 0.5
    affine_scale_min: float = 0.9
    affine_scale_max: float = 1.1
    affine_translate_percent: float = 0.1
    affine_rotate: int = 15
    affine_shear: int = 5
    affine_prob: float = 0.5

    optical_distortion_prob: float = 0.5
    lighting_variations_prob: float = 0.5
    color_variations_prob: float = 0.5
    texture_variations_prob: float = 0.3
    perspective_prob: float = 0.3
    noise_prob: float = 0.3
    blur_prob: float = 0.3
    clahe_prob: float = 0.5

    # Training setup
    num_workers: int = 8
    prefetch_factor: int = 16
    seed: int = 42

    # Cross-validation settings
    cv_type: str = "5fold_cv"  # '5fold_cv' or 'holdout_cv'
    n_folds: int = 5
    single_fold: bool = False
    fold_number: int = 0

    # Hyperparameter search
    hp_search: bool = False
    hp_trials: int = 20
    optuna_study_name: str = None
    optuna_storage: str = None
    hp_search_loss: bool = False  # include loss function search in hyperparameter optimization

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """
        Create config from command line arguments, with backward-compatible
        mapping from deprecated --use_*_loss flags to loss_type.
        """
        raw = vars(args).copy()

        # Normalize booleans that were provided as flags with default=None in parser
        # so that dataclass defaults apply when value is None.
        cleaned = {k: v for k, v in raw.items() if v is not None}

        cfg = cls(**cleaned)
        cfg.validate_and_normalize()
        return cfg

    def validate_and_normalize(self) -> None:
        """Sanity checks and normalization of dependent fields."""
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got '{self.loss_type}'.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()


def get_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(description="Modular Training Script")

    # Data settings
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--splits_dir", type=str, help="Path to splits directory")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--huggingface_cache_dir", type=str, help="Cache dir for huggingface")

    # Wandb info
    parser.add_argument("--wandb_project", type=str, help="Name of wandb project")
    parser.add_argument("--wandb_group", type=str, help="Name of wandb group")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--num_workers", type=int, help="Number of data loader workers")
    parser.add_argument("--prefetch_factor", type=int, help="Prefetch factor in the dataloader")

    # Image settings
    parser.add_argument("--im_size", type=str, help="Size of image to be loaded")

    # Loss selection (new)
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=list(LOSS_TYPES),
        help=f"Loss function to use. One of {LOSS_TYPES}",
    )

    # Loss-specific parameters
    parser.add_argument("--ppv_lambda", type=float, help="Lambda parameter for PPV loss")
    parser.add_argument("--ppv_margin", type=float, help="Margin parameter for PPV loss")
    parser.add_argument("--ppv_beta", type=float, help="EMA beta for PPV loss")
    parser.add_argument("--surrogate_lambda", type=float, help="Lambda penalty for surrogate loss")


    # Imbalance handling
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=None,
        help="Use class weights to handle imbalance",
    )
    parser.add_argument(
        "--no_class_weights", dest="use_class_weights", action="store_false", help="Disable class weights"
    )
    parser.add_argument(
        "--use_weighted_sampling",
        action="store_true",
        default=None,
        help="Use weighted sampling to handle imbalance",
    )
    parser.add_argument(
        "--no_weighted_sampling", dest="use_weighted_sampling", action="store_false", help="Disable weighted sampling"
    )

    # Training setup
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Cross-validation settings
    parser.add_argument(
        "--cv_type",
        type=str,
        choices=["5fold_cv", "holdout_cv", "center1", "center2"],
        help="Cross-validation type",
    )
    parser.add_argument("--n_folds", type=int, help="Number of folds for k-fold CV")
    parser.add_argument("--single_fold", action="store_true", default=None, help="Run only one CV fold")
    parser.add_argument("--fold_number", type=int, help="Fold number to be used")

    # Hyperparameter search
    parser.add_argument("--hp_search", action="store_true", default=None, help="Enable hyperparameter search")
    parser.add_argument("--hp_trials", type=int, help="Number of hyperparameter trials per fold")
    parser.add_argument("--optuna_study_name", type=str, help="Optuna study name for persistent storage")
    parser.add_argument("--optuna_storage", type=str, help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument(
        "--hp_search_loss",
        action="store_true",
        default=None,
        help="Include loss function search in hyperparameter optimization",
    )

    # Model type
    parser.add_argument(
    "--model_type", type=str, choices=["resnet50", "vit_small", "vit_large", "dinov3", "vit_lora", "efficientnetv2", "convnext"], help="Model architecture type"
    )
    parser.add_argument("--lora_rank", type=int, help="LoRA rank for vit_lora model")
    
    parser.add_argument(
        "--model_filename", type=str, help="Filename of the model checkpoint"
    )
    parser.add_argument(
        "--our_weights", type=bool, help="Use our weights, not from huggingface"
    )
    parser.add_argument("--num_classes", type=int, help="Dimension of final model layer")

    # Augmentation presets
    parser.add_argument("--aug_preset", type=str, help="Preset to be used")

    parser.add_argument("--backbone_lr_multiplier", type=float, help="LR multiplier for backbone layers")
    parser.add_argument("--dinov3_unfreeze_blocks", type=int, help="Number of last DINOv3 blocks to unfreeze")

    return parser