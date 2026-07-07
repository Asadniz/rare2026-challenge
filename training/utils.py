"""Utility functions for training."""

import os
import json
import random
import datetime
import logging
import sys
import numpy as np
import torch

logger = logging.getLogger("training")


class _FlushStreamHandler(logging.StreamHandler):
    """Stream handler that flushes after every log record (needed for Kaggle/bash)."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def log_print(msg, level=logging.INFO, logger_instance=None):
    """Print immediately to stdout (single line, no duplicate logger output)."""
    print(f"[training] {msg}", flush=True)


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    log_print(f"Random seeds set to {seed} for reproducibility")


def create_checkpoint_dir(config):
    """Create checkpoint directory with parameter info."""
    # Create timestamp
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Create directory name with key parameters
    dir_name = f"checkpoints_{timestamp}_model{config.model_type}_ep{config.epochs}_bs{config.batch_size}_lr{config.learning_rate}_seed{config.seed}"
    
        # Add HP search method
    if config.hp_search:
        dir_name += f"_hp_optuna_{config.hp_trials}"
    
    dir_name += config.loss_type

    if config.loss_type=="cross_entropy" or config.loss_type=="ppv":
        dir_name += "_weighted" if config.use_class_weights else "_unweighted"

    # Add sampling method
    dir_name += "_wsample" if config.use_weighted_sampling else "_shuffle"
    
    # Add CV type
    dir_name += f"_{config.cv_type}"
    
    # Create full path
    checkpoint_dir = os.path.join(config.output_dir, dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save run configuration
    run_config = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': config.to_dict(),
        'pytorch_version': torch.__version__,
        'device': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    }
    
    config_path = os.path.join(checkpoint_dir, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    
    return checkpoint_dir


def setup_logging(level=logging.INFO):
    """Setup logging configuration with unbuffered stdout for bash/Kaggle."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = _FlushStreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(handler)

    for name in (
        "training",
        "training.train",
        "training.training",
        "training.data",
        "training.models",
        "training.evaluation",
        "training.utils",
    ):
        logging.getLogger(name).setLevel(level)

    log_print("Logging configured (stdout, line-buffered, flush-on-write)")
    return logging.getLogger("training.train")