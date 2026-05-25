"""Core training logic and trainer class."""

import torch
import torch.optim as optim
import numpy as np
import os
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import itertools
import random
import datetime

import optuna
from optuna.samplers import TPESampler
import wandb
import copy

from torch.utils.data import DataLoader

from .models import create_model, create_loss_function
from .data import create_data_loaders, CachedGastroDataset, get_transforms
from .evaluation import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for handling model training."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.evaluator = Evaluator(device)

    def train_epoch(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        epoch,
    ):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            # Logging
            log_every = 50
            if batch_idx % log_every == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss:.4f}, Acc: {100.0 * correct / max(total,1):.2f}%"
                )

        if scheduler:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / max(total, 1)

        # Log epoch metrics to wandb
        if wandb.run is not None:
            wandb.log({
                "train/loss": epoch_loss,
                "train/accuracy": epoch_acc,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })

        return epoch_loss, epoch_acc

    def predict_on_test_set(self, fold, test_paths, test_labels, checkpoint_dir):
        """Load best model for fold and predict on test set."""
        logger.info(f"Predicting on test set for fold {fold}")

        # Load the best model for this fold
        checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}_best_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create model and load weights
        model = create_model(self.config, phase="test")
        model = model.to(self.device)

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        self._load_checkpoint(model, checkpoint)

        # Create test dataset and loader
        test_dataset = CachedGastroDataset(
            test_paths, 
            test_labels, 
            transform=get_transforms("val", self.config), 
            data_root=None, 
            im_size=self.config.im_size,
            phase="test")

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=True
        )

        # Create dummy criterion (not used for inference but needed for evaluator)
        criterion = create_loss_function(self.config).to(self.device)

        # Get predictions
        _, _, _, test_preds, test_labels_true, test_logits, test_paths_list = self.evaluator.validate_epoch(
            model, test_loader, criterion
        )

        return test_preds, test_labels_true, test_logits, test_paths_list


    def train_fold(self, fold, train_paths, train_labels, val_paths, val_labels, data_root, checkpoint_dir):
        """Train a single fold."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold}")
        logger.info(f"{'='*50}")

        # Initialize wandb for this fold
        wandb_config = {
            "fold": fold,
            "loss_type": self.config.loss_type,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "num_train_samples": len(train_paths),
            "num_val_samples": len(val_paths),
        }

        wandb.init(
            project=getattr(self.config, "wandb_project", "gastro-training"),
            group=getattr(self.config, "wandb_group", "baseline"),
            name=f"fold_{fold}",
            config=wandb_config,
            tags=[f"fold_{fold}", self.config.loss_type],
            reinit=True
        )

        logger.info(f"FOLD {fold} - TRAINING WITH CONFIG:")
        logger.info(f"  Loss type: {self.config.loss_type}")
        logger.info(f"  Learning Rate: {self.config.learning_rate}")
        logger.info(f"  Weight Decay: {self.config.weight_decay}")
        logger.info(f"  Batch Size: {self.config.batch_size}")

        # Sanity check for no leakage
        self._check_data_leakage(train_paths, val_paths, fold)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_paths, train_labels, val_paths, val_labels, self.config, data_root
        )

        # Initialize model
        model = create_model(self.config, phase="train")
        model = model.to(self.device)

        if hasattr(model, 'get_parameter_groups'):
            all_params = model.get_parameter_groups(
            self.config.learning_rate,
            getattr(self.config, 'backbone_lr_multiplier', 0.1)
        )
            for pg in all_params:
                pg["weight_decay"] = self.config.weight_decay
        else:
            all_params = [
                {
                    "params": model.parameters(),
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.learning_rate,
                }
            ]

        # Calculate class weights for loss function (if enabled)
        class_weights = None
        if self.config.use_class_weights:
            class_counts = np.bincount(train_labels, minlength=2)
            # Avoid division by zero
            class_counts = np.clip(class_counts, 1, None)
            class_weights = torch.FloatTensor([1.0 / class_counts[0], 1.0 / class_counts[1]]).to(self.device)
            logger.info(f"Using class weights in loss: {class_weights}")
            wandb.log({"class_weights": class_weights.tolist()})
        else:
            logger.info("Class weights in loss disabled")

        # Create loss function
        criterion = create_loss_function(self.config, class_weights).to(self.device)

        logger.info(f"FOLD {fold} - CREATING OPTIMIZER WITH:")
        logger.info(f"  Learning Rate: {self.config.learning_rate}")
        logger.info(f"  Weight Decay: {self.config.weight_decay}")

        # Optimizer and scheduler
        optimizer = optim.AdamW(all_params)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate * 0.01,
        )

        # Training loop
        best_ppv = 0.0
        best_model_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold}.pth")

        for epoch in range(self.config.epochs):
            
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, scheduler, epoch,
            )

            val_loss, val_acc, val_ppv, val_preds, val_labels_true, val_logits, val_paths = self.evaluator.validate_epoch(
                model, val_loader, criterion, 
            )

            # Log validation metrics to wandb
            wandb.log({
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/ppv": val_ppv,
                "epoch": epoch
            })

            logger.info(
                f"Fold {fold}, Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val PPV: {val_ppv:.4f}"
            )

            # Save best model
            if val_ppv > best_ppv:
                best_ppv = val_ppv
                self._save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    fold,
                    best_ppv,
                    val_loss,
                    val_acc,
                    train_loss,
                    train_acc,
                    best_model_path,
                )
                # Log best metrics to wandb
                wandb.log({
                    "best/ppv": best_ppv,
                    "best/epoch": epoch,
                    "best/val_loss": val_loss,
                    "best/val_accuracy": val_acc
                })
                logger.info(f"New best model saved with PPV: {best_ppv:.4f}")

        # Load best model and get final predictions
        checkpoint = torch.load(best_model_path, weights_only=False, map_location="cpu")
        self._load_checkpoint(model, checkpoint)

        val_loss, val_acc, val_ppv, val_preds, val_labels_true, val_logits, val_paths = self.evaluator.validate_epoch(
            model, val_loader, criterion,
        )

        # Log final results
        wandb.log({
            "final/accuracy": val_acc,
            "final/ppv": val_ppv,
            "final/loss": val_loss
        })

        logger.info(f"Fold {fold} Final Results: Acc: {val_acc:.2f}%, PPV: {val_ppv:.4f}")

        # Print classification report
        self.evaluator.print_classification_report(val_labels_true, val_preds, fold)

        # Save checkpoint to final location
        final_checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}_best_checkpoint.pth")
        if os.path.exists(best_model_path):
            os.rename(best_model_path, final_checkpoint_path)
            logger.info(f"Checkpoint saved to: {final_checkpoint_path}")

        # Save fold paths
        torch.save(val_paths, os.path.join(checkpoint_dir, f"fold_{fold}_val_paths.pt"))
        torch.save(train_paths, os.path.join(checkpoint_dir, f"fold_{fold}_train_paths.pt"))

        # Finish wandb run for this fold
        wandb.finish()

        return val_preds, val_labels_true, val_logits, val_paths, val_ppv

    def hyperparameter_search(self, fold, train_paths, train_labels, val_paths, val_labels, checkpoint_dir):
        """Perform Optuna hyperparameter search for a single fold with W&B logging."""
        logger.info(f"Starting Optuna hyperparameter search for fold {fold}")

        # Force fold-specific study name
        base_study_name = self.config.optuna_study_name or "gastro_study"
        study_name = f"{base_study_name}_fold_{fold}"

        # Use fold-specific storage if using SQLite
        storage = self.config.optuna_storage
        if storage and storage.startswith("sqlite:///"):
            db_path = storage.replace("sqlite:///", "")
            db_name, db_ext = os.path.splitext(db_path)
            storage = f"sqlite:///{db_name}{db_ext}"

        if storage:
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                sampler=TPESampler(seed=self.config.seed + fold),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
            )
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=self.config.seed + fold),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
            )

        def objective(trial):
            """Optuna objective function with W&B logging."""
            hp_config = self._sample_optuna_hyperparameters(trial)
            logger.info(f"Fold {fold}, Trial {trial.number}: {hp_config}")

            # Initialize W&B for this trial
            wandb_config = {
                "fold": fold,
                "trial_number": trial.number,
                "optuna_study": study_name,
                **hp_config
            }

            wandb.init(
                project=getattr(self.config, "wandb_project", "gastro-training"),
                group=getattr(self.config, "wandb_group", "baseline"),
                name=f"fold_{fold}_trial_{trial.number}",
                config=wandb_config,
                tags=[f"fold_{fold}", f"trial_{trial.number}", "optuna", hp_config.get("loss_type", "unknown")],
                reinit=True
            )

            try:
                # Use actual train/val split (not additional split)
                score = self._train_optuna_trial(
                    hp_config, train_paths, train_labels, val_paths, val_labels, fold, trial, checkpoint_dir
                )

                # Log final trial results to wandb
                wandb.log({
                    "trial/final_score": score,
                    "trial/number": trial.number,
                    "trial/state": "completed"
                })

                logger.info(f"Fold {fold}, Trial {trial.number} completed with score: {score:.4f}")
                
                # Finish this trial's wandb run
                wandb.finish()
                
                return score

            except optuna.exceptions.TrialPruned:
                wandb.log({
                    "trial/final_score": 0.0,
                    "trial/number": trial.number,
                    "trial/state": "pruned"
                })
                wandb.finish()
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                wandb.log({
                    "trial/final_score": 0.0,
                    "trial/number": trial.number,
                    "trial/state": "failed",
                    "trial/error": str(e)
                })
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

        try:
            # Run optimization
            study.optimize(objective, n_trials=self.config.hp_trials)

            # Get best parameters
            best_hp = study.best_params
            best_score = study.best_value
            best_trial_number = study.best_trial.number

            best_temp_checkpoint_path = os.path.join(checkpoint_dir, f"temp_optuna_trial_{best_trial_number}_fold_{fold}.pth")

            # Save the best checkpoint to the results directory
            final_checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}_best_optuna_checkpoint.pth")

            if os.path.exists(best_temp_checkpoint_path):
                best_checkpoint = torch.load(best_temp_checkpoint_path, weights_only=False, map_location="cpu")
                best_full_config = best_checkpoint["full_config"]

                # Apply full config to the trainer
                logger.info(f"Applying FULL best config to trainer (not just Optuna params)")
                for key, value in best_full_config.items():
                    if hasattr(self.config, key):
                        old_value = getattr(self.config, key)
                        setattr(self.config, key, value)
                        if old_value != value:
                            logger.info(f"  Updated self.config.{key}: {old_value} -> {value}")

                # Log the key derived parameters
                logger.info("Key derived parameters applied:")
                logger.info(f"  loss_type: {self.config.loss_type}")

                # Save final checkpoint with complete config
                final_checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}_best_optuna_checkpoint.pth")

                # Update checkpoint with final info
                best_checkpoint["optuna_best_trial"] = best_trial_number
                best_checkpoint["optuna_best_score"] = best_score
                best_checkpoint["saved_to"] = final_checkpoint_path
                best_checkpoint["trainer_config_updated"] = True  # Flag to indicate trainer was updated

                torch.save(best_checkpoint, final_checkpoint_path)
                logger.info(f"Best Optuna model saved to: {final_checkpoint_path}")
            else:
                logger.error(f"Best checkpoint not found: {best_temp_checkpoint_path}")
                final_checkpoint_path = None

            # Clean up ALL temporary checkpoint files for this fold
            import glob

            temp_files = glob.glob(os.path.join(checkpoint_dir, f"temp_optuna_trial_*_fold_{fold}.pth"))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    logger.debug(f"Removed temp file: {temp_file}")
                except OSError as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")

            logger.info(f"Cleaned up {len(temp_files)} temporary checkpoint files for fold {fold}")

            # Save Optuna study results with W&B info
            study_results = {
                "best_params": best_hp,
                "best_score": best_score,
                "best_trial_number": best_trial_number,
                "best_checkpoint_path": final_checkpoint_path,
                "study_name": study_name,
                "n_trials": len(study.trials),
                "wandb_project": getattr(self.config, "wandb_project", "gastro-training"),
                "all_trials": [
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "state": trial.state.name,
                    }
                    for trial in study.trials
                ],
            }

            study_results_path = os.path.join(checkpoint_dir, f"fold_{fold}_optuna_study.json")
            with open(study_results_path, "w") as f:
                json.dump(study_results, f, indent=2)

            logger.info(f"Fold {fold} best hyperparameters: {best_hp} (score: {best_score:.4f})")
        except Exception as e:
            logger.error(f"Optuna optimization failed for fold {fold}: {e}")
            raise
        return best_hp, final_checkpoint_path

    def _sample_optuna_hyperparameters(self, trial):
        """Sample hyperparameters using Optuna with loss function search."""
        hp_config = {}

        # Learning rate
        hp_config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

        # Weight decay
        hp_config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # Loss function search
        if self.config.hp_search_loss:
            loss_type = trial.suggest_categorical(
                "loss_type",
                ["cross_entropy", "ppv", "surrogate"],
            )
        else:
            # Keep current loss if not searching over it
            loss_type = self.config.loss_type
        hp_config["loss_type"] = loss_type

        # Loss-specific params
        if loss_type == "ppv":
            hp_config["ppv_lambda"] = trial.suggest_float("ppv_lambda", 0.01, 1.0, log=True)
            hp_config["ppv_margin"] = trial.suggest_float("ppv_margin", 0.01, 0.2)
            hp_config["ppv_beta"] = trial.suggest_float("ppv_beta", 0.9, 0.999)
        elif loss_type == "surrogate":
            hp_config["surrogate_lambda"] = trial.suggest_float("surrogate_lambda", 0.01, 100.0, log=True)

        # Batch size: restrict  if needed
        if self.config.im_size == "large":
            hp_config["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])
        else:
            hp_config["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

        return hp_config

    def _train_optuna_trial(
        self, hp_config, train_paths, train_labels, val_paths, val_labels, fold, trial, checkpoint_dir=""
    ):
        """Full training run for Optuna hyperparameter evaluation with W&B logging."""

        # Create temporary config with HP parameters
        temp_config = copy.deepcopy(self.config)
        for key, value in hp_config.items():
            setattr(temp_config, key, value)

        # Create data loaders with HP config
        train_loader, val_loader = create_data_loaders(train_paths, train_labels, val_paths, val_labels, temp_config)

        # Create model
        model = create_model(temp_config, phase="train")
        model = model.to(self.device)

        if hasattr(model, 'get_parameter_groups'):
            all_params = model.get_parameter_groups(
                self.config.learning_rate,
                getattr(self.config, 'backbone_lr_multiplier', 0.1)
            )
            for pg in all_params:
                pg["weight_decay"] = self.config.weight_decay
        else:
            all_params = [
                {
                    "params": model.parameters(),
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.learning_rate,
                }
            ]

        # Optimizer with HP parameters
        optimizer = optim.AdamW(all_params)

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=temp_config.epochs,
            eta_min=temp_config.learning_rate * 0.01,
        )

        # Loss function with HP parameters
        class_weights = None
        if temp_config.use_class_weights:
            class_counts = np.bincount(train_labels, minlength=2)
            class_counts = np.clip(class_counts, 1, None)
            class_weights = torch.FloatTensor([1.0 / class_counts[0], 1.0 / class_counts[1]]).to(self.device)

        criterion = create_loss_function(temp_config, class_weights).to(self.device)

        # Log the loss function being used
        loss_name = temp_config.loss_type
        logger.info(f"Trial {trial.number}: Using loss '{loss_name}'")

        # Full training run
        best_val_ppv = 0.0
        best_model_state = None
        patience = 10  # Early stopping patience
        patience_counter = 0

        for epoch in range(temp_config.epochs): 
            # Training
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, scheduler, epoch,
            )

            # Validation
            val_loss, val_acc, val_ppv, _, _, _, _ = self.evaluator.validate_epoch(
                model, val_loader, criterion
            )

            # Log metrics to wandb
            if (wandb.run is not None) and (epoch % 5 == 0):
                wandb.log({
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/ppv": val_ppv,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

            # Update best score
            if val_ppv > best_val_ppv:
                best_val_ppv = val_ppv
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                
                # Log best metrics to wandb
                if wandb.run is not None:
                    wandb.log({
                        "best/ppv": best_val_ppv,
                        "best/epoch": epoch
                    })
            else:
                patience_counter += 1

            # Report to Optuna for pruning
            trial.report(val_ppv, epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                if wandb.run is not None:
                    wandb.log({
                        "trial/pruned_at_epoch": epoch,
                        "trial/state": "pruned"
                    })
                raise optuna.exceptions.TrialPruned()

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} for trial {trial.number}")
                if wandb.run is not None:
                    wandb.log({
                        "trial/early_stopped_at_epoch": epoch,
                        "trial/state": "early_stopped"
                    })
                break

            # Log progress every 10 epochs
            if epoch % 10 == 0:
                logger.info(
                    f"Trial {trial.number}, Epoch {epoch}: "
                    f"Val PPV: {val_ppv:.4f}, Best: {best_val_ppv:.4f}, Loss: {loss_name}"
                )

        if best_model_state is not None:
            trial_checkpoint = {
                "model_state_dict": best_model_state,
                "best_ppv": best_val_ppv,
                "hp_config": hp_config,
                "full_config": temp_config.to_dict(),
                "fold": fold,
                "trial_number": trial.number,
                "loss_function": loss_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Use temporary path (will be cleaned up later)
            trial_checkpoint_path = os.path.join(checkpoint_dir, f"temp_optuna_trial_{trial.number}_fold_{fold}.pth")
            torch.save(trial_checkpoint, trial_checkpoint_path)

        return best_val_ppv

    def _check_data_leakage(self, train_paths, val_paths, fold):
        """Check for data leakage between train and validation sets."""
        train_set = set(train_paths)
        val_set = set(val_paths)
        overlap = train_set.intersection(val_set)

        if overlap:
            logger.error(f"FOLD {fold}: Found {len(overlap)} overlapping samples between train/val!")
            logger.error(f"Overlapping files: {list(overlap)[:5]}...")
        else:
            logger.info(f"FOLD {fold}: No overlap between train/val sets ✓")

    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        fold,
        best_ppv,
        val_loss,
        val_acc,
        train_loss,
        train_acc,
        path,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_ppv": best_ppv,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "config": self.config.to_dict(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(checkpoint, path)
        # threading.Thread(target=torch.save, args=(checkpoint, path), daemon=True).start()
    
    def _load_checkpoint(self, model, checkpoint):
        model.load_state_dict(checkpoint["model_state_dict"])

    def load_and_evaluate_best_model(self, fold, val_paths, val_labels, checkpoint_path, final_checkpoint_dir):
        """Load the best model from Optuna and evaluate it."""
        logger.info(f"Loading best Optuna model for fold {fold}: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

        # Create model with the HP config used for this checkpoint
        hp_config = checkpoint["hp_config"]
        temp_config = copy.deepcopy(self.config)
        for key, value in hp_config.items():
            setattr(temp_config, key, value)

        # Create model and load weights
        model = create_model(temp_config, phase="val")
        model = model.to(self.device)

        self._load_checkpoint(model, checkpoint)

        # Create validation loader
        val_dataset = CachedGastroDataset(val_paths, val_labels, transform=get_transforms("val", self.config), 
                                          data_root=None, 
                                          im_size=self.config.im_size, phase="val")
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=temp_config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=True
        )

        # Evaluate
        criterion = create_loss_function(temp_config).to(self.device)  # Dummy criterion for evaluation
        val_loss, val_acc, val_ppv, val_preds, val_labels_true, val_logits, val_paths_list = self.evaluator.validate_epoch(
            model, val_loader, criterion,
        )

        logger.info(f"Fold {fold} Best Optuna Results: Acc: {val_acc:.2f}%, PPV: {val_ppv:.4f}")
        self.evaluator.print_classification_report(val_labels_true, val_preds, fold)

        # Copy checkpoint to final location
        final_checkpoint_path = os.path.join(final_checkpoint_dir, f"fold_{fold}_best_checkpoint.pth")
        torch.save(checkpoint, final_checkpoint_path)

        return val_preds, val_labels_true, val_logits, val_paths_list, val_ppv