"""Validation and evaluation functions."""

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import logging

from validation.metrics import (
    prevalence_corrected_ppv_at_90_recall_gpu,
    compute_youden_threshold_gpu,
    compute_extended_metrics,
)
from .utils import log_print

logger = logging.getLogger("training.evaluation")


class Evaluator:
    """Handles model evaluation and metrics calculation."""

    def __init__(self, device):
        self.device = device

    @staticmethod
    def logits_to_probs(logits):
        """Convert model logits to positive-class probabilities."""
        logits_tensor = torch.as_tensor(logits)
        if logits_tensor.ndim == 1:
            return logits_tensor
        if logits_tensor.shape[1] == 2:
            return torch.nn.functional.softmax(logits_tensor, dim=1)[:, 1]
        return logits_tensor.squeeze()

    def compute_split_metrics(self, logits, labels, threshold):
        """Compute probability-based and class-based metrics for a split."""
        scores = self.logits_to_probs(logits)
        return compute_extended_metrics(scores, labels, threshold)

    def evaluate_split(self, model, loader, criterion, threshold):
        """Run inference on a split and return scores, labels, and metrics."""
        _, _, _, _, labels, logits, _, _ = self.validate_epoch(model, loader, criterion)
        scores = self.logits_to_probs(logits)
        metrics = compute_extended_metrics(scores, labels, threshold)
        return scores.numpy(), labels, metrics

    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch and return predictions."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(val_loader)

        predictions_list = []
        labels_list = []
        logits_list = []
        all_paths = []

        log_print(f"Validation: starting ({num_batches} batches)")

        with torch.no_grad():
            for batch_idx, (images, labels, paths) in enumerate(val_loader):
                if batch_idx == 0:
                    log_print(f"Validation: first batch loaded — images={tuple(images.shape)}")
                elif num_batches > 10 and batch_idx % max(1, num_batches // 5) == 0:
                    log_print(f"Validation: batch {batch_idx + 1}/{num_batches}")

                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                old_labels = labels

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                if outputs.shape[1] > 1:
                    _, predicted = torch.max(outputs, 1)
                else:
                    predicted = (outputs.squeeze() > 0).long()

                total += labels.size(0)
                correct += (predicted == old_labels).sum().item()

                predictions_list.append(predicted)
                labels_list.append(old_labels)
                logits_list.append(outputs)
                all_paths.extend(paths)

        all_predictions = torch.cat(predictions_list)
        all_labels = torch.cat(labels_list)
        all_logits = torch.cat(logits_list)

        epoch_loss = running_loss / num_batches
        epoch_acc = 100. * correct / total

        probs = self.logits_to_probs(all_logits.cpu().numpy())
        log_print(f"Validation: done — loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%")

        try:
            ppv = prevalence_corrected_ppv_at_90_recall_gpu(
                probs.to(self.device),
                torch.tensor(all_labels).to(self.device),
                prevalence=1 / 101,
            )
        except Exception as e:
            logger.warning(f"PPV calculation failed: {e}")
            ppv = 0.0

        return (
            epoch_loss,
            epoch_acc,
            ppv,
            all_predictions.cpu().numpy(),
            all_labels.cpu().numpy(),
            all_logits.cpu().numpy(),
            all_paths,
            probs.cpu().numpy(),
        )

    def create_predictions_dataframe(
        self,
        val_paths_list,
        val_labels_true,
        val_logits,
        fold,
        split_type='val',
        youden_threshold=None,
    ):
        """Create predictions DataFrame for a fold."""
        probs = self.logits_to_probs(val_logits).numpy()
        df = pd.DataFrame({
            'image_path': val_paths_list,
            'sample_id': [Path(p).name for p in val_paths_list],
            'center': [
                Path(p).parts[-3].split('_')[1]
                if len(Path(p).parts) >= 3 and '_' in Path(p).parts[-3]
                else 'unknown'
                for p in val_paths_list
            ],
            'target': val_labels_true,
            'logits_0': [logits[0] if len(logits) == 2 else 0 for logits in val_logits],
            'logits_1': [logits[1] if len(logits) == 2 else logits for logits in val_logits],
            'probability': probs,
            'fold': fold,
            'split_type': split_type,
        })

        if youden_threshold is not None:
            df['youden_threshold'] = youden_threshold
            df['decision'] = (probs >= youden_threshold).astype(int)

        return df

    def print_classification_report(self, y_true, y_pred, fold=None, threshold=None):
        """Print classification report."""
        fold_str = f" for Fold {fold}" if fold is not None else ""
        threshold_str = f" (Youden threshold={threshold:.4f})" if threshold is not None else ""
        logger.info(f'\nClassification Report{fold_str}{threshold_str}:')
        logger.info('\n' + classification_report(y_true, y_pred, target_names=['NBDE', 'NEO']))

    def log_extended_metrics(self, metrics, split_name, wandb_module=None):
        """Log extended metrics with a split prefix."""
        if wandb_module is None:
            import wandb as wandb_module

        for metric_name, value in metrics.items():
            wandb_module.log({f"{split_name}/{metric_name}": value})
