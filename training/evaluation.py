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

CHALLENGE_PREVALENCE = 1 / 101
ORIGINAL_THRESHOLD = 0.5

# Score-based metrics (identical in both columns) and class-based metrics (differ by threshold).
SCORE_METRIC_ROWS = (
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
)
CLASS_METRIC_ROWS = (
    ("accuracy", "Accuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("f1", "F1"),
    ("balanced_accuracy", "Balanced accuracy"),
)
CHALLENGE_PPV_LABEL = f"Challenge PPV@90% recall (prev={CHALLENGE_PREVALENCE:.4f})"


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
        analysis = self.validate_and_analyze(model, loader, criterion)
        scores = analysis["probs"]
        metrics = compute_extended_metrics(scores, analysis["labels"], threshold)
        return scores, analysis["labels"], metrics

    def _run_inference(self, model, loader, criterion, split_name="val", verbose=False):
        """Run forward pass over a loader and collect predictions."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(loader)

        predictions_list = []
        labels_list = []
        logits_list = []
        all_paths = []

        if verbose:
            log_print(f"{split_name}: inference starting ({num_batches} batches)")

        with torch.no_grad():
            for batch_idx, (images, labels, paths) in enumerate(loader):
                if verbose and batch_idx == 0:
                    log_print(f"{split_name}: first batch loaded — images={tuple(images.shape)}")
                elif verbose and num_batches > 10 and batch_idx % max(1, num_batches // 5) == 0:
                    log_print(f"{split_name}: batch {batch_idx + 1}/{num_batches}")

                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                if outputs.shape[1] > 1:
                    _, predicted = torch.max(outputs, 1)
                else:
                    predicted = (outputs.squeeze() > 0).long()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predictions_list.append(predicted)
                labels_list.append(labels)
                logits_list.append(outputs)
                all_paths.extend(paths)

        all_predictions = torch.cat(predictions_list)
        all_labels = torch.cat(labels_list)
        all_logits = torch.cat(logits_list)

        epoch_loss = running_loss / max(num_batches, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        probs = self.logits_to_probs(all_logits.cpu().numpy())

        if verbose:
            log_print(f"{split_name}: inference done — loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%")

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "predictions": all_predictions.cpu().numpy(),
            "labels": all_labels.cpu().numpy(),
            "logits": all_logits.cpu().numpy(),
            "paths": all_paths,
            "probs": probs.cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs),
        }

    def analyze_predictions(self, logits, labels, prevalence=CHALLENGE_PREVALENCE):
        """
        Compute metrics for two decision rules on the same raw scores:

        - original: fixed 0.5 probability threshold (previous pipeline default)
        - youden:   data-driven Youden J threshold on this split

        Score-based metrics (AUROC, AUPRC, Challenge PPV) use the same
        continuous scores for both columns; only class-based metrics differ.
        """
        labels_arr = np.asarray(labels).flatten()
        probs = self.logits_to_probs(logits)
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()
        probs = np.asarray(probs).flatten()

        probs_t = torch.tensor(probs, device=self.device, dtype=torch.float32)
        labels_t = torch.tensor(labels_arr, device=self.device, dtype=torch.float32)

        original_threshold = ORIGINAL_THRESHOLD
        youden_threshold = compute_youden_threshold_gpu(probs_t, labels_t)

        original_metrics = compute_extended_metrics(probs, labels_arr, original_threshold)
        youden_metrics = compute_extended_metrics(probs, labels_arr, youden_threshold)

        try:
            challenge_ppv = prevalence_corrected_ppv_at_90_recall_gpu(
                probs_t, labels_t, prevalence=prevalence,
            )
        except Exception as e:
            logger.warning(f"Challenge PPV calculation failed: {e}")
            challenge_ppv = 0.0

        return {
            "original_threshold": original_threshold,
            "youden_threshold": youden_threshold,
            "original_metrics": original_metrics,
            "youden_metrics": youden_metrics,
            "challenge_ppv": challenge_ppv,
            "probs": probs,
            "labels": labels_arr,
            "logits": np.asarray(logits),
        }

    def validate_and_analyze(self, model, loader, criterion, split_name="val", prevalence=CHALLENGE_PREVALENCE, verbose=False):
        """Run inference and compute original vs Youden metric suites."""
        inference = self._run_inference(model, loader, criterion, split_name=split_name, verbose=verbose)
        analysis = self.analyze_predictions(inference["logits"], inference["labels"], prevalence=prevalence)
        analysis.update(inference)
        return analysis

    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch and return predictions (backward-compatible tuple)."""
        analysis = self.validate_and_analyze(model, val_loader, criterion, split_name="val")
        return (
            analysis["loss"],
            analysis["accuracy"],
            analysis["challenge_ppv"],
            analysis["predictions"],
            analysis["labels"],
            analysis["logits"],
            analysis["paths"],
            analysis["probs"],
        )

    @staticmethod
    def _format_metric_value(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "   —   "
        return f"{value:.4f}"

    def log_epoch_analysis(self, analysis, split_name, epoch=None, fold=None, wandb_log_fn=None):
        """Print a compact side-by-side table: original (0.5) | youden."""
        parts = []
        if fold is not None:
            parts.append(f"Fold {fold}")
        if epoch is not None:
            parts.append(f"Epoch {epoch}")
        parts.append(split_name.upper())
        header = " — ".join(parts)

        col_w = 44
        log_print("=" * 72)
        log_print(header)
        log_print(f"{'':<{col_w}} original (t=0.5) | youden (t={analysis['youden_threshold']:.4f})")
        log_print("-" * 72)
        log_print(f"{'loss':<{col_w}} {analysis['loss']:.4f}")
        log_print(f"{'argmax accuracy':<{col_w}} {analysis['accuracy']:.2f}%")

        for key, label in SCORE_METRIC_ROWS:
            value = analysis["original_metrics"].get(key, float("nan"))
            formatted = self._format_metric_value(value)
            log_print(f"{label:<{col_w}} {formatted} | {formatted}")

        ppv = analysis["challenge_ppv"]
        ppv_fmt = self._format_metric_value(ppv)
        log_print(f"{CHALLENGE_PPV_LABEL:<{col_w}} {ppv_fmt} | {ppv_fmt}")

        for key, label in CLASS_METRIC_ROWS:
            orig = analysis["original_metrics"].get(key, float("nan"))
            youd = analysis["youden_metrics"].get(key, float("nan"))
            log_print(
                f"{label:<{col_w}} {self._format_metric_value(orig)} | {self._format_metric_value(youd)}"
            )

        log_print("=" * 72)

        if wandb_log_fn is None:
            return

        payload = {
            f"{split_name}/loss": analysis["loss"],
            f"{split_name}/accuracy": analysis["accuracy"],
            f"{split_name}/challenge_ppv": analysis["challenge_ppv"],
            f"{split_name}/original/threshold": analysis["original_threshold"],
            f"{split_name}/youden/threshold": analysis["youden_threshold"],
        }
        for key, _ in (*SCORE_METRIC_ROWS, *CLASS_METRIC_ROWS):
            payload[f"{split_name}/original/{key}"] = analysis["original_metrics"].get(key, 0.0)
            payload[f"{split_name}/youden/{key}"] = analysis["youden_metrics"].get(key, 0.0)
        if epoch is not None:
            payload["epoch"] = epoch
        wandb_log_fn(payload)

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
