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
from .calibration import fit_calibrated_positive_scores
from .utils import log_print

logger = logging.getLogger("training.evaluation")

CHALLENGE_PREVALENCE = 1 / 101


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

    def _run_inference(self, model, loader, criterion, split_name="val"):
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

        log_print(f"{split_name}: inference starting ({num_batches} batches)")

        with torch.no_grad():
            for batch_idx, (images, labels, paths) in enumerate(loader):
                if batch_idx == 0:
                    log_print(f"{split_name}: first batch loaded — images={tuple(images.shape)}")
                elif num_batches > 10 and batch_idx % max(1, num_batches // 5) == 0:
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
        Compute challenge PPV plus full metric suites for raw (Youden) and calibrated scores.
        """
        labels_arr = np.asarray(labels).flatten()
        probs = self.logits_to_probs(logits)
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()
        probs = np.asarray(probs).flatten()

        probs_t = torch.tensor(probs, device=self.device, dtype=torch.float32)
        labels_t = torch.tensor(labels_arr, device=self.device, dtype=torch.float32)

        try:
            challenge_ppv = prevalence_corrected_ppv_at_90_recall_gpu(
                probs_t, labels_t, prevalence=prevalence,
            )
        except Exception as e:
            logger.warning(f"Challenge PPV calculation failed: {e}")
            challenge_ppv = 0.0

        youden_threshold = compute_youden_threshold_gpu(probs_t, labels_t)
        youden_metrics = compute_extended_metrics(probs, labels_arr, youden_threshold)

        calibrated_probs = None
        cal_t = float("nan")
        cal_b = float("nan")
        calibrated_youden_threshold = float("nan")
        calibrated_metrics = {k: 0.0 for k in youden_metrics}

        try:
            calibrated_probs, cal_t, cal_b = fit_calibrated_positive_scores(logits, labels_arr)
            cal_probs_t = torch.tensor(calibrated_probs, device=self.device, dtype=torch.float32)
            calibrated_youden_threshold = compute_youden_threshold_gpu(cal_probs_t, labels_t)
            calibrated_metrics = compute_extended_metrics(
                calibrated_probs, labels_arr, calibrated_youden_threshold,
            )
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            calibrated_probs = np.full_like(probs, np.nan)

        return {
            "challenge_ppv": challenge_ppv,
            "youden_threshold": youden_threshold,
            "youden_metrics": youden_metrics,
            "calibrated_probs": calibrated_probs,
            "cal_t": cal_t,
            "cal_b": cal_b,
            "calibrated_youden_threshold": calibrated_youden_threshold,
            "calibrated_metrics": calibrated_metrics,
            "probs": probs,
            "labels": labels_arr,
            "logits": np.asarray(logits),
        }

    def validate_and_analyze(self, model, loader, criterion, split_name="val", prevalence=CHALLENGE_PREVALENCE):
        """Run inference and compute raw-Youden + calibrated metric suites."""
        inference = self._run_inference(model, loader, criterion, split_name=split_name)
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

    def log_epoch_analysis(self, analysis, split_name, epoch=None, fold=None, wandb_log_fn=None):
        """Print and optionally log all per-epoch metrics (raw Youden + calibrated)."""
        parts = []
        if fold is not None:
            parts.append(f"Fold {fold}")
        if epoch is not None:
            parts.append(f"Epoch {epoch}")
        parts.append(split_name)
        header = ", ".join(parts) + " metrics"

        log_print("=" * 60)
        log_print(header)
        log_print(f"  loss={analysis['loss']:.4f}, argmax_accuracy={analysis['accuracy']:.2f}%")
        log_print(
            f"  challenge_ppv@90_recall (prevalence-corrected, prev={CHALLENGE_PREVALENCE:.6f})="
            f"{analysis['challenge_ppv']:.4f}"
        )

        log_print(f"  --- raw scores / Youden threshold={analysis['youden_threshold']:.4f} ---")
        for key, value in sorted(analysis["youden_metrics"].items()):
            log_print(f"    youden/{key}: {value:.4f}")

        log_print(
            f"  --- calibrated scores (t={analysis['cal_t']:.4f}, b={analysis['cal_b']:.4f}) / "
            f"Youden threshold={analysis['calibrated_youden_threshold']:.4f} ---"
        )
        for key, value in sorted(analysis["calibrated_metrics"].items()):
            log_print(f"    calibrated/{key}: {value:.4f}")
        log_print("=" * 60)

        if wandb_log_fn is None:
            return

        payload = {
            f"{split_name}/loss": analysis["loss"],
            f"{split_name}/accuracy": analysis["accuracy"],
            f"{split_name}/challenge_ppv": analysis["challenge_ppv"],
            f"{split_name}/youden/threshold": analysis["youden_threshold"],
            f"{split_name}/calibrated/threshold": analysis["calibrated_youden_threshold"],
            f"{split_name}/calibrated/t": analysis["cal_t"],
            f"{split_name}/calibrated/b": analysis["cal_b"],
        }
        for key, value in analysis["youden_metrics"].items():
            payload[f"{split_name}/youden/{key}"] = value
        for key, value in analysis["calibrated_metrics"].items():
            payload[f"{split_name}/calibrated/{key}"] = value
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
