import numpy as np
import torch
from typing import Dict, Union, Tuple

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

def compute_prevalence_corrected_ppv_at_90_recall_gpu(
    y_pred: Union[torch.Tensor, None],
    y_true: Union[torch.Tensor, None],
    prevalence: float = None,
    thresholds: Union[torch.Tensor, None] = None,
    min_recall: float = 0.9,
    mode: str = "max",  # "max" or "max_threshold"
) -> Tuple[float, float]:
    """
    Compute prevalence-corrected PPV@90%Recall metric on GPU, vectorized.
    
    Args:
        y_pred: Predicted probabilities/scores. Shape: (N,)
        y_true: True binary labels (0 or 1). Shape: (N,)
        prevalence: prevalence of positive class. If None, computed from y_true.
        thresholds: Tensor of thresholds. If None, uses unique y_pred values with boundaries.
        min_recall: Minimum recall threshold (default 0.9).
        mode: "max" or "max_threshold" to select PPV.
        
    Returns:
        (ppv_corrected, threshold) tuple, both floats on CPU.
    """

    if y_pred is None or y_true is None:
        raise ValueError("y_pred and y_true cannot be None")
    if not torch.is_tensor(y_pred) or not torch.is_tensor(y_true):
        raise ValueError("y_pred and y_true must be torch tensors")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} must match")
    
    device = y_pred.device
    y_pred = y_pred.flatten()
    y_true = y_true.flatten().float()

    # Validate labels are binary (0 or 1)
    unique_labels = torch.unique(y_true)
    if not ((unique_labels == 0) | (unique_labels == 1)).all():
        raise ValueError("y_true must contain only binary 0 and 1 values")
    
    if prevalence is None:
        prevalence = y_true.mean().item()
    if not (0 < prevalence < 1):
        raise ValueError("Prevalence must be between 0 and 1")
    if not (0 <= min_recall <= 1):
        raise ValueError("min_recall must be between 0 and 1")
    
    # Prepare thresholds
    if thresholds is None:
        unique_preds = torch.unique(y_pred)
        # Add boundaries slightly outside min/max
        boundaries = torch.tensor([y_pred.min() - 1e-8, y_pred.max() + 1e-8], device=device)
        thresholds = torch.cat((boundaries[:1], unique_preds, boundaries[1:]))
    
    # Sort descending
    thresholds, _ = torch.sort(thresholds, descending=True)
    
    # Number of positives and negatives
    num_positive = (y_true == 1).sum().item()
    num_negative = (y_true == 0).sum().item()
    if num_positive == 0 or num_negative == 0:
        raise ValueError("y_true must have at least one positive and one negative sample")

    # Expand predictions and thresholds for vectorized thresholding
    # y_pred shape: (N,), thresholds shape: (T,)
    # After comparison: (T, N)
    y_pred_expand = y_pred.unsqueeze(0)  # shape (1, N)
    thresholds_expand = thresholds.unsqueeze(1)  # shape (T, 1)

    # Binary predictions per threshold (T, N)
    y_pred_binary = (y_pred_expand >= thresholds_expand).float()

    # True positives per threshold: sum over samples where pred=1 and true=1
    tp = (y_pred_binary * y_true).sum(dim=1)  # shape (T,)
    fp = (y_pred_binary * (1 - y_true)).sum(dim=1)  # shape (T,)
    fn = ((1 - y_pred_binary) * y_true).sum(dim=1)  # shape (T,)
    tn = ((1 - y_pred_binary) * (1 - y_true)).sum(dim=1)  # shape (T,)

    recall = tp / (tp + fn + 1e-12)  # add epsilon to avoid div0
    specificity = tn / (tn + fp + 1e-12)

    numerator = recall * prevalence
    denominator = recall * prevalence + (1 - specificity) * (1 - prevalence)
    ppv_corrected = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(denominator))

    # Find indices where recall >= min_recall
    valid_indices = torch.where(recall >= min_recall)[0]
    if valid_indices.numel() == 0:
        raise ValueError(f"No thresholds achieve recall >= {min_recall}. Max recall: {recall.max().item():.4f}")

    if mode == "max":
        # Max PPV_corrected among valid thresholds
        max_idx_in_valid = torch.argmax(ppv_corrected[valid_indices])
        best_idx = valid_indices[max_idx_in_valid]
    elif mode == "max_threshold":
        # Take the threshold with highest recall >= min_recall (i.e. first valid threshold)
        best_idx = valid_indices[0]
    else:
        raise ValueError(f"Unknown mode {mode}")

    best_ppv = ppv_corrected[best_idx].item()
    best_threshold = thresholds[best_idx].item()

    return best_ppv, best_threshold


def prevalence_corrected_ppv_at_90_recall_gpu(
    y_pred: Union[torch.Tensor, None],
    y_true: Union[torch.Tensor, None],
    prevalence: float = None,
    mode: str = "max",
) -> float:
    """
    Wrapper returning just the max prevalence-corrected PPV@90%Recall float on CPU.
    """
    ppv, _ = compute_prevalence_corrected_ppv_at_90_recall_gpu(y_pred, y_true, prevalence, mode=mode)
    return ppv


def _prepare_thresholds(y_pred: torch.Tensor, thresholds: Union[torch.Tensor, None] = None) -> torch.Tensor:
    """Build candidate thresholds from unique prediction values with boundary padding."""
    if thresholds is not None:
        return thresholds

    device = y_pred.device
    unique_preds = torch.unique(y_pred)
    boundaries = torch.tensor([y_pred.min() - 1e-8, y_pred.max() + 1e-8], device=device)
    thresholds = torch.cat((boundaries[:1], unique_preds, boundaries[1:]))
    thresholds, _ = torch.sort(thresholds, descending=True)
    return thresholds


def _confusion_counts_per_threshold(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    thresholds: Union[torch.Tensor, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return TP/FP/TN/FN per threshold and the threshold tensor."""
    device = y_pred.device
    y_pred = y_pred.flatten()
    y_true = y_true.flatten().float()

    unique_labels = torch.unique(y_true)
    if not ((unique_labels == 0) | (unique_labels == 1)).all():
        raise ValueError("y_true must contain only binary 0 and 1 values")

    num_positive = (y_true == 1).sum().item()
    num_negative = (y_true == 0).sum().item()
    if num_positive == 0 or num_negative == 0:
        raise ValueError("y_true must have at least one positive and one negative sample")

    thresholds = _prepare_thresholds(y_pred, thresholds)
    y_pred_expand = y_pred.unsqueeze(0)
    thresholds_expand = thresholds.unsqueeze(1)
    y_pred_binary = (y_pred_expand >= thresholds_expand).float()

    tp = (y_pred_binary * y_true).sum(dim=1)
    fp = (y_pred_binary * (1 - y_true)).sum(dim=1)
    fn = ((1 - y_pred_binary) * y_true).sum(dim=1)
    tn = ((1 - y_pred_binary) * (1 - y_true)).sum(dim=1)
    return tp, fp, tn, fn, thresholds


def compute_youden_threshold_gpu(
    y_pred: Union[torch.Tensor, None],
    y_true: Union[torch.Tensor, None],
    thresholds: Union[torch.Tensor, None] = None,
) -> float:
    """
    Select the threshold that maximizes Youden's J statistic (sensitivity + specificity - 1).

    Args:
        y_pred: Predicted probabilities/scores. Shape: (N,)
        y_true: True binary labels (0 or 1). Shape: (N,)
        thresholds: Optional tensor of candidate thresholds.

    Returns:
        Best threshold as a float on CPU.
    """
    if y_pred is None or y_true is None:
        raise ValueError("y_pred and y_true cannot be None")
    if not torch.is_tensor(y_pred) or not torch.is_tensor(y_true):
        raise ValueError("y_pred and y_true must be torch tensors")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} must match")

    tp, fp, tn, fn, thresholds = _confusion_counts_per_threshold(y_pred, y_true, thresholds)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    youden_index = recall + specificity - 1.0

    best_idx = torch.argmax(youden_index)
    return thresholds[best_idx].item()


def compute_extended_metrics(
    scores: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    threshold: float,
) -> Dict[str, float]:
    """
    Compute probability-based and class-based metrics for a split.

    Probability-based metrics (AUPRC, AUROC, PPV@90% Recall) are computed on
    continuous scores without binarization. Class-based metrics are computed on
    hard predictions obtained by thresholding scores at ``threshold``.
    """
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()

    scores = np.asarray(scores).flatten()
    y_true = np.asarray(y_true).flatten().astype(int)

    metrics: Dict[str, float] = {}

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true=y_true, y_score=scores))
        metrics["auprc"] = float(average_precision_score(y_true=y_true, y_score=scores))
        precisions, recalls, _ = precision_recall_curve(y_true=y_true, y_score=scores)
        metrics["ppv_at_90_recall"] = float(np.interp(0.9, recalls[::-1], precisions[::-1]))
    else:
        metrics["auroc"] = 0.0
        metrics["auprc"] = 0.0
        metrics["ppv_at_90_recall"] = 0.0

    y_pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    metrics["accuracy"] = (tp + tn) / max(tp + tn + fp + fn, 1)
    metrics["sensitivity"] = tp / max(tp + fn, 1)
    metrics["specificity"] = tn / max(tn + fp, 1)
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    return metrics
