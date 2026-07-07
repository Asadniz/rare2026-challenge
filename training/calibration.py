"""Affine probability calibration via psrcal (matches recalibrate_files.py)."""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

DEFAULT_CALIBRATION_PRIORS = (100 / 101, 1 / 101)


def fit_calibrated_positive_scores(
    logits: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    priors: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit affine calibration on validation logits and return calibrated positive-class scores.

    Uses the same settings as training/recalibrate_files.py (AffineCalLogLoss, challenge priors).
    """
    from psrcal.calibration import AffineCalLogLoss, calibrate

    if priors is None:
        priors = list(DEFAULT_CALIBRATION_PRIORS)

    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels)
    if logits_t.ndim != 2 or logits_t.shape[1] < 2:
        raise ValueError(f"Expected logits shape (N, 2), got {tuple(logits_t.shape)}")

    calibrated, (t, b) = calibrate(
        trnscores=logits_t,
        trnlabels=labels_t,
        tstscores=logits_t,
        calclass=AffineCalLogLoss,
        bias=True,
        priors=list(priors),
        quiet=True,
    )

    calibrated_t = torch.as_tensor(calibrated)
    if calibrated_t.ndim == 2 and calibrated_t.shape[1] >= 2:
        pos_scores = calibrated_t[:, 1]
    else:
        pos_scores = calibrated_t.squeeze()

    t_val = float(t.item()) if torch.is_tensor(t) else float(t)
    b_tensor = torch.as_tensor(b).flatten()
    b_val = float(b_tensor[0].item())

    return pos_scores.detach().cpu().numpy(), t_val, b_val
