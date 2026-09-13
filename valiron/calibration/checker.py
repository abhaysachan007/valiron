"""Probability calibration checker — ECE, MCE, reliability data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    well_calibrated: bool


def check_calibration(y_true: Any, y_prob: Any, n_bins: int = 10) -> CalibrationResult:
    """Compute ECE and MCE for probability calibration assessment.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equal-width bins (default 10).

    Returns:
        CalibrationResult with ece, mce, and well_calibrated flag (ece < 0.10).
    """
    _, frac_pos, counts = reliability_data(y_true, y_prob, n_bins=n_bins)
    mean_pred_arr, _, _ = reliability_data(y_true, y_prob, n_bins=n_bins)

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        if mask.sum() == 0:
            continue
        gap = abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
        ece += gap * mask.sum() / n
        mce = max(mce, gap)

    return CalibrationResult(
        ece=round(float(ece), 4),
        mce=round(float(mce), 4),
        well_calibrated=bool(ece < 0.1),
    )


def reliability_data(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
) -> Tuple[list, list, list]:
    """Compute bin-level data for a reliability (calibration) diagram.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equal-width bins (default 10).

    Returns:
        Tuple of (mean_predicted_prob, fraction_positives, counts) — one entry
        per non-empty bin. Suitable for plotting a reliability diagram.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    mean_pred: list = []
    frac_pos: list = []
    counts: list = []

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred.append(round(float(np.mean(y_prob[mask])), 4))
        frac_pos.append(round(float(np.mean(y_true[mask])), 4))
        counts.append(n)

    return mean_pred, frac_pos, counts
