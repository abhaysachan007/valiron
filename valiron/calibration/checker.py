"""Probability calibration checker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        if mask.sum() == 0:
            continue
        gap = abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
        ece += gap * mask.sum() / n
        mce = max(mce, gap)

    return CalibrationResult(
        ece=round(ece, 4),
        mce=round(mce, 4),
        well_calibrated=ece < 0.1,
    )
