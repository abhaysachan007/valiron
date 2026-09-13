"""Binary classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


Interval = Tuple[float, float]
_SUPPORTED = ("accuracy", "f1", "sensitivity", "specificity", "auc")


@dataclass
class MetricsResult:
    accuracy: float
    f1: float
    sensitivity: float
    specificity: float
    auc: Optional[float] = None
    accuracy_ci: Optional[Interval] = None
    f1_ci: Optional[Interval] = None
    sensitivity_ci: Optional[Interval] = None
    specificity_ci: Optional[Interval] = None
    auc_ci: Optional[Interval] = None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    bootstrap_n: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> MetricsResult:
    """Compute binary classification metrics with optional bootstrap CIs.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        y_prob: Predicted probabilities for the positive class. Required for AUC.
        bootstrap_n: Bootstrap resamples for CI estimation. 0 disables CIs.
        ci_level: Confidence level for intervals, default 0.95.
        seed: Random seed for reproducibility.

    Returns:
        MetricsResult with point estimates and, when bootstrap_n > 0, CIs.

    Raises:
        ValueError: If y_true contains more than 2 unique classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    if len(classes) > 2:
        raise ValueError(f"compute_metrics requires binary labels; got {classes.tolist()}")

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    sens, spec = _sens_spec(y_true, y_pred)
    auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None

    result = MetricsResult(
        accuracy=round(acc, 4),
        f1=round(f1, 4),
        sensitivity=round(sens, 4),
        specificity=round(spec, 4),
        auc=round(auc, 4) if auc is not None else None,
    )

    if bootstrap_n > 0:
        _attach_cis(result, y_true, y_pred, y_prob, bootstrap_n, ci_level, seed)

    return result


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    y_prob: Optional[np.ndarray] = None,
    n: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Compute a bootstrap percentile confidence interval for one metric.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        metric: One of 'accuracy', 'f1', 'sensitivity', 'specificity', 'auc'.
        y_prob: Predicted probabilities; required when metric='auc'.
        n: Number of bootstrap resamples.
        ci_level: Confidence level, default 0.95.
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) tuple rounded to 4 decimal places.

    Raises:
        ValueError: If metric is not in the supported set.
    """
    if metric not in _SUPPORTED:
        raise ValueError(f"Unknown metric '{metric}'. Supported: {_SUPPORTED}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp = y_true[idx], y_pred[idx]
        if metric == "accuracy":
            scores.append(accuracy_score(yt, yp))
        elif metric == "f1":
            scores.append(f1_score(yt, yp, zero_division=0))
        elif metric == "sensitivity":
            s, _ = _sens_spec(yt, yp)
            scores.append(s)
        elif metric == "specificity":
            _, s = _sens_spec(yt, yp)
            scores.append(s)
        elif metric == "auc" and y_prob is not None:
            yp_b = y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                scores.append(roc_auc_score(yt, yp_b))
            except Exception:
                continue

    alpha = (1.0 - ci_level) / 2
    arr = np.asarray(scores)
    return round(float(np.percentile(arr, alpha * 100)), 4), round(float(np.percentile(arr, (1 - alpha) * 100)), 4)


def _sens_spec(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (sensitivity, specificity) from a binary confusion matrix."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) < 2:
        return (1.0, 0.0) if classes[0] == 1 else (0.0, 1.0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens), float(spec)


def _attach_cis(
    result: MetricsResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    n: int,
    ci_level: float,
    seed: int,
) -> None:
    """Populate all CI fields on *result* in-place."""
    kw: dict = dict(n=n, ci_level=ci_level, seed=seed)
    result.accuracy_ci    = bootstrap_ci(y_true, y_pred, "accuracy",    **kw)
    result.f1_ci          = bootstrap_ci(y_true, y_pred, "f1",          **kw)
    result.sensitivity_ci = bootstrap_ci(y_true, y_pred, "sensitivity", **kw)
    result.specificity_ci = bootstrap_ci(y_true, y_pred, "specificity", **kw)
    if y_prob is not None:
        result.auc_ci = bootstrap_ci(y_true, y_pred, "auc", y_prob=y_prob, **kw)
