"""Subgroup fairness analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix


@dataclass
class SubgroupResult:
    accuracy: float
    f1: float
    sensitivity: float
    specificity: float
    n: int


def _compute_group_metrics(yt: np.ndarray, yp: np.ndarray) -> Dict[str, Any]:
    n = len(yt)
    acc = round(float(np.mean(yt == yp)), 4)
    f1 = round(float(f1_score(yt, yp, zero_division=0)), 4)

    classes = np.unique(np.concatenate([yt, yp]))
    if len(classes) < 2:
        sens, spec = (1.0, 0.0) if (len(classes) == 1 and classes[0] == 1) else (0.0, 1.0)
    else:
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {"accuracy": acc, "f1": f1, "sensitivity": round(sens, 4), "specificity": round(spec, 4), "n": n}


def analyze_subgroups(
    y_true: Any,
    y_pred: Any,
    sensitive_df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute per-subgroup metrics (accuracy, f1, sensitivity, specificity, n) for each sensitive feature.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        sensitive_df: DataFrame with one column per sensitive attribute.
        features: Subset of columns to analyse; None = all columns.

    Returns:
        Nested dict: {feature -> {group_value -> metric_dict}}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cols = features or list(sensitive_df.columns)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for col in cols:
        group_metrics: Dict[str, Dict[str, Any]] = {}
        for val in sensitive_df[col].unique():
            mask = (sensitive_df[col] == val).values
            if mask.sum() == 0:
                continue
            group_metrics[str(val)] = _compute_group_metrics(y_true[mask], y_pred[mask])
        results[col] = group_metrics

    return results


def disparity_report(
    y_true: Any,
    y_pred: Any,
    sensitive_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    metric: str = "accuracy",
) -> Dict[str, Dict[str, Any]]:
    """Compute worst-group disparity (max - min) for each sensitive feature.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        sensitive_df: DataFrame with sensitive attribute columns.
        features: Subset of columns; None = all.
        metric: Which metric to compute disparity on (default 'accuracy').

    Returns:
        {feature -> {accuracy_disparity, worst_group, best_group, group_values}}.
    """
    subgroup_results = analyze_subgroups(y_true, y_pred, sensitive_df, features)
    report: Dict[str, Dict[str, Any]] = {}

    for col, groups in subgroup_results.items():
        scores = {val: metrics[metric] for val, metrics in groups.items()}
        best = max(scores, key=scores.__getitem__)
        worst = min(scores, key=scores.__getitem__)
        disparity = round(scores[best] - scores[worst], 4)
        report[col] = {
            f"{metric}_disparity": disparity,
            "worst_group": worst,
            "best_group": best,
            "group_values": scores,
        }

    return report
