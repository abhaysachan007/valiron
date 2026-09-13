"""Subgroup fairness analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def analyze_subgroups(
    y_true: Any,
    y_pred: Any,
    sensitive_df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-subgroup accuracy for each sensitive feature."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cols = features or list(sensitive_df.columns)
    results: Dict[str, Dict[str, float]] = {}

    for col in cols:
        group_accs: Dict[str, float] = {}
        for val in sensitive_df[col].unique():
            mask = sensitive_df[col] == val
            if mask.sum() == 0:
                continue
            acc = float(np.mean(y_true[mask] == y_pred[mask]))
            group_accs[str(val)] = round(acc, 4)
        results[col] = group_accs

    return results
