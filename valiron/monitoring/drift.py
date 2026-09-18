"""Population Stability Index (PSI) drift detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np
import pandas as pd

from valiron.evaluate.metrics import MetricsResult


_PSI_MILD = 0.1
_PSI_SEVERE = 0.2
_PERF_ALERT_THRESHOLD = 0.05  # >5% relative drop → alert


@dataclass
class DriftReport:
    baseline_date: str
    current_date: str
    psi_scores: Dict[str, float]
    performance_delta: Dict[str, float]
    drift_detected: bool
    severity: str  # "none" | "mild" | "severe"
    alerts: List[str]


def _psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two 1-D arrays."""
    # Build bin edges from combined data
    combined = np.concatenate([baseline, current])
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)
    bins[0] -= 1e-9  # include left edge
    bins[-1] += 1e-9

    eps = 1e-8
    b_counts, _ = np.histogram(baseline, bins=bins)
    c_counts, _ = np.histogram(current, bins=bins)

    b_pct = b_counts / (len(baseline) + eps)
    c_pct = c_counts / (len(current) + eps)

    # Avoid log(0)
    b_pct = np.where(b_pct == 0, eps, b_pct)
    c_pct = np.where(c_pct == 0, eps, c_pct)

    psi = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
    return float(np.clip(psi, 0, None))


def monitor(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    baseline_metrics: MetricsResult,
    current_metrics: MetricsResult,
    threshold: float = 0.2,
) -> DriftReport:
    """Compute drift between baseline and current datasets.

    Args:
        baseline_data: Reference distribution DataFrame.
        current_data: Current production DataFrame.
        baseline_metrics: MetricsResult from baseline period.
        current_metrics: MetricsResult from current period.
        threshold: PSI threshold for severe drift (default 0.2).

    Returns:
        DriftReport with PSI scores, performance deltas, and alerts.
    """
    numeric_cols = baseline_data.select_dtypes(include=[np.number]).columns.tolist()
    # Only include columns present in both
    cols = [c for c in numeric_cols if c in current_data.columns]

    psi_scores: Dict[str, float] = {}
    for col in cols:
        psi_scores[col] = round(_psi(baseline_data[col].values, current_data[col].values), 6)

    # Severity based on max PSI
    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    if max_psi >= _PSI_SEVERE:
        severity = "severe"
        drift_detected = True
    elif max_psi >= _PSI_MILD:
        severity = "mild"
        drift_detected = True
    else:
        severity = "none"
        drift_detected = False

    # Performance delta
    bm, cm = baseline_metrics, current_metrics
    perf_metrics = {
        "accuracy": (bm.accuracy, cm.accuracy),
        "f1": (bm.f1, cm.f1),
        "sensitivity": (bm.sensitivity, cm.sensitivity),
        "specificity": (bm.specificity, cm.specificity),
    }
    perf_delta: Dict[str, float] = {}
    alerts: List[str] = []

    for name, (base, curr) in perf_metrics.items():
        delta = round(curr - base, 6)
        perf_delta[name] = delta
        if base > 0 and (base - curr) / base > _PERF_ALERT_THRESHOLD:
            pct = (base - curr) / base * 100
            alerts.append(f"PERFORMANCE: {name} dropped {pct:.1f}% (baseline={base:.4f}, current={curr:.4f})")

    return DriftReport(
        baseline_date="",
        current_date="",
        psi_scores=psi_scores,
        performance_delta=perf_delta,
        drift_detected=drift_detected,
        severity=severity,
        alerts=alerts,
    )


def generate_drift_report(drift: DriftReport) -> str:
    """Generate a markdown drift monitoring report.

    Returns:
        Markdown string.
    """
    lines = [
        "# Drift Monitoring Report",
        "",
        f"**Baseline date:** {drift.baseline_date or 'N/A'}  ",
        f"**Current date:** {drift.current_date or 'N/A'}  ",
        f"**Drift detected:** {'Yes' if drift.drift_detected else 'No'}  ",
        f"**Severity:** {drift.severity}",
        "",
        "---",
        "",
        "## PSI Scores (Population Stability Index)",
        "",
        "| Feature | PSI | Status |",
        "|---------|-----|--------|",
    ]

    for feat, psi in drift.psi_scores.items():
        if psi >= _PSI_SEVERE:
            status = "🔴 Severe drift"
        elif psi >= _PSI_MILD:
            status = "🟡 Mild drift"
        else:
            status = "🟢 No drift"
        lines.append(f"| {feat} | {psi:.4f} | {status} |")

    lines += [
        "",
        "---",
        "",
        "## Performance Delta",
        "",
        "| Metric | Delta |",
        "|--------|-------|",
    ]

    for metric, delta in drift.performance_delta.items():
        lines.append(f"| {metric} | {delta:+.4f} |")

    lines += ["", "---", "", "## Alerts", ""]

    if drift.alerts:
        for alert in drift.alerts:
            lines.append(f"- {alert}")
    else:
        lines.append("No alerts. All metrics within acceptable thresholds.")

    lines.append("")
    return "\n".join(lines)
