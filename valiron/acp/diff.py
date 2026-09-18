"""ACP diff — compare two ModelVersions and generate ACP documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from valiron.acp.version import ModelVersion

_REGRESSION_THRESHOLD = 0.02  # >2% drop = regression


@dataclass
class VersionDiff:
    baseline: str
    candidate: str
    metric_changes: Dict[str, float]
    subgroup_changes: Dict[str, Any]
    regression_detected: bool
    regression_details: List[str]


def diff_versions(v1: ModelVersion, v2: ModelVersion) -> VersionDiff:
    """Compare two ModelVersions and detect regressions.

    Args:
        v1: Baseline ModelVersion.
        v2: Candidate ModelVersion.

    Returns:
        VersionDiff with metric deltas and regression assessment.
    """
    m1, m2 = v1.metrics, v2.metrics
    scalar_metrics = {
        "accuracy": (m1.accuracy, m2.accuracy),
        "f1": (m1.f1, m2.f1),
        "sensitivity": (m1.sensitivity, m2.sensitivity),
        "specificity": (m1.specificity, m2.specificity),
    }

    changes: Dict[str, float] = {}
    details: List[str] = []

    for name, (base, cand) in scalar_metrics.items():
        delta = cand - base
        changes[name] = round(delta, 6)
        if base > 0 and (base - cand) / base > _REGRESSION_THRESHOLD:
            pct = (base - cand) / base * 100
            details.append(f"{name} dropped {pct:.1f}% (baseline={base:.4f}, candidate={cand:.4f})")

    # Subgroup changes: compare common keys
    sg_changes: Dict[str, Any] = {}
    for feat in set(v1.subgroups) & set(v2.subgroups):
        sg_changes[feat] = {
            grp: {
                metric: round(v2.subgroups[feat].get(grp, {}).get(metric, 0)
                               - v1.subgroups[feat].get(grp, {}).get(metric, 0), 4)
                for metric in ["accuracy", "f1", "sensitivity", "specificity"]
                if metric in v1.subgroups[feat].get(grp, {})
            }
            for grp in set(v1.subgroups[feat]) & set(v2.subgroups[feat])
        }

    return VersionDiff(
        baseline=v1.id,
        candidate=v2.id,
        metric_changes=changes,
        subgroup_changes=sg_changes,
        regression_detected=bool(details),
        regression_details=details,
    )


def generate_acp_document(diff: VersionDiff) -> str:
    """Generate a markdown ACP document from a VersionDiff.

    Sections: 1. Change Description, 2. Performance Comparison Table,
              3. Subgroup Impact Analysis, 4. Regression Assessment,
              5. Recommendation (APPROVE/REVIEW/REJECT).

    Returns:
        Markdown string.
    """
    lines = [
        "# Algorithm Change Protocol (ACP) Document",
        "",
        f"**Baseline version:** `{diff.baseline}`  ",
        f"**Candidate version:** `{diff.candidate}`",
        "",
        "---",
        "",
        "## 1. Change Description",
        "",
        f"Comparison between baseline `{diff.baseline}` and candidate `{diff.candidate}`.",
        "",
        "---",
        "",
        "## 2. Performance Comparison Table",
        "",
        "| Metric | Change | Direction |",
        "|--------|--------|-----------|",
    ]

    for metric, delta in diff.metric_changes.items():
        direction = "▲ improved" if delta > 0 else ("▼ degraded" if delta < 0 else "→ unchanged")
        lines.append(f"| {metric} | {delta:+.4f} | {direction} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Subgroup Impact Analysis",
        "",
    ]

    if diff.subgroup_changes:
        for feat, groups in diff.subgroup_changes.items():
            lines.append(f"**Feature: {feat}**")
            for grp, deltas in groups.items():
                lines.append(f"- {grp}: " + ", ".join(f"{k}={v:+.4f}" for k, v in deltas.items()))
        lines.append("")
    else:
        lines += ["No subgroup data available.", ""]

    lines += [
        "---",
        "",
        "## 4. Regression Assessment",
        "",
    ]

    if diff.regression_detected:
        lines.append("**Regressions detected:**")
        for detail in diff.regression_details:
            lines.append(f"- {detail}")
    else:
        lines.append("No regressions detected (no metric dropped more than 2%).")

    lines += [
        "",
        "---",
        "",
        "## 5. Recommendation",
        "",
    ]

    if not diff.regression_detected:
        recommendation = "**APPROVE** — no regressions detected, candidate meets performance baseline."
    elif len(diff.regression_details) == 1:
        recommendation = "**REVIEW** — one metric regression detected. Manual review required before deployment."
    else:
        recommendation = "**REJECT** — multiple metric regressions detected. Candidate does not meet baseline."

    lines.append(recommendation)
    lines.append("")

    return "\n".join(lines)
