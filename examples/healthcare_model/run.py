"""CDSCO MDSW end-to-end demo — one command generates a full compliance report.

Usage:
    python examples/healthcare_model/run.py

Outputs:
    healthcare_compliance.html — full CDSCO report with metrics, subgroups, calibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import valiron
from valiron.evaluate.metrics import compute_metrics
from valiron.subgroups.analyzer import analyze_subgroups
from valiron.calibration.checker import check_calibration
from valiron.report.builder import ReportInput

# ── Synthetic dataset ────────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=600, n_features=15, n_informative=8,
    random_state=0, class_sep=1.2,
)
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

rng = np.random.default_rng(42)
sensitive_df = pd.DataFrame({
    "gender":    rng.choice(["M", "F"], size=len(X_test)),
    "age_group": rng.choice(["18-35", "36-55", "55+"], size=len(X_test)),
})

# ── Train model ──────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# ── Regulatory evaluation ────────────────────────────────────────────────────
eval_result = valiron.evaluate(
    model=model,
    X_test=X_test,
    y_test=y_test,
    regulation="cdsco_mdsw",
    use_case="diagnostic_imaging_aid",
    sensitive_features=["gender", "age_group"],
)

print(f"Regulation : CDSCO MDSW")
print(f"Compliant  : {eval_result.compliant}")
print(f"Score      : {eval_result.score:.1%}")
print(f"Passing    : {len(eval_result.passing_checks)} checks")
print(f"Failing    : {eval_result.failing_checks or 'none'}")

# ── Detailed metrics with bootstrap CIs ─────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
metrics = compute_metrics(y_test, y_pred, y_prob=y_prob, bootstrap_n=200, seed=0)
print(f"\nAccuracy   : {metrics.accuracy}  95%CI {metrics.accuracy_ci}")
print(f"Sensitivity: {metrics.sensitivity}  95%CI {metrics.sensitivity_ci}")
print(f"Specificity: {metrics.specificity}  95%CI {metrics.specificity_ci}")
print(f"AUC-ROC    : {metrics.auc}  95%CI {metrics.auc_ci}")

# ── Subgroup fairness ────────────────────────────────────────────────────────
subgroups = analyze_subgroups(y_test, y_pred, sensitive_df)
for feature, groups in subgroups.items():
    accs = {g: m["accuracy"] for g, m in groups.items()}
    print(f"\nSubgroup [{feature}]: {accs}")

# ── Calibration ──────────────────────────────────────────────────────────────
calibration = check_calibration(y_test, y_prob)
print(f"\nECE: {calibration.ece}  MCE: {calibration.mce}  "
      f"Well-calibrated: {calibration.well_calibrated}")

# ── Generate HTML report ─────────────────────────────────────────────────────
report_input = ReportInput(
    eval_result=eval_result,
    metrics=metrics,
    subgroups=subgroups,
    calibration=calibration,
)
out_path = valiron.report(report_input, format="html", output="healthcare_compliance.html")
print(f"\nReport saved → {out_path}")
