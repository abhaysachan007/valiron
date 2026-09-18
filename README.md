# Valiron

[![Tests](https://github.com/abhaysachan007/valiron/actions/workflows/ci.yml/badge.svg)](https://github.com/abhaysachan007/valiron/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/abhaysachan007/valiron/branch/main/graph/badge.svg)](https://codecov.io/gh/abhaysachan007/valiron)
[![PyPI](https://img.shields.io/pypi/v/valiron.svg)](https://pypi.org/project/valiron/)
[![Python](https://img.shields.io/pypi/pyversions/valiron.svg)](https://pypi.org/project/valiron/)

**AI Regulatory Compliance Validation for Python**

Valiron is a Python library that helps ML engineers and data scientists validate their AI/ML models against global regulatory frameworks — before deployment.

---

## Problem

Deploying AI in regulated industries (healthcare, finance, HR) requires compliance with frameworks like the EU AI Act, FDA AI/ML SaMD guidance, CDSCO MDSW, and RBI ML Model Risk guidelines. Manual compliance checks are slow, inconsistent, and expensive.

## Solution

Valiron automates compliance validation with a simple API:

```python
import valiron

result = valiron.evaluate(
    model=my_sklearn_model,
    X_test=X_test,
    y_test=y_test,
    regulation="eu_ai_act",
    use_case="medical_diagnosis"
)

report = valiron.report(result, format="pdf")
```

---

## Compliance Coverage

| Regulation | Status | Key Checks |
|---|---|---|
| EU AI Act Annex III | ✅ | High-risk classification, transparency, human oversight |
| FDA AI/ML SaMD | ✅ | Predetermined change control, performance monitoring |
| CDSCO MDSW (India) | ✅ | Software as medical device validation |
| RBI ML Model Risk | ✅ | Model governance, bias detection, explainability |

---

## Quickstart

```bash
pip install valiron
```

```python
import valiron
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Validate compliance
result = valiron.evaluate(
    model=model,
    X_test=X_test,
    y_test=y_test,
    regulation="cdsco_mdsw",
    use_case="diagnostic_aid",
    sensitive_features=["age", "gender"]
)

# Check results
print(result.compliant)          # True/False
print(result.score)              # 0.0 - 1.0
print(result.failing_checks)    # list of failed requirements

# Generate report
valiron.report(result, format="html", output="compliance_report.html")
```

---

## Supported Frameworks

- scikit-learn
- PyTorch
- ONNX
- Raw predictions (numpy arrays)

---

## Data Privacy Architecture

- **No data leaves your machine.** All validation runs locally.
- No model weights are transmitted.
- Reports are generated on-device.
- Optional: anonymize test data before validation with `valiron.anonymize()`.

---

## Algorithm Change Protocol (ACP) — v0.2

Track model versions and detect regressions before deployment.

```python
from valiron.acp import save_version, load_version, diff_versions, generate_acp_document
from valiron.evaluate.metrics import MetricsResult

# Save a version after training
mv1 = save_version(model_v1, {
    "id": "v1.0.0",
    "name": "risk-model",
    "version_string": "1.0.0",
    "metrics": metrics_v1,      # MetricsResult from compute_metrics()
    "subgroups": subgroups,     # dict from analyze_subgroups()
    "regulation": "eu_ai_act",
    "notes": "initial production model",
})

# Later: save candidate version and diff
mv2 = save_version(model_v2, {"id": "v1.1.0", ...})
diff = diff_versions(mv1, mv2)

print(diff.regression_detected)   # True/False
print(generate_acp_document(diff)) # markdown ACP report

# List all saved versions
from valiron.acp import list_versions
print(list_versions())
```

Versions are stored as JSON in `.valiron/` in your working directory. Regression = any metric drops > 2% relative to baseline.

---

## EU AI Act Report — v0.2

Dedicated HTML report for Regulation (EU) 2024/1689 Annex III high-risk AI systems, covering Articles 9, 10, 13, 14, and 15.

```python
from valiron.report.builder import ReportInput, report

ri = ReportInput(
    eval_result=valiron.evaluate(model, X_test, y_test, regulation="eu_ai_act", ...),
    metrics=metrics,
    subgroups=subgroups,
    calibration=calibration,
)

html = report(ri, format="html", output="eu_ai_act_compliance.html")
```

The template maps Valiron data to EU AI Act obligations:
- **Article 9** (Risk Management) — compliance check results
- **Article 10** (Data Governance) — subgroup bias analysis
- **Article 13** (Transparency) — warnings and audit trail
- **Article 14** (Human Oversight) — calibration-aware oversight guidance
- **Article 15** (Accuracy & Robustness) — metrics table with 95% CIs

---

## Drift Monitoring — v0.2

Detect data and performance drift in production using Population Stability Index (PSI).

```python
from valiron.monitoring import monitor, generate_drift_report

drift = monitor(
    baseline_data=X_train_df,      # pandas DataFrame
    current_data=X_production_df,
    baseline_metrics=metrics_train,
    current_metrics=metrics_production,
    threshold=0.2,                  # PSI threshold for severe drift
)

print(drift.drift_detected)        # True/False
print(drift.severity)              # "none" | "mild" | "severe"
print(drift.psi_scores)            # {"feature": psi_value, ...}
print(drift.alerts)                # performance degradation alerts

print(generate_drift_report(drift)) # markdown report
```

PSI thresholds: < 0.1 = no drift, 0.1–0.2 = mild, > 0.2 = severe.
Performance alert fires when any metric drops > 5% relative to baseline.

---

## Roadmap

- [ ] ISO 42001 (AI Management Systems)
- [ ] HIPAA AI compliance checks
- [ ] SEBI AI governance framework
- [ ] CI/CD integration (GitHub Actions, GitLab CI)
- [ ] VS Code extension

---

## License

MIT — see [LICENSE](LICENSE)
