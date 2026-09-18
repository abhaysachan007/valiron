<div align="center">

# Valiron

[![PyPI version](https://img.shields.io/pypi/v/valiron.svg?color=blue)](https://pypi.org/project/valiron/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/valiron/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/abhaysachan007/valiron/actions/workflows/ci.yml/badge.svg)](https://github.com/abhaysachan007/valiron/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)](https://github.com/abhaysachan007/valiron)

[![CDSCO](https://img.shields.io/badge/CDSCO-MDSW%20compliant-orange.svg)](#supported-regulations)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Annex%20III-blue.svg)](#supported-regulations)
[![FDA](https://img.shields.io/badge/FDA-AI%2FML%20SaMD-red.svg)](#supported-regulations)
[![RBI](https://img.shields.io/badge/RBI-ML%20Model%20Risk-purple.svg)](#supported-regulations)

**The open-source toolkit for AI compliance evidence generation.**

</div>

---

> Deploying AI in healthcare or finance means months of manual compliance work, expensive consultants, and evidence packages built in spreadsheets. Valiron does it in a function call — generating audit-ready reports with bootstrap confidence intervals, subgroup fairness tables, calibration analysis, and version diff documents that regulators actually ask for.

<!-- demo.gif here — terminal showing valiron.evaluate() running → CDSCO report generated in 4.2s -->

---

## Quickstart

```bash
pip install valiron
```

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import valiron
from valiron.evaluate.metrics import compute_metrics
from valiron.subgroups.analyzer import analyze_subgroups
from valiron.calibration.checker import check_calibration
from valiron.report.builder import ReportInput

# --- Train ---
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=0
)
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# --- Evaluate against CDSCO MDSW ---
result = valiron.evaluate(
    model=model,
    X_test=X_test,
    y_test=y_test,
    regulation="cdsco_mdsw",
    use_case="diagnostic_imaging_aid",
    sensitive_features=["age_group"],
)
print(result.compliant)   # True
print(result.score)       # 1.0

# --- Metrics with 95% bootstrap CIs (1 000 resamples) ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
metrics = compute_metrics(y_test, y_pred, y_prob=y_prob, bootstrap_n=1000)

# --- Subgroup fairness ---
rng = np.random.default_rng(42)
sensitive_df = pd.DataFrame({"age_group": rng.choice(["<50", "50+"], size=len(y_test))})
subgroups = analyze_subgroups(y_test, y_pred, sensitive_df)

# --- Calibration ---
calibration = check_calibration(y_test, y_prob)

# --- Generate CDSCO-ready HTML report ---
valiron.report(
    ReportInput(eval_result=result, metrics=metrics,
                subgroups=subgroups, calibration=calibration),
    format="html",
    output="cdsco_report.html",
)
# → cdsco_report.html written in under 5 seconds. Zero data leaves your machine.
```

---

## What it generates

**Metrics table — with bootstrap confidence intervals**

```
Metric       Value    95% CI
-----------  -------  ---------------
accuracy     0.9650   (0.9350, 0.9850)
f1           0.9652   (0.9354, 0.9866)
sensitivity  0.9798   (0.9459, 1.0000)
specificity  0.9505   (0.9036, 0.9900)
AUC-ROC      0.9819   (0.9532, 0.9971)
```

**Subgroup fairness table**

```
Feature: gender
  Group  Accuracy  F1      Sensitivity  Specificity  N
  -----  --------  ------  -----------  -----------  ---
  M      0.9697    0.9677  0.9808       0.9583        99
  F      0.9604    0.9630  0.9787       0.9412       101

Feature: age_group
  Group   Accuracy  F1      N
  ------  --------  ------  ---
  18-35   1.0000    1.0000   77
  55+     0.9677    0.9643   62
  36-55   0.9180    0.9254   61   ← worst group

Accuracy disparity  gender:    0.0093
Accuracy disparity  age_group: 0.0820
```

**CDSCO compliance checks**

```
✓ CDSCO_MDSW_PERFORMANCE: accuracy=0.965 >=0.70
✓ CDSCO_MDSW_SENSITIVITY: sensitivity=0.980 >=0.60
✓ CDSCO_MDSW_SPECIFICITY: specificity=0.951 >=0.60
✓ CDSCO_MDSW_DATA_PRIVACY: local evaluation — no patient data transmitted
⚠ CDSCO_MDSW_REGISTRATION: MDSW registration with CDSCO required

compliant=True   score=1.0
```

**Calibration**

```
ECE: 0.0871   MCE: 0.4550   well_calibrated: True
```

---

## Modules

| Module | What it does | Key function |
|--------|-------------|-------------|
| `valiron.evaluate` | Run per-regulation compliance checks | `evaluate(model, X_test, y_test, regulation, use_case)` |
| `valiron.evaluate.metrics` | Binary metrics with bootstrap CIs | `compute_metrics(y_true, y_pred, y_prob, bootstrap_n)` |
| `valiron.subgroups` | Per-subgroup accuracy / F1 / sensitivity / specificity + disparity | `analyze_subgroups(y_true, y_pred, sensitive_df)` |
| `valiron.calibration` | ECE, MCE, reliability diagram data | `check_calibration(y_true, y_prob, n_bins)` |
| `valiron.report` | Render Jinja2 templates to HTML or PDF | `report(ReportInput, format, output)` |
| `valiron.acp` | Version-tracked model diffs with APPROVE / REVIEW / REJECT | `save_version()`, `diff_versions()`, `generate_acp_document()` |
| `valiron.monitoring` | PSI-based data drift + performance degradation alerts | `monitor(baseline_data, current_data, baseline_metrics, current_metrics)` |

---

## Supported regulations

| Regulation | Region | Status | Thresholds |
|-----------|--------|--------|-----------|
| CDSCO MDSW (Medical Device Software) | 🇮🇳 India | ✅ | accuracy ≥ 0.70, sensitivity ≥ 0.60, specificity ≥ 0.60 |
| EU AI Act Annex III (Regulation 2024/1689) | 🇪🇺 Europe | ✅ | accuracy ≥ 0.70, Articles 9/10/13/14/15 |
| FDA AI/ML SaMD Guidance | 🇺🇸 USA | ✅ | accuracy ≥ 0.75, PDCP flag, real-world monitoring flag |
| RBI ML Model Risk Framework | 🇮🇳 India | ✅ | accuracy ≥ 0.65, explainability required |

```python
SUPPORTED_REGULATIONS = ("cdsco_mdsw", "eu_ai_act", "fda_samd", "rbi_ml_risk")
```

---

## Architecture

```
                   ┌──────────────────────────┐
                   │   your model + test data  │
                   └────────────┬─────────────┘
                                │
                   ┌────────────▼─────────────┐
                   │     valiron.evaluate()   │
                   │  regulation checks       │
                   │  use_case heuristics     │
                   └──┬──────────┬────────┬───┘
                      │          │        │
          ┌───────────▼──┐  ┌────▼───┐  ┌─▼──────────┐
          │   metrics    │  │subgroups│  │ calibration │
          │ (bootstrap   │  │fairness │  │  ECE / MCE  │
          │   95% CIs)   │  │analysis │  │             │
          └───────────┬──┘  └────┬───┘  └─┬──────────┘
                      └────┬─────┘         │
                           │               │
                ┌──────────▼───────────────▼──┐
                │         ReportInput          │
                └──────────┬──────────────────┘
                           │
                ┌──────────▼──────────────────┐
                │       valiron.report()       │
                │  Jinja2 → HTML / PDF         │
                │  cdsco.html.j2               │
                │  eu_ai_act.html.j2           │
                └─────────────────────────────┘

ACP (parallel pipeline):
  save_version() → .valiron/<id>.json
  diff_versions() → VersionDiff → generate_acp_document() → Markdown
```

---

## Algorithm Change Protocol (ACP)

CDSCO MDSW § 5 and FDA PDCP mandate that every algorithm change be documented before redeployment. No other open-source compliance tool generates these documents automatically.

```python
from valiron.acp import save_version, diff_versions, generate_acp_document

# After training v1 — persist a snapshot
v1 = save_version(model_v1, {
    "id":             "rf-v1",
    "name":           "DiagnosticRF",
    "version_string": "1.0.0",
    "metrics":        metrics_v1,    # MetricsResult from compute_metrics()
    "subgroups":      subgroups_v1,  # dict from analyze_subgroups()
    "regulation":     "cdsco_mdsw",
    "notes":          "initial production model, AIIMS dataset",
})
# Stored as .valiron/rf-v1.json

# Retrain, re-evaluate, save v2
v2 = save_version(model_v2, {"id": "rf-v2", "version_string": "1.1.0", ...})

# Generate the ACP document
diff = diff_versions(v1, v2)
print(diff.regression_detected)    # True — metrics dropped > 2%
doc  = generate_acp_document(diff)
```

**ACP output (Markdown, ready to attach to regulatory submission):**

```markdown
# Algorithm Change Protocol (ACP) Document

**Baseline version:** `rf-v1`
**Candidate version:** `rf-v2`

## 2. Performance Comparison Table

| Metric      | Change  | Direction  |
|-------------|---------|------------|
| accuracy    | -0.0440 | ▼ degraded |
| f1          | -0.0456 | ▼ degraded |
| sensitivity | -0.0539 | ▼ degraded |
| specificity | -0.0347 | ▼ degraded |

## 4. Regression Assessment

Regressions detected:
- accuracy dropped 4.5%   (baseline=0.9650, candidate=0.9210)
- sensitivity dropped 5.5% (baseline=0.9798, candidate=0.9259)

## 5. Recommendation

**REJECT** — multiple metric regressions detected.
Candidate does not meet baseline.
```

**Recommendation logic:** zero regressions → `APPROVE` · one → `REVIEW` · multiple → `REJECT`.  
Regression = any metric drops > 2% relative to baseline.

---

## Drift monitoring

```python
from valiron.monitoring import monitor, generate_drift_report

drift = monitor(
    baseline_data=X_train_df,           # pandas DataFrame, reference distribution
    current_data=X_production_df,       # current production window
    baseline_metrics=metrics_train,     # MetricsResult
    current_metrics=metrics_production,
    threshold=0.2,                      # PSI threshold for severe drift
)

print(drift.severity)      # "none" | "mild" | "severe"
print(drift.psi_scores)    # {"feature_0": 0.031, "feature_1": 0.214, ...}
print(drift.alerts)        # ["PERFORMANCE: accuracy dropped 6.2% ..."]

print(generate_drift_report(drift))    # markdown report
```

**PSI:** `< 0.10` no drift · `0.10–0.20` mild · `> 0.20` severe  
**Performance alert:** any metric drops > 5% relative to baseline.

---

## Installation

```bash
# Core (scikit-learn models, HTML reports, all 4 regulations)
pip install valiron

# PDF export
pip install "valiron[pdf]"

# ONNX model support
pip install "valiron[onnx]"

# PyTorch support
pip install "valiron[torch]"

# Everything
pip install "valiron[pdf,onnx,torch]"
```

**Requirements:** Python 3.9+ · numpy · pandas · scikit-learn · scipy · jinja2

---

## Supported frameworks

| Framework | Usage |
|-----------|-------|
| scikit-learn | Pass any fitted estimator with `.predict()` / `.predict_proba()` |
| PyTorch | Wrap in a callable: `lambda X: model(torch.tensor(X)).argmax(1).numpy()` |
| ONNX | Use `onnxruntime.InferenceSession`; pass predictions as numpy arrays |
| Raw numpy | Pass `y_pred` and `y_prob` arrays directly to `compute_metrics()` |

---

## Contributing

```bash
git clone https://github.com/abhaysachan007/valiron
cd valiron
pip install -e ".[dev]"
pytest --cov=valiron
```

Open an issue first for anything beyond a small bug fix.  
We especially want feedback from **clinical AI teams** and **regulatory consultants** — the frameworks are complex, real-world edge cases matter.

---

## Roadmap

| Version | Status | What |
|---------|--------|------|
| v0.1.0 | ✅ Released | `evaluate()`, CDSCO / EU / FDA / RBI checks, bootstrap CIs, subgroups, calibration, HTML reports |
| v0.2.0 | ✅ Released | ACP version tracking, EU AI Act Jinja2 template (Articles 9–15), PSI drift monitoring |
| v0.3.0 | 🔄 Planned | ISO 42001, CI/CD integration (GitHub Actions), SEBI AI governance |
| v1.0.0 | 🔄 Planned | Stable API, HIPAA checks, VS Code extension, multi-model comparison dashboard |

---

## Citation

```bibtex
@software{valiron2025,
  author  = {Sachan, Abhay},
  title   = {Valiron: Open-Source AI Compliance Evidence Generation},
  year    = {2025},
  url     = {https://github.com/abhaysachan007/valiron},
  version = {0.2.0},
  license = {MIT}
}
```

---

## License

[MIT](LICENSE) — use it, modify it, ship it. Zero data leaves your system. Ever.
