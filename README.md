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

## Roadmap

- [ ] ISO 42001 (AI Management Systems)
- [ ] HIPAA AI compliance checks
- [ ] SEBI AI governance framework
- [ ] CI/CD integration (GitHub Actions, GitLab CI)
- [ ] VS Code extension

---

## License

MIT — see [LICENSE](LICENSE)
