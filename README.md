<div align="center">

<!-- Logo here -->

# Valiron

> Generate regulatory-grade AI validation evidence in minutes, not months.

[![PyPI version](https://img.shields.io/pypi/v/valiron.svg?color=blue)](https://pypi.org/project/valiron/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/valiron/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/abhaysachan007/valiron/actions/workflows/ci.yml/badge.svg)](https://github.com/abhaysachan007/valiron/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)](https://github.com/abhaysachan007/valiron)
[![Downloads](https://img.shields.io/pypi/dm/valiron.svg?color=blue)](https://pypi.org/project/valiron/)

[![CDSCO](https://img.shields.io/badge/CDSCO-MDSW%202022-orange.svg)](#supported-regulations)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Annex%20III-blue.svg)](#supported-regulations)
[![FDA](https://img.shields.io/badge/FDA-AI%2FML%20SaMD-red.svg)](#supported-regulations)
[![RBI](https://img.shields.io/badge/RBI-ML%20Model%20Risk-purple.svg)](#supported-regulations)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhaysachan007/valiron/blob/main/colab_demo.ipynb)

<!--
demo.gif — shows terminal running valiron.evaluate() → CDSCO report in 4.2s
Coming soon
-->

</div>

---

Clinical AI companies spend 3–6 months and ₹15–40 lakh proving their models are safe. CDSCO, FDA, and EU AI Act all demand the same evidence. Valiron generates it automatically.

---

## Quickstart

```bash
pip install valiron
```

```python
import valiron
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Your model and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier().fit(X_train, y_train)

# Subgroup metadata
metadata = pd.DataFrame({
    'age_group': ['18-35', '36-55', '55+', ...],
    'gender':    ['M', 'F', 'M', ...]
})

# One command → full evidence pack
result = valiron.evaluate(
    model=model,
    X_test=X_test,
    y_test=y_test,
    regulation='cdsco_mdsw',
    sensitive_features=['age_group', 'gender']
)

# Generate regulator-formatted report
from valiron.report.builder import ReportInput, report
report(
    ReportInput(eval_result=result),
    format='html',
    output='cdsco_report.html'
)

# Output:
# ✅ COMPLIANT — CDSCO MDSW | score=1.000
# ✅ Report: cdsco_report.html (8.1 KB)
```

---

## What it generates

### Performance Analysis

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | **0.965** | 0.935–0.985 |
| Sensitivity | **0.980** | 0.946–1.000 |
| Specificity | **0.951** | 0.904–0.990 |
| AUC-ROC | **0.982** | 0.953–0.997 |

### CDSCO Compliance Checks

| Check | Status |
|-------|--------|
| CDSCO_MDSW_PERFORMANCE | ✅ PASS |
| CDSCO_MDSW_SENSITIVITY | ✅ PASS |
| CDSCO_MDSW_SPECIFICITY | ✅ PASS |
| CDSCO_MDSW_DATA_PRIVACY | ✅ PASS |

---

## Modules

| Module | What it does | Key function |
|--------|-------------|-------------|
| `evaluate` | Metrics + Bootstrap CIs | `compute_metrics()` |
| `subgroups` | Bias detection across demographics | `analyze_subgroups()` |
| `calibration` | ECE, MCE, reliability diagram | `check_calibration()` |
| `report` | PDF/HTML generation | `report()` |
| `acp` | Algorithm Change Protocol | `diff_versions()` |
| `monitoring` | PSI drift detection | `monitor()` |

---

## Supported regulations

| Regulation | Region | Coverage | Status |
|-----------|--------|----------|--------|
| CDSCO MDSW 2022 | 🇮🇳 India | §5–§8 | ✅ v0.2 |
| EU AI Act Annex III | 🇪🇺 Europe | Art. 9–15 | ✅ v0.2 |
| FDA AI/ML SaMD | 🇺🇸 USA | PCCP | 🔄 v0.3 |
| RBI ML Model Risk | 🇮🇳 India | SR 11-7 | 🔄 v0.3 |

---

## Architecture

```
┌─────────────────────────────────────┐
│         Your AI Model + Data        │
└──────────────┬──────────────────────┘
               │
     valiron.evaluate()
               │
    ┌──────────▼──────────┐
    │                     │
  ┌─▼──────────┐  ┌───────▼──────┐
  │  evaluate/ │  │  subgroups/  │
  │  metrics   │  │  bias detect │
  └─────┬──────┘  └──────┬───────┘
        │                │
  ┌─────▼──────┐  ┌──────▼───────┐
  │calibration/│  │   report/    │
  │  ECE / MCE │  │  CDSCO HTML  │
  └────────────┘  └──────────────┘
        │
  ┌─────▼──────────────────────────┐
  │    acp/ + monitoring/          │
  │  Version diff + PSI drift      │
  └────────────────────────────────┘
```

---

## Algorithm Change Protocol (ACP)

Mandatory under CDSCO MDSW §5 and FDA PDCP for AI/ML SaMD updates. No other open-source compliance tool has automated this.

```python
from valiron.acp import save_version, diff_versions, generate_acp_document

# Save baseline snapshot
v1 = save_version(model_v1, {
    'id': 'rf-v1', 'name': 'DiagnosticRF', 'version_string': '1.0.0',
    'metrics': metrics_v1, 'regulation': 'cdsco_mdsw',
    'notes': 'Initial production model',
})

# After retraining, save candidate
v2 = save_version(model_v2, {'id': 'rf-v2', 'version_string': '1.1.0', ...})

# Generate the ACP document
diff = diff_versions(v1, v2)
doc  = generate_acp_document(diff)
print(doc)
```

**ACP output (Markdown, ready to attach to regulatory submission):**

```markdown
# Algorithm Change Protocol (ACP) Document

**Baseline:** rf-v1  |  **Candidate:** rf-v2

## Performance Comparison

| Metric      | Change  | Direction  |
|-------------|---------|------------|
| accuracy    | -0.0440 | ▼ degraded |
| sensitivity | -0.0539 | ▼ degraded |

## Recommendation

**REJECT** — multiple metric regressions detected.
```

Recommendation logic: zero regressions → `APPROVE` · one → `REVIEW` · multiple → `REJECT`.

---

## Data privacy

Your data never leaves your system.

```
Your server          Valiron
──────────           ────────────────────
Patient data  →     Runs entirely locally
Model weights        ↓
                    Only metrics computed
                    ↓
                    Report generated
                    on your machine
                    ↓
                    Zero data transmitted
```

---

## Installation

```bash
# Standard
pip install valiron

# With ONNX support
pip install "valiron[onnx]"

# With PDF export
pip install "valiron[pdf]"

# Everything
pip install "valiron[all]"
```

**Requirements:** Python 3.9+ · numpy · pandas · scikit-learn · scipy · jinja2

**Framework support:** scikit-learn · PyTorch · ONNX · raw numpy arrays

---

## Roadmap

- [x] v0.1 — Core metrics + CDSCO report + Bootstrap CIs
- [x] v0.2 — ACP + EU AI Act Annex III template + PSI drift monitoring
- [ ] v0.3 — FDA template + PDF export + REST API
- [ ] v1.0 — Hosted platform + monitoring dashboard + ISO 42001

---

## Contributing

We especially want feedback from **clinical AI teams** and **regulatory consultants** — the frameworks are complex and real-world edge cases matter enormously.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code style, and PR checklist.

Open an issue or email **abhay@valiron.ai**

---

## Citation

```bibtex
@software{valiron2026,
  author  = {Sachan, Abhay},
  title   = {Valiron: Automated AI Compliance Evidence Generation},
  year    = {2026},
  url     = {https://github.com/abhaysachan007/valiron},
  version = {0.2.0},
  license = {MIT}
}
```

---

## License

[MIT](LICENSE) — use it, modify it, ship it. Zero data leaves your machine. Ever.

Built with ❤️ in Dehradun, India
