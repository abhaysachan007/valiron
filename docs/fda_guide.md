# FDA AI/ML SaMD Compliance Guide

## Overview

The FDA regulates AI/ML-based Software as a Medical Device (SaMD). The 2021 Action Plan emphasizes a "predetermined change control plan" (PCCP) for adaptive models.

## Checks Performed by Valiron

| Check | Threshold | Description |
|---|---|---|
| `FDA_SAMD_PERFORMANCE` | accuracy ≥ 0.75 | Higher bar for medical device context |
| `FDA_SAMD_PDCP` | warning | Predetermined change control plan required |
| `FDA_SAMD_MONITORING` | warning | Real-world performance monitoring plan |

## Usage

```python
result = valiron.evaluate(
    model=model, X_test=X_test, y_test=y_test,
    regulation="fda_samd", use_case="diagnostic_imaging_aid",
)
```

## Key Requirements

- **PCCP**: document how model updates without new 510(k) submission
- **Real-World Monitoring**: define post-market drift thresholds
- **Labelling**: indicate AI/ML nature of device
