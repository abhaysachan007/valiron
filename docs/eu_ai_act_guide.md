# EU AI Act Compliance Guide

## Overview

The EU AI Act classifies AI systems by risk. Annex III (high-risk) includes medical devices, employment screening, credit scoring, and biometric identification. High-risk systems face mandatory requirements before EU market placement.

## Checks Performed by Valiron

| Check | Threshold | Description |
|---|---|---|
| `EU_AI_ACT_PERFORMANCE` | accuracy ≥ 0.70 | Minimum acceptable performance |
| `EU_AI_ACT_TRANSPARENCY` | model attribute check | Feature importances or coefficients present |
| `EU_AI_ACT_HIGH_RISK` | use_case keyword scan | Flags Annex III use cases |
| `EU_AI_ACT_BIAS` | sensitive_features provided | Fairness audit feasibility |

## Usage

```python
result = valiron.evaluate(
    model=model, X_test=X_test, y_test=y_test,
    regulation="eu_ai_act", use_case="recruitment_screening",
    sensitive_features=["gender", "age_group"],
)
```

## Key Requirements for High-Risk AI

- Human oversight mechanisms
- Transparency and documentation
- Data governance and bias checks
- Technical documentation maintained
