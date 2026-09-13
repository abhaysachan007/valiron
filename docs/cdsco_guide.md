# CDSCO MDSW Compliance Guide

## What is CDSCO MDSW?

CDSCO (Central Drugs Standard Control Organisation) regulates Medical Device Software (MDSW) in India under the Medical Devices Rules, 2017. AI/ML software used for clinical diagnosis or treatment requires compliance.

## Checks Performed by Valiron

| Check | Threshold | Description |
|---|---|---|
| `CDSCO_MDSW_PERFORMANCE` | accuracy ≥ 0.70 | Minimum diagnostic accuracy |
| `CDSCO_MDSW_DATA_PRIVACY` | local only | No patient data transmitted |
| `CDSCO_MDSW_REGISTRATION` | warning | Reminder to register with CDSCO |

## Usage

```python
result = valiron.evaluate(
    model=model,
    X_test=X_test,
    y_test=y_test,
    regulation="cdsco_mdsw",
    use_case="diagnostic_aid",
)
```

## Pre-Deployment Checklist

- [ ] Accuracy ≥ 70% on representative Indian patient population
- [ ] MDSW registered with CDSCO before clinical use
- [ ] Clinical validation study conducted
- [ ] Post-market surveillance plan in place
