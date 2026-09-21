# Security Policy

## Data privacy

Valiron never transmits patient data. All computation runs locally on your machine. The only output is metrics and a report file — both stay on your server.

```
Your server          Valiron
──────────           ────────────────────
Patient data  →     Runs entirely locally
Model weights        ↓
                    Metrics computed
                    in memory only
                    ↓
                    Report written to disk
                    on your machine
                    ↓
                    Zero data transmitted
```

This is validation tooling, not a medical device itself. It does not make clinical decisions.

---

## Responsible disclosure

If you find a security vulnerability, please report it privately:

**Email:** abhay@valiron.ai  
**Subject:** `[SECURITY] <brief description>`

Please do not open a public GitHub issue for security vulnerabilities. We will acknowledge within 48 hours and patch within 14 days.

---

## Scope

In scope:
- Code execution vulnerabilities in report generation
- Path traversal in file output
- Dependency vulnerabilities in core packages

Out of scope:
- Issues in optional dependencies (weasyprint, onnxruntime)
- Vulnerabilities in the user's own model or data pipeline
