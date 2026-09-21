# Contributing to Valiron

Thanks for your interest. Valiron is most useful when it covers the regulations that clinical AI teams actually face — so contributions from regulatory consultants, clinical AI engineers, and compliance specialists are especially welcome.

---

## How to add a new regulation template

1. Add the regulation ID to `valiron/evaluate/regulations/` — copy an existing one (e.g., `cdsco_mdsw.py`) as a starting point.
2. Define the thresholds, check IDs, and pass/fail logic in `run_checks()`.
3. Add a Jinja2 HTML template to `valiron/report/templates/` — follow `cdsco.html.j2`.
4. Register it in `valiron/evaluate/registry.py`.
5. Add at least 5 tests covering passing and failing cases.

The key invariant: a `run_checks()` call must be deterministic — same inputs, same result.

---

## How to run tests locally

```bash
git clone https://github.com/abhaysachan007/valiron
cd valiron
pip install -e ".[dev]"

# Run all tests with coverage
pytest --cov=valiron --cov-report=term-missing

# Run a specific module
pytest tests/test_evaluate.py -v

# Check types
mypy valiron

# Lint
ruff check valiron
ruff format valiron
```

Coverage must stay at or above **80%**. New regulation modules must hit **90%+**.

---

## Code style

- **Formatter:** `ruff format` (line length 100)
- **Linter:** `ruff check` — no warnings allowed
- **Types:** All public functions must have type annotations; `mypy --strict` must pass
- **Immutability:** Prefer returning new objects over mutating in place
- **No magic numbers:** Use named constants for all thresholds

---

## PR checklist

Before opening a PR:

- [ ] `pytest` passes with coverage ≥ 80%
- [ ] `mypy valiron` passes with no errors
- [ ] `ruff check valiron` passes with no warnings
- [ ] New regulation: template tested against at least 2 real scenarios (passing + failing)
- [ ] No secrets or credentials committed
- [ ] CHANGELOG updated if adding a public API

---

## Issue templates

Use the issue templates in `.github/ISSUE_TEMPLATE/` for:

- **Bug reports** — include the regulation, Python version, and full traceback
- **Feature requests** — describe the use case before proposing an API
- **Regulation requests** — which regulation, which sections, do you have access to the official guidance?

---

## Questions

Open a GitHub Discussion or email **abhay@valiron.ai**.
