"""Compliance report builder — HTML and optional PDF output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader

from valiron.evaluate.runner import EvaluationResult
from valiron.evaluate.metrics import MetricsResult


_TEMPLATES_DIR = Path(__file__).parent / "templates"

_REGULATION_TEMPLATES: Dict[str, str] = {
    "cdsco_mdsw": "cdsco.html.j2",
}
_DEFAULT_TEMPLATE = "report.html.j2"


@dataclass
class ReportInput:
    """All data needed to render a compliance report.

    Args:
        eval_result: Output of valiron.evaluate().
        metrics: Binary classification metrics (MetricsResult). Optional.
        subgroups: Per-subgroup metric dicts from analyze_subgroups(). Optional.
        calibration: CalibrationResult from check_calibration(). Optional.
    """

    eval_result: EvaluationResult
    metrics: Optional[MetricsResult] = None
    subgroups: Optional[Dict[str, Any]] = None
    calibration: Optional[Any] = None


def report(
    input: ReportInput,
    format: str = "html",
    output: Optional[str] = None,
) -> str:
    """Render a compliance report from a ReportInput.

    Args:
        input: ReportInput containing EvaluationResult and optional metrics/subgroups/calibration.
        format: 'html' (default) or 'pdf'. PDF requires weasyprint.
        output: File path to write output. Returns path string when set.

    Returns:
        Rendered HTML string, or output file path when *output* is provided.

    Raises:
        ValueError: If format is not 'html' or 'pdf'.
        ImportError: If format='pdf' and weasyprint is not installed.
    """
    if format not in ("html", "pdf"):
        raise ValueError(f"Unsupported format '{format}'. Use 'html' or 'pdf'.")

    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)
    template_name = _REGULATION_TEMPLATES.get(input.eval_result.regulation, _DEFAULT_TEMPLATE)
    template = env.get_template(template_name)

    html = template.render(
        result=input.eval_result,
        metrics=input.metrics,
        subgroups=input.subgroups,
        calibration=input.calibration,
    )

    if format == "pdf":
        try:
            from weasyprint import HTML as WeasyprintHTML
        except ImportError:
            raise ImportError("PDF export requires weasyprint: pip install valiron[pdf]")
        pdf_bytes = WeasyprintHTML(string=html).write_pdf()
        out_path = output or "compliance_report.pdf"
        Path(out_path).write_bytes(pdf_bytes)
        return out_path

    if output:
        Path(output).write_text(html, encoding="utf-8")
        return output

    return html
