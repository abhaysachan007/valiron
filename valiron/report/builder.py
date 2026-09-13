"""Compliance report builder."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from valiron.evaluate.runner import EvaluationResult


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def report(
    result: EvaluationResult,
    format: str = "html",
    output: Optional[str] = None,
) -> str:
    """Generate a compliance report from an EvaluationResult."""
    if format not in ("html", "pdf"):
        raise ValueError(f"Unsupported format '{format}'. Use 'html' or 'pdf'.")

    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("report.html.j2")
    html = template.render(result=result)

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
