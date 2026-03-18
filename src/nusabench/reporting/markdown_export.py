from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from nusabench.results import EvaluationResult


def export_markdown(result: EvaluationResult, output_path: str | Path) -> None:
    """Save EvaluationResult as a Markdown table."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = [
        "# NusaBench Evaluation Results",
        "",
        f"**Model**: {result.model}  ",
        f"**Timestamp**: {timestamp}",
        "",
        "| Task | Metric | Score |",
        "|------|--------|-------|",
    ]

    for task_name, task_result in result.results.items():
        for metric_name, score in task_result.metrics.items():
            lines.append(f"| {task_name} | {metric_name} | {score:.4f} |")

    lines.append("")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
