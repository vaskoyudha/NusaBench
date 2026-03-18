from __future__ import annotations

from rich.console import Console
from rich.table import Table

from nusabench.results import EvaluationResult


def print_results(result: EvaluationResult) -> None:
    """Print EvaluationResult as a Rich table to stdout."""
    table = Table(title="Evaluation Results")
    table.add_column("Task", style="cyan")
    table.add_column("Metric", style="green")
    table.add_column("Score", style="yellow")

    for task_name, task_result in result.results.items():
        for metric_name, score in task_result.metrics.items():
            table.add_row(task_name, metric_name, f"{score:.4f}")

    console = Console()
    console.print(table)
