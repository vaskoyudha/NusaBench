from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="nusabench", help="NLP Evaluation Suite for Bahasa Indonesia")
console = Console()


@app.command()
def evaluate(
    model: str = typer.Option(..., help="Model backend name (dummy, hf, gemini)"),
    model_args: str = typer.Option(
        "",
        help="Comma-separated key=value model arguments, e.g. pretrained=gpt2,dtype=float16",
    ),
    task: list[str] = typer.Option(..., help="Task name(s) to evaluate"),  # noqa: B008
    limit: int | None = typer.Option(None, help="Max samples per task"),
    output: str = typer.Option("results.json", help="Output file path"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),  # noqa: FBT001
) -> None:
    """Run evaluation for one or more tasks with a given model."""
    import nusabench as nb

    try:
        result = nb.evaluate(
            model=model,
            tasks=task,
            model_args=model_args or None,
            limit=limit,
            verbose=verbose,
        )
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    result_dict = result.to_dict()
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, indent=2)

    table = Table(title="Evaluation Results")
    table.add_column("Task", style="cyan")
    table.add_column("Metric", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Samples", style="white")

    for task_name, task_result in result.results.items():
        for metric_name, score in task_result.metrics.items():
            table.add_row(
                task_name,
                metric_name,
                f"{score:.4f}",
                str(task_result.num_samples),
            )

    console.print(table)
    console.print(f"\n[bold green]Results saved to:[/] {output}")


@app.command("list-tasks")
def list_tasks() -> None:
    """List all registered evaluation tasks."""
    from nusabench.tasks import TaskRegistry

    names = TaskRegistry.list()
    table = Table(title="Available Tasks")
    table.add_column("Task Name", style="cyan")

    for name in names:
        table.add_row(name)

    console.print(table)

    if not names:
        console.print("[yellow]No tasks registered yet.[/]")


@app.command("list-models")
def list_models() -> None:
    """List all registered model backends."""
    from nusabench.models import ModelRegistry

    names = ModelRegistry.list()
    table = Table(title="Available Model Backends")
    table.add_column("Model Name", style="cyan")

    for name in names:
        table.add_row(name)

    console.print(table)
