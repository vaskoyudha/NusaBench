from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nusabench import __version__
from nusabench.results import EvaluationResult


def export_json(result: EvaluationResult, output_path: str | Path) -> None:
    """Save EvaluationResult as formatted JSON."""
    results_dict: dict[str, object] = {}
    for task_name, task_result in result.results.items():
        entry: dict[str, object] = {**task_result.metrics, "num_samples": task_result.num_samples}
        results_dict[task_name] = entry

    config = result.metadata.get("config", {"limit": None, "seed": 42})

    payload: dict[str, object] = {
        "model": result.model,
        "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nusabench_version": __version__,
        "results": results_dict,
        "config": config,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
