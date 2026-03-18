"""Data loading and prompt formatting utilities for NusaBench."""
from __future__ import annotations

from typing import Any

import jinja2


def load_hf_dataset(
    path: str,
    name: str | None = None,
    split: str = "test",
    limit: int | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset and return as a list of dicts.
    
    Args:
        path: HuggingFace dataset identifier (e.g. "indonlp/indonlu")
        name: Dataset configuration name (e.g. "smsa")
        split: Dataset split to load (default: "test")
        limit: Maximum number of examples to load (None for all)
        cache_dir: Directory for caching downloaded datasets
    
    Returns:
        List of document dicts
    
    Raises:
        RuntimeError: If the dataset cannot be loaded
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        
        dataset = load_dataset(path, name, split=split, cache_dir=cache_dir)
        docs = [dict(example) for example in dataset]
        if limit is not None:
            docs = docs[:limit]
        return docs
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset {path!r} (config={name!r}): {exc}") from exc


def format_prompt_jinja(template: str, doc: dict[str, Any]) -> str:
    """Render a Jinja2 template string with document fields.
    
    Args:
        template: Jinja2 template string (e.g. "Sentimen: {{text}}")
        doc: Document dict with field values
    
    Returns:
        Rendered string
    """
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    return env.from_string(template).render(**doc)
