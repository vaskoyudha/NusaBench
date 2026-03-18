"""Global configuration dataclass for NusaBench."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class NusaBenchConfig:
    """Global configuration for a NusaBench evaluation run."""
    cache_dir: str = field(
        default_factory=lambda: os.path.join(os.path.expanduser("~"), ".cache", "nusabench")
    )
    verbose: bool = False
    seed: int = 42
    limit: int | None = None  # None means use all samples
