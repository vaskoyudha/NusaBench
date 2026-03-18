from __future__ import annotations

from nusabench.models.base import Model, ModelRegistry, register_model
from nusabench.models.dummy import DummyModel

__all__ = ["DummyModel", "Model", "ModelRegistry", "register_model"]
