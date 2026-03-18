from __future__ import annotations

import importlib
from typing import cast

from nusabench.models.base import Model, ModelRegistry, register_model
from nusabench.models.dummy import DummyModel

__all__ = ["DummyModel", "Model", "ModelRegistry", "register_model"]

try:
    _huggingface_module = importlib.import_module("nusabench.models.huggingface")
    HuggingFaceModel = cast(type[Model], _huggingface_module.__dict__["HuggingFaceModel"])

    __all__.append("HuggingFaceModel")
except ImportError:
    pass

try:
    _gemini_module = importlib.import_module("nusabench.models.gemini")
    GeminiModel = cast(type[Model], _gemini_module.__dict__["GeminiModel"])

    __all__.append("GeminiModel")
except ImportError:
    pass
