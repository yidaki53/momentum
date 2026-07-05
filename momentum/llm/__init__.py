"""Local LLM integration for AI Coach and encouragement features."""

from momentum.llm.disclaimer import DISCLAIMER, SHORT_DISCLAIMER
from momentum.llm.engine import LlmEngine, get_engine
from momentum.llm.downloader import ensure_model, get_model_path, MODEL_REPO, MODEL_FILENAME

__all__ = [
    "DISCLAIMER",
    "SHORT_DISCLAIMER",
    "LlmEngine",
    "get_engine",
    "ensure_model",
    "get_model_path",
    "MODEL_REPO",
    "MODEL_FILENAME",
]
