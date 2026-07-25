"""Local LLM integration for AI Coach and encouragement features."""

from momentum.llm.disclaimer import DISCLAIMER, SHORT_DISCLAIMER
from momentum.llm.downloader import (
    MODEL_FILENAME,
    MODEL_REPO,
    ensure_model,
    get_model_path,
)
from momentum.llm.engine import LLM_AVAILABLE, LlmEngine, get_engine, is_llm_available

__all__ = [
    "DISCLAIMER",
    "SHORT_DISCLAIMER",
    "LlmEngine",
    "get_engine",
    "is_llm_available",
    "LLM_AVAILABLE",
    "ensure_model",
    "get_model_path",
    "MODEL_REPO",
    "MODEL_FILENAME",
]
