"""Local LLM integration for AI Coach and encouragement features."""

from momentum.llm.assist import (
    clear_cache as clear_assistance_cache,
)
from momentum.llm.assist import (
    is_enabled as is_assistance_enabled,
)
from momentum.llm.assist import (
    is_ready as is_assistance_ready,
)
from momentum.llm.assist import (
    request_assistance,
)
from momentum.llm.assist import (
    status as assistance_status,
)
from momentum.llm.disclaimer import DISCLAIMER, SHORT_DISCLAIMER
from momentum.llm.downloader import (
    MODEL_FILENAME,
    MODEL_REPO,
    delete_model,
    ensure_model,
    get_model_path,
)
from momentum.llm.engine import (
    LLM_AVAILABLE,
    LlmEngine,
    get_engine,
    is_llm_available,
    native_diagnostics,
    reset_engine,
)

__all__ = [
    "DISCLAIMER",
    "SHORT_DISCLAIMER",
    "LlmEngine",
    "get_engine",
    "is_llm_available",
    "native_diagnostics",
    "reset_engine",
    "LLM_AVAILABLE",
    "request_assistance",
    "is_assistance_enabled",
    "is_assistance_ready",
    "assistance_status",
    "clear_assistance_cache",
    "ensure_model",
    "delete_model",
    "get_model_path",
    "MODEL_REPO",
    "MODEL_FILENAME",
]
