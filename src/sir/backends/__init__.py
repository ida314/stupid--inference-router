"""Backend implementations. The mock stands in for a GPU; vLLM is a real one."""

from sir.backends.mock import MockBackend
from sir.backends.vllm import VllmBackend

__all__ = ["MockBackend", "VllmBackend"]
