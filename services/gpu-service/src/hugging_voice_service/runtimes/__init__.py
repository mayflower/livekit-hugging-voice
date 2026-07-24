"""Concrete local model runtimes; no provider registry or fallback path."""

from .llama_cpp_chat import LlamaCppChatRuntime, TextDelta, TextUsage
from .parakeet import ParakeetRuntime
from .qwen_tts import QwenTTSRuntime
from .silero import SessionVAD, VADSignal

__all__ = [
    "LlamaCppChatRuntime",
    "ParakeetRuntime",
    "QwenTTSRuntime",
    "SessionVAD",
    "TextDelta",
    "TextUsage",
    "VADSignal",
]
