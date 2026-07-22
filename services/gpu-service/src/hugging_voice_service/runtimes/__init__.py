"""Concrete local model runtimes; no provider registry or fallback path."""

from .gemma import GemmaRuntime, TextDelta, TextUsage
from .parakeet import ParakeetRuntime
from .qwen_tts import QwenTTSRuntime
from .silero import SessionVAD, VADSignal

__all__ = [
    "GemmaRuntime",
    "ParakeetRuntime",
    "QwenTTSRuntime",
    "SessionVAD",
    "TextDelta",
    "TextUsage",
    "VADSignal",
]
