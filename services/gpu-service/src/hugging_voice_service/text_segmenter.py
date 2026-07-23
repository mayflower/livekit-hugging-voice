"""Small deterministic visible-text segmenter for multilingual TTS."""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    "bzw.",
    "ca.",
    "d.h.",
    "dr.",
    "etc.",
    "nr.",
    "prof.",
    "u.a.",
    "usw.",
    "z.b.",
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
}


class SpeechTextSegmenter:
    def __init__(self, *, max_characters: int = 200) -> None:
        if not 80 <= max_characters <= 220:
            raise ValueError("TTS segment maximum must be between 80 and 220 characters")
        self._max_characters = max_characters
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        segments: list[str] = []
        while True:
            boundary = self._sentence_boundary()
            if boundary is None and len(self._buffer) > self._max_characters:
                boundary = self._hard_boundary()
            if boundary is None:
                break
            segment = self._take(boundary)
            if segment:
                segments.append(segment)
        return segments

    def flush(self) -> list[str]:
        segment = self._normalize(self._buffer)
        self._buffer = ""
        return [] if not segment else [segment]

    def _sentence_boundary(self) -> int | None:
        for match in re.finditer(r"[.!?:;](?:[\"'»“”)]*)(?:\s+|$)", self._buffer):
            end = match.end()
            candidate = self._buffer[:end].strip().lower()
            word = candidate.split()[-1] if candidate.split() else ""
            punctuation = self._buffer[match.start()]
            if punctuation == ".":
                if word in _ABBREVIATIONS or re.search(r"\d+\.\d+$", candidate):
                    continue
                if re.search(r"(?:\b[A-Za-zÄÖÜäöü]\.){2,}$", candidate):
                    continue
            return end
        return None

    def _hard_boundary(self) -> int:
        prefix = self._buffer[: self._max_characters + 1]
        whitespace = max(prefix.rfind(" "), prefix.rfind("\n"), prefix.rfind("\t"))
        return whitespace + 1 if whitespace >= 80 else self._max_characters

    def _take(self, boundary: int) -> str:
        raw = self._buffer[:boundary]
        self._buffer = self._buffer[boundary:]
        return self._normalize(raw)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.split())
