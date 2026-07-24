"""Small deterministic visible-text segmenter for multilingual TTS."""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    # German
    "bzw.",
    "ca.",
    "d.h.",
    "dr.",
    "nr.",
    "prof.",
    "u.a.",
    "usw.",
    "z.b.",
    # English
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    # French
    "env.",
    "mme.",
    "mlles.",
    "m.",
    # Italian
    "dott.",
    "sig.",
    "sig.ra.",
    "sig.na.",
    # Shared
    "etc.",
}


class SpeechTextSegmenter:
    """Segment streamed text with a short first chunk and a strict hard cap."""

    def __init__(
        self,
        *,
        first_segment_characters: int = 72,
        next_segment_characters: int = 140,
        hard_max_characters: int = 160,
    ) -> None:
        if not 32 <= first_segment_characters <= 160:
            raise ValueError("first TTS segment target must be between 32 and 160 characters")
        if not 64 <= next_segment_characters <= 200:
            raise ValueError("later TTS segment target must be between 64 and 200 characters")
        if not 80 <= hard_max_characters <= 220:
            raise ValueError("TTS segment hard maximum must be between 80 and 220 characters")
        if first_segment_characters > next_segment_characters:
            raise ValueError("first TTS segment target cannot exceed the later target")
        if next_segment_characters > hard_max_characters:
            raise ValueError("later TTS segment target cannot exceed the hard maximum")
        self._first_target = first_segment_characters
        self._next_target = next_segment_characters
        self._hard_max = hard_max_characters
        self._buffer = ""
        self._segments_emitted = 0

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        segments: list[str] = []
        while True:
            boundary = self._sentence_boundary()
            if boundary is not None and boundary <= self._hard_max:
                pass
            else:
                boundary = self._length_boundary()
            if boundary is None:
                break
            segment = self._take(boundary)
            if segment:
                segments.append(segment)
                self._segments_emitted += 1
        return segments

    def flush(self) -> list[str]:
        segments: list[str] = []
        while len(self._normalize(self._buffer)) > self._hard_max:
            boundary = self._word_boundary(self._hard_max)
            if boundary is None:
                boundary = self._hard_max
            segment = self._take(boundary)
            if segment:
                segments.append(segment)
                self._segments_emitted += 1
        segment = self._normalize(self._buffer)
        self._buffer = ""
        if segment:
            segments.append(segment)
            self._segments_emitted += 1
        return segments

    def _sentence_boundary(self) -> int | None:
        for match in re.finditer(r"[.!?:;](?:[\"'»“”)]*)(?:\s+|$)", self._buffer):
            end = match.end()
            candidate = self._buffer[:end].strip().lower()
            word = candidate.split()[-1] if candidate.split() else ""
            punctuation = self._buffer[match.start()]
            if punctuation == ".":
                if word in _ABBREVIATIONS or re.search(r"\d+\.\d+$", candidate):
                    continue
                if re.search(r"(?:\b[A-Za-zÀ-ÖØ-öø-ÿ]\.){2,}$", candidate):
                    continue
            return end
        return None

    def _length_boundary(self) -> int | None:
        target = self._first_target if self._segments_emitted == 0 else self._next_target
        if len(self._buffer) <= target:
            return None
        boundary = self._word_boundary(target)
        if boundary is not None:
            return boundary
        if len(self._buffer) > self._hard_max:
            return self._hard_max
        return None

    def _word_boundary(self, limit: int) -> int | None:
        prefix = self._buffer[: limit + 1]
        whitespace = max(prefix.rfind(" "), prefix.rfind("\n"), prefix.rfind("\t"))
        return whitespace + 1 if whitespace >= 0 else None

    def _take(self, boundary: int) -> str:
        raw = self._buffer[:boundary]
        self._buffer = self._buffer[boundary:]
        return self._normalize(raw)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.split())
