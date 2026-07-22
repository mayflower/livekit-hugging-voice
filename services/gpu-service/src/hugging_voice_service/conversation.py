"""Bounded, completed-text-only ephemeral conversation state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from .runtimes.gemma import GemmaMessage

ConversationRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ConversationEntry:
    item_id: str
    role: ConversationRole
    content: str


class Conversation:
    def __init__(self, *, max_messages: int = 30, max_characters: int = 48_000) -> None:
        if max_messages < 1 or max_characters < 1:
            raise ValueError("conversation limits must be positive")
        self._entries: deque[ConversationEntry] = deque()
        self._max_messages = max_messages
        self._max_characters = max_characters
        self._characters = 0

    @property
    def entries(self) -> tuple[ConversationEntry, ...]:
        return tuple(self._entries)

    def append(self, *, item_id: str, role: ConversationRole, content: str) -> None:
        normalized = content.strip()
        if not normalized:
            raise ValueError("completed conversation items must contain text")
        if len(normalized) > self._max_characters:
            raise ValueError("conversation item exceeds total character limit")
        entry = ConversationEntry(item_id=item_id, role=role, content=normalized)
        self._entries.append(entry)
        self._characters += len(normalized)
        self._trim()

    def messages(self) -> list[GemmaMessage]:
        return [GemmaMessage(role=entry.role, content=entry.content) for entry in self._entries]

    def clear(self) -> None:
        self._entries.clear()
        self._characters = 0

    def _trim(self) -> None:
        while len(self._entries) > self._max_messages or self._characters > self._max_characters:
            removed = self._entries.popleft()
            self._characters -= len(removed.content)
