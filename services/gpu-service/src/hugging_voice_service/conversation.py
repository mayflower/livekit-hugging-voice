"""Bounded ephemeral conversation with atomic tool exchanges."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, TypeAlias

from hugging_voice_protocol.events import canonical_json

from .runtimes.llama_cpp_chat import GemmaMessage, GemmaToolCall

ConversationRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ConversationEntry:
    item_id: str
    role: ConversationRole
    content: str

    @property
    def item_count(self) -> int:
        return 1

    @property
    def character_count(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class FunctionCallEntry:
    item_id: str
    call_id: str
    name: str
    arguments: str
    turn_id: str
    turn_revision: int
    generation_id: str
    response_id: str


@dataclass(frozen=True, slots=True)
class FunctionCallOutputEntry:
    item_id: str
    call_id: str
    name: str
    output: str
    is_error: bool
    turn_id: str
    turn_revision: int
    generation_id: str
    response_id: str


@dataclass(frozen=True, slots=True)
class ToolExchangeGroup:
    call: FunctionCallEntry
    output: FunctionCallOutputEntry

    @property
    def item_count(self) -> int:
        return 2

    @property
    def character_count(self) -> int:
        return len(self.call.arguments) + len(self.output.output)


ConversationGroup: TypeAlias = ConversationEntry | ToolExchangeGroup


class Conversation:
    def __init__(self, *, max_messages: int = 30, max_characters: int = 48_000) -> None:
        if max_messages < 1 or max_characters < 1:
            raise ValueError("conversation limits must be positive")
        self._groups: deque[ConversationGroup] = deque()
        self._max_items = max_messages
        self._max_characters = max_characters
        self._items = 0
        self._characters = 0

    @property
    def groups(self) -> tuple[ConversationGroup, ...]:
        return tuple(self._groups)

    @property
    def entries(self) -> tuple[ConversationEntry, ...]:
        """Compatibility view containing only normal text messages."""

        return tuple(group for group in self._groups if isinstance(group, ConversationEntry))

    def append(self, *, item_id: str, role: ConversationRole, content: str) -> None:
        normalized = content.strip()
        if not normalized:
            raise ValueError("completed conversation items must contain text")
        if len(normalized) > self._max_characters:
            raise ValueError("conversation item exceeds total character limit")
        self._append_group(ConversationEntry(item_id=item_id, role=role, content=normalized))

    def commit_tool_exchange(
        self,
        *,
        call: FunctionCallEntry,
        output: FunctionCallOutputEntry,
    ) -> None:
        if call.call_id != output.call_id or call.name != output.name:
            raise ValueError("tool call and output do not match")
        if (
            call.turn_id,
            call.turn_revision,
            call.generation_id,
            call.response_id,
        ) != (
            output.turn_id,
            output.turn_revision,
            output.generation_id,
            output.response_id,
        ):
            raise ValueError("tool call and output correlations do not match")
        if self.has_call(call.call_id):
            raise ValueError("tool call output is already committed")
        self._append_group(ToolExchangeGroup(call=call, output=output))

    def has_call(self, call_id: str) -> bool:
        return any(
            isinstance(group, ToolExchangeGroup) and group.call.call_id == call_id
            for group in self._groups
        )

    def messages(self) -> list[GemmaMessage]:
        messages: list[GemmaMessage] = []
        for group in self._groups:
            if isinstance(group, ConversationEntry):
                messages.append(GemmaMessage(role=group.role, content=group.content))
                continue
            messages.extend(
                (
                    GemmaMessage(
                        role="assistant",
                        content=None,
                        tool_calls=(
                            GemmaToolCall(
                                call_id=group.call.call_id,
                                name=group.call.name,
                                arguments=group.call.arguments,
                            ),
                        ),
                    ),
                    GemmaMessage(
                        role="tool",
                        content=canonical_json(
                            {
                                "is_error": group.output.is_error,
                                "output": group.output.output,
                            }
                        ),
                        tool_call_id=group.call.call_id,
                        name=group.call.name,
                    ),
                )
            )
        return messages

    def clear(self) -> None:
        self._groups.clear()
        self._items = 0
        self._characters = 0

    def _append_group(self, group: ConversationGroup) -> None:
        if group.item_count > self._max_items or group.character_count > self._max_characters:
            raise ValueError("conversation group exceeds total limits")
        self._groups.append(group)
        self._items += group.item_count
        self._characters += group.character_count
        self._trim()

    def _trim(self) -> None:
        while self._items > self._max_items or self._characters > self._max_characters:
            removed = self._groups.popleft()
            self._items -= removed.item_count
            self._characters -= removed.character_count
