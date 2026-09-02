"""The item cap must not end a call that simply got long.

The cap mirrored the service's ``Conversation.max_messages`` (30) as if the
service rejected an overflowing context. It does not: it drops the oldest
groups (``Conversation._trim``) and keeps going. Raising instead turned a long
call into a dead one, because the raise happens inside
``update_chat_ctx`` *while a tool result is being committed* — the output is
never confirmed, the next response fails with "pending tool call requires a
confirmed output before another response", and the session treats that as
unrecoverable.

Observed in voicebot room ``dev-martin-vds-web-call-1788338729362``
(2026-09-02): the agent answered normally for three minutes, hit 33 items on a
``web_search`` result that had already come back fine, and went silent for the
rest of the call.
"""

from __future__ import annotations

from typing import Literal

import pytest
from livekit.agents import APIConnectOptions
from livekit.agents.llm import ChatContext, RealtimeError
from livekit.plugins.hugging_voice.realtime import (
    _MAX_CHAT_CTX_ITEMS,
    RealtimeModel,
    RealtimeSession,
)


def _unconnected_session() -> tuple[RealtimeModel, RealtimeSession]:
    """A session that never opens a socket: update_chat_ctx stops at the check."""
    model = RealtimeModel(
        base_url="ws://127.0.0.1:1",
        token="secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=0.01),
    )
    return model, model.session()


def _conversation(turns: int) -> ChatContext:
    ctx = ChatContext.empty()
    for i in range(turns):
        role: Literal["user", "assistant"] = "user" if i % 2 == 0 else "assistant"
        ctx.add_message(id=f"item-{i}", role=role, content=f"turn {i}")
    return ctx


@pytest.mark.asyncio
async def test_a_call_past_the_old_30_item_cap_survives() -> None:
    """40 items is a normal few-minute call, not an error."""
    model, session = _unconnected_session()
    try:
        await session.update_chat_ctx(_conversation(40))
        assert len(session.chat_ctx.items) == 40
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_the_cap_still_refuses_a_runaway_context() -> None:
    """The guard is raised, not removed — something has to catch a real runaway."""
    model, session = _unconnected_session()
    try:
        with pytest.raises(RealtimeError, match="limited to"):
            await session.update_chat_ctx(_conversation(_MAX_CHAT_CTX_ITEMS + 1))
    finally:
        await model.aclose()
