"""The append-only check must compare item identity, not field equality.

The service keeps one atomic model context, so the plugin only accepts context
updates that append. It enforced that by comparing the incoming prefix to its own
items with ``!=`` — a Pydantic field comparison, on models whose ``created_at`` is
``default_factory=time.time`` and whose ``metrics`` the framework fills in as a
turn completes.

Both sides build their own instance of the same message: the plugin adds the
assistant message for a filler itself (``_finish_response``), and the framework
builds its own from the same generation, with the same ``id``. Those two
instances are never field-equal, so from the first assistant turn onwards *every*
context update was rejected as "not append-only" — including every finished
async-tool result, which is how a caller ends up never hearing one.

Observed in room ``dev-martin-vds-web-call-1788173758880`` (2026-08-31): three
tool calls, three rejections, no result spoken. Nothing else in the call logged a
problem, because the rejection travels in a ``tool_execution_updated`` event that
only shows up with LOG_REALTIME_EVENTS on.

So the comparison has to ignore the fields that say *when* an item was recorded
rather than *what* it says — ``created_at``, ``metrics``,
``transcript_confidence`` and the free-form ``extra`` both sides write to.
Content still counts: a rewritten or reordered history is not an append and
cannot be applied to an atomic model context, so it must still be refused rather
than silently dropped. The tests below pin both halves of that line.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents import APIConnectOptions
from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput, RealtimeError
from livekit.plugins.hugging_voice.realtime import RealtimeModel, RealtimeSession


def _unconnected_session() -> tuple[RealtimeModel, RealtimeSession]:
    """A session that never opens a socket: update_chat_ctx stops at the check."""
    model = RealtimeModel(
        base_url="ws://127.0.0.1:1",
        token="secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=0.01),
    )
    return model, model.session()


def _assistant(text: str, item_id: str) -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(id=item_id, role="assistant", content=text, interrupted=False)
    return ctx


@pytest.mark.asyncio
async def test_the_same_message_built_twice_is_still_append_only() -> None:
    """The exact shape of the failure: one filler, two instances, one new result."""
    model, session = _unconnected_session()
    try:
        # what the plugin recorded when the filler finished
        session._chat_ctx = _assistant("Ich schaue gerade nach.", "item_filler")

        # what the framework sends back: its own instance of that message,
        # plus the finished tool result appended
        await asyncio.sleep(0.01)  # guarantees a different created_at
        incoming = _assistant("Ich schaue gerade nach.", "item_filler")
        call = FunctionCall(call_id="call_1", name="web_search", arguments="{}")
        incoming.insert(
            [
                call,
                FunctionCallOutput(
                    call_id="call_1",
                    name="web_search",
                    output="Drei Veranstaltungen heute Abend.",
                    is_error=False,
                ),
            ]
        )

        assert incoming.items[0] != session.chat_ctx.items[0], (
            "premise: the two instances are not field-equal"
        )
        assert incoming.items[0].id == session.chat_ctx.items[0].id

        # disconnected, so this stops right after the check and records the context
        await session.update_chat_ctx(incoming)

        assert [item.id for item in session.chat_ctx.items] == [
            "item_filler",
            call.id,
            incoming.items[2].id,
        ]
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_a_reordered_history_is_still_refused() -> None:
    """The guard must keep catching what it was written for."""
    model, session = _unconnected_session()
    try:
        first = ChatContext.empty()
        first.add_message(id="item_a", role="assistant", content="eins")
        first.add_message(id="item_b", role="user", content="zwei")
        session._chat_ctx = first

        swapped = ChatContext.empty()
        swapped.add_message(id="item_b", role="user", content="zwei")
        swapped.add_message(id="item_a", role="assistant", content="eins")

        with pytest.raises(RealtimeError, match="append-only"):
            await session.update_chat_ctx(swapped)
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_a_dropped_item_is_still_refused() -> None:
    """Removing history is not an append either — the service cannot forget."""
    model, session = _unconnected_session()
    try:
        first = ChatContext.empty()
        first.add_message(id="item_a", role="assistant", content="eins")
        first.add_message(id="item_b", role="user", content="zwei")
        session._chat_ctx = first

        without_first = ChatContext.empty()
        without_first.add_message(id="item_b", role="user", content="zwei")

        with pytest.raises(RealtimeError, match="append-only"):
            await session.update_chat_ctx(without_first)
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_bookkeeping_on_a_tool_call_does_not_break_the_append() -> None:
    """``extra`` is written by both sides and must not decide append-only.

    The framework marks a tool call non-blocking in ``extra`` the moment our
    acknowledgement fires (``RunContext.update`` sets
    ``__livekit_agents_tool_non_blocking``), while this plugin keeps the turn ids
    it needs in the same dict. Comparing it would reproduce the same outage in a
    different disguise.
    """
    model, session = _unconnected_session()
    try:
        call = FunctionCall(
            call_id="call_1",
            name="web_search",
            arguments="{}",
            extra={"hugging_voice": {"turn_id": "turn_1"}},
        )
        session._chat_ctx = ChatContext([call])

        marked = call.model_copy(
            update={
                "extra": {
                    "hugging_voice": {"turn_id": "turn_1"},
                    "__livekit_agents_tool_non_blocking": True,
                }
            }
        )
        incoming = ChatContext([marked])
        incoming.insert(
            FunctionCallOutput(
                call_id="call_1", name="web_search", output="Drei Treffer.", is_error=False
            )
        )

        await session.update_chat_ctx(incoming)

        assert len(session.chat_ctx.items) == 2
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_changed_content_is_still_refused() -> None:
    """Tolerating metadata must not tolerate a rewritten message."""
    model, session = _unconnected_session()
    try:
        session._chat_ctx = _assistant("Ich schaue gerade nach.", "item_filler")

        rewritten = _assistant("Etwas voellig anderes.", "item_filler")

        with pytest.raises(RealtimeError, match="append-only"):
            await session.update_chat_ctx(rewritten)
    finally:
        await model.aclose()
