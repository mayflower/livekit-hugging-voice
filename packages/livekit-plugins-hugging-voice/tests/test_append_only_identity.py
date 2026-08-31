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


@pytest.mark.asyncio
async def test_the_rejection_says_which_item_and_field_diverged() -> None:
    """A bare "not append-only" is the same sentence for four different causes.

    Naming the item and the differing fields is what turns the next occurrence
    into one log line instead of a debugging round.
    """
    model, session = _unconnected_session()
    try:
        session._chat_ctx = _assistant("Ich schaue gerade nach.", "item_filler")
        rewritten = _assistant("Etwas anderes.", "item_filler")

        with pytest.raises(RealtimeError) as excinfo:
            await session.update_chat_ctx(rewritten)

        message = str(excinfo.value)
        assert "item 0 (item_filler)" in message
        assert "content" in message
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_the_rejection_says_when_history_was_lost() -> None:
    model, session = _unconnected_session()
    try:
        first = ChatContext.empty()
        first.add_message(id="item_a", role="assistant", content="eins")
        first.add_message(id="item_b", role="user", content="zwei")
        session._chat_ctx = first

        with pytest.raises(RealtimeError, match="lost 1 of 2 items"):
            await session.update_chat_ctx(_assistant("eins", "item_a"))
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_an_interruption_the_plugin_cannot_see_does_not_break_the_append() -> None:
    """``interrupted`` is the framework's to know, not this plugin's.

    ``_finish_response`` records every finished response with a constant
    ``interrupted=False`` — it has no view of what the caller actually heard —
    while the framework fills in the real value from the speech handle. Demanding
    agreement there would refuse every later append.
    """
    model, session = _unconnected_session()
    try:
        session._chat_ctx = _assistant("Ich schaue gerade nach.", "item_filler")

        interrupted = ChatContext.empty()
        interrupted.add_message(
            id="item_filler",
            role="assistant",
            content="Ich schaue gerade nach.",
            interrupted=True,
        )
        interrupted.insert(
            FunctionCallOutput(
                call_id="call_1", name="web_search", output="Drei Treffer.", is_error=False
            )
        )

        await session.update_chat_ctx(interrupted)

        assert len(session.chat_ctx.items) == 2
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_the_frameworks_session_config_record_does_not_break_the_append() -> None:
    """The failure that survived three fixes: a versat, not a diverging field.

    ``AgentConfigUpdate`` is a ChatContext item the framework writes when a
    session starts, recording the instructions and the initial tool list
    (``AgentActivity`` inserts it into both the agent's and the session's
    context). ``ChatContext.insert`` orders by ``created_at``, so it is created
    before the first spoken word and stays at index 0 for the whole call.

    This plugin never holds it: instructions and tools travel over the session
    configuration, not as conversation items, and it records its own responses in
    ``_finish_response``, which starts the context at the first assistant
    message. So the two lists are offset by one from the greeting onwards, and
    comparing index against index rejects every append — including every finished
    async-tool result.

    Observed in room ``dev-martin-vds-web-call-1788183135400`` (2026-08-31):
    "item 0 (item_4e0aaef8...) differs in content, id, instructions, role,
    tools_added, type". ``id`` and ``type`` in that list are the tell: those are
    not two versions of one item, they are two different kinds of item.
    """
    from livekit.agents.llm import AgentConfigUpdate

    model, session = _unconnected_session()
    try:
        # what the plugin holds: its own record of the greeting it just spoke
        session._chat_ctx = _assistant("Hallo! Hier ist die Telefonassistenz.", "item_greeting")

        # what the framework sends: its session-config record first, then the
        # same greeting, then the finished tool call
        incoming = ChatContext.empty()
        incoming.insert(
            AgentConfigUpdate(
                instructions="Du bist die Telefonassistenz.",
                tools_added=["web_search", "knowledge_base_search"],
            )
        )
        incoming.add_message(
            id="item_greeting",
            role="assistant",
            content="Hallo! Hier ist die Telefonassistenz.",
            interrupted=False,
        )
        incoming.insert(
            [
                FunctionCall(call_id="call_1", name="web_search", arguments="{}"),
                FunctionCallOutput(
                    call_id="call_1",
                    name="web_search",
                    output="Drei Veranstaltungen heute Abend.",
                    is_error=False,
                ),
            ]
        )

        assert incoming.items[0].type == "agent_config_update", (
            "premise: the config record sorts to index 0"
        )

        await session.update_chat_ctx(incoming)

        # the config record is not conversation and must not enter the context
        assert [item.id for item in session.chat_ctx.items] == [
            "item_greeting",
            incoming.items[2].id,
            incoming.items[3].id,
        ]
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_an_agent_handoff_record_is_ignored_too() -> None:
    """Same category: a framework record the service has no representation for.

    Which agent is active is bookkeeping; what the caller and the model said is
    conversation. Ignoring the record is safe because the parts that do affect
    the model — instructions and tools — arrive through ``update_instructions``
    and ``update_tools``.
    """
    from livekit.agents.llm import AgentHandoff

    model, session = _unconnected_session()
    try:
        session._chat_ctx = _assistant("Ich verbinde Sie weiter.", "item_a")

        incoming = ChatContext.empty()
        incoming.add_message(
            id="item_a", role="assistant", content="Ich verbinde Sie weiter.", interrupted=False
        )
        incoming.insert(AgentHandoff(new_agent_id="agent_2"))
        incoming.add_message(id="item_b", role="user", content="Danke.")

        await session.update_chat_ctx(incoming)

        assert [item.id for item in session.chat_ctx.items] == ["item_a", "item_b"]
    finally:
        await model.aclose()
