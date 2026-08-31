"""A rejected generation request must travel in the future, never as a raise.

Why this is not a style question. The framework calls ``generate_reply`` from two
places with very different protection:

* ``AgentActivity._realtime_reply_task`` guards only the ``await`` on the
  returned future (``except llm.RealtimeError`` around ``await
  generate_reply_fut``), so a *synchronous* raise walks straight past it.
* ``_ToolExecutor._deliver_reply`` — the path every deferred async-tool result
  takes — has no guard at all, and ``_create_speech_task`` attaches no error
  handler. A synchronous raise there ends the delivery task as
  ``Task exception was never retrieved``: the caller loses the finished tool
  result and not one line is logged.

Every plugin shipped with the framework reports this way — OpenAI's realtime
model sends ``response.create`` unconditionally and puts failures into the
future. This one checked its preconditions and raised, and on 2026-08-31 that
silently cost two finished tool calls on staging (room
``staging-vds-web-call-1788157577901``): the caller heard neither result, and the
service's turn kept waiting until its inbound audio queue overflowed.

Because the service runs its own server-side VAD it can start a response at any
moment, so "a response is already pending or active" is a race the client cannot
avoid by checking first — which is exactly why it has to be reportable.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents import APIConnectOptions
from livekit.agents.llm import RealtimeError
from livekit.plugins.hugging_voice.realtime import RealtimeModel


def _unconnected_model() -> RealtimeModel:
    """A model whose session never opens a socket — no service needed here."""
    return RealtimeModel(
        base_url="ws://127.0.0.1:1",
        token="secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=0.01),
    )


@pytest.mark.asyncio
async def test_a_busy_session_rejects_through_the_future() -> None:
    """The case that hit staging: a response is already in flight."""
    model = _unconnected_model()
    session = model.session()
    try:
        in_flight: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        session._pending_generation = in_flight  # type: ignore[assignment]

        rejected = session.generate_reply()

        assert isinstance(rejected, asyncio.Future), "the caller needs a future to await"
        with pytest.raises(RealtimeError, match="already pending or active"):
            await rejected

        in_flight.cancel()
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_over_long_instructions_reject_through_the_future() -> None:
    """The protocol's 8,000-character ceiling is a rejection like any other.

    The deferred tool reply carries per-response instructions, so this ceiling is
    reachable on exactly the path that cannot survive a raise.
    """
    model = _unconnected_model()
    session = model.session()
    try:
        rejected = session.generate_reply(instructions="x" * 8_001)

        with pytest.raises(RealtimeError, match="instructions exceed"):
            await rejected
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_a_disconnected_session_rejects_through_the_future() -> None:
    model = _unconnected_model()
    session = model.session()
    try:
        session._ever_connected = True

        rejected = session.generate_reply()

        with pytest.raises(RealtimeError, match="while disconnected"):
            await rejected
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_the_deferred_reply_path_survives_a_rejection() -> None:
    """Reproduce the framework's unguarded call site, minus the framework.

    ``_deliver_reply`` does ``speech = session.generate_reply(...)`` with no
    ``try``. Standing in for it here proves the shape the framework needs: the
    call returns, and the failure is discoverable by awaiting.
    """
    model = _unconnected_model()
    session = model.session()
    try:
        in_flight: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        session._pending_generation = in_flight  # type: ignore[assignment]

        async def deliver_reply() -> str:
            # no try/except here, exactly as the framework has none
            future = session.generate_reply(instructions="Summarise the tool result.")
            try:
                await future
            except RealtimeError as exc:
                return f"reported: {exc}"
            return "generated"

        assert (await deliver_reply()).startswith("reported: ")

        in_flight.cancel()
    finally:
        await model.aclose()
