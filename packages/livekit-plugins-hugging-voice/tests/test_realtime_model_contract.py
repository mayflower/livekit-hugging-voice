"""The model's side of the LiveKit RealtimeModel contract.

livekit-agents 1.6.8 added a ``turn_detection_disabled`` keyword to
``RealtimeModel.session``. AgentActivity passes it on every call, so a plugin
that omits it raises TypeError at session start -- after the room join, which
makes the call look silent rather than broken. These tests pin the calling
convention independently of the livekit-agents version the plugin resolves
against, because that version differs between this repo and its consumers.
"""

from __future__ import annotations

import pytest
from livekit.plugins.hugging_voice import RealtimeModel


def build_model() -> RealtimeModel:
    return RealtimeModel(base_url="ws://127.0.0.1:9/v1/realtime", token="test-token")


@pytest.mark.asyncio
async def test_session_accepts_the_turn_detection_disabled_keyword() -> None:
    model = build_model()
    try:
        session = model.session(turn_detection_disabled=False)
        assert session is not None
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_session_ignores_turn_detection_disabled_when_it_is_requested() -> None:
    # The service always runs its own server-side VAD, so the plugin honours
    # nothing here. It stays truthful by never claiming the capability.
    model = build_model()
    try:
        assert model.capabilities.turn_detection is True
        assert getattr(model.capabilities, "can_disable_turn_detection", False) is False
        session = model.session(turn_detection_disabled=True)
        assert session is not None
    finally:
        await model.aclose()
