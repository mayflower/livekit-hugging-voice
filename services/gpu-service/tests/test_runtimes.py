from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from aiohttp import web
from hugging_voice_protocol.audio import OUTPUT_FRAME_BYTES
from hugging_voice_protocol.events import FunctionDefinition, FunctionTool
from hugging_voice_service.config import VADSettings
from hugging_voice_service.llm_profiles import LLM_PROFILES
from hugging_voice_service.runtimes.llama_cpp_chat import (
    BASE_PROMPT,
    GemmaMessage,
    LlamaCppChatRuntime,
    TextDelta,
    TextUsage,
    ToolCall,
)
from hugging_voice_service.runtimes.parakeet import ParakeetRuntime
from hugging_voice_service.runtimes.qwen_tts import QwenCudaGraphTTSRuntime, QwenTTSRuntime
from hugging_voice_service.runtimes.silero import SessionVAD


def test_llm_profiles_are_closed_and_qwen_2507_is_explicitly_non_thinking() -> None:
    assert set(LLM_PROFILES) == {
        "compat_gemma31",
        "gemma4_26b_a4b",
        "qwen3_30b_a3b_2507",
    }
    qwen = LLM_PROFILES["qwen3_30b_a3b_2507"]
    assert qwen.reasoning_mode == "off"
    assert dict(qwen.chat_template_kwargs) == {}
    assert "enable_thinking" not in qwen.chat_template_kwargs
    assert all(profile.readiness_probe == "two_step_tool" for profile in LLM_PROFILES.values())


class Probability:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class ScriptedVADModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = iter(probabilities)
        self.reset_count = 0

    def __call__(self, samples: object, sample_rate: int) -> Probability:
        assert sample_rate == 16_000
        assert len(samples) == 512  # type: ignore[arg-type]
        return Probability(next(self._probabilities))

    def reset_states(self) -> None:
        self.reset_count += 1


class FakeCudaGraphQwenModel:
    def __init__(self) -> None:
        self.warmups: list[int] = []
        self.prepared: list[tuple[str, str]] = []
        self.prompts_used: list[object] = []
        self.closed_streams = 0
        self.non_finite = False

    def warmup(self, *, prefill_len: int = 100) -> None:
        self.warmups.append(prefill_len)

    def create_voice_clone_prompt(self, *, ref_audio: str, ref_text: str) -> object:
        self.prepared.append((ref_audio, ref_text))
        return {"reference": ref_audio, "text": ref_text}

    def generate_voice_clone_streaming(
        self, **kwargs: Any
    ) -> Iterator[tuple[np.ndarray, int, dict[str, Any]]]:
        self.prompts_used.append(kwargs["voice_clone_prompt"])

        def generate() -> Iterator[tuple[np.ndarray, int, dict[str, Any]]]:
            try:
                value = np.nan if self.non_finite else 0.1
                yield np.full(480, value, dtype=np.float32), 24_000, {}
                yield np.full(480, 0.2, dtype=np.float32), 24_000, {}
            finally:
                self.closed_streams += 1

        return generate()


def test_session_vad_has_isolated_state_remainder_and_deterministic_boundaries() -> None:
    # Thresholds are pinned here on purpose: this covers remainder handling and
    # state isolation, so it must not need recomputing whenever VADSettings is
    # recalibrated. Turn calibration is covered separately below.
    model = ScriptedVADModel([0.9] * 12 + [0.1] * 16)
    vad = SessionVAD(
        threshold=0.6,
        min_speech_ms=384,
        min_speech_continuation_ms=192,
        min_silence_ms=500,
        speech_pad_ms=30,
        model_factory=lambda: model,
        sample_tensor_factory=lambda samples: samples,
    )
    payload = bytes(SessionVAD.window_bytes * 28 + 10)
    signals = vad.process_pcm16(payload)
    assert [(signal.kind, signal.sample_index) for signal in signals] == [
        ("speech_started", 0),
        ("speech_stopped", 6_624),
    ]
    assert vad.buffered_bytes == 10
    assert not vad.speaking
    vad.reset()
    assert model.reset_count == 1
    assert vad.buffered_bytes == 0


def test_a_short_word_opens_a_turn_and_keeps_its_onset_and_coda() -> None:
    """A bare "ja" has to reach the STT — the 2026-09-02 call regression.

    Seven 32 ms windows is ~224 ms, the length of a spoken "ja". Under the
    Silero reference defaults it produced no ``speech_started`` at all (384 ms
    of accumulated speech was never reached), so the answer was dropped before
    transcription while the separate Deepgram transcript still showed it in the
    dashboard. The padding also has to outlast the speech on both ends: 30 ms
    is a single window, and cutting the onset off a short word is what turned
    "ja, das" into "does".

    The expectations are stated against the speech boundaries rather than
    absolute sample counts, so this stays meaningful under recalibration and
    only fails if a short word stops surviving.
    """
    settings = VADSettings()
    silence_before, speech, silence_after = 10, 7, 16
    model = ScriptedVADModel([0.1] * silence_before + [0.9] * speech + [0.1] * silence_after)
    vad = SessionVAD(
        threshold=settings.threshold,
        min_speech_ms=settings.min_speech_ms,
        min_speech_continuation_ms=settings.min_speech_continuation_ms,
        min_silence_ms=settings.min_silence_ms,
        speech_pad_ms=settings.speech_pad_ms,
        model_factory=lambda: model,
        sample_tensor_factory=lambda samples: samples,
    )
    windows = silence_before + speech + silence_after
    signals = vad.process_pcm16(bytes(SessionVAD.window_bytes * windows))

    assert [signal.kind for signal in signals] == ["speech_started", "speech_stopped"]
    speech_starts_at = silence_before * SessionVAD.window_samples
    speech_ends_at = (silence_before + speech) * SessionVAD.window_samples
    pad_samples = settings.speech_pad_ms * SessionVAD.sample_rate // 1_000
    # A lower bound on the calibration, not just on the mechanism: anything
    # near the 30 ms reference default is a single window and clips the word.
    assert settings.speech_pad_ms >= 100
    assert signals[0].sample_index == speech_starts_at - pad_samples
    assert signals[1].sample_index == speech_ends_at + pad_samples


def test_silero_recurrent_model_is_constructed_per_session() -> None:
    created: list[ScriptedVADModel] = []

    def factory() -> ScriptedVADModel:
        model = ScriptedVADModel([0.1])
        created.append(model)
        return model

    first = SessionVAD(model_factory=factory, sample_tensor_factory=lambda samples: samples)
    second = SessionVAD(model_factory=factory, sample_tensor_factory=lambda samples: samples)
    assert len(created) == 2
    first.reset()
    assert (created[0].reset_count, created[1].reset_count) == (1, 0)
    assert first is not second


def test_bundled_silero_model_satisfies_the_runtime_contract() -> None:
    """Guard the real ONNX build; every other VAD test injects a fake.

    Loading it here is what would catch a ``silero-vad`` bump that renames the
    ONNX artifact or changes what the wrapper returns.
    """

    pytest.importorskip("torch", reason="Silero needs the gpu extra")
    pytest.importorskip("silero_vad", reason="Silero needs the gpu extra")

    vad = SessionVAD()
    assert vad.process_pcm16(bytes(SessionVAD.window_bytes * 4)) == []
    assert vad.flush() == []


class FakeParakeetModel:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray[Any, Any]] = []

    def transcribe(self, audio: Any, timestamps: bool = False) -> str:
        assert timestamps is False
        self.inputs.append(audio)
        return "  Guten Tag.  "


@pytest.mark.asyncio
async def test_parakeet_loads_once_uses_local_file_and_runs_off_loop(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.nemo"
    checkpoint.write_bytes(b"checkpoint")
    model = FakeParakeetModel()
    runtime = ParakeetRuntime(
        checkpoint,
        model_factory=lambda path: model,
        cuda_probe=lambda: None,
    )
    runtime.load()
    assert runtime.load_count == 1
    with pytest.raises(RuntimeError, match="already loaded"):
        runtime.load()
    assert await runtime.transcribe_final(bytes(3_200)) == "Guten Tag."
    assert model.inputs[0].dtype == np.float32
    runtime.close()


class FakeQwenModel:
    def __init__(self) -> None:
        self.warmups = 0
        self.closed_streams = 0
        self.calls: list[dict[str, Any]] = []
        self.clone_calls: list[dict[str, Any]] = []

    def warmup(self, *, prefill_len: int = 100) -> None:
        assert prefill_len == 100
        self.warmups += 1

    def generate_voice_design_streaming(
        self, **kwargs: Any
    ) -> Iterator[tuple[np.ndarray[Any, Any], int, dict[str, Any]]]:
        self.calls.append(kwargs)
        try:
            yield np.full(600, 0.5, dtype=np.float32), 24_000, {}
            yield np.full(100, -0.5, dtype=np.float32), 24_000, {}
        finally:
            self.closed_streams += 1

    def generate_voice_clone_streaming(
        self, **kwargs: Any
    ) -> Iterator[tuple[np.ndarray[Any, Any], int, dict[str, Any]]]:
        self.clone_calls.append(kwargs)
        yield np.full(600, 0.5, dtype=np.float32), 24_000, {}
        yield np.full(100, -0.5, dtype=np.float32), 24_000, {}


@pytest.mark.asyncio
async def test_qwen_forwards_speech_options_and_emits_exact_pcm_frames(tmp_path: Path) -> None:
    talker = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"
    talker.write_bytes(b"talker")
    codec.write_bytes(b"codec")
    model = FakeQwenModel()
    runtime = QwenTTSRuntime(
        talker,
        codec,
        model_factory=lambda talker_path, codec_path: model,
        cuda_probe=lambda: None,
    )
    runtime.load()
    runtime.warmup()
    assert runtime.load_count == 1
    assert model.closed_streams == 1
    frames = [
        frame
        async for frame in runtime.stream_pcm16_frames(
            "Hello.",
            language="English",
            instructions="Speak warmly.",
        )
    ]
    assert [len(frame) for frame in frames] == [OUTPUT_FRAME_BYTES, OUTPUT_FRAME_BYTES]
    call = model.calls[-1]
    assert "speaker" not in call
    assert call["language"] == "English"
    assert call["instruct"] == "Speak warmly."
    assert call["do_sample"] is True
    assert call["temperature"] == 0.9
    assert call["top_k"] == 50
    assert call["top_p"] == 1.0
    assert call["repetition_penalty"] == 1.05
    assert model.closed_streams == 2


@pytest.mark.asyncio
async def test_qwen_closes_native_iterator_when_audio_consumer_stops_early(
    tmp_path: Path,
) -> None:
    talker = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"
    talker.write_bytes(b"talker")
    codec.write_bytes(b"codec")
    model = FakeQwenModel()
    runtime = QwenTTSRuntime(
        talker,
        codec,
        model_factory=lambda talker_path, codec_path: model,
        cuda_probe=lambda: None,
    )
    runtime.load()
    stream = runtime.stream_pcm16_frames(
        "Hello.",
        language="English",
        instructions="Speak warmly.",
    )

    assert len(await anext(stream)) == OUTPUT_FRAME_BYTES
    await cast(AsyncGenerator[bytes, None], stream).aclose()

    assert model.closed_streams == 1


@pytest.mark.asyncio
async def test_qwen_voice_clone_streams_from_the_frozen_reference(tmp_path: Path) -> None:
    talker = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"
    reference = tmp_path / "warm_female.de.wav"
    talker.write_bytes(b"talker")
    codec.write_bytes(b"codec")
    reference.write_bytes(b"reference")
    model = FakeQwenModel()
    runtime = QwenTTSRuntime(
        talker,
        codec,
        model_factory=lambda talker_path, codec_path: model,
        cuda_probe=lambda: None,
        mode="voice_clone",
        voice_references=(
            ("German", reference, "Willkommen bei unserem Sprachassistenten!"),
            ("English", reference, "Welcome to our voice assistant!"),
        ),
    )
    runtime.load()
    runtime.warmup()
    assert not model.calls
    # Warmup pre-extracts every frozen reference once, speaking its own
    # transcript so the stream cannot legitimately be empty.
    assert len(model.clone_calls) == 2
    assert model.clone_calls[0]["language"] == "German"
    assert model.clone_calls[0]["text"] == "Willkommen bei unserem Sprachassistenten!"
    assert model.clone_calls[1]["language"] == "English"
    assert model.clone_calls[1]["text"] == "Welcome to our voice assistant!"
    frames = [
        frame
        async for frame in runtime.stream_pcm16_frames(
            "Hello.",
            language="English",
            instructions="Delivery style only: excited.",
            ref_audio=reference,
            ref_text="Willkommen bei unserem Sprachassistenten!",
        )
    ]
    assert [len(frame) for frame in frames] == [OUTPUT_FRAME_BYTES, OUTPUT_FRAME_BYTES]
    call = model.clone_calls[-1]
    assert call["ref_audio"] == str(reference)
    assert call["ref_text"] == "Willkommen bei unserem Sprachassistenten!"
    assert "instruct" not in call
    assert call["do_sample"] is True
    with pytest.raises(RuntimeError, match="reference"):
        async for _ in runtime.stream_pcm16_frames("Hi.", language="English", instructions=""):
            pass


def test_qwen_voice_clone_requires_the_frozen_references(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="voice references"):
        QwenTTSRuntime(
            tmp_path / "talker.gguf",
            tmp_path / "codec.gguf",
            model_factory=lambda talker_path, codec_path: FakeQwenModel(),
            cuda_probe=lambda: None,
            mode="voice_clone",
        )


def _cuda_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "qwen-cuda"
    (model_dir / "speech_tokenizer").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"model")
    (model_dir / "speech_tokenizer" / "model.safetensors").write_bytes(b"codec")
    return model_dir


@pytest.mark.asyncio
async def test_qwen_cuda_graph_prepares_all_references_and_reuses_prompts(
    tmp_path: Path,
) -> None:
    model = FakeCudaGraphQwenModel()
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    runtime = QwenCudaGraphTTSRuntime(
        _cuda_model_dir(tmp_path),
        model_factory=lambda path: model,
        cuda_probe=lambda: None,
        inference_context_factory=lambda: nullcontext(),
        voice_references=(
            ("German", first, "Erste Referenz."),
            ("Italian", second, "Secondo riferimento."),
        ),
        chunk_size=4,
    )
    runtime.load()
    runtime.warmup()
    frames = [
        frame
        async for frame in runtime.stream_pcm16_frames(
            "Hallo.",
            language="German",
            instructions="ignored",
            ref_audio=first,
            ref_text="Erste Referenz.",
        )
    ]

    assert runtime.load_count == 1
    assert model.warmups == [100]
    assert model.prepared == [
        (str(first), "Erste Referenz."),
        (str(second), "Secondo riferimento."),
    ]
    assert model.prompts_used[-1] == {"reference": str(first), "text": "Erste Referenz."}
    assert frames and all(len(frame) == 960 for frame in frames)
    runtime.close()


@pytest.mark.asyncio
async def test_qwen_cuda_graph_closes_generator_on_cancellation_and_rejects_nonfinite(
    tmp_path: Path,
) -> None:
    model = FakeCudaGraphQwenModel()
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"voice")
    runtime = QwenCudaGraphTTSRuntime(
        _cuda_model_dir(tmp_path),
        model_factory=lambda path: model,
        cuda_probe=lambda: None,
        inference_context_factory=lambda: nullcontext(),
        voice_references=(("French", reference, "Une référence."),),
    )
    runtime.load()
    runtime.warmup()
    cancelled = False
    stream = runtime.stream_pcm16_frames(
        "Bonjour.",
        language="French",
        instructions="",
        ref_audio=reference,
        ref_text="Une référence.",
        cancelled=lambda: cancelled,
    )
    assert len(await anext(stream)) == 960
    cancelled = True
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert model.closed_streams >= 2  # startup streaming probe plus cancelled job

    model.non_finite = True
    with pytest.raises(RuntimeError, match="non-finite"):
        async for _ in runtime.stream_pcm16_frames(
            "Encore.",
            language="French",
            instructions="",
            ref_audio=reference,
            ref_text="Une référence.",
        ):
            pass
    runtime.close()


@pytest.mark.asyncio
async def test_gemma_suppresses_reasoning_and_disables_thinking_in_request() -> None:
    received: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        received.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        events = [
            {"choices": [{"delta": {"reasoning_content": "geheim"}}]},
            {"choices": [{"delta": {"content": "Antwort "}}]},
            {"choices": [{"delta": {"content": "sichtbar."}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        ]
        for event in events:
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]
    runtime = LlamaCppChatRuntime(port=port)
    try:
        events = [
            event
            async for event in runtime.stream_response(
                messages=[GemmaMessage(role="user", content="Sag etwas.")]
            )
        ]
    finally:
        await runtime.aclose()
        await runner.cleanup()

    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == (
        "Antwort sichtbar."
    )
    assert [event for event in events if isinstance(event, TextUsage)] == [TextUsage(4, 2, 6)]
    assert runtime.reasoning_violations == 1
    assert received[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert received[0]["messages"][0]["role"] == "system"
    assert "hidden analysis" in received[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_gemma_prefix_prefill_uses_zero_tokens_fixed_slot_and_cache() -> None:
    received: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.Response:
        received.append(await request.json())
        return web.json_response(
            {
                "choices": [],
                "usage": {"prompt_tokens": 23, "completion_tokens": 0, "total_tokens": 23},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    runtime = LlamaCppChatRuntime(port=sockets[0].getsockname()[1], parallel_slots=4)
    try:
        tokens = await runtime.prefill(
            instructions="Keep it short.",
            language_instruction="Respond in Italian.",
            slot_id=3,
        )
    finally:
        await runtime.aclose()
        await runner.cleanup()

    assert tokens == 23
    assert received == [
        {
            "model": "gemma-4-31b",
            "messages": [
                {"role": "system", "content": BASE_PROMPT},
                {"role": "system", "content": "Respond in Italian."},
                {"role": "system", "content": "Keep it short."},
            ],
            "stream": False,
            "max_tokens": 0,
            "temperature": 0,
            "cache_prompt": True,
            "id_slot": 3,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]


@pytest.mark.asyncio
async def test_gemma_strips_fragmented_leading_thinking_block() -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for content in (" <thi", "nk>nicht sichtbar", "</think>Ja."):
            payload = {"choices": [{"delta": {"content": content}}]}
            await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    runtime = LlamaCppChatRuntime(port=sockets[0].getsockname()[1])
    try:
        events = [
            event
            async for event in runtime.stream_response(
                messages=[GemmaMessage(role="user", content="Test")]
            )
        ]
    finally:
        await runtime.aclose()
        await runner.cleanup()
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["Ja."]
    assert runtime.reasoning_violations == 1


@pytest.mark.asyncio
async def test_gemma_allows_exactly_two_parallel_isolated_streams() -> None:
    active = 0
    maximum = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal active, maximum
        payload = await request.json()
        canary = payload["messages"][-1]["content"]
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            two_started.set()
        await release.wait()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        event = {"choices": [{"delta": {"content": canary}}]}
        await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        active -= 1
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    runtime = LlamaCppChatRuntime(port=sockets[0].getsockname()[1])

    async def consume(canary: str) -> str:
        chunks = [
            event.text
            async for event in runtime.stream_response(
                messages=[GemmaMessage(role="user", content=canary)]
            )
            if isinstance(event, TextDelta)
        ]
        return "".join(chunks)

    tasks = [asyncio.create_task(consume(canary)) for canary in ("ALPHA", "BETA", "GAMMA")]
    try:
        await asyncio.wait_for(two_started.wait(), timeout=1.0)
        assert maximum == 2
        assert not tasks[2].done()
        release.set()
        assert await asyncio.gather(*tasks) == ["ALPHA", "BETA", "GAMMA"]
        assert maximum == 2
    finally:
        release.set()
        await runtime.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gemma_aggregates_one_structured_tool_call_with_slot_cache() -> None:
    requests: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        requests.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        fragments = [
            {"index": 0, "id": "call_add", "function": {"name": "add_", "arguments": '{"a":'}},
            {"index": 0, "function": {"name": "numbers", "arguments": '19,"b":23}'}},
        ]
        for fragment in fragments:
            event = {"choices": [{"delta": {"tool_calls": [fragment]}}]}
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    runtime = LlamaCppChatRuntime(port=sockets[0].getsockname()[1], parallel_slots=4)
    tool = FunctionTool(
        function=FunctionDefinition(
            name="add_numbers",
            description="Add two integers.",
            parameters={"type": "object", "properties": {}},
        )
    )
    try:
        events = [
            event
            async for event in runtime.stream_response(
                messages=[GemmaMessage(role="user", content="Addiere 19 und 23")],
                tools=[tool],
                tool_choice="required",
                slot_id=3,
            )
        ]
        calls = [event for event in events if isinstance(event, ToolCall)]
        assert calls == [
            ToolCall(call_id="call_add", name="add_numbers", arguments='{"a":19,"b":23}')
        ]
        assert requests[0]["id_slot"] == 3
        assert requests[0]["cache_prompt"] is True
        assert requests[0]["parallel_tool_calls"] is False
        assert requests[0]["temperature"] == 0.0
        assert requests[0]["max_tokens"] == 128
    finally:
        await runtime.aclose()
        await runner.cleanup()


async def _sse_runtime(
    deltas: list[dict[str, Any]],
) -> tuple[LlamaCppChatRuntime, web.AppRunner]:
    """Serve a fixed list of `choices[0].delta` payloads as one SSE stream."""

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for delta in deltas:
            event = {"choices": [{"delta": delta}]}
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    runtime = LlamaCppChatRuntime(port=sockets[0].getsockname()[1])
    return runtime, runner


def _search_tool(name: str = "knowledge_base_search") -> FunctionTool:
    return FunctionTool(
        function=FunctionDefinition(
            name=name,
            description="Search the knowledge base.",
            parameters={"type": "object", "properties": {}},
        )
    )


@pytest.mark.asyncio
async def test_gemma_streams_text_then_tool_call_and_counts_the_violation() -> None:
    """An announcement followed by a tool call must survive, not abort the call.

    Gemma does this on its own: it says "let me look that up" and appends the
    call to the same completion. Raising here killed the whole session.
    """
    runtime, runner = await _sse_runtime(
        [
            {"content": "Ich schaue das nach. "},
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_kb",
                        "function": {
                            "name": "knowledge_base_search",
                            "arguments": '{"query":"Muenchen"}',
                        },
                    }
                ]
            },
        ]
    )
    try:
        events = [
            event
            async for event in runtime.stream_response(
                messages=[GemmaMessage(role="user", content="Was gibt es zu sehen?")],
                tools=[_search_tool()],
                slot_id=0,
            )
        ]
        assert [type(event).__name__ for event in events] == ["TextDelta", "ToolCall"]
        assert cast(TextDelta, events[0]).text == "Ich schaue das nach. "
        assert cast(ToolCall, events[1]).name == "knowledge_base_search"
        assert runtime.mixed_output_violations == 1
    finally:
        await runtime.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gemma_drops_trailing_text_after_a_tool_call() -> None:
    """Text after the call comments on a result that does not exist yet."""
    runtime, runner = await _sse_runtime(
        [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_kb",
                        "function": {
                            "name": "knowledge_base_search",
                            "arguments": '{"query":"Muenchen"}',
                        },
                    }
                ]
            },
            {"content": "Einen Moment bitte."},
        ]
    )
    try:
        events = [
            event
            async for event in runtime.stream_response(
                messages=[GemmaMessage(role="user", content="Was gibt es zu sehen?")],
                tools=[_search_tool()],
                slot_id=0,
            )
        ]
        assert [type(event).__name__ for event in events] == ["ToolCall"]
        assert runtime.mixed_output_violations == 1
    finally:
        await runtime.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gemma_keeps_spoken_text_when_the_trailing_tool_call_is_malformed() -> None:
    """Once text is on the wire, a broken call must not retroactively fail the turn.

    The 96-token tool budget makes truncated arguments a real possibility after
    an announcement has eaten into it.
    """
    runtime, runner = await _sse_runtime(
        [
            {"content": "Ich schaue das nach. "},
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_kb",
                        "function": {"name": "unknown_tool", "arguments": "{"},
                    }
                ]
            },
        ]
    )
    try:
        events = [
            event
            async for event in runtime.stream_response(
                messages=[GemmaMessage(role="user", content="Was gibt es zu sehen?")],
                tools=[_search_tool()],
                slot_id=0,
            )
        ]
        assert [type(event).__name__ for event in events] == ["TextDelta"]
        assert runtime.mixed_output_violations == 1
    finally:
        await runtime.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gemma_keeps_spoken_text_when_tool_arguments_are_not_finite_json() -> None:
    """NaN in the arguments must not retroactively kill a turn that already spoke.

    json.loads accepts NaN, so the value survives push and only canonical_json
    rejects it — with a bare ValueError, not a ToolCallValidationError. Guarding
    the narrow type left the fatal path open for exactly this input.
    """
    runtime, runner = await _sse_runtime(
        [
            {"content": "Ich schaue das nach. "},
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_kb",
                        "function": {
                            "name": "knowledge_base_search",
                            "arguments": '{"query": NaN}',
                        },
                    }
                ]
            },
        ]
    )
    try:
        events = [
            event
            async for event in runtime.stream_response(
                messages=[GemmaMessage(role="user", content="Was gibt es zu sehen?")],
                tools=[_search_tool()],
                slot_id=0,
            )
        ]
        assert [type(event).__name__ for event in events] == ["TextDelta"]
        assert runtime.mixed_output_violations == 1
    finally:
        await runtime.aclose()
        await runner.cleanup()
