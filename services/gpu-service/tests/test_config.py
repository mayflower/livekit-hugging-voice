import json
import wave
from pathlib import Path

import pytest
from hugging_voice_service.config import (
    SegmentationSettings,
    ServiceSettings,
    SpeechSettings,
    VoiceReference,
    VoiceSettings,
    default_voice_reference_dir,
    load_settings,
)
from pydantic import ValidationError

DEFAULT_CONFIG = Path(__file__).parents[1] / "config" / "default.yaml"


def test_default_config_matches_fixed_audio_and_capacity_contract() -> None:
    settings = load_settings(DEFAULT_CONFIG)
    assert settings.server.max_sessions == 2
    assert settings.audio.input_sample_rate == 16_000
    assert settings.audio.output_sample_rate == 24_000
    assert settings.audio.vad_window_samples == 512
    assert settings.speech.resolve_language("en").model_language == "English"
    rendered = settings.speech.resolve_voice("warm_female").render("German")
    assert "native German speaker" in rendered
    assert "Keep the speaker identity unchanged across every utterance" in rendered
    assert settings.speech.generation.do_sample is True
    assert settings.transcription.partial_enabled is False
    assert settings.transcription.partial_interval_ms == 1_000
    assert settings.transcription.partial_max_audio_ms == 4_000
    assert settings.speech.segmentation.first_segment_max_characters == 72
    assert settings.speech.segmentation.next_segment_max_characters == 140
    assert settings.speech.segmentation.hard_max_characters == 160
    assert settings.speech.llm.tool_decision_max_tokens == 96
    assert settings.speech.llm.voice_reply_max_tokens == 128
    assert settings.models.llama_flash_attention == "auto"
    assert settings.models.llama_continuous_batching is True
    assert settings.models.llama_batch_size == 2_048
    assert settings.models.llama_ubatch_size == 512
    assert settings.models.llama_cache_type_k == "f16"
    assert settings.models.llama_cache_type_v == "f16"
    assert settings.models.llama_cache_reuse == 0
    assert settings.models.llama_metrics is True
    assert settings.tts.profile == "compat_qwen3_tts_1_7b_ggml"
    assert settings.tts.chunk_size == 12
    assert settings.tts.worker_count == 1
    assert settings.speech.tts_mode == "voice_clone"
    for voice_id, voice in settings.speech.voices.items():
        assert set(voice.refs) == set(settings.speech.languages)
        for language_id, reference in voice.refs.items():
            assert reference.audio == f"{voice_id}.{language_id}.wav"
            assert reference.text
    assert set(settings.speech.languages) == {"de", "en", "fr", "it"}
    assert len(settings.speech.voices) == 5


@pytest.mark.parametrize(
    ("filename", "profile_id", "llm_profile", "sessions", "workers"),
    [
        ("compat.yaml", "compat_gemma31_qwen17_ggml", "compat_gemma31", 2, 1),
        (
            "multisession-gemma-a4b.yaml",
            "multisession_gemma_a4b_qwen06_cuda",
            "gemma4_26b_a4b",
            4,
            2,
        ),
        (
            "multisession-qwen-a3b.yaml",
            "multisession_qwen_a3b_qwen06_cuda",
            "qwen3_30b_a3b_2507",
            4,
            2,
        ),
    ],
)
def test_complete_version_03_profiles_validate(
    filename: str,
    profile_id: str,
    llm_profile: str,
    sessions: int,
    workers: int,
) -> None:
    settings = load_settings(DEFAULT_CONFIG.parent / "profiles" / filename)
    assert settings.profile_id == profile_id
    assert settings.models.llm_profile == llm_profile
    assert settings.server.max_sessions == sessions
    assert settings.models.llama_parallel_slots == sessions
    assert settings.tts.worker_count == workers


def test_voice_style_is_scoped_without_replacing_the_fixed_identity() -> None:
    rendered = (
        SpeechSettings().resolve_voice("warm_female").render("German", "Speak more excitedly.")
    )
    assert "Speaker identity:" in rendered
    assert "Delivery style only: Speak more excitedly." in rendered
    assert "must not alter the speaker identity" in rendered


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HV_SERVER__MAX_SESSIONS", "8")
    monkeypatch.setenv("HV_MODELS__LLAMA_PARALLEL_SLOTS", "8")
    monkeypatch.setenv("HV_MODELS__LLAMA_CONTEXT_SIZE", "65536")
    monkeypatch.setenv("HV_SERVER__PORT", "9000")
    monkeypatch.setenv("HV_TTS__CHUNK_SIZE", "8")
    settings = load_settings(DEFAULT_CONFIG)
    assert settings.server.max_sessions == 8
    assert settings.models.llama_parallel_slots == 8
    assert settings.models.llama_context_size == 65_536
    assert settings.server.port == 9000
    assert settings.tts.chunk_size == 8


def test_capacity_is_bounded_and_matches_llama_slots() -> None:
    configured = ServiceSettings(
        server={"max_sessions": 20},  # type: ignore[arg-type]
        models={"llama_parallel_slots": 20, "llama_context_size": 65_536},  # type: ignore[arg-type]
    )
    assert configured.server.max_sessions == 20
    assert configured.models.llama_parallel_slots == 20

    with pytest.raises(ValidationError, match="cannot exceed"):
        ServiceSettings(server={"max_sessions": 3})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ServiceSettings(
            server={"max_sessions": 65},  # type: ignore[arg-type]
            models={"llama_parallel_slots": 64, "llama_context_size": 131_072},  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="at least 2048 tokens"):
        ServiceSettings(
            models={"llama_parallel_slots": 20, "llama_context_size": 32_768}  # type: ignore[arg-type]
        )


def test_latency_configuration_is_strictly_bounded() -> None:
    with pytest.raises(ValidationError):
        ServiceSettings(vad={"min_silence_ms": 249})  # type: ignore[arg-type]
    for silence_ms in (300, 350, 500):
        assert (
            ServiceSettings(vad={"min_silence_ms": silence_ms}).vad.min_silence_ms  # type: ignore[arg-type]
            == silence_ms
        )
    with pytest.raises(ValidationError):
        ServiceSettings(
            transcription={"partial_max_audio_ms": 999}  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="first_segment_max_characters"):
        SpeechSettings(
            segmentation=SegmentationSettings(
                first_segment_max_characters=150,
                next_segment_max_characters=140,
                hard_max_characters=160,
            )
        )
    with pytest.raises(ValidationError, match="ubatch"):
        ServiceSettings(
            models={"llama_batch_size": 256, "llama_ubatch_size": 512}  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        ServiceSettings(
            models={"llama_extra_args": ["--dangerous"]}  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="requires qwen3_tts_0_6b_cuda"):
        ServiceSettings(tts={"worker_count": 2})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match=r"cannot exceed server\.max_sessions"):
        ServiceSettings(
            server={"max_sessions": 1},  # type: ignore[arg-type]
            tts={
                "profile": "qwen3_tts_0_6b_cuda",
                "worker_count": 2,
            },  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="production TTS worker_count"):
        ServiceSettings(
            server={"max_sessions": 4},  # type: ignore[arg-type]
            models={
                "llama_parallel_slots": 4,
                "llama_context_size": 32_768,
            },  # type: ignore[arg-type]
            tts={
                "profile": "qwen3_tts_0_6b_cuda",
                "worker_count": 3,
            },  # type: ignore[arg-type]
        )
    benchmark = ServiceSettings(
        profile_id="multisession_gemma_a4b_qwen06_cuda",
        server={"max_sessions": 4},  # type: ignore[arg-type]
        models={
            "llm_profile": "gemma4_26b_a4b",
            "llama_parallel_slots": 4,
            "llama_context_size": 32_768,
        },  # type: ignore[arg-type]
        tts={
            "profile": "qwen3_tts_0_6b_cuda",
            "worker_count": 4,
            "deployment_mode": "benchmark",
            "chunk_size": 4,
        },  # type: ignore[arg-type]
    )
    assert benchmark.tts.worker_count == 4
    with pytest.raises(ValidationError, match="profile_id does not match"):
        ServiceSettings(
            models={"llm_profile": "gemma4_26b_a4b"},  # type: ignore[arg-type]
        )


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ServiceSettings(backend="cloud")  # type: ignore[call-arg]


def test_speech_defaults_must_resolve_and_unknown_session_options_are_rejected() -> None:
    with pytest.raises(ValidationError, match="default_language"):
        SpeechSettings(default_language="ar")
    speech = SpeechSettings()
    with pytest.raises(ValueError, match="unsupported language"):
        speech.resolve_language("ar")
    with pytest.raises(ValueError, match="unsupported voice"):
        speech.resolve_voice("unknown")
    with pytest.raises(ValidationError):
        SpeechSettings(
            languages={
                "de": {  # type: ignore[dict-item]
                    "model_language": "Klingon",
                    "response_instruction": "Respond in Klingon.",
                }
            }
        )


def test_yaml_root_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_settings(path)


def test_voice_clone_requires_a_reference_per_language() -> None:
    with pytest.raises(ValidationError, match="lacks a voice_clone reference"):
        SpeechSettings(
            tts_mode="voice_clone",
            voices={"warm_female": VoiceSettings(instructions="A {language} voice.")},
        )


def test_voice_design_mode_does_not_require_references() -> None:
    speech = SpeechSettings(
        tts_mode="voice_design",
        voices={"warm_female": VoiceSettings(instructions="A {language} voice.")},
    )
    assert speech.tts_mode == "voice_design"


def test_voice_reference_filenames_are_plain_wav_names() -> None:
    for bad in ("../warm.wav", "warm/../x.wav", "warm.mp3", "/abs.wav", "a..b.wav"):
        with pytest.raises(ValidationError):
            VoiceReference(audio=bad, text="transcript")


def test_voice_reference_resolution_and_path_override(tmp_path: Path) -> None:
    speech = SpeechSettings(voice_ref_dir=tmp_path)
    reference = speech.resolve_voice_reference("warm_female", "de")
    assert speech.voice_reference_path(reference) == tmp_path / "warm_female.de.wav"
    with pytest.raises(ValueError, match="no voice_clone reference"):
        speech.resolve_voice_reference("warm_female", "es")
    with pytest.raises(ValueError, match="unsupported voice"):
        speech.resolve_voice_reference("unknown_voice", "de")


def test_packaged_voice_references_are_complete_and_valid() -> None:
    speech = SpeechSettings()
    for voice in speech.voices.values():
        for reference in voice.refs.values():
            path = speech.voice_reference_path(reference)
            with wave.open(str(path), "rb") as recording:
                assert recording.getnchannels() == 1
                assert recording.getsampwidth() == 2
                assert recording.getframerate() == 24_000
                duration = recording.getnframes() / recording.getframerate()
                assert 3.0 <= duration <= 15.0


def test_reference_transcripts_are_identical_across_code_yaml_and_recordings() -> None:
    """The transcript must match the frozen audio exactly (it anchors the ICL
    clone), so the code defaults, default.yaml, and the recording provenance
    must never drift apart."""

    code = SpeechSettings()
    from_yaml = load_settings(DEFAULT_CONFIG).speech
    metadata = json.loads(
        (default_voice_reference_dir() / "metadata.json").read_text(encoding="utf-8")
    )
    recorded = {
        (artifact["voice"], artifact["language"]): artifact["text"]
        for artifact in metadata["artifacts"]
    }
    assert set(from_yaml.voices) == set(code.voices)
    for voice_id, voice in from_yaml.voices.items():
        assert set(voice.refs) == set(code.voices[voice_id].refs)
        for language, reference in voice.refs.items():
            assert reference.text == code.voices[voice_id].refs[language].text
            assert reference.text == recorded[(voice_id, language)]
            assert reference.audio == code.voices[voice_id].refs[language].audio
