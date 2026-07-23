import json
import wave
from pathlib import Path

import pytest
from hugging_voice_service.config import (
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
    assert settings.speech.tts_mode == "voice_clone"
    for voice_id, voice in settings.speech.voices.items():
        assert set(voice.refs) == set(settings.speech.languages)
        for language_id, reference in voice.refs.items():
            assert reference.audio == f"{voice_id}.{language_id}.wav"
            assert reference.text
    assert set(settings.speech.languages) == {"de", "en", "fr", "it"}
    assert len(settings.speech.voices) == 5


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
    settings = load_settings(DEFAULT_CONFIG)
    assert settings.server.max_sessions == 8
    assert settings.models.llama_parallel_slots == 8
    assert settings.models.llama_context_size == 65_536
    assert settings.server.port == 9000


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
