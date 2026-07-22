from pathlib import Path

import pytest
from hugging_voice_service.config import ServiceSettings, load_settings
from pydantic import ValidationError

DEFAULT_CONFIG = Path(__file__).parents[1] / "config" / "default.yaml"


def test_default_config_matches_fixed_audio_and_capacity_contract() -> None:
    settings = load_settings(DEFAULT_CONFIG)
    assert settings.server.max_sessions == 2
    assert settings.audio.input_sample_rate == 16_000
    assert settings.audio.output_sample_rate == 24_000
    assert settings.audio.vad_window_samples == 512


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HV_SERVER__MAX_SESSIONS", "1")
    monkeypatch.setenv("HV_SERVER__PORT", "9000")
    settings = load_settings(DEFAULT_CONFIG)
    assert settings.server.max_sessions == 1
    assert settings.server.port == 9000


def test_capacity_and_unknown_backend_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ServiceSettings(server={"max_sessions": 3})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ServiceSettings(backend="cloud")  # type: ignore[call-arg]


def test_yaml_root_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_settings(path)
