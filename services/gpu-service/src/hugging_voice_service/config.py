"""Validated service configuration with explicit YAML and HV_ overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerSettings(StrictConfig):
    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1, le=65_535)
    max_sessions: int = Field(default=2, ge=1, le=2)
    token_file: Path = Path("/run/secrets/hugging_voice_token")
    inbound_queue_size: int = Field(default=128, ge=8, le=1_024)
    outbound_queue_size: int = Field(default=256, ge=8, le=2_048)
    drain_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)


class ModelSettings(StrictConfig):
    root: Path = Path("/models")
    lock_file: Path = Path("/models/manifest.lock.json")
    llama_server_binary: Path = Path("/usr/local/bin/llama-server")
    llama_cpp_commit: Literal["3ce7da2c852c538c4c5f9806da27029cf8c9cc4a"] = (
        "3ce7da2c852c538c4c5f9806da27029cf8c9cc4a"
    )
    llama_port: int = Field(default=8081, ge=1, le=65_535)
    llama_context_size: Literal[32_768] = 32_768
    llama_parallel_slots: Literal[2] = 2
    llama_startup_timeout_seconds: float = Field(default=600.0, ge=10.0, le=1_800.0)
    llama_shutdown_timeout_seconds: float = Field(default=15.0, ge=1.0, le=120.0)


class AudioSettings(StrictConfig):
    input_encoding: Literal["pcm_s16le"] = "pcm_s16le"
    input_sample_rate: Literal[16_000] = 16_000
    input_channels: Literal[1] = 1
    input_chunk_ms: Literal[40] = 40
    output_encoding: Literal["pcm_s16le"] = "pcm_s16le"
    output_sample_rate: Literal[24_000] = 24_000
    output_channels: Literal[1] = 1
    output_frame_ms: Literal[20] = 20
    vad_window_samples: Literal[512] = 512


class VADSettings(StrictConfig):
    threshold: float = Field(default=0.6, ge=0.1, le=0.95)
    min_speech_ms: int = Field(default=384, ge=96, le=2_000)
    min_speech_continuation_ms: int = Field(default=192, ge=0, le=1_000)
    min_silence_ms: int = Field(default=500, ge=100, le=3_000)
    speech_pad_ms: int = Field(default=30, ge=0, le=500)
    short_segment_merge_ms: Literal[0] = 0
    interrupt_response: bool = True


class ServiceSettings(BaseSettings):
    """Complete fixed service configuration.

    Environment variables use nested names such as
    ``HV_SERVER__MAX_SESSIONS=1`` and take precedence over YAML values.
    """

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=True,
        env_prefix="HV_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls, dotenv_settings
        return env_settings, init_settings, file_secret_settings


def load_settings(
    path: Path | str = Path("services/gpu-service/config/default.yaml"),
) -> ServiceSettings:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read service config {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"service config {config_path} must contain a YAML mapping")
    return ServiceSettings(**_string_key_mapping(raw, config_path=config_path))


def _string_key_mapping(value: dict[Any, Any], *, config_path: Path) -> dict[str, Any]:
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"service config {config_path} contains a non-string key")
    return value
