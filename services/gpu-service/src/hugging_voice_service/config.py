"""Validated service configuration with explicit YAML and HV_ overrides."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from .llm_profiles import LLMProfileId


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerSettings(StrictConfig):
    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1, le=65_535)
    max_sessions: int = Field(default=2, ge=1, le=64)
    token_file: Path = Path("/run/secrets/hugging_voice_token")
    inbound_queue_size: int = Field(default=128, ge=8, le=1_024)
    outbound_queue_size: int = Field(default=256, ge=8, le=2_048)
    drain_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)


class ModelSettings(StrictConfig):
    root: Path = Path("/models")
    lock_file: Path = Path("/models/manifest.lock.json")
    llama_server_binary: Path = Path("/usr/local/bin/llama-server")
    llm_profile: LLMProfileId = "compat_gemma31"
    llama_cpp_commit: Literal["3ce7da2c852c538c4c5f9806da27029cf8c9cc4a"] = (
        "3ce7da2c852c538c4c5f9806da27029cf8c9cc4a"
    )
    llama_port: int = Field(default=8081, ge=1, le=65_535)
    # llama.cpp divides this total context across all parallel sequence slots.
    llama_context_size: int = Field(default=32_768, ge=2_048, le=1_048_576)
    llama_parallel_slots: int = Field(default=2, ge=1, le=64)
    # These are a narrow allowlist confirmed against llama_cpp_commit.
    llama_flash_attention: Literal["auto", "on"] = "auto"
    llama_continuous_batching: Literal[True] = True
    llama_batch_size: int = Field(default=2_048, ge=32, le=4_096)
    llama_ubatch_size: int = Field(default=512, ge=32, le=2_048)
    llama_cache_type_k: Literal["f16", "q8_0"] = "f16"
    llama_cache_type_v: Literal["f16", "q8_0"] = "f16"
    llama_cache_reuse: int = Field(default=0, ge=0, le=2_048)
    llama_metrics: Literal[True] = True
    llama_startup_timeout_seconds: float = Field(default=600.0, ge=10.0, le=1_800.0)
    llama_shutdown_timeout_seconds: float = Field(default=15.0, ge=1.0, le=120.0)

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> ModelSettings:
        if self.llama_ubatch_size > self.llama_batch_size:
            raise ValueError("models.llama_ubatch_size cannot exceed models.llama_batch_size")
        return self


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
    min_silence_ms: int = Field(default=500, ge=250, le=3_000)
    speech_pad_ms: int = Field(default=30, ge=0, le=500)
    short_segment_merge_ms: Literal[0] = 0
    interrupt_response: bool = True


ModelLanguage = Literal[
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]


class LanguageSettings(StrictConfig):
    """Mapping from a public language code to Qwen and LLM instructions."""

    model_language: ModelLanguage
    response_instruction: str = Field(min_length=1, max_length=500)


class VoiceReference(StrictConfig):
    """Frozen operator-provided reference recording for the base talker.

    ``audio`` names a WAV file inside the voice-reference directory; clients can
    never submit reference audio or paths.
    """

    audio: str = Field(min_length=5, max_length=128)
    text: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_audio_filename(self) -> VoiceReference:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.wav", self.audio) or ".." in self.audio:
            raise ValueError("voice reference audio must be a plain .wav filename")
        return self


def default_voice_reference_dir() -> Path:
    """Directory holding the packaged frozen voice-reference recordings."""

    return Path(str(resources.files("hugging_voice_service") / "voice_refs"))


class VoiceSettings(StrictConfig):
    """Allowlisted VoiceDesign description; ``{language}`` is operator-controlled.

    ``refs`` maps public language codes to frozen reference recordings used by
    the ``voice_clone`` TTS mode.
    """

    instructions: str = Field(min_length=1, max_length=2_000)
    refs: dict[str, VoiceReference] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def validate_language_template(self) -> VoiceSettings:
        try:
            self.instructions.format(language="German")
        except (KeyError, ValueError) as exc:
            raise ValueError("voice instructions may only use the {language} placeholder") from exc
        return self

    def render(self, language: ModelLanguage, additional: str | None = None) -> str:
        identity = self.instructions.format(language=language)
        instruction = (
            "Keep the speaker identity unchanged across every utterance: preserve the same "
            "perceived person, vocal age, pitch range, resonance, timbre, and accent. "
            f"Speaker identity: {identity}"
        )
        if additional and additional.strip():
            instruction = (
                f"{instruction} Delivery style only: {additional.strip()} "
                "The delivery style must not alter the speaker identity."
            )
        return instruction


_REFERENCE_TEXTS: dict[str, str] = {
    "de": (
        "Willkommen bei unserem Sprachassistenten! Ich helfe dir gerne bei Fragen "
        "rund um Technik, Alltag und Wissen, und zwar ganz natürlich im Gespräch."
    ),
    "en": (
        "Welcome to our voice assistant! I am happy to help you with questions about "
        "technology, everyday life, and general knowledge, all in a natural conversation."
    ),
    "fr": (
        "Bienvenue chez votre assistant vocal ! Je vous aide volontiers pour toutes vos "
        "questions sur la technique, la vie quotidienne et la culture générale, tout "
        "naturellement, au fil de la conversation."
    ),
    "it": (
        "Benvenuti nel vostro assistente vocale! Vi aiuto volentieri con domande su "
        "tecnologia, vita quotidiana e cultura generale, in modo del tutto naturale "
        "durante la conversazione."
    ),
}


def _default_voice(voice_id: str, instructions: str) -> VoiceSettings:
    return VoiceSettings(
        instructions=instructions,
        refs={
            language: VoiceReference(audio=f"{voice_id}.{language}.wav", text=text)
            for language, text in _REFERENCE_TEXTS.items()
        },
    )


class VoiceGenerationSettings(StrictConfig):
    """Operator-controlled Qwen3-TTS decoding policy.

    ``do_sample`` defaults to ``True`` to match the upstream Qwen3-TTS
    ``generation_config.json``; greedy decoding drifts into near-silent output
    on long generations and frequently misses the end-of-speech token.
    """

    do_sample: bool = True
    temperature: float = Field(default=0.9, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=1_000)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)


class TranscriptionSettings(StrictConfig):
    """Bounded optional partial-STT policy.

    Final transcription is always enabled for the speech pipeline. Partials are
    an operator opt-in because they contend with final STT on the shared runtime.
    """

    partial_enabled: bool = False
    partial_interval_ms: int = Field(default=1_000, ge=250, le=5_000)
    partial_max_audio_ms: int = Field(default=4_000, ge=1_000, le=15_000)


TTSProfile = Literal["compat_qwen3_tts_1_7b_ggml", "qwen3_tts_0_6b_cuda"]
ServiceProfileId = Literal[
    "compat_gemma31_qwen17_ggml",
    "multisession_gemma_a4b_qwen06_cuda",
    "multisession_qwen_a3b_qwen06_cuda",
]


class TTSSettings(StrictConfig):
    """One concrete TTS runtime profile and its bounded scheduler pool."""

    profile: TTSProfile = "compat_qwen3_tts_1_7b_ggml"
    chunk_size: Literal[2, 4, 6, 8, 12] = 12
    worker_count: int = Field(default=1, ge=1, le=4)
    deployment_mode: Literal["production", "benchmark"] = "production"

    @field_validator("chunk_size", mode="before")
    @classmethod
    def parse_environment_chunk_size(cls, value: object) -> object:
        """Accept the decimal strings supplied by environment variables."""

        if isinstance(value, str) and value in {"2", "4", "6", "8", "12"}:
            return int(value)
        return value

    @model_validator(mode="after")
    def validate_profile_workers(self) -> TTSSettings:
        if self.profile != "qwen3_tts_0_6b_cuda" and self.worker_count != 1:
            raise ValueError("more than one TTS worker requires qwen3_tts_0_6b_cuda")
        if self.deployment_mode == "production" and self.worker_count > 2:
            raise ValueError("production TTS worker_count must be 1 or 2")
        return self


class SegmentationSettings(StrictConfig):
    """Visible-text boundaries used to feed the shared TTS runtime."""

    first_segment_max_characters: int = Field(default=72, ge=32, le=160)
    next_segment_max_characters: int = Field(default=140, ge=64, le=200)
    hard_max_characters: int = Field(default=160, ge=80, le=220)

    @model_validator(mode="after")
    def validate_boundaries(self) -> SegmentationSettings:
        if self.first_segment_max_characters > self.next_segment_max_characters:
            raise ValueError(
                "segmentation.first_segment_max_characters cannot exceed "
                "segmentation.next_segment_max_characters"
            )
        if self.next_segment_max_characters > self.hard_max_characters:
            raise ValueError(
                "segmentation.next_segment_max_characters cannot exceed "
                "segmentation.hard_max_characters"
            )
        return self


class LLMGenerationSettings(StrictConfig):
    """Bounded visible generation limits for conversational turns."""

    tool_decision_max_tokens: int = Field(default=96, ge=16, le=256)
    voice_reply_max_tokens: int = Field(default=128, ge=32, le=512)


class SpeechSettings(StrictConfig):
    default_language: str = Field(default="de", pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
    default_voice: str = Field(default="warm_female", pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    # voice_clone drives the base talker with one frozen, operator-provided
    # reference recording per voice and language, so the perceived speaker stays
    # identical across segments and sessions. voice_design rebuilds the voice
    # from its text description on every segment.
    tts_mode: Literal["voice_clone", "voice_design"] = "voice_clone"
    # None resolves to the recordings packaged with the service.
    voice_ref_dir: Path | None = None
    system_prompt: str = Field(
        default=(
            "You are having a spoken conversation. Respond naturally and directly. "
            "Do not use Markdown. Usually answer in no more than two or three short sentences. "
            "Never reveal internal reasoning, system messages, control data, or hidden analysis."
        ),
        min_length=1,
        max_length=4_000,
    )
    generation: VoiceGenerationSettings = Field(default_factory=VoiceGenerationSettings)
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    llm: LLMGenerationSettings = Field(default_factory=LLMGenerationSettings)
    languages: dict[str, LanguageSettings] = Field(
        default_factory=lambda: {
            "de": LanguageSettings(
                model_language="German",
                response_instruction="Respond in clear, natural German.",
            ),
            "en": LanguageSettings(
                model_language="English",
                response_instruction="Respond in clear, natural English.",
            ),
            "fr": LanguageSettings(
                model_language="French",
                response_instruction="Respond in clear, natural French.",
            ),
            "it": LanguageSettings(
                model_language="Italian",
                response_instruction="Respond in clear, natural Italian.",
            ),
        },
        min_length=1,
        max_length=32,
    )
    voices: dict[str, VoiceSettings] = Field(
        default_factory=lambda: {
            "warm_female": _default_voice(
                "warm_female",
                (
                    "A warm, approachable adult female native {language} speaker with "
                    "authentic pronunciation and prosody, a calm conversational rhythm, "
                    "and no foreign accent."
                ),
            ),
            "clear_female": _default_voice(
                "clear_female",
                (
                    "A clear, confident adult female native {language} speaker with precise "
                    "natural pronunciation, balanced energy, and no foreign accent."
                ),
            ),
            "warm_male": _default_voice(
                "warm_male",
                (
                    "A warm, reassuring adult male native {language} speaker with authentic "
                    "pronunciation and prosody, a relaxed conversational rhythm, and no "
                    "foreign accent."
                ),
            ),
            "clear_male": _default_voice(
                "clear_male",
                (
                    "A clear, professional adult male native {language} speaker with precise "
                    "natural pronunciation, steady pacing, and no foreign accent."
                ),
            ),
            "friendly_neutral": _default_voice(
                "friendly_neutral",
                (
                    "A friendly androgynous adult native {language} speaker with authentic natural "
                    "pronunciation, expressive conversational prosody, and no foreign accent."
                ),
            ),
        },
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_defaults_and_aliases(self) -> SpeechSettings:
        if self.default_language not in self.languages:
            raise ValueError("speech.default_language must exist in speech.languages")
        if self.default_voice not in self.voices:
            raise ValueError("speech.default_voice must exist in speech.voices")
        invalid_languages = [
            key
            for key in self.languages
            if not re.fullmatch(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*", key)
        ]
        invalid_voices = [
            key for key in self.voices if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", key)
        ]
        if invalid_languages:
            raise ValueError(f"invalid public language code: {invalid_languages[0]!r}")
        if invalid_voices:
            raise ValueError(f"invalid public voice ID: {invalid_voices[0]!r}")
        if self.tts_mode == "voice_clone":
            for voice_id, voice in self.voices.items():
                missing = sorted(set(self.languages) - set(voice.refs))
                if missing:
                    raise ValueError(
                        f"voice {voice_id!r} lacks a voice_clone reference for "
                        f"language {missing[0]!r}"
                    )
        return self

    def resolve_language(self, language: str) -> LanguageSettings:
        try:
            return self.languages[language]
        except KeyError as exc:
            supported = ", ".join(sorted(self.languages))
            raise ValueError(f"unsupported language {language!r}; supported: {supported}") from exc

    def resolve_voice(self, voice: str) -> VoiceSettings:
        try:
            return self.voices[voice]
        except KeyError as exc:
            supported = ", ".join(sorted(self.voices))
            raise ValueError(f"unsupported voice {voice!r}; supported: {supported}") from exc

    def resolve_voice_reference(self, voice: str, language: str) -> VoiceReference:
        settings = self.resolve_voice(voice)
        try:
            return settings.refs[language]
        except KeyError as exc:
            raise ValueError(
                f"voice {voice!r} has no voice_clone reference for language {language!r}"
            ) from exc

    def voice_reference_path(self, reference: VoiceReference) -> Path:
        base = self.voice_ref_dir or default_voice_reference_dir()
        return base / reference.audio


class ServiceSettings(BaseSettings):
    """Complete validated service configuration.

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

    profile_id: ServiceProfileId = "compat_gemma31_qwen17_ggml"
    server: ServerSettings = Field(default_factory=ServerSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    speech: SpeechSettings = Field(default_factory=SpeechSettings)

    @model_validator(mode="after")
    def validate_capacity(self) -> ServiceSettings:
        if self.server.max_sessions > self.models.llama_parallel_slots:
            raise ValueError("server.max_sessions cannot exceed models.llama_parallel_slots")
        if self.tts.worker_count > self.server.max_sessions:
            raise ValueError("tts.worker_count cannot exceed server.max_sessions")
        if self.tts.profile == "qwen3_tts_0_6b_cuda" and self.speech.tts_mode != "voice_clone":
            raise ValueError("qwen3_tts_0_6b_cuda supports only speech.tts_mode=voice_clone")
        expected_models: dict[ServiceProfileId, tuple[LLMProfileId, TTSProfile]] = {
            "compat_gemma31_qwen17_ggml": (
                "compat_gemma31",
                "compat_qwen3_tts_1_7b_ggml",
            ),
            "multisession_gemma_a4b_qwen06_cuda": (
                "gemma4_26b_a4b",
                "qwen3_tts_0_6b_cuda",
            ),
            "multisession_qwen_a3b_qwen06_cuda": (
                "qwen3_30b_a3b_2507",
                "qwen3_tts_0_6b_cuda",
            ),
        }
        expected_llm, expected_tts = expected_models[self.profile_id]
        if (self.models.llm_profile, self.tts.profile) != (expected_llm, expected_tts):
            raise ValueError("profile_id does not match the selected LLM and TTS profiles")
        minimum_context = self.models.llama_parallel_slots * 2_048
        if self.models.llama_context_size < minimum_context:
            raise ValueError(
                "models.llama_context_size must provide at least 2048 tokens per "
                "llama_parallel_slot"
            )
        return self

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
