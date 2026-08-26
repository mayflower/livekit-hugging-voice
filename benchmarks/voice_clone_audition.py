#!/usr/bin/env python3
"""Render every voice_clone profile through the production synthesis path.

``voice_audition.py`` renders the VoiceDesign *descriptions*. That is the wrong
question for a deployment running ``speech.tts_mode: voice_clone``, where the
description never reaches the model and the frozen recording defines the
speaker. This script calls the same ``generate_voice_clone_streaming`` the
service calls, with the packaged references and the configured decoding
parameters, so what you hear is what a caller hears.

It matters most for a voice whose recording came from outside: a clip validated
against one talker on one backend is evidence, not proof, for another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from faster_qwen3_tts import GGMLQwen3TTS
from hugging_voice_service.config import load_settings
from hugging_voice_service.model_manifest import LockedModel, load_lock, verify_lock

# Reused verbatim from voice_audition.py so the two auditions are comparable
# by ear rather than only by construction.
from voice_audition import SENTENCES, locked_path, pcm16, write_wav

TALKER = "qwen-talker-1.7b-base-BF16.gguf"
TOKENIZER = "qwen-tokenizer-12hz-BF16.gguf"


def _resolve_qwen(lock: Any) -> LockedModel:
    for model in lock.models:
        if any(item.path == TALKER for item in model.files):
            return model
    raise RuntimeError(
        f"no locked model provides {TALKER}; the base talker is what voice_clone uses"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path(".models"))
    parser.add_argument("--lock", type=Path, default=Path("models/manifest.lock.json"))
    parser.add_argument(
        "--config", type=Path, default=Path("services/gpu-service/config/default.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--only-voice",
        help="render just this voice (the usual case when auditioning a new one)",
    )
    parser.add_argument(
        "--only-language",
        help="render just this language",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("voice audition requires an NVIDIA GPU; CPU fallback is disabled")

    settings = load_settings(args.config)
    if settings.speech.tts_mode != "voice_clone":
        raise RuntimeError(
            f"config uses tts_mode={settings.speech.tts_mode!r}; "
            "use voice_audition.py for voice_design"
        )
    if args.only_voice and args.only_voice not in settings.speech.voices:
        parser.error(f"unknown voice {args.only_voice!r}")
    if args.only_language and args.only_language not in settings.speech.languages:
        parser.error(f"unknown language {args.only_language!r}")

    lock = load_lock(args.lock)
    verify_lock(lock, args.model_root)
    qwen = _resolve_qwen(lock)
    model = GGMLQwen3TTS.from_gguf(
        locked_path(args.model_root, qwen, TALKER),
        locked_path(args.model_root, qwen, TOKENIZER),
        use_fa=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError("audition output directory must be empty")

    generation = settings.speech.generation
    artifacts: list[dict[str, Any]] = []
    for language_id, language in settings.speech.languages.items():
        if args.only_language and language_id != args.only_language:
            continue
        source_text = " ".join(SENTENCES[language_id])
        for voice_id in settings.speech.voices:
            if args.only_voice and voice_id != args.only_voice:
                continue
            reference = settings.speech.resolve_voice_reference(voice_id, language_id)
            recording = settings.speech.voice_reference_path(reference)
            chunks: list[Any] = []
            sample_rate: int | None = None
            for chunk, rate, _timing in model.generate_voice_clone_streaming(
                text=source_text,
                language=language.model_language,
                ref_audio=str(recording),
                ref_text=reference.text,
                # The deployed TTS profile's chunk size, not a default: it is
                # what shapes streaming granularity in production.
                chunk_size=settings.tts.chunk_size,
                max_new_tokens=2_048,
                do_sample=generation.do_sample,
                temperature=generation.temperature,
                top_k=generation.top_k,
                top_p=generation.top_p,
                repetition_penalty=generation.repetition_penalty,
            ):
                if sample_rate is not None and rate != sample_rate:
                    raise RuntimeError(f"sample rate changed during {language_id}/{voice_id}")
                sample_rate = rate
                chunks.append(chunk)
            if not chunks or sample_rate is None:
                raise RuntimeError(f"Qwen produced no audio for {language_id}/{voice_id}")
            audio = pcm16(chunks)
            output = args.output_dir / f"{language_id}-{voice_id}.wav"
            write_wav(output, audio, sample_rate)
            artifacts.append(
                {
                    "language": language_id,
                    "voice": voice_id,
                    "reference": reference.audio,
                    "reference_text": reference.text,
                    "file": output.name,
                    "sample_rate": sample_rate,
                    "duration_seconds": len(audio) / 2 / sample_rate,
                    "sha256": hashlib.sha256(audio).hexdigest(),
                }
            )
            print(f"{language_id}/{voice_id}: {artifacts[-1]['duration_seconds']:.2f}s")

    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "host": platform.node(),
                "mode": "voice_clone",
                "talker": TALKER,
                "tts_profile": settings.tts.profile,
                "chunk_size": settings.tts.chunk_size,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
