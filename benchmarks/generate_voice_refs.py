#!/usr/bin/env python3
"""Render the frozen voice_clone reference recordings on a real GPU.

For every configured voice and language this renders candidate takes of the
voice's reference transcript with the VoiceDesign talker, keeps the best take
that passes the acoustic checks, and writes it as ``<voice>.<language>.wav``.
The chosen recordings are meant to be reviewed by ear and then committed as
package data; they define the frozen speaker identity of the voice_clone mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from faster_qwen3_tts import GGMLQwen3TTS
from hugging_voice_service.config import load_settings
from hugging_voice_service.model_manifest import LockedModel, load_lock, verify_lock

MIN_DURATION_SECONDS = 4.0
MAX_DURATION_SECONDS = 14.0
MIN_MEAN_DBFS = -30.0
MAX_HALVES_DRIFT_DB = 6.0
MAX_INTERNAL_SILENCE_SECONDS = 0.8


def locked_path(root: Path, model: LockedModel, name: str) -> Path:
    locked = next((item for item in model.files if item.path == name), None)
    if locked is None:
        raise RuntimeError(f"lock for {model.id} does not contain {name}")
    return root / model.id / locked.path


@dataclass(frozen=True)
class TakeReport:
    duration_seconds: float
    mean_dbfs: float
    halves_drift_db: float
    longest_internal_silence_seconds: float
    failures: tuple[str, ...]


def _dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-9))


def assess_take(audio: np.ndarray, sample_rate: int) -> TakeReport:
    duration = audio.size / sample_rate
    voiced = np.flatnonzero(np.abs(audio) > 1e-3)
    trimmed = audio[voiced[0] : voiced[-1] + 1] if voiced.size else audio
    mean_dbfs = _dbfs(trimmed)
    half = trimmed.size // 2
    drift = abs(_dbfs(trimmed[:half]) - _dbfs(trimmed[half:])) if half else 0.0
    window = max(sample_rate // 10, 1)
    silent = 0
    longest_silence = 0
    for start in range(0, trimmed.size, window):
        if _dbfs(trimmed[start : start + window]) < -45.0:
            silent += 1
            longest_silence = max(longest_silence, silent)
        else:
            silent = 0
    longest_silence_seconds = longest_silence * window / sample_rate
    failures = []
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        failures.append(f"duration {duration:.1f}s outside bounds")
    if mean_dbfs < MIN_MEAN_DBFS:
        failures.append(f"mean level {mean_dbfs:.1f} dBFS too quiet")
    if drift > MAX_HALVES_DRIFT_DB:
        failures.append(f"level drift {drift:.1f} dB between halves")
    if longest_silence_seconds > MAX_INTERNAL_SILENCE_SECONDS:
        failures.append(f"internal silence {longest_silence_seconds:.1f}s too long")
    return TakeReport(
        duration_seconds=duration,
        mean_dbfs=mean_dbfs,
        halves_drift_db=drift,
        longest_internal_silence_seconds=longest_silence_seconds,
        failures=tuple(failures),
    )


def render_take(
    model: GGMLQwen3TTS,
    *,
    text: str,
    model_language: str,
    instruction: str,
    generation: Any,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    for chunk, rate, _timing in model.generate_voice_design_streaming(
        text=text,
        language=model_language,
        instruct=instruction,
        chunk_size=12,
        max_new_tokens=2_048,
        do_sample=generation.do_sample,
        temperature=generation.temperature,
        top_k=generation.top_k,
        top_p=generation.top_p,
        repetition_penalty=generation.repetition_penalty,
    ):
        if sample_rate is not None and rate != sample_rate:
            raise RuntimeError("sample rate changed inside one take")
        sample_rate = rate
        chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    if not chunks or sample_rate is None:
        raise RuntimeError("Qwen produced no audio for a reference take")
    audio = np.concatenate(chunks)
    if not np.all(np.isfinite(audio)):
        raise RuntimeError("Qwen returned non-finite reference audio")
    return audio, sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(pcm)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path(".models"))
    parser.add_argument("--lock", type=Path, default=Path("models/manifest.lock.json"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("services/gpu-service/config/default.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--takes", type=int, default=4)
    parser.add_argument("--only-voice", default=None)
    parser.add_argument("--only-language", default=None)
    args = parser.parse_args()
    if args.takes < 1:
        parser.error("--takes must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("reference rendering requires an NVIDIA GPU")
    lock = load_lock(args.lock)
    verify_lock(lock, args.model_root)
    qwen = next(
        (model for model in lock.models if model.id == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        None,
    )
    if qwen is None:
        raise RuntimeError("Qwen3-TTS is absent from the verified model lock")
    talker = locked_path(args.model_root, qwen, "qwen-talker-1.7b-voicedesign-BF16.gguf")
    tokenizer = locked_path(args.model_root, qwen, "qwen-tokenizer-12hz-BF16.gguf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = GGMLQwen3TTS.from_gguf(talker, tokenizer, use_fa=True)
    settings = load_settings(args.config)
    if args.only_voice and args.only_voice not in settings.speech.voices:
        parser.error(f"unknown voice {args.only_voice!r}")
    if args.only_language and args.only_language not in settings.speech.languages:
        parser.error(f"unknown language {args.only_language!r}")
    generation = settings.speech.generation
    artifacts: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for voice_id, voice in settings.speech.voices.items():
        if args.only_voice and voice_id != args.only_voice:
            continue
        for language_id, language in settings.speech.languages.items():
            if args.only_language and language_id != args.only_language:
                continue
            reference = voice.refs.get(language_id)
            if reference is None:
                raise RuntimeError(f"voice {voice_id} has no reference text for {language_id}")
            instruction = voice.render(language.model_language)
            best: tuple[int, np.ndarray, int, TakeReport] | None = None
            reports: list[dict[str, Any]] = []
            for take in range(args.takes):
                audio, sample_rate = render_take(
                    model,
                    text=reference.text,
                    model_language=language.model_language,
                    instruction=instruction,
                    generation=generation,
                )
                report = assess_take(audio, sample_rate)
                reports.append(
                    {
                        "take": take,
                        "duration_seconds": round(report.duration_seconds, 2),
                        "mean_dbfs": round(report.mean_dbfs, 1),
                        "halves_drift_db": round(report.halves_drift_db, 1),
                        "longest_internal_silence_seconds": round(
                            report.longest_internal_silence_seconds, 2
                        ),
                        "failures": list(report.failures),
                    }
                )
                if not report.failures and (
                    best is None or report.halves_drift_db < best[3].halves_drift_db
                ):
                    best = (take, audio, sample_rate, report)
            name = f"{voice_id}.{language_id}.wav"
            if best is None:
                unresolved.append(name)
                print(f"FAILED {name}: no take passed the acoustic checks")
                continue
            chosen_take, audio, sample_rate, report = best
            output = args.output_dir / name
            write_wav(output, audio, sample_rate)
            print(
                f"wrote {name}: {report.duration_seconds:.1f}s "
                f"mean {report.mean_dbfs:.1f} dBFS drift {report.halves_drift_db:.1f} dB"
            )
            artifacts.append(
                {
                    "voice": voice_id,
                    "language": language_id,
                    "file": name,
                    "text": reference.text,
                    "instruction": instruction,
                    "sample_rate": sample_rate,
                    "chosen_take": chosen_take,
                    "takes": reports,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            )
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        # Only the device name: the metadata ships as package data, so it must
        # not carry host names, GPU UUIDs, or other machine identifiers.
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "qwen_revision": qwen.revision,
        "artifacts": artifacts,
        "review_note": (
            "Listen to every recording before committing it as package data; "
            "these files freeze the public speaker identities."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if unresolved:
        print(f"{len(unresolved)} recordings unresolved: {', '.join(unresolved)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
