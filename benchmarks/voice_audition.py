#!/usr/bin/env python3
"""Render the four fixed Qwen speakers with identical German material on a real GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from faster_qwen3_tts import GGMLQwen3TTS
from hugging_voice_service.model_manifest import LockedModel, load_lock, verify_lock

SPEAKERS = ("Aiden", "Ryan", "Serena", "Sohee")
SENTENCES = (
    "Guten Morgen, wie kann ich Ihnen heute helfen?",
    "Der nächste Zug nach München fährt um siebzehn Uhr dreißig.",
    "Natürlichkeit, Verständlichkeit und ruhige Betonung sind für diesen Test entscheidend.",
)
INSTRUCTION = (
    "Sprich in klarem, ruhigem Hochdeutsch. Natürlich, freundlich und professionell. "
    "Keine übertriebene Emotionalität."
)


def locked_path(root: Path, model: LockedModel, name: str) -> Path:
    locked = next((item for item in model.files if item.path == name), None)
    if locked is None:
        raise RuntimeError(f"lock for {model.id} does not contain {name}")
    return root / model.id / locked.path


def pcm16(chunks: list[Any]) -> bytes:
    audio = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks])
    if not np.all(np.isfinite(audio)):
        raise RuntimeError("Qwen returned non-finite audition audio")
    return np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def write_wav(path: Path, audio: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(audio)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path(".models"))
    parser.add_argument("--lock", type=Path, default=Path("models/manifest.lock.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("voice audition requires an NVIDIA GPU; CPU fallback is disabled")
    lock = load_lock(args.lock)
    verify_lock(lock, args.model_root)
    qwen = next(
        (model for model in lock.models if model.id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        None,
    )
    if qwen is None:
        raise RuntimeError("Qwen3-TTS is absent from the verified model lock")
    talker = locked_path(args.model_root, qwen, "qwen-talker-1.7b-customvoice-BF16.gguf")
    tokenizer = locked_path(args.model_root, qwen, "qwen-tokenizer-12hz-BF16.gguf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError("audition output directory must be empty")
    model = GGMLQwen3TTS.from_gguf(talker, tokenizer, use_fa=True)
    source_text = " ".join(SENTENCES)
    artifacts: list[dict[str, Any]] = []
    for speaker in SPEAKERS:
        chunks: list[Any] = []
        sample_rate: int | None = None
        for chunk, rate, _timing in model.generate_custom_voice_streaming(
            text=source_text,
            speaker=speaker,
            language="German",
            instruct=INSTRUCTION,
            chunk_size=12,
            max_new_tokens=2_048,
        ):
            if sample_rate is not None and rate != sample_rate:
                raise RuntimeError(f"sample rate changed during {speaker} audition")
            sample_rate = rate
            chunks.append(chunk)
        if not chunks or sample_rate is None:
            raise RuntimeError(f"Qwen produced no audition audio for {speaker}")
        audio = pcm16(chunks)
        output = args.output_dir / f"{speaker.lower()}-german.wav"
        write_wav(output, audio, sample_rate)
        artifacts.append(
            {
                "speaker": speaker,
                "file": output.name,
                "sample_rate": sample_rate,
                "duration_seconds": len(audio) / 2 / sample_rate,
                "sha256": hashlib.sha256(audio).hexdigest(),
            }
        )
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "qwen_revision": qwen.revision,
        "sentences": SENTENCES,
        "instruction": INSTRUCTION,
        "artifacts": artifacts,
        "selection": None,
        "selection_note": (
            "Complete a blinded listening review; this script does not choose a voice."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
