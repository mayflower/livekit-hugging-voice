from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import cast

import pytest

REPO_ROOT = Path(__file__).parents[3]


def assert_single_shared_model_loads(records: list[dict[str, object]]) -> None:
    after = next(
        record
        for record in records
        if record.get("record_type") == "prometheus" and record.get("phase") == "after"
    )
    metrics = cast(str, after["text"])
    for model in ("gemma", "parakeet", "qwen_tts"):
        pattern = rf'^hugging_voice_model_loads_total\{{model="{model}"\}} 1(?:\.0)?$'
        assert re.search(pattern, metrics, re.MULTILINE), model


def external_assets() -> tuple[Path, Path, Path]:
    if os.environ.get("HV_RUN_GPU_TESTS") != "1":
        pytest.skip("set HV_RUN_GPU_TESTS=1 to run external real-GPU tests")
    required = {
        "token": os.environ.get("HV_GPU_TOKEN_FILE"),
        "wav_a": os.environ.get("HV_GPU_WAV_A"),
        "wav_b": os.environ.get("HV_GPU_WAV_B"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        pytest.skip(f"external GPU assets are not configured: {', '.join(missing)}")
    paths = (
        Path(str(required["token"])),
        Path(str(required["wav_a"])),
        Path(str(required["wav_b"])),
    )
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        pytest.skip(f"external GPU assets are absent: {', '.join(absent)}")
    return paths


async def invoke_soak(
    tmp_path: Path,
    *,
    duration: float,
    cancel_every: int = 0,
    reconnect_every: int = 0,
) -> list[dict[str, object]]:
    token, wav_a, wav_b = external_assets()
    output = tmp_path / "raw.jsonl"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "two_session_soak.py"),
        "--service-url",
        os.environ.get("HV_GPU_SERVICE_URL", "http://127.0.0.1:8765"),
        "--token-file",
        str(token),
        "--wav-a",
        str(wav_a),
        "--wav-b",
        str(wav_b),
        "--duration",
        str(duration),
        "--cancel-every",
        str(cancel_every),
        "--reconnect-every",
        str(reconnect_every),
        "--output",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=REPO_ROOT,
    )
    stdout, stderr = await process.communicate()
    assert process.returncode == 0, (stdout + stderr).decode(errors="replace")
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.asyncio
async def test_external_two_session_e2e(tmp_path: Path) -> None:
    records = await invoke_soak(tmp_path, duration=0.01)
    turns = [record for record in records if record["record_type"] == "turn"]
    assert {record["session_label"] for record in turns} == {"alpha", "beta"}
    assert len({record["session_id"] for record in turns}) == 2
    assert all(cast(int, record["transcript_chars"]) > 0 for record in turns)
    assert all(cast(int, record["response_chars"]) > 0 for record in turns)
    assert all(
        "speech_stop_to_first_audio_frame_seconds" in cast(dict[str, float], record["metrics"])
        for record in turns
    )
    assert all(record["cross_session_leak"] is False for record in turns)
    assert {record["isolation_canary"] for record in turns} == {"ALPHAEINS", "BETAZWEI"}
    assert not [record for record in records if record["record_type"] == "error"]
    assert_single_shared_model_loads(records)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.asyncio
async def test_external_two_session_soak(tmp_path: Path) -> None:
    if os.environ.get("HV_RUN_GPU_SOAK") != "1":
        pytest.skip("set HV_RUN_GPU_SOAK=1 for the 30-minute two-session soak")
    records = await invoke_soak(
        tmp_path,
        duration=1_800.0,
        cancel_every=7,
        reconnect_every=11,
    )
    turns = [record for record in records if record["record_type"] == "turn"]
    assert {record["session_label"] for record in turns} == {"alpha", "beta"}
    assert all(record["cross_session_leak"] is False for record in turns)
    assert not [record for record in records if record["record_type"] == "error"]
    assert_single_shared_model_loads(records)
