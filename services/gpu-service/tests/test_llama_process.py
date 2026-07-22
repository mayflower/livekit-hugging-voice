from pathlib import Path

import pytest
from hugging_voice_service.llama_process import LlamaProcess, LlamaProcessError, LlamaProcessState


def test_llama_command_is_loopback_local_file_two_slots_and_no_hub(tmp_path: Path) -> None:
    process = LlamaProcess(
        binary=tmp_path / "llama-server",
        model=tmp_path / "gemma.gguf",
        parallel_slots=2,
    )
    command = process.command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--parallel") + 1] == "2"
    assert command[command.index("--ctx-size") + 1] == "32768"
    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert "-hf" not in command
    assert "--hf-repo" not in command
    assert command[command.index("--reasoning-format") + 1] == "deepseek"


def test_llama_rejects_any_slot_count_other_than_two(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        LlamaProcess(binary=tmp_path / "server", model=tmp_path / "model", parallel_slots=1)


@pytest.mark.asyncio
async def test_missing_llama_binary_fails_before_process_start(tmp_path: Path) -> None:
    process = LlamaProcess(binary=tmp_path / "missing", model=tmp_path / "missing.gguf")
    with pytest.raises(LlamaProcessError, match="missing or not executable"):
        await process.start()
    assert process.state is LlamaProcessState.STOPPED
