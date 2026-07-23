from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

import pytest
from hugging_voice_service.llama_process import LlamaProcess, LlamaProcessError, LlamaProcessState


def write_llama_stub(tmp_path: Path, *, mode: str = "ready") -> tuple[Path, Path]:
    binary = tmp_path / "llama-server-stub"
    binary.write_text(
        f"""#!{sys.executable}
import argparse
import http.server
import json
import os
import signal
import socketserver
import threading

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()
mode = {mode!r}
if mode == "ignore_term":
    signal.signal(signal.SIGTERM, lambda *_: None)

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def send_json(self, status, value):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.send_json(200 if mode != "unready" else 503, {{}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if request["messages"][-1]["role"] == "tool":
            response = {{"choices": [{{"message": {{"content": "Das Ergebnis ist 42."}}}}]}}
            if mode == "exit_after_ready":
                threading.Timer(0.1, lambda: os._exit(9)).start()
        else:
            response = {{
                "choices": [{{
                    "message": {{
                        "tool_calls": [{{
                            "id": "call_stub",
                            "function": {{
                                "name": "add_numbers",
                                "arguments": "{{\\"a\\":19,\\"b\\":23}}"
                            }}
                        }}]
                    }}
                }}]
            }}
        self.send_json(200, response)

with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as server:
    server.serve_forever()
""",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "gemma.gguf"
    model.write_bytes(b"model")
    return binary, model


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
    assert process.state.value == LlamaProcessState.STOPPED.value


@pytest.mark.asyncio
async def test_llama_process_runs_real_readiness_probe_and_stops_cleanly(
    tmp_path: Path,
    unused_tcp_port: int,
) -> None:
    binary, model = write_llama_stub(tmp_path)
    process = LlamaProcess(
        binary=binary,
        model=model,
        port=unused_tcp_port,
        startup_timeout=2.0,
        shutdown_timeout=0.2,
    )

    await process.start()
    assert process.state is LlamaProcessState.READY
    assert process.returncode is None
    await process.stop()

    assert process.state.value == LlamaProcessState.STOPPED.value
    assert process.returncode is None


@pytest.mark.asyncio
async def test_llama_process_reports_unexpected_child_exit(
    tmp_path: Path,
    unused_tcp_port: int,
) -> None:
    binary, model = write_llama_stub(tmp_path, mode="exit_after_ready")
    process = LlamaProcess(
        binary=binary,
        model=model,
        port=unused_tcp_port,
        startup_timeout=2.0,
        shutdown_timeout=0.2,
    )
    await process.start()

    await asyncio.wait_for(process.failure_event.wait(), timeout=2.0)
    assert process.state is LlamaProcessState.FAILED
    assert process.failure == "llama-server exited unexpectedly with code 9"
    await process.stop()

    assert process.state.value == LlamaProcessState.STOPPED.value


@pytest.mark.asyncio
async def test_llama_process_kills_child_that_ignores_terminate(
    tmp_path: Path,
    unused_tcp_port: int,
) -> None:
    binary, model = write_llama_stub(tmp_path, mode="ignore_term")
    process = LlamaProcess(
        binary=binary,
        model=model,
        port=unused_tcp_port,
        startup_timeout=2.0,
        shutdown_timeout=0.05,
    )
    await process.start()

    started = asyncio.get_running_loop().time()
    await process.stop()

    assert process.state is LlamaProcessState.STOPPED
    assert asyncio.get_running_loop().time() - started < 1.0


@pytest.mark.asyncio
async def test_llama_startup_timeout_cleans_child_process(
    tmp_path: Path,
    unused_tcp_port: int,
) -> None:
    binary, model = write_llama_stub(tmp_path, mode="unready")
    process = LlamaProcess(
        binary=binary,
        model=model,
        port=unused_tcp_port,
        startup_timeout=0.1,
        shutdown_timeout=0.2,
    )

    with pytest.raises(LlamaProcessError, match="did not become ready"):
        await process.start()

    assert process.state is LlamaProcessState.FAILED
    assert process.returncode is None
